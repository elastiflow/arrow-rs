// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowPrimitiveType, Float32Type, Float64Type, Int32Type, Int64Type, IntervalMonthDayNanoType,
    TimestampMicrosecondType,
};
use arrow_array::{
    Array, FixedSizeBinaryArray, GenericBinaryArray, LargeListArray, LargeStringArray, ListArray,
    PrimitiveArray, RecordBatch, StringArray, StructArray,
};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType::{Duration as ADuration, Interval};
use arrow_schema::{ArrowError, DataType, Field, IntervalUnit, TimeUnit};
use std::io::Write;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Default)]
/// Behavior knobs for the Avro encoder.
///
/// When `impala_mode` is `true`, optional/nullable values are encoded
/// as Avro unions with **null second** (`[T, "null"]`). When `false`
/// (default), we use **null first** (`["null", T]`).
pub struct EncoderOptions {
    pub(crate) impala_mode: bool,
}

/// Encode a single Avro-`long` using ZigZag + variable length, buffered.
///
/// Spec: https://avro.apache.org/docs/1.11.0/spec.html (Binary Encoding)
#[inline]
pub fn write_long<W: Write + ?Sized>(writer: &mut W, value: i64) -> Result<(), ArrowError> {
    let mut zz = ((value << 1) ^ (value >> 63)) as u64;
    // At most 10 bytes for 64-bit varint
    let mut buf = [0u8; 10];
    let mut i = 0;
    while (zz & !0x7F) != 0 {
        buf[i] = ((zz & 0x7F) as u8) | 0x80;
        i += 1;
        zz >>= 7;
    }
    buf[i] = (zz & 0x7F) as u8;
    i += 1;
    writer
        .write_all(&buf[..i])
        .map_err(|e| ArrowError::IoError(format!("write long: {e}"), e))
}

#[inline]
fn write_int<W: Write + ?Sized>(writer: &mut W, value: i32) -> Result<(), ArrowError> {
    write_long(writer, value as i64)
}

#[inline]
fn write_len_prefixed<W: Write + ?Sized>(writer: &mut W, bytes: &[u8]) -> Result<(), ArrowError> {
    write_long(writer, bytes.len() as i64)?;
    writer
        .write_all(bytes)
        .map_err(|e| ArrowError::IoError(format!("write bytes: {e}"), e))
}

#[inline]
fn write_bool<W: Write + ?Sized>(writer: &mut W, v: bool) -> Result<(), ArrowError> {
    writer
        .write_all(&[v as u8])
        .map_err(|e| ArrowError::IoError(format!("write bool: {e}"), e))
}

/// Write the union branch index for an optional field.
///
/// Branch index is 0-based per Avro unions:
/// - Null-first (default): null => 0, value => 1
/// - Null-second (Impala): value => 0, null => 1
///
/// Spec says to write an **int** value for union position.
/// See: https://avro.apache.org/docs/1.11.0/spec.html#Unions
#[inline]
fn write_optional_branch<W: Write + ?Sized>(
    writer: &mut W,
    is_null: bool,
    impala_mode: bool,
) -> Result<(), ArrowError> {
    let branch = if impala_mode == is_null { 1 } else { 0 };
    write_int(writer, branch)
}

/// Encode a `RecordBatch` in Avro binary format using **default options**.
pub fn encode_record_batch<W: Write>(batch: &RecordBatch, out: &mut W) -> Result<(), ArrowError> {
    encode_record_batch_with_options(batch, out, &EncoderOptions::default())
}

/// Encode a `RecordBatch` with explicit `EncoderOptions`.
pub fn encode_record_batch_with_options<W: Write>(
    batch: &RecordBatch,
    out: &mut W,
    opts: &EncoderOptions,
) -> Result<(), ArrowError> {
    let schema = batch.schema();
    let fields = schema.fields();

    // Build per-column encoders once.
    let mut encoders = fields
        .iter()
        .zip(batch.columns())
        .map(|(field_ref, array)| {
            let field: &Field = field_ref.as_ref();
            let enc = make_encoder(array.as_ref(), field)?;
            Ok::<_, ArrowError>((field.is_nullable(), enc))
        })
        .collect::<Result<Vec<_>, ArrowError>>()?;

    // Precompute the "value" branch for nullable fields (0 if Impala-mode, else 1)
    let value_branch: i32 = if opts.impala_mode { 0 } else { 1 };

    (0..batch.num_rows()).try_for_each(|row| {
        encoders.iter_mut().try_for_each(|(is_nullable, enc)| {
            if *is_nullable {
                if enc.has_nulls() {
                    let is_null = enc.is_null(row);
                    write_optional_branch(out, is_null, opts.impala_mode)?;
                    if is_null {
                        return Ok(());
                    }
                } else {
                    // Column is nullable but has no nulls; skip per-row null checks
                    write_int(out, value_branch)?;
                }
            }
            enc.encode(row, out)
        })
    })
}

/// An encoder + a null buffer for nullable fields.
///
/// This stores the inner `Encoder` **by value**. To break recursive size
/// cycles, the `Encoder` enum boxes only **recursive** variants (Struct/List).
pub struct NullableEncoder<'a> {
    encoder: Encoder<'a>,
    nulls: Option<NullBuffer>,
    has_nulls: bool,
}

impl<'a> NullableEncoder<'a> {
    /// Create a new nullable encoder, wrapping a non-null encoder and a null buffer.
    #[inline]
    fn new(encoder: Encoder<'a>, nulls: Option<NullBuffer>, has_nulls: bool) -> Self {
        Self {
            encoder,
            nulls,
            has_nulls,
        }
    }

    /// Encode the value at `idx`, assuming it's not-null.
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        self.encoder.encode(idx, out)
    }

    /// Check if the value at `idx` is null.
    #[inline]
    fn is_null(&self, idx: usize) -> bool {
        self.nulls.as_ref().is_some_and(|nulls| nulls.is_null(idx))
    }

    /// Whether this column contains any nulls at all.
    #[inline]
    fn has_nulls(&self) -> bool {
        self.has_nulls
    }
}

/// Creates an Avro encoder for the given `array`, using `field` to inspect
/// logical type / extension metadata so that the binary encoding matches the
/// emitted Avro schema (e.g., `fixed` vs `string`+`logicalType=uuid`,
/// and `fixed(12)`+`logicalType=duration` for MonthDayNano).
pub fn make_encoder<'a>(
    array: &'a dyn Array,
    field: &Field,
) -> Result<NullableEncoder<'a>, ArrowError> {
    let nulls = array.nulls().cloned();
    let has_nulls = array.null_count() > 0;

    let enc = match array.data_type() {
        DataType::Boolean => {
            let arr = array.as_boolean();
            NullableEncoder::new(Encoder::Boolean(BooleanEncoder(arr)), nulls, has_nulls)
        }
        DataType::Utf8 => {
            let arr = array.as_string::<i32>();
            NullableEncoder::new(Encoder::Utf8(Utf8Encoder(arr)), nulls, has_nulls)
        }
        DataType::LargeUtf8 => {
            let arr = array.as_string::<i64>();
            NullableEncoder::new(Encoder::Utf8Large(Utf8LargeEncoder(arr)), nulls, has_nulls)
        }
        DataType::Int32 => {
            let arr = array.as_primitive::<Int32Type>();
            NullableEncoder::new(Encoder::Int(IntEncoder(arr)), nulls, has_nulls)
        }
        DataType::Int64 => {
            let arr = array.as_primitive::<Int64Type>();
            NullableEncoder::new(Encoder::Long(LongEncoder(arr)), nulls, has_nulls)
        }
        DataType::Float32 => {
            let arr = array.as_primitive::<Float32Type>();
            NullableEncoder::new(Encoder::Float32(F32Encoder(arr)), nulls, has_nulls)
        }
        DataType::Float64 => {
            let arr = array.as_primitive::<Float64Type>();
            NullableEncoder::new(Encoder::Float64(F64Encoder(arr)), nulls, has_nulls)
        }
        DataType::Binary => {
            let arr = array.as_binary::<i32>();
            NullableEncoder::new(Encoder::Binary(BinaryEncoder(arr)), nulls, has_nulls)
        }
        DataType::LargeBinary => {
            let arr = array.as_binary::<i64>();
            NullableEncoder::new(Encoder::LargeBinary(BinaryEncoder(arr)), nulls, has_nulls)
        }
        // ---- FixedSizeBinary & UUID logical type -------------------------------------------
        DataType::FixedSizeBinary(len) => {
            // Decide between Avro `fixed` (raw bytes) and `uuid` logical string
            // based on Field metadata, mirroring schema generation rules.
            let arr = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| ArrowError::SchemaError("Expected FixedSizeBinaryArray".into()))?;

            let md = field.metadata();
            let is_uuid = md
                .get("logicalType")
                .is_some_and(|v| v == "uuid")
                // Honor common Arrow extension marker if present
                || (*len == 16
                && md.get("ARROW:extension:name")
                .is_some_and(|v| v == "uuid"));

            if is_uuid {
                if *len != 16 {
                    return Err(ArrowError::InvalidArgumentError(
                        "logicalType=uuid requires FixedSizeBinary(16)".into(),
                    ));
                }
                NullableEncoder::new(Encoder::Uuid(UuidEncoder(arr)), nulls, has_nulls)
            } else {
                NullableEncoder::new(Encoder::Fixed(FixedEncoder(arr)), nulls, has_nulls)
            }
        }
        // ---- Interval / Duration (Avro `duration`) -----------------------------------------
        Interval(IntervalUnit::MonthDayNano) => {
            let arr = array.as_primitive::<IntervalMonthDayNanoType>();
            NullableEncoder::new(
                Encoder::IntervalMonthDayNano(IntervalMonthDayNanoEncoder(arr)),
                nulls,
                has_nulls,
            )
        }
        Interval(unit) => {
            return Err(ArrowError::NotYetImplemented(format!(
                "Avro writer: Interval({unit:?}) is not supported; cast to Interval(MonthDayNano) to write Avro 'duration'"
            )));
        }
        ADuration(_) => {
            return Err(ArrowError::NotYetImplemented(
                "Avro writer: Arrow Duration(TimeUnit) has no standard Avro mapping; cast to Interval(MonthDayNano) to use Avro 'duration'".into(),
            ));
        }
        // ---- Lists / Structs ---------------------------------------------------------------
        DataType::List(_) => {
            let arr = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| ArrowError::SchemaError("Expected ListArray".into()))?;
            let enc = ListEncoder32::try_new(arr)?;
            NullableEncoder::new(Encoder::List(Box::new(enc)), nulls, has_nulls)
        }
        DataType::LargeList(_) => {
            let arr = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| ArrowError::SchemaError("Expected LargeListArray".into()))?;
            let enc = ListEncoder64::try_new(arr)?;
            NullableEncoder::new(Encoder::LargeList(Box::new(enc)), nulls, has_nulls)
        }
        DataType::Struct(_) => {
            let arr = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| ArrowError::SchemaError("Expected StructArray".into()))?;
            let enc = StructEncoder::try_new(arr)?;
            NullableEncoder::new(Encoder::Struct(Box::new(enc)), nulls, has_nulls)
        }
        // ---- Timestamps --------------------------------------------------------------------
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let arr = array.as_primitive::<TimestampMicrosecondType>();
            NullableEncoder::new(Encoder::Timestamp(LongEncoder(arr)), nulls, has_nulls)
        }
        // ---- Fallback ----------------------------------------------------------------------
        other => {
            return Err(ArrowError::NotYetImplemented(format!(
                "Unsupported data type for Avro encoding in slim build: {other:?}"
            )))
        }
    };
    Ok(enc)
}

enum Encoder<'a> {
    Boolean(BooleanEncoder<'a>),
    Int(IntEncoder<'a, Int32Type>),
    Long(LongEncoder<'a, Int64Type>),
    Timestamp(LongEncoder<'a, TimestampMicrosecondType>),
    Float32(F32Encoder<'a>),
    Float64(F64Encoder<'a>),
    Binary(BinaryEncoder<'a, i32>),
    LargeBinary(BinaryEncoder<'a, i64>),
    /// Avro `fixed` encoder (raw bytes, no length)
    Fixed(FixedEncoder<'a>),
    /// Avro `uuid` logical type encoder (string with RFC‑4122 hyphenated text)
    Uuid(UuidEncoder<'a>),
    /// Avro `duration` logical type (Arrow Interval(MonthDayNano)) encoder
    IntervalMonthDayNano(IntervalMonthDayNanoEncoder<'a>),
    Utf8(Utf8Encoder<'a>),
    Utf8Large(Utf8LargeEncoder<'a>),

    // Box only the recursive variants to keep the enum sized
    Struct(Box<StructEncoder<'a>>),
    List(Box<ListEncoder32<'a>>),
    LargeList(Box<ListEncoder64<'a>>),
}

impl<'a> Encoder<'a> {
    /// Encode the value at `idx`.
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        match self {
            Encoder::Boolean(e) => e.encode(idx, out),
            Encoder::Int(e) => e.encode(idx, out),
            Encoder::Long(e) => e.encode(idx, out),
            Encoder::Timestamp(e) => e.encode(idx, out),
            Encoder::Float32(e) => e.encode(idx, out),
            Encoder::Float64(e) => e.encode(idx, out),
            Encoder::Binary(e) => e.encode(idx, out),
            Encoder::LargeBinary(e) => e.encode(idx, out),
            Encoder::Fixed(e) => e.encode(idx, out),
            Encoder::Uuid(e) => e.encode(idx, out),
            Encoder::IntervalMonthDayNano(e) => e.encode(idx, out),
            Encoder::Utf8(e) => e.encode(idx, out),
            Encoder::Utf8Large(e) => e.encode(idx, out),
            Encoder::Struct(e) => e.encode(idx, out),
            Encoder::List(e) => e.encode(idx, out),
            Encoder::LargeList(e) => e.encode(idx, out),
        }
    }
}

struct BooleanEncoder<'a>(&'a arrow_array::BooleanArray);
impl BooleanEncoder<'_> {
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        write_bool(out, self.0.value(idx))
    }
}

/// Generic Avro `int` encoder for primitive arrays with `i32` native values.
struct IntEncoder<'a, P: ArrowPrimitiveType<Native = i32>>(&'a PrimitiveArray<P>);
impl<'a, P: ArrowPrimitiveType<Native = i32>> IntEncoder<'a, P> {
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        write_int(out, self.0.value(idx))
    }
}

/// Generic Avro `long` encoder for primitive arrays with `i64` native values.
struct LongEncoder<'a, P: ArrowPrimitiveType<Native = i64>>(&'a PrimitiveArray<P>);
impl<'a, P: ArrowPrimitiveType<Native = i64>> LongEncoder<'a, P> {
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        write_long(out, self.0.value(idx))
    }
}

/// Unified binary encoder generic over offset size (i32/i64).
/// Avro `bytes` are encoded as a length-prefixed block using a long for the length.
struct BinaryEncoder<'a, O: arrow_array::OffsetSizeTrait>(&'a GenericBinaryArray<O>);
impl<'a, O: arrow_array::OffsetSizeTrait> BinaryEncoder<'a, O> {
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        let v = self.0.value(idx);
        write_len_prefixed(out, v)
    }
}

/// Avro `fixed` encoder for Arrow `FixedSizeBinaryArray`.
/// Spec: a fixed is encoded as exactly `size` bytes, with no length prefix.
/// See: https://avro.apache.org/docs/1.11.0/spec.html#Fixed
struct FixedEncoder<'a>(&'a FixedSizeBinaryArray);
impl FixedEncoder<'_> {
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        let v = self.0.value(idx); // &[u8] of fixed width
        out.write_all(v)
            .map_err(|e| ArrowError::IoError(format!("write fixed bytes: {e}"), e))
    }
}

/// Avro UUID logical type encoder: Arrow FixedSizeBinary(16) → Avro string (UUID).
/// Spec: uuid is a logical type over string (RFC‑4122). We output hyphenated form.
/// See: https://avro.apache.org/docs/1.11.0/spec.html#UUID
struct UuidEncoder<'a>(&'a FixedSizeBinaryArray);
impl UuidEncoder<'_> {
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        let v = self.0.value(idx);
        if v.len() != 16 {
            return Err(ArrowError::InvalidArgumentError(
                "logicalType=uuid requires FixedSizeBinary(16)".into(),
            ));
        }
        let u = Uuid::from_slice(v)
            .map_err(|e| ArrowError::InvalidArgumentError(format!("Invalid UUID bytes: {e}")))?;
        // Format UUID into a fixed-size stack buffer to avoid allocation.
        // Hyphenated form is 36 bytes.
        let mut tmp = [0u8; uuid::fmt::Hyphenated::LENGTH];
        let s = u.hyphenated().encode_lower(&mut tmp);
        write_len_prefixed(out, s.as_bytes())
    }
}

/// Avro `duration` encoder for Arrow `Interval(IntervalUnit::MonthDayNano)`.
/// Spec: `duration` annotates Avro fixed(12) with three **little‑endian u32**:
/// months, days, milliseconds (no negatives).
/// See: https://avro.apache.org/docs/1.11.0/spec.html#Duration
struct IntervalMonthDayNanoEncoder<'a>(&'a PrimitiveArray<IntervalMonthDayNanoType>);
impl IntervalMonthDayNanoEncoder<'_> {
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        let native = self.0.value(idx);
        let (months, days, nanos) = IntervalMonthDayNanoType::to_parts(native);

        // Validation per Avro 'duration' constraints: unsigned components, ms granularity
        if months < 0 || days < 0 || nanos < 0 {
            return Err(ArrowError::InvalidArgumentError(
                "Avro 'duration' cannot encode negative months/days/nanoseconds".into(),
            ));
        }
        if nanos % 1_000_000 != 0 {
            return Err(ArrowError::InvalidArgumentError(
                "Avro 'duration' requires whole milliseconds; nanoseconds must be divisible by 1_000_000"
                    .into(),
            ));
        }

        let millis = nanos / 1_000_000;
        if millis > u32::MAX as i64 {
            return Err(ArrowError::InvalidArgumentError(
                "Avro 'duration' milliseconds exceed u32::MAX".into(),
            ));
        }

        let mut buf = [0u8; 12];
        buf[0..4].copy_from_slice(&(months as u32).to_le_bytes());
        buf[4..8].copy_from_slice(&(days as u32).to_le_bytes());
        buf[8..12].copy_from_slice(&((millis as u32).to_le_bytes()));
        out.write_all(&buf)
            .map_err(|e| ArrowError::IoError(format!("write duration: {e}"), e))
    }
}

struct F32Encoder<'a>(&'a arrow_array::Float32Array);
impl F32Encoder<'_> {
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        let bits = self.0.value(idx).to_bits();
        out.write_all(&bits.to_le_bytes())
            .map_err(|e| ArrowError::IoError(format!("write f32: {e}"), e))
    }
}

struct F64Encoder<'a>(&'a arrow_array::Float64Array);
impl F64Encoder<'_> {
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        // Avro double: 8 bytes, IEEE-754 little-endian
        let bits = self.0.value(idx).to_bits();
        out.write_all(&bits.to_le_bytes())
            .map_err(|e| ArrowError::IoError(format!("write f64: {e}"), e))
    }
}

/// Avro `string` encoder for Arrow `Utf8` (StringArray).
/// Spec: a string is encoded as a long followed by that many bytes of UTF‑8 data.
struct Utf8Encoder<'a>(&'a StringArray);
impl Utf8Encoder<'_> {
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        let s = self.0.value(idx); // &str
        write_len_prefixed(out, s.as_bytes())
    }
}

/// Avro `string` encoder for Arrow `LargeUtf8` (LargeStringArray).
struct Utf8LargeEncoder<'a>(&'a LargeStringArray);
impl Utf8LargeEncoder<'_> {
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        let s = self.0.value(idx); // &str
        write_len_prefixed(out, s.as_bytes())
    }
}

/// Avro `record` encoder for Arrow `StructArray`
/// For each field in schema order:
/// - If the field is nullable: write union branch (null-first: null=0, value=1) and skip if null
/// - Then encode the field's value using its child encoder
struct StructEncoder<'a> {
    fields: Vec<(bool, NullableEncoder<'a>)>,
}

impl<'a> StructEncoder<'a> {
    fn try_new(array: &'a StructArray) -> Result<Self, ArrowError> {
        let fields = match array.data_type() {
            DataType::Struct(fs) => fs,
            _ => return Err(ArrowError::SchemaError("Expected Struct".into())),
        };
        let mut encs = Vec::with_capacity(fields.len());
        for (f_ref, col) in fields.iter().zip(array.columns().iter()) {
            let f: &Field = f_ref.as_ref();
            let child = make_encoder(col.as_ref(), f)?;
            encs.push((f.is_nullable(), child));
        }
        Ok(Self { fields: encs })
    }

    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        for (is_nullable, enc) in self.fields.iter_mut() {
            if *is_nullable {
                let is_null = enc.is_null(idx);
                // Struct field unions use null-first encoding (existing behavior).
                write_optional_branch(out, is_null, false)?;
                if is_null {
                    continue;
                }
            }
            enc.encode(idx, out)?;
        }
        Ok(())
    }
}

/// Shared, allocation-free encode path for Arrow `ListArray` and `LargeListArray`.
///
/// Encodes a single list element given its `[start, end)` bounds in the (possibly sliced)
/// child `values()` array. Handles nullable list **items** by writing the union branch
/// per item when `items_nullable` is true.
///
/// This function purposely does **not** depend on the actual list type, keeping
/// the `ListEncoder32::encode` and `ListEncoder64::encode` bodies trivial.
#[inline]
fn encode_list_range<W: Write + ?Sized>(
    out: &mut W,
    start: usize,
    end: usize,
    values_offset: usize,
    items_nullable: bool,
    values_encoder: &mut NullableEncoder<'_>,
) -> Result<(), ArrowError> {
    let len = end.saturating_sub(start);
    if len == 0 {
        // Single zero-length block terminator per Avro spec
        write_long(out, 0)?;
        return Ok(());
    }
    // Emit a single block with item count, followed by the items, then the end marker.
    write_long(out, len as i64)?;
    // Iterate `[start, end)` translating to local indices into `values()`.
    for j in start..end {
        // Safety: Arrow guarantees `values_offset <= j` for slices; keep a debug check and
        // use saturating_sub in release to avoid UB if invariants are violated upstream.
        debug_assert!(
            j >= values_offset,
            "List values offset invariant violated: j < values_offset"
        );
        let j_local = j.saturating_sub(values_offset);
        if items_nullable {
            let is_null = values_encoder.is_null(j_local);
            // For list items we always use null-first unions (consistent with existing behavior).
            write_optional_branch(out, is_null, false)?;
            if is_null {
                continue;
            }
        }
        values_encoder.encode(j_local, out)?;
    }
    // End-of-array marker
    write_long(out, 0)?;
    Ok(())
}

struct ListEncoder<'a, O: arrow_array::OffsetSizeTrait> {
    list: &'a arrow_array::array::GenericListArray<O>,
    values: NullableEncoder<'a>,
    items_nullable: bool,
    values_offset: usize,
}

// Keep your public surface the same via type aliases:
type ListEncoder32<'a> = ListEncoder<'a, i32>;
type ListEncoder64<'a> = ListEncoder<'a, i64>;

impl<'a, O: arrow_array::OffsetSizeTrait> ListEncoder<'a, O> {
    fn try_new(list: &'a arrow_array::array::GenericListArray<O>) -> Result<Self, ArrowError> {
        // Item nullability is defined on the list's item Field and we also need the child Field
        // itself to select the appropriate encoder (e.g., UUID vs fixed).
        let (child_field, items_nullable) = match list.data_type() {
            DataType::List(field) => (field.as_ref(), field.is_nullable()),
            DataType::LargeList(field) => (field.as_ref(), field.is_nullable()),
            _ => {
                return Err(ArrowError::SchemaError(
                    "Expected List or LargeList for ListEncoder".into(),
                ))
            }
        };

        // Build the encoder for the child values() array using the child field metadata
        let values_enc = make_encoder(list.values().as_ref(), child_field)?;
        Ok(Self {
            list,
            values: values_enc,
            items_nullable,
            values_offset: list.values().offset(), // cache once
        })
    }

    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        let offsets = self.list.offsets();
        let start = offsets[idx].to_usize().ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("Error converting offset[{idx}] to usize"))
        })?;
        let end = offsets[idx + 1].to_usize().ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "Error converting offset[{}] to usize",
                idx + 1
            ))
        })?;
        encode_list_range(
            out,
            start,
            end,
            self.values_offset,
            self.items_nullable,
            &mut self.values,
        )
    }
}
