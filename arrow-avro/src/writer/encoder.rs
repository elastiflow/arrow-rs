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

use crate::codec::{AvroDataType, AvroField as CodecAvroField, Codec as AvroCodec, Nullability};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowPrimitiveType, Float32Type, Float64Type, Int32Type, Int64Type, IntervalMonthDayNanoType,
    TimestampMicrosecondType,
};
use arrow_array::{
    Array, Decimal128Array, Decimal256Array, Decimal32Array, Decimal64Array, DictionaryArray,
    FixedSizeBinaryArray, GenericBinaryArray, GenericStringArray, LargeListArray, ListArray,
    MapArray, PrimitiveArray, RecordBatch, StringArray, StructArray,
};
use arrow_buffer::NullBuffer;
use arrow_schema::{ArrowError, DataType, Field, IntervalUnit, Schema as ArrowSchema, TimeUnit};
use std::io::Write;
use std::sync::Arc;
use uuid::Uuid;

/// Optional plan reference passed to the unified encoder constructor.
type PlanRef<'p> = Option<&'p FieldPlan>;

/// Encode a single Avro-`long` using ZigZag + variable length, buffered.
///
/// Avro binary encoding uses variable-length zig-zag encoding for integral types.
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

/// Minimal two's-complement big-endian representation helper for Avro decimal (bytes).
///
/// For positive numbers, trim leading 0x00 while the next byte's MSB is 0.
/// For negative numbers, trim leading 0xFF while the next byte's MSB is 1.
/// The resulting slice still encodes the same signed value.
///
/// See Avro spec: decimal over `bytes` uses two's-complement big-endian
/// representation of the unscaled integer value. 1.11.1 specification.
#[inline]
fn minimal_twos_complement(be: &[u8]) -> &[u8] {
    if be.is_empty() {
        return be;
    }
    let mut i = 0usize;
    let sign = (be[0] & 0x80) != 0;
    while i + 1 < be.len() {
        let b = be[i];
        let next = be[i + 1];
        let trim_pos = !sign && b == 0x00 && (next & 0x80) == 0;
        let trim_neg = sign && b == 0xFF && (next & 0x80) != 0;
        if trim_pos || trim_neg {
            i += 1;
        } else {
            break;
        }
    }
    &be[i..]
}

/// Sign-extend (or validate/truncate) big-endian integer bytes to exactly `n` bytes.
///
/// If `src_be` is longer than `n`, ensure that dropped leading bytes are all sign bytes,
/// and that the MSB of the first kept byte matches the sign; otherwise return an overflow error.
/// If shorter than `n`, left-pad with the sign byte.
///
/// Used for Avro decimal over `fixed(N)`.
#[inline]
fn sign_extend_to_exact(src_be: &[u8], n: usize) -> Result<Vec<u8>, ArrowError> {
    let len = src_be.len();
    let sign_byte = if len > 0 && (src_be[0] & 0x80) != 0 {
        0xFF
    } else {
        0x00
    };
    if len == n {
        return Ok(src_be.to_vec());
    }
    if len > n {
        let extra = len - n;
        if src_be[..extra].iter().any(|&b| b != sign_byte) {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Decimal value with {} bytes cannot be represented in {} bytes without overflow",
                len, n
            )));
        }
        if n > 0 {
            let first_kept = src_be[extra];
            if ((first_kept ^ sign_byte) & 0x80) != 0 {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Decimal value with {} bytes cannot be represented in {} bytes without overflow",
                    len, n
                )));
            }
        }
        return Ok(src_be[extra..].to_vec());
    }
    let mut out = vec![sign_byte; n];
    out[n - len..].copy_from_slice(src_be);
    Ok(out)
}

/// Write the union branch index for an optional site with the specified `order`.
///
/// Branch index is 0‑based per Avro unions. We special-case 0 or 1 which
/// are single-byte varints: `0x00` (index 0) and `0x02` (index 1).
#[inline]
fn write_optional_index<W: Write + ?Sized>(
    writer: &mut W,
    is_null: bool,
    order: Nullability,
) -> Result<(), ArrowError> {
    // For NullFirst: null => 0x00, value => 0x02
    // For NullSecond: value => 0x00, null => 0x02
    let byte = match order {
        Nullability::NullFirst => {
            if is_null {
                0x00
            } else {
                0x02
            }
        }
        Nullability::NullSecond => {
            if is_null {
                0x02
            } else {
                0x00
            }
        }
    };
    writer
        .write_all(&[byte])
        .map_err(|e| ArrowError::IoError(format!("write union branch: {e}"), e))
}

/// Per‑site encoder plan for a field. This mirrors Avro structure so nested
/// optional branch order can be honored exactly as declared by the schema.
#[derive(Debug, Clone)]
enum FieldPlan {
    /// Non-nested scalar/logical type
    Scalar,
    /// Record/Struct with Avro‑ordered encodings
    Struct { encodings: Vec<StructFieldPlan> },
    /// Array with item‑site nullability and nested plan
    List {
        items_nullability: Option<Nullability>,
        item_plan: Box<FieldPlan>,
    },
    /// Avro map with value‑site nullability and nested plan
    Map {
        values_nullability: Option<Nullability>,
        value_plan: Box<FieldPlan>,
    },
    /// Avro decimal logical type (bytes or fixed). `size=None` => bytes(decimal), `Some(n)` => fixed(n)
    Decimal { size: Option<usize> },
    /// Avro enum; maps to Arrow Dictionary<Int32, Utf8> with dictionary values
    /// exactly equal and ordered as the Avro enum `symbols`.
    Enum { symbols: Arc<[String]> },
}

#[derive(Debug, Clone)]
struct StructFieldPlan {
    /// Encoding field name (Avro)
    name: String,
    /// Index of the encoding within the Arrow struct's Fields
    arrow_index: usize,
    /// Nullability/order for this encoding (None if not optional)
    nullability: Option<Nullability>,
    /// Nested plan for this encoding
    plan: FieldPlan,
}

#[derive(Debug, Clone)]
struct ColumnPlan {
    /// Index of the top‑level Arrow column that corresponds to this Avro field
    arrow_index: usize,
    /// Nullability/order for this column (None if not optional)
    nullability: Option<Nullability>,
    /// Nested plan for the field
    plan: FieldPlan,
}

/// Builder for `RecordEncoder` write plan
#[derive(Debug)]
pub struct RecordEncoderBuilder<'a> {
    avro_root: &'a CodecAvroField,
    arrow_schema: &'a ArrowSchema,
}

impl<'a> RecordEncoderBuilder<'a> {
    /// Create a new builder from the Avro root and Arrow schema.
    pub fn new(avro_root: &'a CodecAvroField, arrow_schema: &'a ArrowSchema) -> Self {
        Self {
            avro_root,
            arrow_schema,
        }
    }

    /// Build the `RecordEncoder` by walking the Avro **record** root in Avro order,
    /// resolving each field to an Arrow index by name.
    pub fn build(self) -> Result<RecordEncoder, ArrowError> {
        let avro_root_dt = self.avro_root.data_type();
        let avro_encodings = match avro_root_dt.codec() {
            AvroCodec::Struct(encodings) => encodings,
            _ => {
                return Err(ArrowError::SchemaError(
                    "Top-level Avro schema must be a record/struct".into(),
                ))
            }
        };
        let mut columns = Vec::with_capacity(avro_encodings.len());
        for avro_encoding in avro_encodings.iter() {
            let name = avro_encoding.name();
            let arrow_index = self.arrow_schema.index_of(name).map_err(|e| {
                ArrowError::SchemaError(format!("Schema mismatch for field '{name}': {e}"))
            })?;
            let arrow_field = self.arrow_schema.field(arrow_index);
            let plan = build_field_plan(avro_encoding.data_type(), arrow_field)?;
            columns.push(ColumnPlan {
                arrow_index,
                nullability: avro_encoding.data_type().nullability(),
                plan,
            });
        }
        Ok(RecordEncoder { columns })
    }
}

/// A pre-computed plan for encoding a `RecordBatch` to Avro.
///
/// Derived from an Avro schema and an Arrow schema. It maps
/// top-level Avro fields to Arrow columns and contains a nested encoding plan
/// for each column.
#[derive(Debug, Clone)]
pub struct RecordEncoder {
    columns: Vec<ColumnPlan>,
}

impl RecordEncoder {
    /// Prepare column encoders for a specific batch using the precomputed plan.
    fn prepare_for_batch<'a>(
        &'a self,
        batch: &'a RecordBatch,
    ) -> Result<Vec<ColumnEncoder<'a>>, ArrowError> {
        // bind schema to extend lifetime of `fields()` borrow
        let schema_binding = batch.schema();
        let fields = schema_binding.fields();
        let arrays = batch.columns();
        let mut out = Vec::with_capacity(self.columns.len());
        for col_plan in self.columns.iter() {
            let arrow_index = col_plan.arrow_index;
            let array = arrays.get(arrow_index).ok_or_else(|| {
                ArrowError::SchemaError(format!("Column index {arrow_index} out of range"))
            })?;
            let field = fields[arrow_index].as_ref();
            let enc = make_encoder(array.as_ref(), field, Some(&col_plan.plan))?;
            let pre = precomputed_union_value_branch(col_plan.nullability, enc.has_nulls());
            out.push(ColumnEncoder {
                nullability: col_plan.nullability,
                pre,
                enc,
            });
        }
        Ok(out)
    }

    /// Encode a `RecordBatch` using this encoder plan.
    ///
    /// Tip: Wrap `out` in a `std::io::BufWriter` to reduce the overhead of many small writes.
    pub fn encode_batch<W: Write>(
        &self,
        batch: &RecordBatch,
        out: &mut W,
    ) -> Result<(), ArrowError> {
        let mut cols = self.prepare_for_batch(batch)?;
        encode_rows_with_prefix_plan(batch.num_rows(), &mut cols, out, |_w, _row| Ok(()))
    }

    /// Encode a `RecordBatch` with a per-row single‑object `prefix`.
    pub fn encode_batch_single_object<W: Write>(
        &self,
        batch: &RecordBatch,
        out: &mut W,
        prefix: &[u8; 10],
    ) -> Result<(), ArrowError> {
        let mut cols = self.prepare_for_batch(batch)?;
        encode_rows_with_prefix_plan(batch.num_rows(), &mut cols, out, |w, _row| {
            w.write_all(prefix)
                .map_err(|e| ArrowError::IoError(format!("write single-object prefix: {e}"), e))
        })
    }
}

fn find_struct_encoding_index(fields: &arrow_schema::Fields, name: &str) -> Option<usize> {
    fields.iter().position(|f| f.name() == name)
}

#[inline]
fn find_map_value_field_index(fields: &arrow_schema::Fields) -> Option<usize> {
    // Prefer common Arrow field names; fall back to second encoding if exactly two
    find_struct_encoding_index(fields, "value")
        .or_else(|| find_struct_encoding_index(fields, "values"))
        .or_else(|| if fields.len() == 2 { Some(1) } else { None })
}

fn build_field_plan(avro_dt: &AvroDataType, arrow_field: &Field) -> Result<FieldPlan, ArrowError> {
    match avro_dt.codec() {
        AvroCodec::Struct(avro_encodings) => {
            let fields = match arrow_field.data_type() {
                DataType::Struct(fs) => fs,
                other => {
                    return Err(ArrowError::SchemaError(format!(
                        "Avro struct maps to Arrow Struct, found: {other:?}"
                    )))
                }
            };
            let mut encs = Vec::with_capacity(avro_encodings.len());
            for avro_encoding in avro_encodings.iter() {
                let name = avro_encoding.name().to_string();
                let idx = find_struct_encoding_index(fields, &name).ok_or_else(|| {
                    ArrowError::SchemaError(format!(
                        "Struct field '{name}' not present in Arrow field '{}'",
                        arrow_field.name()
                    ))
                })?;
                let arrow_field_in_struct = fields[idx].as_ref();
                let encoding_plan =
                    build_field_plan(avro_encoding.data_type(), arrow_field_in_struct)?;
                encs.push(StructFieldPlan {
                    name,
                    arrow_index: idx,
                    nullability: avro_encoding.data_type().nullability(),
                    plan: encoding_plan,
                });
            }
            Ok(FieldPlan::Struct { encodings: encs })
        }
        AvroCodec::List(items_dt) => {
            // Map Avro array -> Arrow List/LargeList. Recurse on the **item field** of the Arrow list.
            match arrow_field.data_type() {
                DataType::List(item_field_ref) => {
                    let item_field: &Field = item_field_ref.as_ref();
                    let item_plan = build_field_plan(items_dt.as_ref(), item_field)?;
                    Ok(FieldPlan::List {
                        items_nullability: items_dt.nullability(),
                        item_plan: Box::new(item_plan),
                    })
                }
                DataType::LargeList(item_field_ref) => {
                    let item_field: &Field = item_field_ref.as_ref();
                    let item_plan = build_field_plan(items_dt.as_ref(), item_field)?;
                    Ok(FieldPlan::List {
                        items_nullability: items_dt.nullability(),
                        item_plan: Box::new(item_plan),
                    })
                }
                other => Err(ArrowError::SchemaError(format!(
                    "Avro array maps to Arrow List/LargeList, found: {other:?}"
                ))),
            }
        }
        _ => Ok(FieldPlan::Scalar),
    }
}

/// Internal representation of a prepared column encoder for a record batch.
/// This caches per-column properties and the encoder itself.
///
/// `nullability` encodes whether the column is optional and, if so, the
/// **branch order** to use for union indices (NullFirst/NullSecond).
struct ColumnEncoder<'a> {
    nullability: Option<Nullability>,
    pre: Option<u8>,
    enc: NullableEncoder<'a>,
}

/// Encode `rows` rows by iterating columns and writing values for each row.
/// A `per_row_prefix` callback can inject data *before* each record (used by
/// the Avro single-object encoding to write the 10-byte prefix).
#[inline]
fn encode_rows_with_prefix_plan<W: Write>(
    rows: usize,
    cols: &mut [ColumnEncoder<'_>],
    out: &mut W,
    mut per_row_prefix: impl FnMut(&mut W, usize) -> Result<(), ArrowError>,
) -> Result<(), ArrowError> {
    for row in 0..rows {
        per_row_prefix(out, row)?;
        for ColumnEncoder {
            nullability,
            pre,
            enc,
        } in cols.iter_mut()
        {
            write_value_with_union(out, enc, *nullability, *pre, row)?;
        }
    }
    Ok(())
}

/// An encoder + a null buffer for nullable fields.
///
/// This stores the inner `Encoder` **by value**. To break recursive size
/// cycles, the `Encoder` enum boxes only **recursive** variants (Struct/List/Map).
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
fn make_encoder<'a>(
    array: &'a dyn Array,
    field: &Field,
    plan: PlanRef<'_>,
) -> Result<NullableEncoder<'a>, ArrowError> {
    let nulls = array.nulls().cloned();
    let has_nulls = array.null_count() > 0;
    // Plan-aware nested handling, otherwise legacy default path.
    let enc = if let Some(plan) = plan {
        match (array.data_type(), plan) {
            (DataType::Struct(_), FieldPlan::Struct { encodings }) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| ArrowError::SchemaError("Expected StructArray".into()))?;
                let enc = StructEncoder::try_new(arr, Some(encodings))?;
                NullableEncoder::new(Encoder::Struct(Box::new(enc)), nulls, has_nulls)
            }
            // Fallthrough to default scalar/logical handling for other combos.
            _ => make_encoder(array, field, None)?,
        }
    } else {
        // Legacy default path (identical behavior to previous make_encoder),
        // including UUID-as-string and duration fixed(12).
        match array.data_type() {
            DataType::Boolean => {
                let arr = array.as_boolean();
                NullableEncoder::new(Encoder::Boolean(BooleanEncoder(arr)), nulls, has_nulls)
            }
            DataType::Utf8 => {
                let arr = array.as_string::<i32>();
                NullableEncoder::new(
                    Encoder::Utf8(Utf8GenericEncoder::<i32>(arr)),
                    nulls,
                    has_nulls,
                )
            }
            DataType::LargeUtf8 => {
                let arr = array.as_string::<i64>();
                NullableEncoder::new(
                    Encoder::Utf8Large(Utf8GenericEncoder::<i64>(arr)),
                    nulls,
                    has_nulls,
                )
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
            DataType::Struct(_) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| ArrowError::SchemaError("Expected StructArray".into()))?;
                let enc = StructEncoder::try_new(arr, None)?;
                NullableEncoder::new(Encoder::Struct(Box::new(enc)), nulls, has_nulls)
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                let arr = array.as_primitive::<TimestampMicrosecondType>();
                NullableEncoder::new(Encoder::Timestamp(LongEncoder(arr)), nulls, has_nulls)
            }
            other => {
                return Err(ArrowError::NotYetImplemented(format!(
                    "Unsupported data type for Avro encoding in slim build: {other:?}"
                )));
            }
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
    /// Avro `string` encoder variants (Utf8/LargeUtf8)
    Utf8(Utf8Encoder<'a>),
    Utf8Large(Utf8LargeEncoder<'a>),
    Struct(Box<StructEncoder<'a>>),
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
            Encoder::Utf8(e) => e.encode(idx, out),
            Encoder::Utf8Large(e) => e.encode(idx, out),
            Encoder::Struct(e) => e.encode(idx, out),
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

/// Avro `string` encoder generic over Arrow `GenericStringArray<O>`.
struct Utf8GenericEncoder<'a, O: arrow_array::OffsetSizeTrait>(&'a GenericStringArray<O>);

impl<'a, O: arrow_array::OffsetSizeTrait> Utf8GenericEncoder<'a, O> {
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        let s = self.0.value(idx); // &str
        write_len_prefixed(out, s.as_bytes())
    }
}

type Utf8Encoder<'a> = Utf8GenericEncoder<'a, i32>;
type Utf8LargeEncoder<'a> = Utf8GenericEncoder<'a, i64>;

/// Avro `record` encoder for Arrow `StructArray`
///
/// The encodings are stored in Avro field order, and each encoding carries its
/// own per‑site `Nullability`, so union indices are written exactly as the
/// Avro header declares.
struct StructFieldEncoder<'a> {
    nullability: Option<Nullability>,
    pre: Option<u8>,
    enc: NullableEncoder<'a>,
}

struct StructEncoder<'a> {
    encodings: Vec<StructFieldEncoder<'a>>,
}

impl<'a> StructEncoder<'a> {
    /// Unified constructor: uses Avro order and per‑encoding `Nullability` when provided,
    /// otherwise legacy defaults (NullFirst if the Arrow field is nullable).
    fn try_new(
        array: &'a StructArray,
        plan_encodings: Option<&[StructFieldPlan]>,
    ) -> Result<Self, ArrowError> {
        let fields = match array.data_type() {
            DataType::Struct(fs) => fs,
            _ => return Err(ArrowError::SchemaError("Expected Struct".into())),
        };
        let cols = array.columns();
        let capacity = plan_encodings.map_or(fields.len(), |c| c.len());
        let mut encodings = Vec::with_capacity(capacity);
        if let Some(field_plans) = plan_encodings {
            for field_plan in field_plans {
                let idx = field_plan.arrow_index;
                let col = cols.get(idx).ok_or_else(|| {
                    ArrowError::SchemaError(format!("Struct encoding index {idx} out of range"))
                })?;
                let field = fields
                    .get(idx)
                    .ok_or_else(|| {
                        ArrowError::SchemaError(format!("Struct encoding index {idx} out of range"))
                    })?
                    .as_ref();
                // Use unified helper for value-site preparation
                let (enc_val, eff_null, pre) = prepare_value_site_encoder(
                    col.as_ref(),
                    field,
                    field_plan.nullability,
                    Some(&field_plan.plan),
                )?;
                encodings.push(StructFieldEncoder {
                    nullability: eff_null,
                    pre,
                    enc: enc_val,
                });
            }
        } else {
            // Legacy default (no plan): Arrow field nullability => NullFirst
            for (f_ref, col) in fields.iter().zip(cols.iter()) {
                let f: &Field = f_ref.as_ref();
                let (enc_val, nb, pre) = prepare_value_site_encoder(col.as_ref(), f, None, None)?;
                encodings.push(StructFieldEncoder {
                    nullability: nb,
                    pre,
                    enc: enc_val,
                });
            }
        }
        Ok(Self { encodings })
    }

    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        for enc in self.encodings.iter_mut() {
            write_value_with_union(out, &mut enc.enc, enc.nullability, enc.pre, idx)?;
        }
        Ok(())
    }
}

/// Small helper: shared value writing incl. union-index fast path.
/// If a nullable value site contains no nulls, we can write a constant
/// branch byte (`0x00` or `0x02`) and skip per-item `is_null` checks.
#[inline]
fn write_value_with_union<W: Write + ?Sized>(
    out: &mut W,
    values: &mut NullableEncoder<'_>,
    values_nullability: Option<Nullability>,
    precomputed_branch: Option<u8>,
    j_local: usize,
) -> Result<(), ArrowError> {
    if let Some(b) = precomputed_branch {
        out.write_all(&[b])
            .map_err(|e| ArrowError::IoError(format!("write union value branch: {e}"), e))?;
        return values.encode(j_local, out);
    }
    if let Some(order) = values_nullability {
        let is_null = values.is_null(j_local);
        write_optional_index(out, is_null, order)?;
        if is_null {
            return Ok(());
        }
    }
    values.encode(j_local, out)
}

/// If a nullable union site has no nulls, return the constant branch byte to write
/// before each value. Otherwise, return `None` so the caller performs per-item checks.
#[inline]
fn precomputed_union_value_branch(order: Option<Nullability>, has_nulls: bool) -> Option<u8> {
    match (order, has_nulls) {
        (Some(Nullability::NullFirst), false) => Some(0x02), // value branch index 1
        (Some(Nullability::NullSecond), false) => Some(0x00), // value branch index 0
        _ => None,
    }
}

/// Prepare a nested value-site encoder along with its effective nullability and
/// a precomputed union-branch byte (when the site is nullable but contains no nulls).
#[inline]
fn prepare_value_site_encoder<'a>(
    values_array: &'a dyn Array,
    value_field: &Field,
    site_nullability: Option<Nullability>,
    plan: PlanRef<'_>,
) -> Result<(NullableEncoder<'a>, Option<Nullability>, Option<u8>), ArrowError> {
    // Make the nested encoder using the provided plan (if any).
    let enc = make_encoder(values_array, value_field, plan)?;
    // Resolve effective nullability:
    // - If a plan is provided *and* site nullability is specified, use it.
    // - If no plan is provided, default to Arrow field nullability to NullFirst.
    // - Otherwise, pass through caller-provided value.
    let effective_nullability = match (site_nullability, plan) {
        (Some(n), Some(_)) => Some(n),
        (_, None) => value_field.is_nullable().then_some(Nullability::NullFirst),
        (x, _) => x,
    };
    // If the site is nullable but has no nulls, compute the constant branch byte.
    let pre = precomputed_union_value_branch(effective_nullability, enc.has_nulls());
    Ok((enc, effective_nullability, pre))
}
