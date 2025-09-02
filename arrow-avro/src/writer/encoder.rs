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

use crate::codec::{AvroDataType, AvroField, Codec};
use crate::schema::Nullability;
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

/// Plan reference passed to the unified encoder constructor (required).
type PlanRef<'p> = &'p FieldPlan;

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
    /// Record/Struct with Avro‑ordered children
    Struct { children: Vec<FieldBinding> },
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

/// Unified binding used for both top‑level columns and struct children.
///
/// This replaces the previous duplication between `StructChildPlan` and `ColumnPlan`.
#[derive(Debug, Clone)]
struct FieldBinding {
    /// Index of the Arrow field/column associated with this Avro field site
    arrow_index: usize,
    /// Nullability/order for this site (None if not optional)
    nullability: Option<Nullability>,
    /// Nested plan for this site
    plan: FieldPlan,
}

/// Builder for `RecordEncoder` write plan
#[derive(Debug)]
pub struct RecordEncoderBuilder<'a> {
    avro_root: &'a AvroField,
    arrow_schema: &'a ArrowSchema,
}

impl<'a> RecordEncoderBuilder<'a> {
    /// Create a new builder from the Avro root and Arrow schema.
    pub fn new(avro_root: &'a AvroField, arrow_schema: &'a ArrowSchema) -> Self {
        Self {
            avro_root,
            arrow_schema,
        }
    }

    /// Build the `RecordEncoder` by walking the Avro **record** root in Avro order,
    /// resolving each field to an Arrow index by name.
    pub fn build(self) -> Result<RecordEncoder, ArrowError> {
        let avro_root_dt = self.avro_root.data_type();
        let avro_children = match avro_root_dt.codec() {
            Codec::Struct(children) => children,
            _ => {
                return Err(ArrowError::SchemaError(
                    "Top-level Avro schema must be a record/struct".into(),
                ))
            }
        };
        let mut columns = Vec::with_capacity(avro_children.len());
        for avro_child in avro_children.iter() {
            let name = avro_child.name();
            let arrow_index = self.arrow_schema.index_of(name).map_err(|e| {
                ArrowError::SchemaError(format!("Schema mismatch for field '{name}': {e}"))
            })?;
            let arrow_field = self.arrow_schema.field(arrow_index);
            let plan = FieldPlan::build(avro_child.data_type(), arrow_field)?;
            columns.push(FieldBinding {
                arrow_index,
                nullability: avro_child.data_type().nullability(),
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
    columns: Vec<FieldBinding>,
}

impl RecordEncoder {
    /// Prepare column encoders for a specific batch using the precomputed plan.
    fn prepare_for_batch<'a>(
        &'a self,
        batch: &'a RecordBatch,
    ) -> Result<Vec<FieldEncoder<'a>>, ArrowError> {
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
            let enc = prepare_value_site_encoder(
                array.as_ref(),
                field,
                col_plan.nullability,
                &col_plan.plan,
            )?;
            out.push(enc);
        }
        Ok(out)
    }

    /// Core row encoder moved from the free function `encode_rows_with_prefix_plan`
    /// and renamed to `RecordEncoder::encode`.
    ///
    /// Encodes `rows` rows by iterating `cols` and writing values for each row.
    /// `per_row_prefix` can inject bytes *before* each record (e.g. single-object encoding).
    #[inline]
    fn encode<W: Write>(
        &self,
        rows: usize,
        cols: &mut [FieldEncoder<'_>],
        out: &mut W,
        mut per_row_prefix: impl FnMut(&mut W, usize) -> Result<(), ArrowError>,
    ) -> Result<(), ArrowError> {
        for row in 0..rows {
            per_row_prefix(out, row)?;
            for enc in cols.iter_mut() {
                enc.write_with_union(row, out)?;
            }
        }
        Ok(())
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
        self.encode(batch.num_rows(), &mut cols, out, |_w, _row| Ok(()))
    }

    /// Encode a `RecordBatch` with a per-row single‑object `prefix`.
    pub fn encode_batch_single_object<W: Write>(
        &self,
        batch: &RecordBatch,
        out: &mut W,
        prefix: &[u8; 10],
    ) -> Result<(), ArrowError> {
        let mut cols = self.prepare_for_batch(batch)?;
        self.encode(batch.num_rows(), &mut cols, out, |w, _row| {
            w.write_all(prefix)
                .map_err(|e| ArrowError::IoError(format!("write single-object prefix: {e}"), e))
        })
    }
}

fn find_struct_child_index(fields: &arrow_schema::Fields, name: &str) -> Option<usize> {
    fields.iter().position(|f| f.name() == name)
}

#[inline]
fn find_map_value_field_index(fields: &arrow_schema::Fields) -> Option<usize> {
    // Prefer common Arrow field names; fall back to second child if exactly two
    find_struct_child_index(fields, "value")
        .or_else(|| find_struct_child_index(fields, "values"))
        .or_else(|| if fields.len() == 2 { Some(1) } else { None })
}

impl FieldPlan {
    /// Build a `FieldPlan` for the provided Avro data type (site) and its
    /// corresponding Arrow field.
    fn build(avro_dt: &AvroDataType, arrow_field: &Field) -> Result<Self, ArrowError> {
        match avro_dt.codec() {
            Codec::Enum(symbols) => match arrow_field.data_type() {
                DataType::Dictionary(key_dt, value_dt) => {
                    // Enforce the exact reader-compatible shape: Dictionary<Int32, Utf8>
                    if **key_dt != DataType::Int32 {
                        return Err(ArrowError::SchemaError(
                            "Avro enum requires Dictionary<Int32, Utf8>".into(),
                        ));
                    }
                    if **value_dt != DataType::Utf8 {
                        return Err(ArrowError::SchemaError(
                            "Avro enum requires Dictionary<Int32, Utf8>".into(),
                        ));
                    }
                    Ok(FieldPlan::Enum {
                        symbols: symbols.clone(),
                    })
                }
                other => Err(ArrowError::SchemaError(format!(
                    "Avro enum maps to Arrow Dictionary<Int32, Utf8>, found: {other:?}"
                ))),
            },
            // decimal site (bytes or fixed(N)) with precision/scale validation
            Codec::Decimal(precision, scale_opt, fixed_size_opt) => {
                let (ap, as_) = match arrow_field.data_type() {
                    DataType::Decimal32(p, s) => (*p as usize, *s as i32),
                    DataType::Decimal64(p, s) => (*p as usize, *s as i32),
                    DataType::Decimal128(p, s) => (*p as usize, *s as i32),
                    DataType::Decimal256(p, s) => (*p as usize, *s as i32),
                    other => {
                        return Err(ArrowError::SchemaError(format!(
                            "Avro decimal requires Arrow decimal, got {other:?} for field '{}'",
                            arrow_field.name()
                        )))
                    }
                };
                let sc = scale_opt.unwrap_or(0) as i32; // Avro scale defaults to 0 if absent
                if ap != *precision || as_ != sc {
                    return Err(ArrowError::SchemaError(format!(
                        "Decimal precision/scale mismatch for field '{}': Avro({precision},{sc}) vs Arrow({ap},{as_})",
                        arrow_field.name()
                    )));
                }
                Ok(FieldPlan::Decimal {
                    size: *fixed_size_opt,
                })
            }
            Codec::Struct(avro_children) => {
                let fields = match arrow_field.data_type() {
                    DataType::Struct(fs) => fs,
                    other => {
                        return Err(ArrowError::SchemaError(format!(
                            "Avro struct maps to Arrow Struct, found: {other:?}"
                        )))
                    }
                };
                let mut kids = Vec::with_capacity(avro_children.len());
                for avro_child in avro_children.iter() {
                    let name = avro_child.name().to_string();
                    let idx = find_struct_child_index(fields, &name).ok_or_else(|| {
                        ArrowError::SchemaError(format!(
                            "Struct field '{name}' not present in Arrow field '{}'",
                            arrow_field.name()
                        ))
                    })?;
                    let arrow_child = fields[idx].as_ref();
                    let child_plan = FieldPlan::build(avro_child.data_type(), arrow_child)?;
                    kids.push(FieldBinding {
                        arrow_index: idx,
                        nullability: avro_child.data_type().nullability(),
                        plan: child_plan,
                    });
                }
                Ok(FieldPlan::Struct { children: kids })
            }
            Codec::List(items_dt) => {
                // Map Avro array -> Arrow List/LargeList. Recurse on the **item field** of the Arrow list.
                match arrow_field.data_type() {
                    DataType::List(child) => {
                        let child_field: &Field = child.as_ref();
                        let item_plan = FieldPlan::build(items_dt.as_ref(), child_field)?;
                        Ok(FieldPlan::List {
                            items_nullability: items_dt.nullability(),
                            item_plan: Box::new(item_plan),
                        })
                    }
                    DataType::LargeList(child) => {
                        let child_field: &Field = child.as_ref();
                        let item_plan = FieldPlan::build(items_dt.as_ref(), child_field)?;
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
            Codec::Map(values_dt) => {
                // Avro map -> Arrow DataType::Map(entries_struct, sorted)
                let entries_field = match arrow_field.data_type() {
                    DataType::Map(entries, _sorted) => entries.as_ref(),
                    other => {
                        return Err(ArrowError::SchemaError(format!(
                            "Avro map maps to Arrow DataType::Map, found: {other:?}"
                        )))
                    }
                };
                let entries_struct_fields = match entries_field.data_type() {
                    DataType::Struct(fs) => fs,
                    other => {
                        return Err(ArrowError::SchemaError(format!(
                            "Arrow Map entries must be Struct, found: {other:?}"
                        )))
                    }
                };
                let value_idx =
                    find_map_value_field_index(entries_struct_fields).ok_or_else(|| {
                        ArrowError::SchemaError("Map entries struct missing value field".into())
                    })?;
                let value_field = entries_struct_fields[value_idx].as_ref();
                let value_plan = FieldPlan::build(values_dt.as_ref(), value_field)?;
                Ok(FieldPlan::Map {
                    values_nullability: values_dt.nullability(),
                    value_plan: Box::new(value_plan),
                })
            }
            _ => Ok(FieldPlan::Scalar),
        }
    }
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
    /// Avro `string` encoder variants (Utf8/LargeUtf8)
    Utf8(Utf8Encoder<'a>),
    Utf8Large(Utf8LargeEncoder<'a>),
    /// Avro `enum` encoder: writes the key (int) as the enum index.
    Enum(EnumEncoder<'a>),
    Decimal32(Decimal32Encoder<'a>),
    Decimal64(Decimal64Encoder<'a>),
    Decimal128(Decimal128Encoder<'a>),
    Decimal256(Decimal256Encoder<'a>),
    Struct(Box<StructEncoder<'a>>),
    List(Box<ListEncoder32<'a>>),
    LargeList(Box<ListEncoder64<'a>>),
    Map(Box<MapEncoder<'a>>),
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
            Encoder::Enum(e) => e.encode(idx, out),
            Encoder::Decimal32(e) => e.encode(idx, out),
            Encoder::Decimal64(e) => e.encode(idx, out),
            Encoder::Decimal128(e) => e.encode(idx, out),
            Encoder::Decimal256(e) => e.encode(idx, out),
            Encoder::Struct(e) => e.encode(idx, out),
            Encoder::List(e) => e.encode(idx, out),
            Encoder::LargeList(e) => e.encode(idx, out),
            Encoder::Map(e) => e.encode(idx, out),
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
        let mut tmp = [0u8; uuid::fmt::Hyphenated::LENGTH];
        let s = u.hyphenated().encode_lower(&mut tmp);
        write_len_prefixed(out, s.as_bytes())
    }
}

/// Avro `duration` encoder for Arrow `Interval(IntervalUnit::MonthDayNano)`.
/// Spec: `duration` annotates Avro fixed(12) with three **little‑endian u32**:
/// months, days, milliseconds (no negatives).
struct IntervalMonthDayNanoEncoder<'a>(&'a PrimitiveArray<IntervalMonthDayNanoType>);
impl IntervalMonthDayNanoEncoder<'_> {
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        let native = self.0.value(idx);
        let (months, days, nanos) = IntervalMonthDayNanoType::to_parts(native);
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
        buf[8..12].copy_from_slice(&(millis as u32).to_le_bytes());
        out.write_all(&buf)
            .map_err(|e| ArrowError::IoError(format!("write duration: {e}"), e))
    }
}

/// Minimal trait to obtain a big-endian fixed-size byte array for a decimal's
/// unscaled integer value at `idx`.
trait DecimalBeBytes<const N: usize> {
    fn value_be_bytes(&self, idx: usize) -> [u8; N];
}

impl DecimalBeBytes<4> for Decimal32Array {
    #[inline]
    fn value_be_bytes(&self, idx: usize) -> [u8; 4] {
        self.value(idx).to_be_bytes()
    }
}
impl DecimalBeBytes<8> for Decimal64Array {
    #[inline]
    fn value_be_bytes(&self, idx: usize) -> [u8; 8] {
        self.value(idx).to_be_bytes()
    }
}
impl DecimalBeBytes<16> for Decimal128Array {
    #[inline]
    fn value_be_bytes(&self, idx: usize) -> [u8; 16] {
        self.value(idx).to_be_bytes()
    }
}
impl DecimalBeBytes<32> for Decimal256Array {
    #[inline]
    fn value_be_bytes(&self, idx: usize) -> [u8; 32] {
        // Arrow i256 → [u8; 32] big-endian
        self.value(idx).to_be_bytes()
    }
}

/// Generic Avro decimal encoder over Arrow decimal arrays.
/// - When `fixed_size` is `None` → Avro `bytes(decimal)`; writes the minimal
///   two's-complement representation with a length prefix.
/// - When `Some(n)` → Avro `fixed(n, decimal)`; sign-extends (or validates)
///   to exactly `n` bytes and writes them directly.
struct DecimalEncoder<'a, const N: usize, A: DecimalBeBytes<N>> {
    arr: &'a A,
    fixed_size: Option<usize>,
}

impl<'a, const N: usize, A: DecimalBeBytes<N>> DecimalEncoder<'a, N, A> {
    #[inline]
    fn new(arr: &'a A, fixed_size: Option<usize>) -> Self {
        Self { arr, fixed_size }
    }

    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        let be = self.arr.value_be_bytes(idx);
        match self.fixed_size {
            Some(n) => {
                let bytes = sign_extend_to_exact(&be, n)?;
                out.write_all(&bytes)
                    .map_err(|e| ArrowError::IoError(format!("write decimal fixed: {e}"), e))
            }
            None => write_len_prefixed(out, minimal_twos_complement(&be)),
        }
    }
}

type Decimal32Encoder<'a> = DecimalEncoder<'a, 4, Decimal32Array>;
type Decimal64Encoder<'a> = DecimalEncoder<'a, 8, Decimal64Array>;
type Decimal128Encoder<'a> = DecimalEncoder<'a, 16, Decimal128Array>;
type Decimal256Encoder<'a> = DecimalEncoder<'a, 32, Decimal256Array>;

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

/// Avro `enum` encoder for Arrow `DictionaryArray<Int32, Utf8>`.
///
/// Per Avro spec, an enum is encoded as an **int** equal to the
/// zero-based position of the symbol in the schema’s `symbols` list.
/// We validate at construction that the dictionary values equal the symbols,
/// so we can directly write the key value here.
struct EnumEncoder<'a> {
    keys: &'a PrimitiveArray<Int32Type>,
}
impl EnumEncoder<'_> {
    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, row: usize, out: &mut W) -> Result<(), ArrowError> {
        let idx = self.keys.value(row);
        write_int(out, idx)
    }
}

/// Unified field encoder:
/// - Holds the inner `Encoder` (by value)
/// - Tracks the column/site null buffer and whether any nulls exist
/// - Carries per-site Avro `Nullability` and precomputed union branch (fast path)
pub struct FieldEncoder<'a> {
    encoder: Encoder<'a>,
    nulls: Option<NullBuffer>,
    has_nulls: bool,
    /// Nullability/order for this site (None if not optional)
    nullability: Option<Nullability>,
    /// Precomputed constant branch byte if site is nullable but contains no nulls
    pre: Option<u8>,
}

impl<'a> FieldEncoder<'a> {
    /// Create a new field encoder from an Arrow array + Field metadata using a required plan.
    ///
    /// Returns `Self` (without setting site nullability yet).
    fn make_encoder(
        array: &'a dyn Array,
        field: &Field,
        plan: PlanRef<'_>,
    ) -> Result<Self, ArrowError> {
        let nulls = array.nulls().cloned();
        let has_nulls = array.null_count() > 0;
        let encoder = match plan {
            FieldPlan::Struct { children } => {
                let arr = array
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| ArrowError::SchemaError("Expected StructArray".into()))?;
                let enc = StructEncoder::try_new(arr, children)?;
                Encoder::Struct(Box::new(enc))
            }
            FieldPlan::List {
                items_nullability,
                item_plan,
            } => match array.data_type() {
                DataType::List(_) => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .ok_or_else(|| ArrowError::SchemaError("Expected ListArray".into()))?;
                    let enc = ListEncoder32::try_new(arr, *items_nullability, item_plan.as_ref())?;
                    Encoder::List(Box::new(enc))
                }
                DataType::LargeList(_) => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<LargeListArray>()
                        .ok_or_else(|| ArrowError::SchemaError("Expected LargeListArray".into()))?;
                    let enc = ListEncoder64::try_new(arr, *items_nullability, item_plan.as_ref())?;
                    Encoder::LargeList(Box::new(enc))
                }
                other => {
                    return Err(ArrowError::SchemaError(format!(
                        "Avro array site requires Arrow List/LargeList, found: {other:?}"
                    )))
                }
            },
            FieldPlan::Map {
                values_nullability,
                value_plan,
            } => {
                let arr = array
                    .as_any()
                    .downcast_ref::<MapArray>()
                    .ok_or_else(|| ArrowError::SchemaError("Expected MapArray".into()))?;
                let enc = MapEncoder::try_new(arr, *values_nullability, value_plan.as_ref())?;
                Encoder::Map(Box::new(enc))
            }
            // plan-aware decimal sites (bytes or fixed)
            FieldPlan::Decimal { size } => match array.data_type() {
                DataType::Decimal32(_, _) => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<Decimal32Array>()
                        .ok_or_else(|| ArrowError::SchemaError("Expected Decimal32Array".into()))?;
                    let dec = DecimalEncoder::<4, Decimal32Array>::new(arr, *size);
                    Encoder::Decimal32(dec)
                }
                DataType::Decimal64(_, _) => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<Decimal64Array>()
                        .ok_or_else(|| ArrowError::SchemaError("Expected Decimal64Array".into()))?;
                    let dec = DecimalEncoder::<8, Decimal64Array>::new(arr, *size);
                    Encoder::Decimal64(dec)
                }
                DataType::Decimal128(_, _) => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<Decimal128Array>()
                        .ok_or_else(|| {
                            ArrowError::SchemaError("Expected Decimal128Array".into())
                        })?;
                    let dec = DecimalEncoder::<16, Decimal128Array>::new(arr, *size);
                    Encoder::Decimal128(dec)
                }
                DataType::Decimal256(_, _) => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<Decimal256Array>()
                        .ok_or_else(|| {
                            ArrowError::SchemaError("Expected Decimal256Array".into())
                        })?;
                    let dec = DecimalEncoder::<32, Decimal256Array>::new(arr, *size);
                    Encoder::Decimal256(dec)
                }
                other => {
                    return Err(ArrowError::SchemaError(format!(
                        "Avro decimal site requires Arrow decimal array, got {other:?}"
                    )))
                }
            },
            FieldPlan::Enum { symbols } => {
                let (key_dt, value_dt) = match array.data_type() {
                    DataType::Dictionary(k, v) => (k, v),
                    other => {
                        return Err(ArrowError::SchemaError(format!(
                            "Avro enum maps to Arrow Dictionary<Int32, Utf8>, found: {other:?}"
                        )))
                    }
                };
                if **key_dt != DataType::Int32 || **value_dt != DataType::Utf8 {
                    return Err(ArrowError::SchemaError(
                        "Avro enum requires Dictionary<Int32, Utf8>".into(),
                    ));
                }
                let dict = array
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()
                    .ok_or_else(|| {
                        ArrowError::SchemaError("Expected DictionaryArray<Int32>".into())
                    })?;

                // Dictionary values must exactly match schema `symbols` (order & content)
                let values = dict
                    .values()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        ArrowError::SchemaError("Dictionary values must be Utf8".into())
                    })?;
                if values.len() != symbols.len() {
                    return Err(ArrowError::SchemaError(format!(
                        "Enum symbol length {} != dictionary size {}",
                        symbols.len(),
                        values.len()
                    )));
                }
                for i in 0..values.len() {
                    if values.value(i) != symbols[i].as_str() {
                        return Err(ArrowError::SchemaError(format!(
                            "Enum symbol mismatch at {i}: schema='{}' dict='{}'",
                            symbols[i],
                            values.value(i)
                        )));
                    }
                }
                // Keys are the Avro enum indices (zero-based position in `symbols`).
                let keys = dict.keys();
                let enc = EnumEncoder { keys };
                Encoder::Enum(enc)
            }
            FieldPlan::Scalar => match array.data_type() {
                DataType::Boolean => {
                    let arr = array.as_boolean();
                    Encoder::Boolean(BooleanEncoder(arr))
                }
                DataType::Utf8 => {
                    let arr = array.as_string::<i32>();
                    Encoder::Utf8(Utf8GenericEncoder::<i32>(arr))
                }
                DataType::LargeUtf8 => {
                    let arr = array.as_string::<i64>();
                    Encoder::Utf8Large(Utf8GenericEncoder::<i64>(arr))
                }
                DataType::Int32 => {
                    let arr = array.as_primitive::<Int32Type>();
                    Encoder::Int(IntEncoder(arr))
                }
                DataType::Int64 => {
                    let arr = array.as_primitive::<Int64Type>();
                    Encoder::Long(LongEncoder(arr))
                }
                DataType::Float32 => {
                    let arr = array.as_primitive::<Float32Type>();
                    Encoder::Float32(F32Encoder(arr))
                }
                DataType::Float64 => {
                    let arr = array.as_primitive::<Float64Type>();
                    Encoder::Float64(F64Encoder(arr))
                }
                DataType::Binary => {
                    let arr = array.as_binary::<i32>();
                    Encoder::Binary(BinaryEncoder(arr))
                }
                DataType::LargeBinary => {
                    let arr = array.as_binary::<i64>();
                    Encoder::LargeBinary(BinaryEncoder(arr))
                }
                DataType::FixedSizeBinary(len) => {
                    // Decide between Avro `fixed` (raw bytes) and `uuid` logical string
                    // based on Field metadata, mirroring schema generation rules.
                    let arr = array
                        .as_any()
                        .downcast_ref::<FixedSizeBinaryArray>()
                        .ok_or_else(|| {
                            ArrowError::SchemaError("Expected FixedSizeBinaryArray".into())
                        })?;
                    let md = field.metadata();
                    let is_uuid = md.get("logicalType").is_some_and(|v| v == "uuid")
                        || (*len == 16
                            && md.get("ARROW:extension:name").is_some_and(|v| v == "uuid"));
                    if is_uuid {
                        if *len != 16 {
                            return Err(ArrowError::InvalidArgumentError(
                                "logicalType=uuid requires FixedSizeBinary(16)".into(),
                            ));
                        }
                        Encoder::Uuid(UuidEncoder(arr))
                    } else {
                        Encoder::Fixed(FixedEncoder(arr))
                    }
                }
                DataType::Interval(IntervalUnit::MonthDayNano) => {
                    let arr = array.as_primitive::<IntervalMonthDayNanoType>();
                    Encoder::IntervalMonthDayNano(IntervalMonthDayNanoEncoder(arr))
                }
                DataType::Interval(unit) => {
                    return Err(ArrowError::NotYetImplemented(format!(
                        "Avro writer: Interval({unit:?}) is not supported; cast to Interval(MonthDayNano) to write Avro 'duration'"
                    )));
                }
                DataType::Duration(_) => {
                    return Err(ArrowError::NotYetImplemented(
                        "Avro writer: Arrow Duration(TimeUnit) has no standard Avro mapping; cast to Interval(MonthDayNano) to use Avro 'duration'".into(),
                    ));
                }
                DataType::Timestamp(TimeUnit::Microsecond, _) => {
                    let arr = array.as_primitive::<TimestampMicrosecondType>();
                    Encoder::Timestamp(LongEncoder(arr))
                }
                // Composite or mismatched types under scalar plan
                DataType::List(_)
                | DataType::LargeList(_)
                | DataType::Map(_, _)
                | DataType::Struct(_)
                | DataType::Dictionary(_, _)
                | DataType::Decimal32(_, _)
                | DataType::Decimal64(_, _)
                | DataType::Decimal128(_, _)
                | DataType::Decimal256(_, _) => {
                    return Err(ArrowError::SchemaError(format!(
                        "Avro scalar site incompatible with Arrow type: {:?}",
                        array.data_type()
                    )))
                }
                other => {
                    return Err(ArrowError::NotYetImplemented(format!(
                        "Unsupported data type for Avro encoding: {other:?}"
                    )));
                }
            },
        };

        Ok(Self {
            encoder,
            nulls,
            has_nulls,
            nullability: None,
            pre: None,
        })
    }

    /// Whether this column contains any nulls at all.
    #[inline]
    fn has_nulls(&self) -> bool {
        self.has_nulls
    }

    /// Check if the value at `idx` is null.
    #[inline]
    fn is_null(&self, idx: usize) -> bool {
        self.nulls.as_ref().is_some_and(|n| n.is_null(idx))
    }

    /// Set effective site nullability and compute precomputed branch (if any), returning `Self`.
    #[inline]
    fn with_effective_nullability(mut self, n: Option<Nullability>) -> Self {
        self.nullability = n;
        self.pre = precomputed_union_value_branch(n, self.has_nulls());
        self
    }

    /// Encode the actual value at `idx` (without union branch handling).
    #[inline]
    fn encode_inner<W: Write + ?Sized>(
        &mut self,
        idx: usize,
        out: &mut W,
    ) -> Result<(), ArrowError> {
        self.encoder.encode(idx, out)
    }

    /// Write union branch (if applicable) and then the value at `idx`.
    #[inline]
    fn write_with_union<W: Write + ?Sized>(
        &mut self,
        idx: usize,
        out: &mut W,
    ) -> Result<(), ArrowError> {
        if let Some(b) = self.pre {
            return out
                .write_all(&[b])
                .map_err(|e| ArrowError::IoError(format!("write union value branch: {e}"), e))
                .and_then(|_| self.encode_inner(idx, out));
        }
        if let Some(order) = self.nullability {
            let is_null = self.is_null(idx);
            write_optional_index(out, is_null, order)?;
            if is_null {
                return Ok(());
            }
        }
        self.encode_inner(idx, out)
    }
}

/// Avro `record` encoder for Arrow `StructArray`
///
/// The children are stored in **Avro order**, and each child carries its
/// own per‑site `Nullability`, so union indices are written exactly as the
/// Avro header declares.
struct StructEncoder<'a> {
    children: Vec<FieldEncoder<'a>>,
}

impl<'a> StructEncoder<'a> {
    /// Constructor that requires Avro‑ordered child plan.
    fn try_new(array: &'a StructArray, plan_children: &[FieldBinding]) -> Result<Self, ArrowError> {
        let fields = match array.data_type() {
            DataType::Struct(fs) => fs,
            _ => return Err(ArrowError::SchemaError("Expected Struct".into())),
        };
        let cols = array.columns();
        let mut children = Vec::with_capacity(plan_children.len());
        for child_plan in plan_children {
            let idx = child_plan.arrow_index;
            let col = cols.get(idx).ok_or_else(|| {
                ArrowError::SchemaError(format!("Struct child index {idx} out of range"))
            })?;
            let field = fields
                .get(idx)
                .ok_or_else(|| {
                    ArrowError::SchemaError(format!("Struct child index {idx} out of range"))
                })?
                .as_ref();
            let child = prepare_value_site_encoder(
                col.as_ref(),
                field,
                child_plan.nullability,
                &child_plan.plan,
            )?;
            children.push(child);
        }
        Ok(Self { children })
    }

    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        for child in self.children.iter_mut() {
            child.write_with_union(idx, out)?;
        }
        Ok(())
    }
}

/// Shared, allocation-free helper to emit a single Avro block for `[start, end)`,
/// followed by the terminating zero-length block.
#[inline]
fn encode_blocked_range<W: Write + ?Sized, F>(
    out: &mut W,
    start: usize,
    end: usize,
    mut write_item: F,
) -> Result<(), ArrowError>
where
    F: FnMut(usize, &mut W) -> Result<(), ArrowError>,
{
    let len = end.saturating_sub(start);
    if len == 0 {
        // Zero-length terminator per Avro spec
        write_long(out, 0)?;
        return Ok(());
    }
    // Emit a single positive block for performance, then the end marker.
    write_long(out, len as i64)?;
    for j in start..end {
        write_item(j, out)?;
    }
    write_long(out, 0)?;
    Ok(())
}

/// Shared, allocation-free encode path for Arrow `ListArray` and `LargeListArray`.
#[inline]
fn encode_list_range<W: Write + ?Sized>(
    out: &mut W,
    start: usize,
    end: usize,
    values_offset: usize,
    values: &mut FieldEncoder<'_>,
) -> Result<(), ArrowError> {
    encode_blocked_range(out, start, end, |j, out| {
        debug_assert!(
            j >= values_offset,
            "List values offset invariant violated: j < values_offset"
        );
        let j_local = j.saturating_sub(values_offset);
        values.write_with_union(j_local, out)
    })
}

struct ListEncoder<'a, O: arrow_array::OffsetSizeTrait> {
    list: &'a arrow_array::array::GenericListArray<O>,
    values: FieldEncoder<'a>,
    values_offset: usize,
}

type ListEncoder32<'a> = ListEncoder<'a, i32>;
type ListEncoder64<'a> = ListEncoder<'a, i64>;

impl<'a, O: arrow_array::OffsetSizeTrait> ListEncoder<'a, O> {
    /// Constructor requiring item plan & nullability.
    fn try_new(
        list: &'a arrow_array::array::GenericListArray<O>,
        items_nullability: Option<Nullability>,
        item_plan: &FieldPlan,
    ) -> Result<Self, ArrowError> {
        let child_field = match list.data_type() {
            DataType::List(field) => field.as_ref(),
            DataType::LargeList(field) => field.as_ref(),
            _ => {
                return Err(ArrowError::SchemaError(
                    "Expected List or LargeList for ListEncoder".into(),
                ))
            }
        };
        let values_enc = prepare_value_site_encoder(
            list.values().as_ref(),
            child_field,
            items_nullability,
            item_plan,
        )?;
        Ok(Self {
            list,
            values: values_enc,
            values_offset: list.values().offset(),
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
        encode_list_range(out, start, end, self.values_offset, &mut self.values)
    }
}

/// Avro `map` encoder for Arrow `MapArray`
///
/// Each row encodes as one (possibly empty) map block sequence:
///   count (long), then for each entry: key (string), value (union index if nullable + value),
///   terminated by a zero count. Keys are strings per Avro spec.
/// Internal key array kind used by Map encoder.
#[derive(Copy, Clone)]
enum KeyKind<'a> {
    Utf8(&'a GenericStringArray<i32>),
    LargeUtf8(&'a GenericStringArray<i64>),
}

struct MapEncoder<'a> {
    map: &'a MapArray,
    keys: KeyKind<'a>,
    values: FieldEncoder<'a>,
    keys_offset: usize,
    values_offset: usize,
}

#[inline]
fn i32_to_usize(i: i32) -> Result<usize, ArrowError> {
    if i < 0 {
        Err(ArrowError::InvalidArgumentError(format!(
            "Negative offset {i}"
        )))
    } else {
        Ok(i as usize)
    }
}

/// Resolve common Map components shared by both constructors:
/// - key array kind & offset
/// - value field reference & values offset
#[inline]
fn resolve_map_components(
    map: &MapArray,
) -> Result<(KeyKind<'_>, usize, &'_ Field, usize), ArrowError> {
    // Keys + offset
    let keys_arr = map.keys();
    let keys_kind = match keys_arr.data_type() {
        DataType::Utf8 => Ok(KeyKind::Utf8(keys_arr.as_string::<i32>())),
        DataType::LargeUtf8 => Ok(KeyKind::LargeUtf8(keys_arr.as_string::<i64>())),
        other => Err(ArrowError::SchemaError(format!(
            "Avro map requires string keys; Arrow key type must be Utf8/LargeUtf8, found: {other:?}"
        ))),
    }?;
    let keys_off = keys_arr.offset();
    // Value field inside entries struct + values offset
    let entries_struct_fields = match map.data_type() {
        DataType::Map(entries, _) => match entries.data_type() {
            DataType::Struct(fs) => Ok(fs),
            other => Err(ArrowError::SchemaError(format!(
                "Arrow Map entries must be Struct, found: {other:?}"
            ))),
        },
        _ => Err(ArrowError::SchemaError(
            "Expected MapArray with DataType::Map".into(),
        )),
    }?;
    let v_idx = find_map_value_field_index(entries_struct_fields)
        .ok_or_else(|| ArrowError::SchemaError("Map entries struct missing value field".into()))?;
    let value_field = entries_struct_fields[v_idx].as_ref();
    let values_off = map.values().offset();
    Ok((keys_kind, keys_off, value_field, values_off))
}

impl<'a> MapEncoder<'a> {
    /// Constructor requiring value plan & nullability.
    fn try_new(
        map: &'a MapArray,
        values_nullability: Option<Nullability>,
        value_plan: &FieldPlan,
    ) -> Result<Self, ArrowError> {
        let (keys, keys_offset, value_field, values_offset) = resolve_map_components(map)?;
        let values_enc = prepare_value_site_encoder(
            map.values().as_ref(),
            value_field,
            values_nullability,
            value_plan,
        )?;
        Ok(Self {
            map,
            keys,
            values: values_enc,
            keys_offset,
            values_offset,
        })
    }

    /// Generic helper that writes `(key, value)` pairs for `[start, end)` using a
    /// `GenericStringArray<O>` for keys. This is monomorphized for Utf8/LargeUtf8
    /// (static dispatch), so the hot loop has no per-entry type branching.
    ///
    /// Note: Implemented as a method to keep the hot path colocated with `MapEncoder`.
    #[inline]
    fn encode_map_entries_for_keys<W, O>(
        &mut self,
        out: &mut W,
        arr: &GenericStringArray<O>,
        start: usize,
        end: usize,
    ) -> Result<(), ArrowError>
    where
        W: Write + ?Sized,
        O: arrow_array::OffsetSizeTrait,
    {
        let keys_offset = self.keys_offset;
        let values_offset = self.values_offset;
        encode_blocked_range(out, start, end, |j, out| {
            debug_assert!(j >= keys_offset && j >= values_offset);
            let j_key = j.saturating_sub(keys_offset);
            let j_val = j.saturating_sub(values_offset);
            // Key (string)
            let s = arr.value(j_key);
            write_len_prefixed(out, s.as_bytes())?;
            // Value (union index if necessary, then payload)
            self.values.write_with_union(j_val, out)
        })
    }

    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        // Compute entry range [start, end)
        let offsets = self.map.offsets();
        // MapArray offsets are i32 and guaranteed >= 0 by Arrow; align style with lists.
        let start = i32_to_usize(offsets[idx])?;
        let end = i32_to_usize(offsets[idx + 1])?;
        // Copy of the keys enum (contains references; cheap and avoids borrow conflicts)
        let keys = self.keys;
        match keys {
            KeyKind::Utf8(arr) => self.encode_map_entries_for_keys(out, arr, start, end),
            KeyKind::LargeUtf8(arr) => self.encode_map_entries_for_keys(out, arr, start, end),
        }
    }
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
) -> Result<FieldEncoder<'a>, ArrowError> {
    // Make the nested encoder using the provided plan (required).
    let enc = FieldEncoder::make_encoder(values_array, value_field, plan)?;
    // Effective nullability is exactly the site's Avro-declared nullability.
    Ok(enc.with_effective_nullability(site_nullability))
}

#[cfg(test)]
mod tests_decimal_helpers {
    use super::*;

    #[test]
    fn test_minimal_twos_complement() {
        // 0 -> [0x00]
        assert_eq!(minimal_twos_complement(&[0x00]), &[0x00]);
        // +127 -> [0x7F]
        assert_eq!(minimal_twos_complement(&[0x7F]), &[0x7F]);
        // +128 -> minimal requires leading 0x00 so the sign bit stays 0
        assert_eq!(minimal_twos_complement(&[0x00, 0x80]), &[0x00, 0x80]);
        // -1 -> [0xFF]
        assert_eq!(minimal_twos_complement(&[0xFF]), &[0xFF]);
        // -128 -> [0x80]
        assert_eq!(minimal_twos_complement(&[0x80]), &[0x80]);
        // -129 (16-bit) -> [0xFF, 0x7F]
        assert_eq!(minimal_twos_complement(&[0xFF, 0x7F]), &[0xFF, 0x7F]);
        // Already minimal multi-byte positive
        assert_eq!(minimal_twos_complement(&[0x01, 0x00]), &[0x01, 0x00]);
        // Already minimal multi-byte negative
        assert_eq!(minimal_twos_complement(&[0xFE, 0xFF]), &[0xFE, 0xFF]);
    }

    #[test]
    fn test_sign_extend_to_exact() {
        // Extend positive: 0x7F -> 0x00 0x7F
        assert_eq!(sign_extend_to_exact(&[0x7F], 2).unwrap(), vec![0x00, 0x7F]);
        // Extend negative: 0x80 -> 0xFF 0x80
        assert_eq!(sign_extend_to_exact(&[0x80], 2).unwrap(), vec![0xFF, 0x80]);
        // Shrink with valid sign bytes
        assert_eq!(sign_extend_to_exact(&[0x00, 0x7F], 1).unwrap(), vec![0x7F]);
        assert_eq!(sign_extend_to_exact(&[0xFF, 0x80], 1).unwrap(), vec![0x80]);
        // Overflow when truncation would change the value/sign:
        // - dropping a non-sign 0x01 from the left
        assert!(sign_extend_to_exact(&[0x01, 0x00], 1).is_err());
        // - dropping a non-sign 0xFE from the left (negative number)
        assert!(sign_extend_to_exact(&[0xFE, 0xFF], 1).is_err());
    }
}
