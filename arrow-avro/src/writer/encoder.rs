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

use crate::codec::{
    AvroDataType, AvroField as CodecAvroField, AvroFieldBuilder, Codec as AvroCodec, Nullability,
};
use crate::schema::{
    AvroSchema as AvroJson, Schema as AvroJsonAst, SchemaGenOptions, SCHEMA_METADATA_KEY,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowPrimitiveType, Float32Type, Float64Type, Int32Type, Int64Type, IntervalMonthDayNanoType,
    TimestampMicrosecondType,
};
use arrow_array::{
    Array, DictionaryArray, FixedSizeBinaryArray, GenericBinaryArray, GenericStringArray,
    LargeListArray, ListArray, MapArray, PrimitiveArray, RecordBatch, StringArray, StructArray,
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
/// See: https://avro.apache.org/docs/1.11.1/specification/ (Binary Encoding)
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

/// Write the union branch index for an optional site with the specified `order`.
///
/// Branch index is 0‑based per Avro unions. We special-case 0 or 1 which
/// are single-byte varints: `0x00` (index 0) and `0x02` (index 1).
/// See: https://avro.apache.org/docs/1.11.1/specification/ (Unions)
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
    Struct { children: Vec<StructChildPlan> },
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
    /// Avro enum; maps to Arrow Dictionary<Int32, Utf8> with dictionary values
    /// exactly equal and ordered as the Avro enum `symbols`.
    Enum { symbols: Arc<[String]> },
}

#[derive(Debug, Clone)]
struct StructChildPlan {
    /// Child field name (Avro)
    name: String,
    /// Index of the child within the Arrow struct's Fields
    arrow_index: usize,
    /// Nullability/order for this child (None if not optional)
    nullability: Option<Nullability>,
    /// Nested plan for this child
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

/// A pre-computed plan for encoding a `RecordBatch` to Avro.
///
/// A `WritePlan` is derived from an Avro schema and an Arrow schema. It maps
/// top-level Avro fields to Arrow columns and contains a nested encoding plan
/// for each column. This allows the encoder to write records efficiently without
/// repeatedly consulting schemas or field names.
#[derive(Debug, Clone)]
pub struct WritePlan {
    columns: Vec<ColumnPlan>,
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

fn build_field_plan(avro_dt: &AvroDataType, arrow_field: &Field) -> Result<FieldPlan, ArrowError> {
    match avro_dt.codec() {
        AvroCodec::Enum(symbols) => {
            match arrow_field.data_type() {
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
            }
        }
        AvroCodec::Struct(avro_children) => {
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
                let child_plan = build_field_plan(avro_child.data_type(), arrow_child)?;
                kids.push(StructChildPlan {
                    name,
                    arrow_index: idx,
                    nullability: avro_child.data_type().nullability(),
                    plan: child_plan,
                });
            }
            Ok(FieldPlan::Struct { children: kids })
        }
        AvroCodec::List(items_dt) => {
            // Map Avro array -> Arrow List/LargeList. IMPORTANT:
            // Recurse on the **item field** of the Arrow list, not the list field itself.
            match arrow_field.data_type() {
                DataType::List(child) => {
                    let child_field: &Field = child.as_ref();
                    let item_plan = build_field_plan(items_dt.as_ref(), child_field)?;
                    Ok(FieldPlan::List {
                        items_nullability: items_dt.nullability(),
                        item_plan: Box::new(item_plan),
                    })
                }
                DataType::LargeList(child) => {
                    let child_field: &Field = child.as_ref();
                    let item_plan = build_field_plan(items_dt.as_ref(), child_field)?;
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
        AvroCodec::Map(values_dt) => {
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
            let value_idx = find_map_value_field_index(entries_struct_fields).ok_or_else(|| {
                ArrowError::SchemaError("Map entries struct missing value field".into())
            })?;
            let value_field = entries_struct_fields[value_idx].as_ref();
            let value_plan = build_field_plan(values_dt.as_ref(), value_field)?;
            Ok(FieldPlan::Map {
                values_nullability: values_dt.nullability(),
                value_plan: Box::new(value_plan),
            })
        }
        _ => Ok(FieldPlan::Scalar),
    }
}

/// Build a [`WritePlan`] by walking the Avro **record** root in Avro order, and
/// resolving each field to an Arrow index by name.
pub fn build_write_plan_from_avro_root(
    root: &CodecAvroField,
    arrow_schema: &ArrowSchema,
) -> Result<WritePlan, ArrowError> {
    let avro_root_dt = root.data_type();
    let avro_children = match avro_root_dt.codec() {
        AvroCodec::Struct(children) => children,
        _ => {
            return Err(ArrowError::SchemaError(
                "Top-level Avro schema must be a record/struct".into(),
            ))
        }
    };
    let mut columns = Vec::with_capacity(avro_children.len());
    for avro_child in avro_children.iter() {
        let name = avro_child.name();
        let arrow_index = arrow_schema.index_of(name).map_err(|e| {
            ArrowError::SchemaError(format!("Schema mismatch for field '{name}': {e}"))
        })?;
        // In this Arrow version, `Schema::field` returns `&Field` directly.
        let arrow_field = arrow_schema.field(arrow_index);
        let plan = build_field_plan(avro_child.data_type(), arrow_field)?;
        columns.push(ColumnPlan {
            arrow_index,
            nullability: avro_child.data_type().nullability(),
            plan,
        });
    }
    Ok(WritePlan { columns })
}

/// Derive the write plan for `batch` from its advertised Avro schema in
/// `SCHEMA_METADATA_KEY`, or generate Avro JSON from Arrow if missing.
///
/// When synthesizing Avro JSON from Arrow (i.e., metadata is absent), the
/// default union order is used (no override).
fn derive_plan_for_batch(batch: &RecordBatch) -> Result<WritePlan, ArrowError> {
    let avro_json = if let Some(json) = batch.schema().metadata.get(SCHEMA_METADATA_KEY) {
        AvroJson::new(json.clone())
    } else {
        let opts = SchemaGenOptions {
            null_union_order: None,
        };
        AvroJson::from_arrow_with_options(batch.schema().as_ref(), opts)?
    };
    let avro_ast: AvroJsonAst<'_> = avro_json.schema()?;
    let root = AvroFieldBuilder::new(&avro_ast).build()?;
    build_write_plan_from_avro_root(&root, batch.schema().as_ref())
}

/// Encode a `RecordBatch` in Avro binary format using the **schema‑driven plan**.
///
/// Tip: Wrap `out` in a `std::io::BufWriter` to reduce the overhead of many small writes.
pub fn encode_record_batch<W: Write>(batch: &RecordBatch, out: &mut W) -> Result<(), ArrowError> {
    let plan = derive_plan_for_batch(batch)?;
    encode_record_batch_with_plan(batch, out, &plan)
}

/// Encode a `RecordBatch` as a stream of Avro **single-object encodings**,
/// writing the provided 10-byte `prefix` (magic + 8-byte fingerprint)
/// **before each record** using a **schema‑driven plan**.
pub fn encode_record_batch_single_object<W: Write>(
    batch: &RecordBatch,
    out: &mut W,
    prefix: &[u8; 10],
) -> Result<(), ArrowError> {
    let plan = derive_plan_for_batch(batch)?;
    encode_record_batch_single_object_with_plan(batch, out, prefix, &plan)
}

/// Encode a `RecordBatch` using a precomputed [`WritePlan`].
///
/// Tip: Wrap `out` in a `std::io::BufWriter` to reduce the overhead of many small writes.
pub fn encode_record_batch_with_plan<W: Write>(
    batch: &RecordBatch,
    out: &mut W,
    plan: &WritePlan,
) -> Result<(), ArrowError> {
    let mut cols = prepare_encoders_for_batch_with_plan(batch, plan)?;
    encode_rows_with_prefix_plan(batch.num_rows(), &mut cols, out, |_w, _row| Ok(()))
}

/// Encode a `RecordBatch` with a plan, adding a single‑object `prefix` before each row.
pub fn encode_record_batch_single_object_with_plan<W: Write>(
    batch: &RecordBatch,
    out: &mut W,
    prefix: &[u8; 10],
    plan: &WritePlan,
) -> Result<(), ArrowError> {
    let mut cols = prepare_encoders_for_batch_with_plan(batch, plan)?;
    encode_rows_with_prefix_plan(batch.num_rows(), &mut cols, out, |w, _row| {
        w.write_all(prefix)
            .map_err(|e| ArrowError::IoError(format!("write single-object prefix: {e}"), e))
    })
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

#[inline]
fn prepare_encoders_for_batch_with_plan<'a>(
    batch: &'a RecordBatch,
    plan: &WritePlan,
) -> Result<Vec<ColumnEncoder<'a>>, ArrowError> {
    // bind schema to extend lifetime of `fields()` borrow
    let schema_binding = batch.schema();
    let fields = schema_binding.fields();
    let arrays = batch.columns();
    let mut out = Vec::with_capacity(plan.columns.len());
    for col_plan in plan.columns.iter() {
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
        for ColumnEncoder { nullability, pre, enc } in cols.iter_mut() {
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
            (DataType::Struct(_), FieldPlan::Struct { children }) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| ArrowError::SchemaError("Expected StructArray".into()))?;
                let enc = StructEncoder::try_new(arr, Some(children))?;
                NullableEncoder::new(Encoder::Struct(Box::new(enc)), nulls, has_nulls)
            }
            (
                DataType::List(_),
                FieldPlan::List {
                    items_nullability,
                    item_plan,
                },
            ) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| ArrowError::SchemaError("Expected ListArray".into()))?;
                let enc =
                    ListEncoder32::try_new(arr, *items_nullability, Some(item_plan.as_ref()))?;
                NullableEncoder::new(Encoder::List(Box::new(enc)), nulls, has_nulls)
            }
            (
                DataType::LargeList(_),
                FieldPlan::List {
                    items_nullability,
                    item_plan,
                },
            ) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<LargeListArray>()
                    .ok_or_else(|| ArrowError::SchemaError("Expected LargeListArray".into()))?;
                let enc =
                    ListEncoder64::try_new(arr, *items_nullability, Some(item_plan.as_ref()))?;
                NullableEncoder::new(Encoder::LargeList(Box::new(enc)), nulls, has_nulls)
            }
            (
                DataType::Map(_, _),
                FieldPlan::Map {
                    values_nullability,
                    value_plan,
                },
            ) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<MapArray>()
                    .ok_or_else(|| ArrowError::SchemaError("Expected MapArray".into()))?;
                let enc =
                    MapEncoder::try_new(arr, *values_nullability, Some(value_plan.as_ref()))?;
                NullableEncoder::new(Encoder::Map(Box::new(enc)), nulls, has_nulls)
            }
            (DataType::Dictionary(key_dt, value_dt), FieldPlan::Enum { symbols }) => {
                // Enforce the same shape we validated during plan build:
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
                NullableEncoder::new(Encoder::Enum(enc), nulls, has_nulls)
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
            DataType::FixedSizeBinary(len) => {
                // Decide between Avro `fixed` (raw bytes) and `uuid` logical string
                // based on Field metadata, mirroring schema generation rules.
                let arr = array
                    .as_any()
                    .downcast_ref::<FixedSizeBinaryArray>()
                    .ok_or_else(|| ArrowError::SchemaError("Expected FixedSizeBinaryArray".into()))?;
                let md = field.metadata();
                let is_uuid = md.get("logicalType").is_some_and(|v| v == "uuid")
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
            DataType::Interval(IntervalUnit::MonthDayNano) => {
                let arr = array.as_primitive::<IntervalMonthDayNanoType>();
                NullableEncoder::new(
                    Encoder::IntervalMonthDayNano(IntervalMonthDayNanoEncoder(arr)),
                    nulls,
                    has_nulls,
                )
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
            DataType::List(_) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| ArrowError::SchemaError("Expected ListArray".into()))?;
                let enc = ListEncoder32::try_new(arr, None, None)?;
                NullableEncoder::new(Encoder::List(Box::new(enc)), nulls, has_nulls)
            }
            DataType::LargeList(_) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<LargeListArray>()
                    .ok_or_else(|| ArrowError::SchemaError("Expected LargeListArray".into()))?;
                let enc = ListEncoder64::try_new(arr, None, None)?;
                NullableEncoder::new(Encoder::LargeList(Box::new(enc)), nulls, has_nulls)
            }
            DataType::Map(_, _) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<MapArray>()
                    .ok_or_else(|| ArrowError::SchemaError("Expected MapArray".into()))?;
                let enc = MapEncoder::try_new(arr, None, None)?;
                NullableEncoder::new(Encoder::Map(Box::new(enc)), nulls, has_nulls)
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
/// Per Avro 1.11.1 spec, an enum is encoded as an **int** equal to the
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

/// Avro `record` encoder for Arrow `StructArray`
///
/// The children are stored in **Avro order**, and each child carries its
/// own per‑site `Nullability`, so union indices are written exactly as the
/// Avro header declares.
struct StructChildEncoder<'a> {
    nullability: Option<Nullability>,
    pre: Option<u8>,
    enc: NullableEncoder<'a>,
}

struct StructEncoder<'a> {
    children: Vec<StructChildEncoder<'a>>,
}

impl<'a> StructEncoder<'a> {
    /// Unified constructor: uses Avro order and per‑child `Nullability` when provided,
    /// otherwise legacy defaults (NullFirst if the Arrow child field is nullable).
    fn try_new(
        array: &'a StructArray,
        plan_children: Option<&[StructChildPlan]>,
    ) -> Result<Self, ArrowError> {
        let fields = match array.data_type() {
            DataType::Struct(fs) => fs,
            _ => return Err(ArrowError::SchemaError("Expected Struct".into())),
        };
        let mut encs = Vec::new();
        match plan_children {
            Some(children_plan) => {
                encs.reserve(children_plan.len());
                for child_plan in children_plan {
                    let idx = child_plan.arrow_index;
                    let col = array.columns().get(idx).ok_or_else(|| {
                        ArrowError::SchemaError(format!("Struct child index {idx} out of range"))
                    })?;
                    let field = fields[idx].as_ref();
                    // Use unified helper for value-site preparation
                    let (child, eff_null, pre) = prepare_value_site_encoder(
                        col.as_ref(),
                        field,
                        child_plan.nullability,
                        Some(&child_plan.plan),
                    )?;
                    encs.push(StructChildEncoder {
                        nullability: eff_null, // same as child_plan.nullability
                        pre,
                        enc: child,
                    });
                }
            }
            None => {
                encs.reserve(fields.len());
                for (f_ref, col) in fields.iter().zip(array.columns().iter()) {
                    let f: &Field = f_ref.as_ref();
                    // Legacy default via unified helper (no plan)
                    let (child, nb, pre) =
                        prepare_value_site_encoder(col.as_ref(), f, None, None)?;
                    encs.push(StructChildEncoder {
                        nullability: nb, // f.is_nullable().then_some(Nullability::NullFirst)
                        pre,
                        enc: child,
                    });
                }
            }
        }
        Ok(Self { children: encs })
    }

    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        for child in self.children.iter_mut() {
            write_value_with_union(out, &mut child.enc, child.nullability, child.pre, idx)?;
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
    items_nullability: Option<Nullability>,
    precomputed_branch: Option<u8>,
    values_encoder: &mut NullableEncoder<'_>,
) -> Result<(), ArrowError> {
    encode_blocked_range(out, start, end, |j, out| {
        debug_assert!(
            j >= values_offset,
            "List values offset invariant violated: j < values_offset"
        );
        let j_local = j.saturating_sub(values_offset);
        write_value_with_union(
            out,
            values_encoder,
            items_nullability,
            precomputed_branch,
            j_local,
        )
    })
}

struct ListEncoder<'a, O: arrow_array::OffsetSizeTrait> {
    list: &'a arrow_array::array::GenericListArray<O>,
    values: NullableEncoder<'a>,
    items_nullability: Option<Nullability>,
    pre: Option<u8>,
    values_offset: usize,
}

type ListEncoder32<'a> = ListEncoder<'a, i32>;
type ListEncoder64<'a> = ListEncoder<'a, i64>;

impl<'a, O: arrow_array::OffsetSizeTrait> ListEncoder<'a, O> {
    /// Unified constructor:
    /// - If `item_plan` is Some(..), use the provided plan & nullability.
    /// - Else, use legacy default (NullFirst when child field is nullable).
    fn try_new(
        list: &'a arrow_array::array::GenericListArray<O>,
        items_nullability: Option<Nullability>,
        item_plan: Option<&FieldPlan>,
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
        let (values_enc, effective_nullability, pre) = prepare_value_site_encoder(
            list.values().as_ref(),
            child_field,
            items_nullability,
            item_plan,
        )?;
        Ok(Self {
            list,
            values: values_enc,
            items_nullability: effective_nullability,
            pre,
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
        encode_list_range(
            out,
            start,
            end,
            self.values_offset,
            self.items_nullability,
            self.pre,
            &mut self.values,
        )
    }
}

/// Avro `map` encoder for Arrow `MapArray`
///
/// Each row encodes as one (possibly empty) map block sequence:
///   count (long), then for each entry: key (string), value (union index if nullable + value),
///   terminated by a zero count. Keys are strings per Avro spec. :contentReference[oaicite:1]{index=1}
/// Internal key array kind used by Map encoder.
enum KeyKind<'a> {
    Utf8(&'a GenericStringArray<i32>),
    LargeUtf8(&'a GenericStringArray<i64>),
}

struct MapEncoder<'a> {
    map: &'a MapArray,
    keys: KeyKind<'a>,
    values: NullableEncoder<'a>,
    values_nullability: Option<Nullability>,
    pre: Option<u8>,
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
    // - If no plan is provided, default to Arrow field nullability → NullFirst.
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

/// Generic helper that writes `(key, value)` pairs for `[start, end)` using a
/// `GenericStringArray<O>` for keys. This is monomorphized for Utf8/LargeUtf8
/// (static dispatch), so the hot loop has no per-entry type branching.
#[inline]
fn encode_map_entries_for_keys<W, O>(
    out: &mut W,
    arr: &GenericStringArray<O>,
    start: usize,
    end: usize,
    keys_offset: usize,
    values_offset: usize,
    values_nullability: Option<Nullability>,
    precomputed_branch: Option<u8>,
    values: &mut NullableEncoder<'_>,
) -> Result<(), ArrowError>
where
    W: Write + ?Sized,
    O: arrow_array::OffsetSizeTrait,
{
    encode_blocked_range(out, start, end, |j, out| {
        debug_assert!(j >= keys_offset && j >= values_offset);
        let j_key = j.saturating_sub(keys_offset);
        let j_val = j.saturating_sub(values_offset);
        // Key (string)
        let s = arr.value(j_key);
        write_len_prefixed(out, s.as_bytes())?;
        // Value (union index if necessary, then payload)
        write_value_with_union(out, values, values_nullability, precomputed_branch, j_val)
    })
}

impl<'a> MapEncoder<'a> {
    /// Unified constructor:
    /// - If `value_plan` is Some(..), use the provided plan & nullability.
    /// - Else, legacy default (NullFirst when Arrow value field is nullable).
    fn try_new(
        map: &'a MapArray,
        values_nullability: Option<Nullability>,
        value_plan: Option<&FieldPlan>,
    ) -> Result<Self, ArrowError> {
        let (keys, keys_offset, value_field, values_offset) = resolve_map_components(map)?;
        // Refactored: use unified helper for value-site preparation
        let (values_enc, effective_nullability, pre) = prepare_value_site_encoder(
            map.values().as_ref(),
            value_field,
            values_nullability,
            value_plan,
        )?;
        Ok(Self {
            map,
            keys,
            values: values_enc,
            values_nullability: effective_nullability,
            pre,
            keys_offset,
            values_offset,
        })
    }

    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        // Compute entry range [start, end)
        let offsets = self.map.offsets();
        // MapArray offsets are i32 and guaranteed >= 0 by Arrow; align style with lists.
        let start = i32_to_usize(offsets[idx])?;
        let end = i32_to_usize(offsets[idx + 1])?;
        match self.keys {
            KeyKind::Utf8(arr) => encode_map_entries_for_keys(
                out,
                arr,
                start,
                end,
                self.keys_offset,
                self.values_offset,
                self.values_nullability,
                self.pre,
                &mut self.values,
            ),
            KeyKind::LargeUtf8(arr) => encode_map_entries_for_keys(
                out,
                arr,
                start,
                end,
                self.keys_offset,
                self.values_offset,
                self.values_nullability,
                self.pre,
                &mut self.values,
            ),
        }
    }
}