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
use crate::codec::{
    AvroDataType, AvroField as CodecAvroField, AvroFieldBuilder, Codec as AvroCodec, Nullability,
};
use crate::schema::{
    AvroSchema as AvroJson, Schema as AvroJsonAst, SchemaGenOptions, SCHEMA_METADATA_KEY,
};

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
                    // Enforce exact reader-compatible shape: Dictionary<Int32, Utf8>
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
        let arrow_index = arrow_schema
            .index_of(name)
            .map_err(|e| ArrowError::SchemaError(format!("Schema mismatch for field '{name}': {e}")))?;
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
/// If `order_override` is `Some`, it is only used when synthesizing Avro JSON
/// from Arrow (i.e., metadata is absent); otherwise the **actual** Avro JSON
/// controls all union orders.
fn derive_plan_for_batch(
    batch: &RecordBatch,
    order_override: Option<Nullability>,
) -> Result<WritePlan, ArrowError> {
    let avro_json = if let Some(json) = batch.schema().metadata.get(SCHEMA_METADATA_KEY) {
        AvroJson::new(json.clone())
    } else {
        let opts = SchemaGenOptions {
            null_union_order: order_override,
        };
        AvroJson::from_arrow_with_options(batch.schema().as_ref(), opts)?
    };
    let avro_ast: AvroJsonAst<'_> = avro_json.schema()?;
    let root = AvroFieldBuilder::new(&avro_ast).build()?;
    build_write_plan_from_avro_root(&root, batch.schema().as_ref())
}

/// Encode a `RecordBatch` in Avro binary format using the **schema‑driven plan**.
pub fn encode_record_batch<W: Write>(batch: &RecordBatch, out: &mut W) -> Result<(), ArrowError> {
    let plan = derive_plan_for_batch(batch, None)?;
    encode_record_batch_with_plan(batch, out, &plan)
}

/// **Deprecated:** legacy options. Kept for compatibility. These no longer
/// directly drive the encoder; instead they are used only to synthesize a
/// temporary Avro schema when none is present in metadata.
#[derive(Debug, Clone, Copy, Default)]
pub struct EncoderOptions {
    /// If `true`, nullable unions are generated as `[T,"null"]` when *creating*
    /// a temporary Avro JSON from Arrow (NullSecond). If `false`, `["null",T]`.
    pub(crate) impala_mode: bool,
}

/// Deprecated: encodes with options by synthesizing an Avro JSON (if needed)
/// matching `opts`, and then following the resulting schema exactly.
pub fn encode_record_batch_with_options<W: Write>(
    batch: &RecordBatch,
    out: &mut W,
    opts: &EncoderOptions,
) -> Result<(), ArrowError> {
    let override_order = if opts.impala_mode {
        Some(Nullability::NullSecond)
    } else {
        Some(Nullability::NullFirst)
    };
    let plan = derive_plan_for_batch(batch, override_order)?;
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
    let plan = derive_plan_for_batch(batch, None)?;
    encode_record_batch_single_object_with_plan(batch, out, prefix, &plan)
}

/// Deprecated: single‑object with legacy options (see note on `encode_record_batch_with_options`)
pub fn encode_record_batch_single_object_with_options<W: Write>(
    batch: &RecordBatch,
    out: &mut W,
    prefix: &[u8; 10],
    opts: &EncoderOptions,
) -> Result<(), ArrowError> {
    let override_order = if opts.impala_mode {
        Some(Nullability::NullSecond)
    } else {
        Some(Nullability::NullFirst)
    };
    let plan = derive_plan_for_batch(batch, override_order)?;
    encode_record_batch_single_object_with_plan(batch, out, prefix, &plan)
}

/// Encode a `RecordBatch` using a precomputed [`WritePlan`].
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
    has_nulls: bool,
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
        let array = arrays
            .get(arrow_index)
            .ok_or_else(|| ArrowError::SchemaError(format!("Column index {arrow_index} out of range")))?;
        let field = fields[arrow_index].as_ref();
        let enc = make_encoder_with_plan(array.as_ref(), field, &col_plan.plan)?;
        let has_nulls = enc.has_nulls();

        out.push(ColumnEncoder {
            nullability: col_plan.nullability,
            has_nulls,
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
        for ColumnEncoder {
            nullability,
            has_nulls,
            enc,
        } in cols.iter_mut()
        {
            if let Some(order) = nullability {
                if *has_nulls {
                    let is_null = enc.is_null(row);
                    write_optional_index(out, is_null, *order)?;
                    if is_null {
                        continue;
                    }
                } else {
                    // Nullable field with no nulls in the column: always write the "value" branch
                    let value_branch_byte = match order {
                        Nullability::NullFirst => 0x02,  // index 1
                        Nullability::NullSecond => 0x00, // index 0
                    };
                    out.write_all(&[value_branch_byte]).map_err(|e| {
                        ArrowError::IoError(format!("write union value branch: {e}"), e)
                    })?;
                }
            }
            enc.encode(row, out)?;
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
fn make_encoder<'a>(array: &'a dyn Array, field: &Field) -> Result<NullableEncoder<'a>, ArrowError> {
    let nulls = array.nulls().cloned();
    let has_nulls = array.null_count() > 0;
    let enc = match array.data_type() {
        DataType::Boolean => {
            let arr = array.as_boolean();
            NullableEncoder::new(Encoder::Boolean(BooleanEncoder(arr)), nulls, has_nulls)
        }
        DataType::Utf8 => {
            let arr = array.as_string::<i32>();
            NullableEncoder::new(Encoder::Utf8(Utf8GenericEncoder::<i32>(arr)), nulls, has_nulls)
        }
        DataType::LargeUtf8 => {
            let arr = array.as_string::<i64>();
            NullableEncoder::new(Encoder::Utf8Large(Utf8GenericEncoder::<i64>(arr)), nulls, has_nulls)
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
            let enc = ListEncoder32::try_new_default(arr)?;
            NullableEncoder::new(Encoder::List(Box::new(enc)), nulls, has_nulls)
        }
        DataType::LargeList(_) => {
            let arr = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| ArrowError::SchemaError("Expected LargeListArray".into()))?;
            let enc = ListEncoder64::try_new_default(arr)?;
            NullableEncoder::new(Encoder::LargeList(Box::new(enc)), nulls, has_nulls)
        }
        DataType::Map(_, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<MapArray>()
                .ok_or_else(|| ArrowError::SchemaError("Expected MapArray".into()))?;
            let enc = MapEncoder::try_new_default(arr)?;
            NullableEncoder::new(Encoder::Map(Box::new(enc)), nulls, has_nulls)
        }
        DataType::Struct(_) => {
            let arr = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| ArrowError::SchemaError("Expected StructArray".into()))?;
            let enc = StructEncoder::try_new_default(arr)?;
            NullableEncoder::new(Encoder::Struct(Box::new(enc)), nulls, has_nulls)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let arr = array.as_primitive::<TimestampMicrosecondType>();
            NullableEncoder::new(Encoder::Timestamp(LongEncoder(arr)), nulls, has_nulls)
        }
        other => {
            return Err(ArrowError::NotYetImplemented(format!(
                "Unsupported data type for Avro encoding in slim build: {other:?}"
            )))
        }
    };
    Ok(enc)
}

/// Plan-aware variant of `make_encoder` that configures nested union orders
/// exactly as specified by `plan`.
fn make_encoder_with_plan<'a>(
    array: &'a dyn Array,
    field: &Field,
    plan: &FieldPlan,
) -> Result<NullableEncoder<'a>, ArrowError> {
    let nulls = array.nulls().cloned();
    let has_nulls = array.null_count() > 0;

    let enc = match (array.data_type(), plan) {
        (DataType::Struct(_), FieldPlan::Struct { children }) => {
            let arr = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| ArrowError::SchemaError("Expected StructArray".into()))?;
            let enc = StructEncoder::try_new_with_plan(arr, children)?;
            NullableEncoder::new(Encoder::Struct(Box::new(enc)), nulls, has_nulls)
        }
        (DataType::List(_), FieldPlan::List { items_nullability, item_plan }) => {
            let arr = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| ArrowError::SchemaError("Expected ListArray".into()))?;
            let enc = ListEncoder32::try_new_with_plan(arr, *items_nullability, item_plan)?;
            NullableEncoder::new(Encoder::List(Box::new(enc)), nulls, has_nulls)
        }
        (DataType::LargeList(_), FieldPlan::List { items_nullability, item_plan }) => {
            let arr = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| ArrowError::SchemaError("Expected LargeListArray".into()))?;
            let enc = ListEncoder64::try_new_with_plan(arr, *items_nullability, item_plan)?;
            NullableEncoder::new(Encoder::LargeList(Box::new(enc)), nulls, has_nulls)
        }
        (DataType::Map(_, _), FieldPlan::Map { values_nullability, value_plan }) => {
            let arr = array
                .as_any()
                .downcast_ref::<MapArray>()
                .ok_or_else(|| ArrowError::SchemaError("Expected MapArray".into()))?;
            let enc = MapEncoder::try_new_with_plan(arr, *values_nullability, value_plan)?;
            NullableEncoder::new(Encoder::Map(Box::new(enc)), nulls, has_nulls)
        }
        // === NEW: Enum support ===
        (DataType::Dictionary(key_dt, value_dt), FieldPlan::Enum { symbols }) => {
            // Enforce the same shape we validated in build_field_plan
            if **key_dt != DataType::Int32 || **value_dt != DataType::Utf8 {
                return Err(ArrowError::SchemaError(
                    "Avro enum requires Dictionary<Int32, Utf8>".into(),
                ));
            }

            let dict = array
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .ok_or_else(|| ArrowError::SchemaError("Expected DictionaryArray<Int32>".into()))?;

            // Validate dictionary values exactly match schema symbols (order & content)
            let values = dict
                .values()
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| ArrowError::SchemaError("Dictionary values must be Utf8".into()))?;

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

            // Keys are the Avro enum indices (zero-based position into symbols)
            // Per Avro spec, enums encode as an int equal to symbol index.
            let keys = dict.keys();
            let enc = EnumEncoder { keys };
            NullableEncoder::new(Encoder::Enum(enc), nulls, has_nulls)
        }
        // Fallback to default scalar encoders for non-nested
        _ => make_encoder(array, field)?,
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
struct StructEncoder<'a> {
    children: Vec<(Option<Nullability>, NullableEncoder<'a>)>,
}

impl<'a> StructEncoder<'a> {
    /// Legacy constructor: preserves previous behavior (NullFirst for nested).
    fn try_new_default(array: &'a StructArray) -> Result<Self, ArrowError> {
        let fields = match array.data_type() {
            DataType::Struct(fs) => fs,
            _ => return Err(ArrowError::SchemaError("Expected Struct".into())),
        };
        let mut encs = Vec::with_capacity(fields.len());
        for (f_ref, col) in fields.iter().zip(array.columns().iter()) {
            let f: &Field = f_ref.as_ref();
            let child = make_encoder(col.as_ref(), f)?;
            encs.push((f.is_nullable().then_some(Nullability::NullFirst), child));
        }
        Ok(Self { children: encs })
    }

    /// Plan‑aware constructor: uses Avro order and per‑child `Nullability`.
    fn try_new_with_plan(
        array: &'a StructArray,
        plan_children: &[StructChildPlan],
    ) -> Result<Self, ArrowError> {
        let fields = match array.data_type() {
            DataType::Struct(fs) => fs,
            _ => return Err(ArrowError::SchemaError("Expected Struct".into())),
        };
        let mut encs = Vec::with_capacity(plan_children.len());
        for child_plan in plan_children {
            let idx = child_plan.arrow_index;
            let col = array
                .columns()
                .get(idx)
                .ok_or_else(|| ArrowError::SchemaError(format!("Struct child index {idx} out of range")))?;
            let field = fields[idx].as_ref();
            let child = make_encoder_with_plan(col.as_ref(), field, &child_plan.plan)?;
            encs.push((child_plan.nullability, child));
        }
        Ok(Self { children: encs })
    }

    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        for (nullable, enc) in self.children.iter_mut() {
            if let Some(order) = nullable {
                let is_null = enc.is_null(idx);
                write_optional_index(out, is_null, *order)?;
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
#[inline]
fn encode_list_range<W: Write + ?Sized>(
    out: &mut W,
    start: usize,
    end: usize,
    values_offset: usize,
    items_nullability: Option<Nullability>,
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
        debug_assert!(
            j >= values_offset,
            "List values offset invariant violated: j < values_offset"
        );
        let j_local = j.saturating_sub(values_offset);
        if let Some(order) = items_nullability {
            let is_null = values_encoder.is_null(j_local);
            write_optional_index(out, is_null, order)?;
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
    items_nullability: Option<Nullability>,
    values_offset: usize,
}

type ListEncoder32<'a> = ListEncoder<'a, i32>;
type ListEncoder64<'a> = ListEncoder<'a, i64>;

impl<'a, O: arrow_array::OffsetSizeTrait> ListEncoder<'a, O> {
    /// Legacy constructor: preserves previous behavior (NullFirst for items).
    fn try_new_default(list: &'a arrow_array::array::GenericListArray<O>) -> Result<Self, ArrowError> {
        let (child_field, items_nullable) = match list.data_type() {
            DataType::List(field) => (field.as_ref(), field.is_nullable()),
            DataType::LargeList(field) => (field.as_ref(), field.is_nullable()),
            _ => {
                return Err(ArrowError::SchemaError(
                    "Expected List or LargeList for ListEncoder".into(),
                ))
            }
        };
        let values_enc = make_encoder(list.values().as_ref(), child_field)?;
        Ok(Self {
            list,
            values: values_enc,
            items_nullability: items_nullable.then_some(Nullability::NullFirst),
            values_offset: list.values().offset(),
        })
    }

    /// Plan‑aware constructor: uses per‑item `Nullability` and nested plan.
    fn try_new_with_plan(
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
        let values_enc =
            make_encoder_with_plan(list.values().as_ref(), child_field, item_plan)?;
        Ok(Self {
            list,
            values: values_enc,
            items_nullability,
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
            &mut self.values,
        )
    }
}

/// Avro `map` encoder for Arrow `MapArray`
///
/// Each row encodes as one (possibly empty) map block sequence:
///   count (long), then for each entry: key (string), value (union index if nullable + value),
///   terminated by a zero count. Keys are strings per Avro spec.
enum KeyArrayRef<'a> {
    Utf8(&'a GenericStringArray<i32>),
    LargeUtf8(&'a GenericStringArray<i64>),
}

struct MapEncoder<'a> {
    map: &'a MapArray,
    keys: KeyArrayRef<'a>,
    values: NullableEncoder<'a>,
    values_nullability: Option<Nullability>,
    keys_offset: usize,
    values_offset: usize,
}

impl<'a> MapEncoder<'a> {
    /// Legacy constructor: default union ordering (NullFirst) if value field nullable.
    fn try_new_default(map: &'a MapArray) -> Result<Self, ArrowError> {
        // Determine key array and offsets
        let keys_arr = map.keys();
        let keys_offset = keys_arr.offset();
        let keys = match keys_arr.data_type() {
            DataType::Utf8 => KeyArrayRef::Utf8(keys_arr.as_ref().as_string::<i32>()),
            DataType::LargeUtf8 => KeyArrayRef::LargeUtf8(keys_arr.as_ref().as_string::<i64>()),
            other => {
                return Err(ArrowError::SchemaError(format!(
                    "Arrow Map keys must be Utf8/LargeUtf8, found: {other:?}"
                )))
            }
        };

        // Find value Field from DataType::Map(entries_struct)
        let (value_field, values_nullable) = match map.data_type() {
            DataType::Map(entries, _sorted) => {
                let entries_struct_fields = match entries.data_type() {
                    DataType::Struct(fs) => fs,
                    other => {
                        return Err(ArrowError::SchemaError(format!(
                            "Arrow Map entries must be Struct, found: {other:?}"
                        )))
                    }
                };
                let v_idx = find_map_value_field_index(entries_struct_fields).ok_or_else(|| {
                    ArrowError::SchemaError("Map entries struct missing value field".into())
                })?;
                let vf = entries_struct_fields[v_idx].as_ref();
                (vf.clone(), vf.is_nullable())
            }
            _ => unreachable!("Validated by MapArray::data_type"),
        };

        let values_enc = make_encoder(map.values().as_ref(), &value_field)?;
        Ok(Self {
            map,
            keys,
            values: values_enc,
            values_nullability: values_nullable.then_some(Nullability::NullFirst),
            keys_offset,
            values_offset: map.values().offset(),
        })
    }

    /// Plan‑aware constructor: preserves Avro union order for values.
    fn try_new_with_plan(
        map: &'a MapArray,
        values_nullability: Option<Nullability>,
        value_plan: &FieldPlan,
    ) -> Result<Self, ArrowError> {
        let keys_arr = map.keys();
        let keys_offset = keys_arr.offset();
        let keys = match keys_arr.data_type() {
            DataType::Utf8 => KeyArrayRef::Utf8(keys_arr.as_ref().as_string::<i32>()),
            DataType::LargeUtf8 => KeyArrayRef::LargeUtf8(keys_arr.as_ref().as_string::<i64>()),
            other => {
                return Err(ArrowError::SchemaError(format!(
                    "Arrow Map keys must be Utf8/LargeUtf8, found: {other:?}"
                )))
            }
        };

        // Locate the Arrow value field from the Map's entries struct
        let value_field = match map.data_type() {
            DataType::Map(entries, _sorted) => {
                let entries_struct_fields = match entries.data_type() {
                    DataType::Struct(fs) => fs,
                    other => {
                        return Err(ArrowError::SchemaError(format!(
                            "Arrow Map entries must be Struct, found: {other:?}"
                        )))
                    }
                };
                let v_idx = find_map_value_field_index(entries_struct_fields).ok_or_else(|| {
                    ArrowError::SchemaError("Map entries struct missing value field".into())
                })?;
                entries_struct_fields[v_idx].as_ref().clone()
            }
            _ => unreachable!("Validated by MapArray::data_type"),
        };

        let values_enc =
            make_encoder_with_plan(map.values().as_ref(), &value_field, value_plan)?;
        Ok(Self {
            map,
            keys,
            values: values_enc,
            values_nullability,
            keys_offset,
            values_offset: map.values().offset(),
        })
    }

    #[inline]
    fn encode<W: Write + ?Sized>(&mut self, idx: usize, out: &mut W) -> Result<(), ArrowError> {
        // Compute entry range [start, end)
        let offsets = self.map.offsets();
        // MapArray offsets are i32 and guaranteed >= 0.
        let start = offsets[idx] as usize;
        let end = offsets[idx + 1] as usize;

        let len = end.saturating_sub(start);
        if len == 0 {
            // Empty map: just write the terminator block
            write_long(out, 0)?;
            return Ok(());
        }

        // Single block containing all entries, then terminator 0
        write_long(out, len as i64)?;

        for j in start..end {
            // Keys
            let j_key = j.saturating_sub(self.keys_offset);
            match self.keys {
                KeyArrayRef::Utf8(arr) => {
                    let s = arr.value(j_key);
                    write_len_prefixed(out, s.as_bytes())?;
                }
                KeyArrayRef::LargeUtf8(arr) => {
                    let s = arr.value(j_key);
                    write_len_prefixed(out, s.as_bytes())?;
                }
            }

            // Values (optional union branch if declared nullable in Avro)
            let j_val = j.saturating_sub(self.values_offset);
            if let Some(order) = self.values_nullability {
                let is_null = self.values.is_null(j_val);
                write_optional_index(out, is_null, order)?;
                if is_null {
                    continue;
                }
            }
            self.values.encode(j_val, out)?;
        }

        // End-of-map marker
        write_long(out, 0)?;
        Ok(())
    }
}