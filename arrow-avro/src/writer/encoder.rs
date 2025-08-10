//! Avro binary encoder for Arrow `RecordBatch`es.
//!
//! Notes & invariants:
//! - Avro `int`/`long` use ZigZag + variable-length encoding.
//! - Strings/bytes are length-prefixed with Avro `long` then raw bytes.
//! - `fixed` types are written as raw bytes with **no** length prefix.
//! - Arrays/Maps are written as a single positive-length block followed by a `0`
//!   block terminator (permitted by the spec), see Avro arrays/maps encoding.
//! - Nullable fields are written as Avro unions of either `["null", T]` (default)
//!   or `[T, "null"]` (Impala/NullSecond) based on `EncoderOptions::impala_mode`.
//! - Some Arrow logical types map to Avro logical types:
//!     * `Time32(Millisecond)` → Avro `time-millis` (int)
//!     * `Time64(Microsecond)` → Avro `time-micros` (long)
//!     * `Timestamp{Milli,Micro}second` → Avro `timestamp-{millis,micros}` (long)
//!     * `Interval(MonthDayNano)` ↔ Avro `duration` (fixed[12], LE: months, days, millis)
//!     * `Decimal{128,256}` over Avro `bytes` (two's complement, minimal length)
//!     * UTF-8 `DictionaryArray` → Avro `enum` (index as int)
//!
//! References:
//! - Avro 1.11.1 Spec: binary encoding of primitives, arrays/maps/unions/enums,
//!   and logical types (decimal/uuid/duration).
//!   https://avro.apache.org/docs/1.11.1/specification/
//! - Arrow JSON writer encoder architecture.
//!   https://github.com/apache/arrow-rs/blob/main/arrow-json/src/writer/encoder.rs

use arrow_array::types::{Int16Type, Int32Type, Int64Type, Int8Type};
use arrow_array::*;
use arrow_schema::{ArrowError, DataType, Field, Fields, IntervalUnit, TimeUnit, UnionMode};
use std::io::Write;

/// Behavior knobs for the Avro encoder.
///
/// Currently only `impala_mode` is exposed. When `true`, optional/nullable
/// values are encoded as Avro unions with **null second** (Impala format),
/// i.e. `[T, "null"]`. When `false` (default), we use `["null", T]`.
#[derive(Debug, Clone, Copy)]
pub struct EncoderOptions {
    /// If `true`, encode nullability as `[T, "null"]` (Impala / null-second).
    /// If `false` (default), encode as `["null", T]` (null-first).
    pub impala_mode: bool,
}

impl Default for EncoderOptions {
    fn default() -> Self {
        Self { impala_mode: false }
    }
}

/// Encode a single Avro-`long` using ZigZag + variable length, buffered.
///
/// Spec: https://avro.apache.org/docs/1.11.1/specification/#binary-encoding
#[inline]
pub fn write_long<W: Write>(writer: &mut W, value: i64) -> Result<(), ArrowError> {
    // ZigZag map i64->u64
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
fn write_int<W: Write>(writer: &mut W, value: i32) -> Result<(), ArrowError> {
    // Per spec, Avro `int` is ZigZag+varint, identical encoding shape to `long` for small values.
    write_long(writer, value as i64)
}

#[inline]
fn write_len_prefixed<W: Write>(writer: &mut W, bytes: &[u8]) -> Result<(), ArrowError> {
    write_long(writer, bytes.len() as i64)?;
    writer
        .write_all(bytes)
        .map_err(|e| ArrowError::IoError(format!("write bytes: {e}"), e))
}

#[inline]
fn write_bool<W: Write>(writer: &mut W, v: bool) -> Result<(), ArrowError> {
    writer
        .write_all(&[if v { 1 } else { 0 }])
        .map_err(|e| ArrowError::IoError(format!("write bool: {e}"), e))
}

#[inline]
fn write_optional_branch<W: Write>(
    writer: &mut W,
    is_null: bool,
    impala_mode: bool,
) -> Result<(), ArrowError> {
    // Branch index: 0-based union arm per Avro unions encoding.
    // - Null-first (default): null => 0, value => 1
    // - Null-second (Impala): value => 0, null => 1
    let branch = if impala_mode {
        if is_null {
            1
        } else {
            0
        }
    } else {
        if is_null {
            0
        } else {
            1
        }
    };
    // Spec says this is an `int`. Using `write_int` (could also be `write_long` for 0/1).
    write_int(writer, branch)
}

/// Public API: encode a `RecordBatch` in Avro binary format using **default options**.
///
/// This is the function called by `writer/mod.rs` and **must preserve** its signature.
pub fn encode_record_batch<W: Write>(batch: &RecordBatch, out: &mut W) -> Result<(), ArrowError> {
    encode_record_batch_with_options(batch, out, &EncoderOptions::default())
}

/// Encode a `RecordBatch` with explicit `EncoderOptions`.
///
/// This variant allows callers to select non-default behaviors (e.g. Impala null-second).
pub fn encode_record_batch_with_options<W: Write>(
    batch: &RecordBatch,
    out: &mut W,
    opts: &EncoderOptions,
) -> Result<(), ArrowError> {
    let schema = batch.schema();
    let fields = schema.fields();
    let columns = batch.columns();

    // Avro records are row-oriented: concat of field values in schema order.
    for row in 0..batch.num_rows() {
        encode_row(fields, columns, row, out, opts)?;
    }
    Ok(())
}

fn encode_row<W: Write>(
    fields: &Fields,
    cols: &[ArrayRef],
    row_idx: usize,
    out: &mut W,
    opts: &EncoderOptions,
) -> Result<(), ArrowError> {
    for (field, col) in fields.iter().zip(cols) {
        encode_field_value(field, col.as_ref(), row_idx, out, opts)?;
    }
    Ok(())
}

fn encode_field_value<W: Write>(
    field: &Field,
    array: &dyn Array,
    index: usize,
    out: &mut W,
    opts: &EncoderOptions,
) -> Result<(), ArrowError> {
    if field.is_nullable() {
        let is_null = array.is_null(index);
        write_optional_branch(out, is_null, opts.impala_mode)?;
        if is_null {
            // For the `null` branch there is no payload to write.
            return Ok(());
        }
    }
    encode_value(array, field.data_type(), index, out, opts)
}

fn encode_value<W: Write>(
    array: &dyn Array,
    dt: &DataType,
    index: usize,
    out: &mut W,
    opts: &EncoderOptions,
) -> Result<(), ArrowError> {
    match dt {
        DataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            write_bool(out, arr.value(index))?;
        }
        DataType::Int8 => {
            let v = array
                .as_any()
                .downcast_ref::<Int8Array>()
                .unwrap()
                .value(index) as i32;
            write_int(out, v)?;
        }
        DataType::Int16 => {
            let v = array
                .as_any()
                .downcast_ref::<Int16Array>()
                .unwrap()
                .value(index) as i32;
            write_int(out, v)?;
        }
        DataType::Int32 => {
            let v = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(index);
            write_int(out, v)?;
        }
        DataType::Int64 => {
            let v = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(index);
            write_long(out, v)?;
        }
        DataType::UInt8 => {
            let v = array
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap()
                .value(index) as i32;
            write_int(out, v)?;
        }
        DataType::UInt16 => {
            let v = array
                .as_any()
                .downcast_ref::<UInt16Array>()
                .unwrap()
                .value(index) as i32;
            write_int(out, v)?;
        }
        DataType::UInt32 => {
            let v = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap()
                .value(index) as i64;
            // Avro `int` is i32; values > i32::MAX must be rejected.
            if v > i32::MAX as i64 {
                return Err(ArrowError::InvalidArgumentError(
                    "UInt32 value exceeds Avro int range".into(),
                ));
            }
            write_int(out, v as i32)?;
        }
        DataType::UInt64 => {
            let v = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(index);
            if v > i64::MAX as u64 {
                return Err(ArrowError::InvalidArgumentError(
                    "UInt64 value exceeds Avro long range".into(),
                ));
            }
            write_long(out, v as i64)?;
        }
        DataType::Float32 => {
            let bits = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(index)
                .to_bits();
            // Avro floats/doubles are IEEE-754 in little-endian. See spec.
            out.write_all(&bits.to_le_bytes())
                .map_err(|e| ArrowError::IoError(format!("write f32: {e}"), e))?;
        }
        DataType::Float64 => {
            let bits = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(index)
                .to_bits();
            out.write_all(&bits.to_le_bytes())
                .map_err(|e| ArrowError::IoError(format!("write f64: {e}"), e))?;
        }
        DataType::Utf8 => {
            let s = array
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(index);
            write_len_prefixed(out, s.as_bytes())?;
        }
        DataType::LargeUtf8 => {
            let s = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .unwrap()
                .value(index);
            write_len_prefixed(out, s.as_bytes())?;
        }
        DataType::Binary => {
            let b = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .value(index);
            write_len_prefixed(out, b)?;
        }
        DataType::LargeBinary => {
            let b = array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .unwrap()
                .value(index);
            write_len_prefixed(out, b)?;
        }
        DataType::FixedSizeBinary(n) => {
            // Avro `fixed(N)` is encoded as **exactly N bytes** (no length prefix).
            // Spec: https://avro.apache.org/docs/1.11.1/specification/#fixed
            let b = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap()
                .value(index);
            debug_assert_eq!(b.len(), *n as usize);
            out.write_all(b)
                .map_err(|e| ArrowError::IoError(format!("write fixed[{n}]: {e}"), e))?;
        }
        DataType::Decimal128(_, _) => {
            // Per Avro decimal (bytes form): two's-complement big-endian of unscaled,
            // minimally sized (strip redundant sign extension bytes).
            let val = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .unwrap()
                .value(index);
            let mut bytes = val.to_be_bytes().to_vec();
            strip_sign_extension(&mut bytes);
            write_len_prefixed(out, &bytes)?;
        }
        DataType::Decimal256(_, _) => {
            let val = array
                .as_any()
                .downcast_ref::<Decimal256Array>()
                .unwrap()
                .value(index);
            let mut bytes = val.to_be_bytes().to_vec();
            strip_sign_extension(&mut bytes);
            write_len_prefixed(out, &bytes)?;
        }
        DataType::Date32 => {
            // Avro `date` is an int (#days since epoch).
            // https://avro.apache.org/docs/1.11.1/specification/#date
            let days = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .unwrap()
                .value(index);
            write_int(out, days)?;
        }
        DataType::Date64 => {
            // Arrow Date64 is ms from epoch. Avro has no `date64` logical type;
            // this will be written as a `long` (or mapped by schema to timestamp-millis
            // if chosen upstream). Keep raw long to match writer-chosen schema.
            let ms = array
                .as_any()
                .downcast_ref::<Date64Array>()
                .unwrap()
                .value(index);
            write_long(out, ms)?;
        }
        DataType::Time32(unit) => {
            match unit {
                TimeUnit::Second => {
                    // Convert seconds since midnight -> millis per Avro time-millis (int)
                    let secs = array
                        .as_any()
                        .downcast_ref::<Time32SecondArray>()
                        .unwrap()
                        .value(index);
                    let millis = secs.checked_mul(1000).ok_or_else(|| {
                        ArrowError::ComputeError("time32(second) overflow".into())
                    })?;
                    write_int(out, millis)?;
                }
                TimeUnit::Millisecond => {
                    let millis = array
                        .as_any()
                        .downcast_ref::<Time32MillisecondArray>()
                        .unwrap()
                        .value(index);
                    write_int(out, millis)?;
                }
                _ => {
                    return Err(ArrowError::NotYetImplemented(
                        "Time32 with micro/nano is not valid".into(),
                    ))
                }
            }
        }
        DataType::Time64(unit) => {
            match unit {
                TimeUnit::Microsecond => {
                    let micros = array
                        .as_any()
                        .downcast_ref::<Time64MicrosecondArray>()
                        .unwrap()
                        .value(index);
                    write_long(out, micros)?;
                }
                TimeUnit::Nanosecond => {
                    // Avro has no time-nanos; to map to time-micros we must be exact.
                    let nanos = array
                        .as_any()
                        .downcast_ref::<Time64NanosecondArray>()
                        .unwrap()
                        .value(index);
                    if nanos % 1_000 != 0 {
                        return Err(ArrowError::InvalidArgumentError(
                            "Cannot encode Time64(Nanosecond) exactly as Avro time-micros: \
                             value not divisible by 1,000"
                                .into(),
                        ));
                    }
                    write_long(out, nanos / 1_000)?;
                }
                _ => {
                    return Err(ArrowError::NotYetImplemented(
                        "Time64 second/millisecond not supported".into(),
                    ))
                }
            }
        }
        DataType::Timestamp(unit, _tz) => {
            // Keep raw units; Avro schema generation (format.rs) selects the logical type
            // (`timestamp-millis` or `timestamp-micros`) when expressible, otherwise a plain `long`.
            let ts = match unit {
                TimeUnit::Second => array
                    .as_any()
                    .downcast_ref::<TimestampSecondArray>()
                    .unwrap()
                    .value(index),
                TimeUnit::Millisecond => array
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .unwrap()
                    .value(index),
                TimeUnit::Microsecond => array
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap()
                    .value(index),
                TimeUnit::Nanosecond => array
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .unwrap()
                    .value(index),
            };
            write_long(out, ts)?;
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            // Avro `duration` is a 12-byte fixed logical type:
            // LE i32 months, LE i32 days, LE i32 milliseconds.
            // https://avro.apache.org/docs/1.11.1/specification/#duration
            let v = array
                .as_any()
                .downcast_ref::<IntervalMonthDayNanoArray>()
                .unwrap()
                .value(index);
            let months = v.months;
            let days = v.days;
            // Require exact millis to avoid loss
            if v.nanoseconds % 1_000_000 != 0 {
                return Err(ArrowError::InvalidArgumentError(
                    "IntervalMonthDayNano cannot be encoded as Avro duration: \
                     nanoseconds not divisible by 1,000,000"
                        .into(),
                ));
            }
            let millis = (v.nanoseconds / 1_000_000) as i32;

            let mut buf = [0u8; 12];
            buf[0..4].copy_from_slice(&months.to_le_bytes());
            buf[4..8].copy_from_slice(&days.to_le_bytes());
            buf[8..12].copy_from_slice(&millis.to_le_bytes());
            out.write_all(&buf)
                .map_err(|e| ArrowError::IoError(format!("write duration: {e}"), e))?;
        }
        DataType::Struct(child_fields) => {
            let struct_arr = array.as_any().downcast_ref::<StructArray>().unwrap();
            for (child_idx, field) in child_fields.iter().enumerate() {
                let child_arr = struct_arr.column(child_idx);
                if field.is_nullable() {
                    let is_null = child_arr.is_null(index);
                    write_optional_branch(out, is_null, opts.impala_mode)?;
                    if is_null {
                        continue;
                    }
                }
                encode_value(child_arr.as_ref(), field.data_type(), index, out, opts)?;
            }
        }
        DataType::List(child) | DataType::LargeList(child) => {
            // Write one positive-length block and a terminating zero block
            let (len, offset, values): (i64, i64, ArrayRef) = match dt {
                DataType::List(_) => {
                    let list_arr = array.as_any().downcast_ref::<ListArray>().unwrap();
                    (
                        list_arr.value_length(index) as i64,
                        list_arr.value_offsets()[index] as i64,
                        list_arr.values().clone(),
                    )
                }
                DataType::LargeList(_) => {
                    let list_arr = array.as_any().downcast_ref::<LargeListArray>().unwrap();
                    (
                        list_arr.value_length(index),
                        list_arr.value_offsets()[index],
                        list_arr.values().clone(),
                    )
                }
                _ => unreachable!(),
            };
            write_long(out, len)?;
            let item_dt = child.data_type();
            for j in 0..len {
                let elem_idx = (offset + j) as usize;
                if child.is_nullable() {
                    let is_null = values.is_null(elem_idx);
                    write_optional_branch(out, is_null, opts.impala_mode)?;
                    if is_null {
                        continue;
                    }
                }
                encode_value(values.as_ref(), item_dt, elem_idx, out, opts)?;
            }
            write_long(out, 0)?; // block terminator
        }
        DataType::Map(value_field, _keys_sorted) => {
            // Avro maps have string keys by definition.
            // Encoded as one positive-length block and a terminating zero block:
            // count, then repeating (key:string, value:<schema>).
            let map_arr = array.as_any().downcast_ref::<MapArray>().unwrap();
            let len = map_arr.value_length(index);
            write_long(out, len.into())?;
            let offset = map_arr.value_offsets()[index];
            let entries = map_arr.entries();
            let key_arr = entries.column(0);
            let val_arr = entries.column(1);
            for j in 0..len {
                let idx = (offset + j) as usize;
                // Key: must be a (non-null) UTF-8 string
                if key_arr.is_null(idx) {
                    return Err(ArrowError::InvalidArgumentError(
                        "Avro map keys cannot be null".into(),
                    ));
                }
                // Arrow Map keys are Utf8 per Arrow spec; support Utf8/LargeUtf8 for robustness.
                match key_arr.data_type() {
                    DataType::Utf8 => {
                        let s = key_arr
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .value(idx);
                        write_len_prefixed(out, s.as_bytes())?;
                    }
                    DataType::LargeUtf8 => {
                        let s = key_arr
                            .as_any()
                            .downcast_ref::<LargeStringArray>()
                            .unwrap()
                            .value(idx);
                        write_len_prefixed(out, s.as_bytes())?;
                    }
                    other => {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "Avro map key must be Utf8/LargeUtf8, got {other:?}"
                        )));
                    }
                }
                // Value (nullable?)
                if value_field.is_nullable() {
                    let is_null = val_arr.is_null(idx);
                    write_optional_branch(out, is_null, opts.impala_mode)?;
                    if is_null {
                        continue;
                    }
                }
                encode_value(val_arr.as_ref(), value_field.data_type(), idx, out, opts)?;
            }
            write_long(out, 0)?; // block terminator
        }
        DataType::Union(field_set, mode) => {
            let union_arr = array.as_any().downcast_ref::<UnionArray>().unwrap();
            let type_id = union_arr.type_id(index);
            // Determine the branch position in the union schema
            let (variant_idx, child_field) = field_set
                .iter()
                .enumerate()
                .find_map(|(i, (id, f))| if id == type_id { Some((i, f)) } else { None })
                .ok_or_else(|| ArrowError::InvalidArgumentError("union type id missing".into()))?;
            // Avro union branch index is an `int` (0-based) and depends on schema order
            write_int(out, variant_idx as i32)?;
            // Access the child array by **type id** (i8), not by variant index.
            let child_arr = union_arr.child(type_id);
            let child_idx = match mode {
                UnionMode::Sparse => index,
                UnionMode::Dense => union_arr.value_offset(index) as usize,
            };
            encode_value(
                child_arr.as_ref(),
                child_field.data_type(),
                child_idx,
                out,
                opts,
            )?;
        }
        DataType::Dictionary(key_type, value_type) => {
            // Encode Arrow dictionary of UTF-8 values as Avro enum (index).
            // Per Avro spec, enums are encoded as `int` symbol position.
            // https://avro.apache.org/docs/1.11.1/specification/#enums
            match value_type.as_ref() {
                DataType::Utf8 | DataType::LargeUtf8 => {
                    // Allow standard dictionary key widths
                    if key_type.as_ref() == &DataType::Int8 {
                        let arr = array
                            .as_any()
                            .downcast_ref::<DictionaryArray<Int8Type>>()
                            .unwrap();
                        if arr.is_null(index) {
                            return Err(ArrowError::InvalidArgumentError(
                                "Unexpected null in non-nullable dictionary. \
                                 (Nullable handled by field-level union.)"
                                    .into(),
                            ));
                        }
                        let k = arr.keys().value(index) as i32;
                        write_int(out, k)?;
                    } else if key_type.as_ref() == &DataType::Int16 {
                        let arr = array
                            .as_any()
                            .downcast_ref::<DictionaryArray<Int16Type>>()
                            .unwrap();
                        if arr.is_null(index) {
                            return Err(ArrowError::InvalidArgumentError(
                                "Unexpected null in non-nullable dictionary.".into(),
                            ));
                        }
                        let k = arr.keys().value(index) as i32;
                        write_int(out, k)?;
                    } else if key_type.as_ref() == &DataType::Int32 {
                        let arr = array
                            .as_any()
                            .downcast_ref::<DictionaryArray<Int32Type>>()
                            .unwrap();
                        if arr.is_null(index) {
                            return Err(ArrowError::InvalidArgumentError(
                                "Unexpected null in non-nullable dictionary.".into(),
                            ));
                        }
                        let k = arr.keys().value(index);
                        write_int(out, k)?;
                    } else if key_type.as_ref() == &DataType::Int64 {
                        let arr = array
                            .as_any()
                            .downcast_ref::<DictionaryArray<Int64Type>>()
                            .unwrap();
                        if arr.is_null(index) {
                            return Err(ArrowError::InvalidArgumentError(
                                "Unexpected null in non-nullable dictionary.".into(),
                            ));
                        }
                        let k = arr.keys().value(index);
                        if k > i32::MAX as i64 {
                            return Err(ArrowError::InvalidArgumentError(
                                "Dictionary index exceeds Avro enum index range".into(),
                            ));
                        }
                        write_int(out, k as i32)?;
                    } else {
                        return Err(ArrowError::NotYetImplemented(format!(
                            "Unsupported dictionary key type for Avro enum: {key_type:?}"
                        )));
                    }
                }
                // Other dictionary value types currently have no direct Avro mapping.
                other => {
                    return Err(ArrowError::NotYetImplemented(format!(
                        "Dictionary of {other:?} not supported for Avro encoding (expect Utf8)"
                    )));
                }
            }
        }
        other => {
            return Err(ArrowError::NotYetImplemented(format!(
                "DataType {:?} not supported in Avro encoder",
                other
            )))
        }
    }

    Ok(())
}

/// Strip redundant sign-extension bytes for Avro decimal (bytes form).
///
/// For two's-complement big-endian representation, remove leading 0x00 (for +)
/// or 0xFF (for -) as long as doing so does not change the sign bit of the next byte.
///
/// See Avro spec "Decimal" logical type over `bytes`.
fn strip_sign_extension(bytes: &mut Vec<u8>) {
    while bytes.len() > 1
        && ((bytes[0] == 0x00 && (bytes[1] & 0x80) == 0)
            || (bytes[0] == 0xFF && (bytes[1] & 0x80) != 0))
    {
        bytes.remove(0);
    }
}
