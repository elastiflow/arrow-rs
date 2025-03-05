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

//! Row-based encoder logic for Avro container files.

use arrow_array::{
    Array, BinaryArray, BooleanArray, Decimal128Array, Decimal256Array, DictionaryArray,
    FixedSizeBinaryArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array,
    Int64Array, LargeListArray, ListArray, MapArray, PrimitiveArray, StringArray,
    StructArray, TimestampMicrosecondArray, TimestampMillisecondArray,
};
use arrow_array::builder::{Decimal128Builder, Decimal256Builder};
use arrow_array::types::{
    Int16Type, Int32Type, Int64Type, Int8Type, IntervalMonthDayNanoType,
    Time32MillisecondType, Time64MicrosecondType,
};
use arrow_buffer::{i256, IntervalMonthDayNano};
use arrow_schema::{ArrowError, DataType, Field, IntervalUnit, TimeUnit};

use crate::codec::Nullability;
use crate::writer::zigzag::write_zigzag_long;

/// A `FieldEncoder` is responsible for writing a single Arrow column's
/// **one row** to Avro bytes.
#[derive(Debug)]
enum FieldEncoder {
    /// Avro null
    Null,
    /// Avro bool
    Boolean,
    /// Avro int32
    Int32,
    /// Avro int64
    Int64,
    /// Avro float32
    Float32,
    /// Avro float64
    Float64,
    /// Avro bytes
    Binary,
    /// Avro string
    Utf8,
    /// Avro record
    Record(Vec<FieldEncoder>),
    /// Avro enum (dictionary)
    Enum(DictKeyEnc),
    /// Avro array
    Array(Box<FieldEncoder>),
    /// Avro map
    Map(Box<FieldEncoder>),
    /// Avro fixed
    Fixed(usize),
    /// Avro decimal128
    Decimal128(usize, usize, Option<usize>),
    /// Avro decimal256
    Decimal256(usize, usize, Option<usize>),
    /// Avro date32
    Date32,
    /// Avro time-millis
    TimeMillis,
    /// Avro time-micros
    TimeMicros,
    /// Avro timestamp-millis (bool indicates whether to store as UTC)
    TimestampMillis(bool),
    /// Avro timestamp-micros (bool indicates whether to store as UTC)
    TimestampMicros(bool),
    /// Avro 16-byte UUID
    Uuid,
    /// Avro 12-byte duration (months, days, ms)
    Duration,
    /// For union-encoded columns. The second param is the union ordering.
    ///
    /// - `NullFirst` => `[ "null", T ]` => branch=0 => null, branch=1 => T
    /// - `NullSecond` => `[ T, "null" ]` => branch=0 => T, branch=1 => null
    Nullable(Box<FieldEncoder>, Nullability),
}

/// The integral type used for dictionary keys in an Avro enum.
#[derive(Debug, Clone, Copy)]
enum DictKeyEnc {
    Int8,
    Int16,
    Int32,
    Int64,
}

/// A `RecordEncoder` converts entire Arrow rows into Avro bytes by
/// calling `encode_one` on each column row.
#[derive(Debug)]
pub struct RecordEncoder {
    fields: Vec<FieldEncoder>,
}

impl RecordEncoder {
    /// Create a new `RecordEncoder` from an Arrow schema,
    /// specifying `impala_mode=true` if we want `[ T, "null" ]` ordering
    /// for nullable fields.
    pub fn try_new(
        schema: &arrow_schema::Schema,
        impala_mode: bool,
    ) -> Result<Self, ArrowError> {
        let mut fields = Vec::with_capacity(schema.fields().len());
        for f in schema.fields() {
            fields.push(FieldEncoder::try_new(f, impala_mode)?);
        }
        Ok(Self { fields })
    }

    /// Encode one row from a `RecordBatch` into `out`.
    pub fn encode_row(
        &mut self,
        _schema: &arrow_schema::Schema,
        batch: &arrow_array::RecordBatch,
        row_idx: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), ArrowError> {
        for (col_idx, field_enc) in self.fields.iter_mut().enumerate() {
            let col = batch.column(col_idx);
            field_enc.encode_one(col.as_ref(), row_idx, out)?;
        }
        Ok(())
    }

    /// Convenience to encode a single row into a new `Vec<u8>`.
    pub fn encode_row_to_vec(
        &mut self,
        schema: &arrow_schema::Schema,
        batch: &arrow_array::RecordBatch,
        row_idx: usize,
    ) -> Result<Vec<u8>, ArrowError> {
        let mut buf = Vec::new();
        self.encode_row(schema, batch, row_idx, &mut buf)?;
        Ok(buf)
    }
}

impl FieldEncoder {
    /// Build a `FieldEncoder` for the given Arrow `Field`.
    ///
    /// If `impala` is true, and the field is nullable, we produce a union
    /// that uses `[ T, "null" ]` ordering (`Nullability::NullSecond`).
    pub fn try_new(field: &Field, impala: bool) -> Result<Self, ArrowError> {
        let dt = field.data_type();
        let enc = match dt {
            DataType::Null => Self::Null,
            DataType::Boolean => Self::Boolean,
            DataType::Int8 | DataType::Int16 | DataType::Int32 => Self::Int32,
            DataType::Int64 => Self::Int64,
            DataType::Float32 => Self::Float32,
            DataType::Float64 => Self::Float64,
            DataType::Binary | DataType::LargeBinary => Self::Binary,
            DataType::Utf8 | DataType::LargeUtf8 => Self::Utf8,
            DataType::Struct(fields) => {
                let mut child_encoders = Vec::with_capacity(fields.len());
                for child_field in fields {
                    child_encoders.push(Self::try_new(child_field.as_ref(), impala)?);
                }
                Self::Record(child_encoders)
            }
            DataType::Dictionary(key_dt, _value_dt) => {
                let valid_key = matches!(
                    key_dt.as_ref(),
                    DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
                );
                // If not recognized as enum => encode as string by default
                if !valid_key {
                    Self::Utf8
                } else if let Some(sym_json_str) = field.metadata().get("avro.enum.symbols") {
                    // parse:
                    let parsed: serde_json::Value = serde_json::from_str(sym_json_str)
                        .map_err(|e| {
                            ArrowError::ParseError(format!(
                                "Invalid JSON in avro.enum.symbols: {e}"
                            ))
                        })?;
                    if !parsed.is_array() {
                        // fallback
                        Self::Utf8
                    } else {
                        let key_enc = match key_dt.as_ref() {
                            DataType::Int8 => DictKeyEnc::Int8,
                            DataType::Int16 => DictKeyEnc::Int16,
                            DataType::Int32 => DictKeyEnc::Int32,
                            DataType::Int64 => DictKeyEnc::Int64,
                            _ => DictKeyEnc::Int32,
                        };
                        Self::Enum(key_enc)
                    }
                } else {
                    Self::Utf8
                }
            }
            DataType::List(child_field) | DataType::LargeList(child_field) => {
                let child_enc = Self::try_new(child_field.as_ref(), impala)?;
                Self::Array(Box::new(child_enc))
            }
            DataType::FixedSizeList(child_field, _sz) => {
                let child_enc = Self::try_new(child_field.as_ref(), impala)?;
                Self::Array(Box::new(child_enc))
            }
            DataType::Map(entry_field, _keys_sorted) => match entry_field.data_type() {
                DataType::Struct(fs) if fs.len() == 2 => {
                    let val_field = &fs[1];
                    let val_enc = Self::try_new(val_field, impala)?;
                    Self::Map(Box::new(val_enc))
                }
                _ => Self::Null,
            },
            DataType::FixedSizeBinary(n) => {
                let md = field.metadata();
                match md.get("logicalType").map(|s| s.as_str()) {
                    Some("uuid") if *n == 16 => Self::Uuid,
                    Some("duration") if *n == 12 => Self::Duration,
                    _ => Self::Fixed(*n as usize),
                }
            }
            DataType::Decimal128(p, s) => {
                Self::Decimal128(*p as usize, *s as usize, Some(16))
            }
            DataType::Decimal256(p, s) => {
                Self::Decimal256(*p as usize, *s as usize, Some(32))
            }
            DataType::Date32 => Self::Date32,
            DataType::Time32(TimeUnit::Millisecond) => Self::TimeMillis,
            DataType::Time64(TimeUnit::Microsecond) => Self::TimeMicros,
            DataType::Timestamp(TimeUnit::Millisecond, tz_opt) => {
                let is_utc = tz_opt.as_deref() == Some("+00:00");
                Self::TimestampMillis(is_utc)
            }
            DataType::Timestamp(TimeUnit::Microsecond, tz_opt) => {
                let is_utc = tz_opt.as_deref() == Some("+00:00");
                Self::TimestampMicros(is_utc)
            }
            DataType::Interval(IntervalUnit::MonthDayNano) => Self::Duration,
            other => {
                eprintln!("WARN: unhandled Arrow type {other:?}, encoding as Null");
                Self::Null
            }
        };
        if field.is_nullable() && !matches!(enc, Self::Null) {
            let nullability = if impala {
                Nullability::NullSecond
            } else {
                Nullability::NullFirst
            };
            Ok(Self::Nullable(Box::new(enc), nullability))
        } else {
            Ok(enc)
        }
    }

    /// Encode exactly one row from an Arrow array into Avro bytes.
    pub fn encode_one(
        &self,
        array: &dyn Array,
        row_idx: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), ArrowError> {
        match self {
            FieldEncoder::Null => Ok(()),
            FieldEncoder::Boolean => {
                let bool_arr = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        ArrowError::ParseError("Not a Boolean array".to_string())
                    })?;
                let val = bool_arr.value(row_idx);
                out.push(val as u8);
                Ok(())
            }
            FieldEncoder::Int32 => {
                let int_arr = array
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| {
                        ArrowError::ParseError("Not an Int32 array".to_string())
                    })?;
                let val = int_arr.value(row_idx);
                write_zigzag_long(val as i64, out)?;
                Ok(())
            }
            FieldEncoder::Int64 => {
                let int_arr = array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        ArrowError::ParseError("Not an Int64 array".to_string())
                    })?;
                let val = int_arr.value(row_idx);
                write_zigzag_long(val, out)?;
                Ok(())
            }
            FieldEncoder::Float32 => {
                let float_arr = array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        ArrowError::ParseError("Not a Float32 array".to_string())
                    })?;
                let val = float_arr.value(row_idx).to_le_bytes();
                out.extend_from_slice(&val);
                Ok(())
            }
            FieldEncoder::Float64 => {
                let float_arr = array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        ArrowError::ParseError("Not a Float64 array".to_string())
                    })?;
                let val = float_arr.value(row_idx).to_le_bytes();
                out.extend_from_slice(&val);
                Ok(())
            }
            FieldEncoder::Binary => {
                let bin_arr = array
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .ok_or_else(|| {
                        ArrowError::ParseError("Not a Binary array".to_string())
                    })?;
                let val = bin_arr.value(row_idx);
                write_zigzag_long(val.len() as i64, out)?;
                out.extend_from_slice(val);
                Ok(())
            }
            FieldEncoder::Utf8 => {
                let str_arr = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        ArrowError::ParseError("Not a String array".to_string())
                    })?;
                let val = str_arr.value(row_idx);
                write_zigzag_long(val.len() as i64, out)?;
                out.extend_from_slice(val.as_bytes());
                Ok(())
            }
            FieldEncoder::Record(child_encoders) => {
                let struct_arr = array
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| {
                        ArrowError::ParseError("Not a Struct array".to_string())
                    })?;
                for (i, child_enc) in child_encoders.iter().enumerate() {
                    let col = struct_arr.column(i);
                    child_enc.encode_one(col.as_ref(), row_idx, out)?;
                }
                Ok(())
            }
            FieldEncoder::Enum(key_enc) => {
                match key_enc {
                    DictKeyEnc::Int8 => {
                        let dict_arr = array
                            .as_any()
                            .downcast_ref::<DictionaryArray<Int8Type>>()
                            .ok_or_else(|| {
                                ArrowError::ParseError(
                                    "Not a Dictionary<Int8> array".to_string(),
                                )
                            })?;
                        let key_usize = dict_arr.key(row_idx).unwrap_or(0);
                        write_zigzag_long(key_usize as i64, out)?;
                    }
                    DictKeyEnc::Int16 => {
                        let dict_arr = array
                            .as_any()
                            .downcast_ref::<DictionaryArray<Int16Type>>()
                            .ok_or_else(|| {
                                ArrowError::ParseError(
                                    "Not a Dictionary<Int16> array".to_string(),
                                )
                            })?;
                        let key_usize = dict_arr.key(row_idx).unwrap_or(0);
                        write_zigzag_long(key_usize as i64, out)?;
                    }
                    DictKeyEnc::Int32 => {
                        let dict_arr = array
                            .as_any()
                            .downcast_ref::<DictionaryArray<Int32Type>>()
                            .ok_or_else(|| {
                                ArrowError::ParseError(
                                    "Not a Dictionary<Int32> array".to_string(),
                                )
                            })?;
                        let key_usize = dict_arr.key(row_idx).unwrap_or(0);
                        write_zigzag_long(key_usize as i64, out)?;
                    }
                    DictKeyEnc::Int64 => {
                        let dict_arr = array
                            .as_any()
                            .downcast_ref::<DictionaryArray<Int64Type>>()
                            .ok_or_else(|| {
                                ArrowError::ParseError(
                                    "Not a Dictionary<Int64> array".to_string(),
                                )
                            })?;
                        let key_usize = dict_arr.key(row_idx).unwrap_or(0);
                        write_zigzag_long(key_usize as i64, out)?;
                    }
                }
                Ok(())
            }
            FieldEncoder::Array(child_enc) => {
                match array.data_type() {
                    DataType::List(_) => {
                        let list_arr = array
                            .as_any()
                            .downcast_ref::<ListArray>()
                            .ok_or_else(|| {
                                ArrowError::ParseError("Not a List array".to_string())
                            })?;

                        let offset = list_arr.value_offsets()[row_idx] as usize;
                        let offset_next = list_arr.value_offsets()[row_idx + 1] as usize;
                        let length = offset_next - offset;
                        write_zigzag_long(length as i64, out)?;
                        let values = list_arr.values();
                        for i in offset..offset_next {
                            child_enc.encode_one(values.as_ref(), i, out)?;
                        }
                        if length > 0 {
                            write_zigzag_long(0, out)?;
                        }
                    }
                    DataType::LargeList(_) => {
                        let ll_arr = array
                            .as_any()
                            .downcast_ref::<LargeListArray>()
                            .ok_or_else(|| {
                                ArrowError::ParseError("Not a LargeList array".to_string())
                            })?;
                        let offset = ll_arr.value_offsets()[row_idx] as usize;
                        let offset_next = ll_arr.value_offsets()[row_idx + 1] as usize;
                        let length = offset_next - offset;
                        write_zigzag_long(length as i64, out)?;
                        let values = ll_arr.values();
                        for i in offset..offset_next {
                            child_enc.encode_one(values.as_ref(), i, out)?;
                        }
                        if length > 0 {
                            write_zigzag_long(0, out)?;
                        }
                    }
                    DataType::FixedSizeList(_, size) => {
                        let fsl_arr = array
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .ok_or_else(|| {
                                ArrowError::ParseError(
                                    "Not a FixedSizeList array".to_string(),
                                )
                            })?;
                        let length = *size;
                        let start = row_idx * *size as usize;
                        let end = start + *size as usize;
                        write_zigzag_long(*size as i64, out)?;
                        let values = fsl_arr.values();
                        for i in start..end {
                            child_enc.encode_one(values.as_ref(), i, out)?;
                        }
                        // Avro array termination
                        if length > 0 {
                            write_zigzag_long(0, out)?;
                        }
                    }
                    dt => {
                        return Err(ArrowError::NotYetImplemented(format!(
                            "Array writer for arrow type {dt:?} not supported"
                        )));
                    }
                }
                Ok(())
            }
            FieldEncoder::Map(val_enc) => {
                let map_array = array
                    .as_any()
                    .downcast_ref::<MapArray>()
                    .ok_or_else(|| {
                        ArrowError::ParseError("Not a Map array".to_string())
                    })?;

                let offset = map_array.value_offsets()[row_idx] as usize;
                let offset_next = map_array.value_offsets()[row_idx + 1] as usize;
                let length = offset_next - offset;

                write_zigzag_long(length as i64, out)?;

                let entries_struct = map_array.entries();
                let key_arr = entries_struct
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        ArrowError::ParseError("Map keys not a String array".to_string())
                    })?;
                let val_arr = entries_struct.column(1);

                for i in offset..offset_next {
                    let key_val = key_arr.value(i);
                    write_zigzag_long(key_val.len() as i64, out)?;
                    out.extend_from_slice(key_val.as_bytes());
                    val_enc.encode_one(val_arr.as_ref(), i, out)?;
                }

                // Only write the '0' terminator if we had a non-empty block
                if length > 0 {
                    write_zigzag_long(0, out)?;
                }
                Ok(())
            }

            FieldEncoder::Fixed(n) => {
                let fsb_arr = array
                    .as_any()
                    .downcast_ref::<FixedSizeBinaryArray>()
                    .ok_or_else(|| {
                        ArrowError::ParseError(
                            "Not a FixedSizeBinary array".to_string(),
                        )
                    })?;
                let val = fsb_arr.value(row_idx);
                if val.len() != *n {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "FixedSizeBinary length mismatch: expected {n}, got {}",
                        val.len()
                    )));
                }
                out.extend_from_slice(val);
                Ok(())
            }
            FieldEncoder::Decimal128(_p, _s, size_opt) => {
                let dec_arr = array
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| {
                        ArrowError::ParseError(
                            "Not a Decimal128 array".to_string(),
                        )
                    })?;
                let value = dec_arr.value(row_idx);
                let be_bytes = value.to_be_bytes();
                let sign_byte = if value >= 0 { 0x00 } else { 0xFF };
                // Trim sign extension
                let mut first_non_extend = 0usize;
                while first_non_extend + 1 < be_bytes.len()
                    && be_bytes[first_non_extend] == sign_byte
                    && (be_bytes[first_non_extend + 1] & 0x80) == (sign_byte & 0x80)
                {
                    first_non_extend += 1;
                }
                let trimmed = &be_bytes[first_non_extend..];
                if let Some(sz) = size_opt {
                    // fixed-size decimal
                    if trimmed.len() > *sz {
                        return Err(ArrowError::InvalidArgumentError(
                            "Decimal128 value doesn't fit fixed size".to_string(),
                        ));
                    }
                    let mut buf = vec![sign_byte; *sz];
                    let start = sz - trimmed.len();
                    buf[start..].copy_from_slice(trimmed);
                    out.extend_from_slice(&buf);
                } else {
                    // variable-size decimal => length-prefix
                    write_zigzag_long(trimmed.len() as i64, out)?;
                    out.extend_from_slice(trimmed);
                }
                Ok(())
            }
            FieldEncoder::Decimal256(_p, _s, size_opt) => {
                let dec_arr = array
                    .as_any()
                    .downcast_ref::<Decimal256Array>()
                    .ok_or_else(|| {
                        ArrowError::ParseError(
                            "Not a Decimal256 array".to_string(),
                        )
                    })?;
                let val_i256 = dec_arr.value(row_idx);
                // Convert to big-endian
                let mut be_bytes = val_i256.to_le_bytes();
                be_bytes.reverse();
                let sign_byte = if val_i256.is_negative() { 0xFF } else { 0x00 };
                // Trim sign extension
                let mut first_non_extend = 0usize;
                while first_non_extend + 1 < be_bytes.len()
                    && be_bytes[first_non_extend] == sign_byte
                    && (be_bytes[first_non_extend + 1] & 0x80)
                    == (sign_byte & 0x80)
                {
                    first_non_extend += 1;
                }
                let trimmed = &be_bytes[first_non_extend..];
                if let Some(sz) = size_opt {
                    // fixed-size
                    if trimmed.len() > *sz {
                        return Err(ArrowError::InvalidArgumentError(
                            "Decimal256 value doesn't fit fixed size".to_string(),
                        ));
                    }
                    let mut buf = vec![sign_byte; *sz];
                    let start = sz - trimmed.len();
                    buf[start..].copy_from_slice(trimmed);
                    out.extend_from_slice(&buf);
                } else {
                    // variable-size => length + data
                    write_zigzag_long(trimmed.len() as i64, out)?;
                    out.extend_from_slice(trimmed);
                }
                Ok(())
            }
            FieldEncoder::Date32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow_array::Date32Array>()
                    .ok_or_else(|| {
                        ArrowError::ParseError("Not a Date32 array".to_string())
                    })?;
                let val = arr.value(row_idx);
                write_zigzag_long(val as i64, out)?;
                Ok(())
            }
            FieldEncoder::TimeMillis => {
                let arr = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Time32MillisecondType>>()
                    .ok_or_else(|| {
                        ArrowError::ParseError(
                            "Not a Time32(Millis) array".to_string(),
                        )
                    })?;
                let val = arr.value(row_idx);
                write_zigzag_long(val as i64, out)?;
                Ok(())
            }
            FieldEncoder::TimeMicros => {
                let arr = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Time64MicrosecondType>>()
                    .ok_or_else(|| {
                        ArrowError::ParseError(
                            "Not a Time64(Micros) array".to_string(),
                        )
                    })?;
                let val = arr.value(row_idx);
                write_zigzag_long(val, out)?;
                Ok(())
            }
            FieldEncoder::TimestampMillis(_is_utc) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .ok_or_else(|| {
                        ArrowError::ParseError(
                            "Not a Timestamp(Millis) array".to_string(),
                        )
                    })?;
                let val = arr.value(row_idx);
                write_zigzag_long(val, out)?;
                Ok(())
            }
            FieldEncoder::TimestampMicros(_is_utc) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .ok_or_else(|| {
                        ArrowError::ParseError(
                            "Not a Timestamp(Micros) array".to_string(),
                        )
                    })?;
                let val = arr.value(row_idx);
                write_zigzag_long(val, out)?;
                Ok(())
            }
            FieldEncoder::Uuid => {
                let fsb_arr = array
                    .as_any()
                    .downcast_ref::<FixedSizeBinaryArray>()
                    .ok_or_else(|| {
                        ArrowError::ParseError(
                            "Not a FixedSizeBinary(16) array".to_string(),
                        )
                    })?;
                let val = fsb_arr.value(row_idx);
                if val.len() != 16 {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "UUID field must be 16 bytes, got {}",
                        val.len()
                    )));
                }
                out.extend_from_slice(val);
                Ok(())
            }
            FieldEncoder::Duration => {
                let arr = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<IntervalMonthDayNanoType>>()
                    .ok_or_else(|| {
                        ArrowError::ParseError(
                            "Not IntervalMonthDayNano array".to_string(),
                        )
                    })?;
                let val: IntervalMonthDayNano = arr.value(row_idx);
                let months = val.months;
                let days = val.days;
                // Convert total nanoseconds to milliseconds
                let ms = (val.nanoseconds / 1_000_000) as i32;
                out.extend_from_slice(&months.to_le_bytes());
                out.extend_from_slice(&days.to_le_bytes());
                out.extend_from_slice(&ms.to_le_bytes());
                Ok(())
            }
            FieldEncoder::Nullable(inner, nb) => {
                match nb {
                    // Standard Avro => [ "null", T ] => branch=0 => null, branch=1 => T
                    Nullability::NullFirst => {
                        if array.is_null(row_idx) {
                            // pick union-variant #0 => null
                            write_zigzag_long(0, out)?;
                        } else {
                            // pick union-variant #1 => T
                            write_zigzag_long(1, out)?;
                            inner.encode_one(array, row_idx, out)?;
                        }
                    }
                    // Impala => [ T, "null" ] => branch=0 => T, branch=1 => null
                    Nullability::NullSecond => {
                        if array.is_null(row_idx) {
                            // => branch=1 => null
                            write_zigzag_long(1, out)?;
                        } else {
                            // => branch=0 => T
                            write_zigzag_long(0, out)?;
                            inner.encode_one(array, row_idx, out)?;
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{BinaryArray, BooleanArray, Date32Array, FixedSizeBinaryArray, Float32Array, Float64Array, Int32Array, Int64Array, Int8Array, RecordBatch, StringArray, Time32MillisecondArray, Time64MicrosecondArray};
    use arrow_schema::{ArrowError, DataType, Field, FieldRef, Fields, Schema as ArrowSchema};
    use std::sync::Arc;
    use arrow_array::builder::{Int32Builder, MapBuilder, StringBuilder};
    use arrow_buffer::{OffsetBuffer, ScalarBuffer, Buffer as ArrowBuffer};
    use arrow_data::ArrayData;

    fn decode_avro_boolean(data: &[u8]) -> bool {
        assert_eq!(data.len(), 1, "boolean => exactly 1 byte");
        data[0] != 0
    }

    fn decode_varint_u64(data: &[u8]) -> (u64, usize) {
        let mut val: u64 = 0;
        let mut shift = 0;
        let mut i = 0;
        for b in data {
            let lower = (b & 0x7F) as u64;
            val |= lower << shift;
            shift += 7;
            i += 1;
            if b & 0x80 == 0 {
                break;
            }
        }
        (val, i)
    }

    fn decode_zigzag_long(data: &[u8]) -> (i64, usize) {
        let (zz, consumed) = decode_varint_u64(data);
        let val = ((zz >> 1) as i64) ^ -((zz & 1) as i64);
        (val, consumed)
    }

    fn decode_avro_f32(data: &[u8]) -> f32 {
        assert_eq!(data.len(), 4);
        f32::from_le_bytes(data.try_into().unwrap())
    }

    fn decode_avro_f64(data: &[u8]) -> f64 {
        assert_eq!(data.len(), 8);
        f64::from_le_bytes(data.try_into().unwrap())
    }

    fn decode_avro_bytes(data: &[u8]) -> Vec<u8> {
        let (len, consumed) = decode_zigzag_long(data);
        let len = len as usize;
        data[consumed..consumed + len].to_vec()
    }

    fn decode_avro_string(data: &[u8]) -> (String, usize) {
        let (len, used) = decode_zigzag_long(data);
        let len_usize = len as usize;
        let start = used;
        let end = used + len_usize;
        let bytes = &data[start..end];
        let s = std::str::from_utf8(bytes).unwrap().to_string();
        (s, end)
    }

    #[test]
    fn test_encode_date32() -> Result<(), ArrowError> {
        let date_arr = Date32Array::from(vec![Some(10), Some(-3)]);
        let field = Field::new("date_col", DataType::Date32, false);
        let schema = ArrowSchema::new(vec![field.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(date_arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        encoder.encode_row(&schema, &batch, 1, &mut out)?;
        let mut offset = 0;
        let (val0, used0) = decode_zigzag_long(&out[offset..]);
        offset += used0;
        assert_eq!(val0, 10, "Expected day=10 in row0");
        let (val1, used1) = decode_zigzag_long(&out[offset..]);
        offset += used1;
        assert_eq!(val1, -3, "Expected day=-3 in row1");
        assert_eq!(offset, out.len(), "Consumed all bytes");
        Ok(())
    }

    #[test]
    fn test_encode_time_millis() -> Result<(), ArrowError> {
        let arr = Time32MillisecondArray::from(vec![Some(1234), Some(99999)]);
        let field = Field::new("time_ms", DataType::Time32(TimeUnit::Millisecond), false);
        let schema = ArrowSchema::new(vec![field.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        encoder.encode_row(&schema, &batch, 1, &mut out)?;
        let mut offset = 0;
        let (row0_val, used0) = decode_zigzag_long(&out[offset..]);
        offset += used0;
        assert_eq!(row0_val, 1234);
        let (row1_val, used1) = decode_zigzag_long(&out[offset..]);
        offset += used1;
        assert_eq!(row1_val, 99999);
        assert_eq!(offset, out.len());
        Ok(())
    }

    #[test]
    fn test_encode_time_micros() -> Result<(), ArrowError> {
        let arr = Time64MicrosecondArray::from(vec![Some(50_000), Some(1_000_000)]);
        let field = Field::new("time_us", DataType::Time64(TimeUnit::Microsecond), false);
        let schema = ArrowSchema::new(vec![field.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        encoder.encode_row(&schema, &batch, 1, &mut out)?;
        let mut offset = 0;
        let (row0_val, used0) = decode_zigzag_long(&out[offset..]);
        offset += used0;
        assert_eq!(row0_val, 50_000);
        let (row1_val, used1) = decode_zigzag_long(&out[offset..]);
        offset += used1;
        assert_eq!(row1_val, 1_000_000);
        assert_eq!(offset, out.len());
        Ok(())
    }

    #[test]
    fn test_encode_timestamp_millis_utc() -> Result<(), ArrowError> {
        let data_type = DataType::Timestamp(TimeUnit::Millisecond, Some(Arc::from("+00:00".to_string())));
        let values = [1000i64, 86400000i64];
        let buf = ArrowBuffer::from_slice_ref(&values);
        let array_data = ArrayData::builder(data_type.clone())
            .len(2)
            .add_buffer(buf)
            .build()?;
        let arr = TimestampMillisecondArray::from(array_data);
        let field = Field::new("ts_ms_utc", data_type, false);
        let schema = ArrowSchema::new(vec![field.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        encoder.encode_row(&schema, &batch, 1, &mut out)?;
        let mut offset = 0;
        let (row0_val, used0) = decode_zigzag_long(&out[offset..]);
        offset += used0;
        assert_eq!(row0_val, 1000);
        let (row1_val, used1) = decode_zigzag_long(&out[offset..]);
        offset += used1;
        assert_eq!(row1_val, 86400000);
        assert_eq!(offset, out.len());
        Ok(())
    }

    #[test]
    fn test_encode_timestamp_millis_local() -> Result<(), ArrowError> {
        let arr = TimestampMillisecondArray::from(vec![Some(5000), Some(1577836800000)]);
        let field = Field::new(
            "ts_ms_local",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        );
        let schema = ArrowSchema::new(vec![field.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        encoder.encode_row(&schema, &batch, 1, &mut out)?;
        let mut offset = 0;
        let (row0_val, used0) = decode_zigzag_long(&out[offset..]);
        offset += used0;
        assert_eq!(row0_val, 5000);
        let (row1_val, used1) = decode_zigzag_long(&out[offset..]);
        offset += used1;
        assert_eq!(row1_val, 1_577_836_800_000);
        assert_eq!(offset, out.len());
        Ok(())
    }

    #[test]
    fn test_encode_timestamp_micros_utc() -> Result<(), ArrowError> {
        let data_type = DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into()));
        let values = [123456i64, 2_000_000i64]; // 2 rows
        let buf = ArrowBuffer::from_slice_ref(&values);
        let array_data = ArrayData::builder(data_type.clone())
            .len(2)
            .add_buffer(buf)
            .build()?;
        let arr = TimestampMicrosecondArray::from(array_data);
        let field = Field::new("ts_us_utc", data_type, false);
        let schema = ArrowSchema::new(vec![field]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        encoder.encode_row(&schema, &batch, 1, &mut out)?;
        let mut offset = 0;
        let (row0_val, used0) = decode_zigzag_long(&out[offset..]);
        offset += used0;
        assert_eq!(row0_val, 123456, "Expected microseconds=123456 for row0");
        let (row1_val, used1) = decode_zigzag_long(&out[offset..]);
        offset += used1;
        assert_eq!(row1_val, 2_000_000, "Expected microseconds=2_000_000 for row1");
        assert_eq!(offset, out.len());
        Ok(())
    }

    #[test]
    fn test_encode_timestamp_micros_local() -> Result<(), ArrowError> {
        let arr = TimestampMicrosecondArray::from(vec![Some(42), Some(9999999999999)]);
        let field = Field::new(
            "ts_us_local",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        );
        let schema = ArrowSchema::new(vec![field.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        encoder.encode_row(&schema, &batch, 1, &mut out)?;
        let mut offset = 0;
        let (row0_val, used0) = decode_zigzag_long(&out[offset..]);
        offset += used0;
        assert_eq!(row0_val, 42);
        let (row1_val, used1) = decode_zigzag_long(&out[offset..]);
        offset += used1;
        assert_eq!(row1_val, 9_999_999_999_999);
        assert_eq!(offset, out.len());
        Ok(())
    }

    #[test]
    fn test_encode_uuid() -> Result<(), ArrowError> {
        let val0 = b"1234567890ABCDEF";
        let val1 = b"abcdefghijklmnop";
        let arr = FixedSizeBinaryArray::from(vec![Some(&val0[..]), Some(&val1[..])]);
        let field = Field::new("uuid_col", DataType::FixedSizeBinary(16), false)
            .with_metadata(std::collections::HashMap::from([
                ("logicalType".to_string(), "uuid".to_string()),
            ]));
        let schema = ArrowSchema::new(vec![field.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        encoder.encode_row(&schema, &batch, 1, &mut out)?;
        assert_eq!(out.len(), 32, "2 rows * 16 bytes each = 32 total");
        assert_eq!(&out[0..16], b"1234567890ABCDEF");
        assert_eq!(&out[16..32], b"abcdefghijklmnop");
        Ok(())
    }

    #[test]
    fn test_encode_duration_interval() {
        let data = vec![Some(IntervalMonthDayNano::new(2, 3, 1_000_000))];
        let arr = PrimitiveArray::<IntervalMonthDayNanoType>::from(data);
        let field = Field::new(
            "duration_test",
            DataType::Interval(IntervalUnit::MonthDayNano),
            false,
        );
        let schema = ArrowSchema::new(vec![field.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(arr)])
            .unwrap();
        let mut encoder = RecordEncoder::try_new(&schema, false).unwrap();
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out).unwrap();
        assert_eq!(
            &out,
            &[
                0x02, 0x00, 0x00, 0x00,
                0x03, 0x00, 0x00, 0x00,
                0x01, 0x00, 0x00, 0x00
            ]
        );
    }

    #[test]
    fn test_encode_enum_dictionary_int8_nullable() -> Result<(), ArrowError> {
        use std::collections::HashMap;
        let keys = Int8Array::from(vec![Some(1i8), Some(0), None, Some(2), Some(2)]);
        let values = StringArray::from(vec!["GREEN", "RED", "BLUE"]);
        let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;
        let mut md = HashMap::new();
        md.insert(
            "avro.enum.symbols".to_string(),
            "[\"GREEN\",\"RED\",\"BLUE\"]".to_string()
        );
        let field = Field::new("enum_col", dict_array.data_type().clone(), true)
            .with_metadata(md);
        let schema = ArrowSchema::new(vec![field]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(dict_array)],
        )?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        for row_idx in 0..5 {
            encoder.encode_row(&schema, &batch, row_idx, &mut out)?;
        }
        let mut offset = 0;
        let (branch0, used0) = decode_zigzag_long(&out[offset..]);
        offset += used0;
        assert_eq!(branch0, 1, "Expected branch=1 => non-null");
        let (val0, used_val0) = decode_zigzag_long(&out[offset..]);
        offset += used_val0;
        assert_eq!(val0, 1);
        let (branch1, used1) = decode_zigzag_long(&out[offset..]);
        offset += used1;
        assert_eq!(branch1, 1);
        let (val1, used_val1) = decode_zigzag_long(&out[offset..]);
        offset += used_val1;
        assert_eq!(val1, 0);
        let (branch2, used2) = decode_zigzag_long(&out[offset..]);
        offset += used2;
        assert_eq!(branch2, 0, "Expected branch=0 => null row");
        let (branch3, used3) = decode_zigzag_long(&out[offset..]);
        offset += used3;
        assert_eq!(branch3, 1);
        let (val3, used_val3) = decode_zigzag_long(&out[offset..]);
        offset += used_val3;
        assert_eq!(val3, 2);
        let (branch4, used4) = decode_zigzag_long(&out[offset..]);
        offset += used4;
        assert_eq!(branch4, 1);
        let (val4, used_val4) = decode_zigzag_long(&out[offset..]);
        offset += used_val4;
        assert_eq!(val4, 2);
        assert_eq!(offset, out.len(), "All encoded data consumed");
        Ok(())
    }

    #[test]
    fn test_encode_enum_dictionary_int64_in_range() -> Result<(), ArrowError> {
        use std::collections::HashMap;
        let keys = Int64Array::from(vec![Some(0), Some(2)]);
        let values = StringArray::from(vec!["FISH", "DOG", "CAT"]);
        let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;
        let mut md = HashMap::new();
        md.insert(
            "avro.enum.symbols".to_string(),
            "[\"FISH\",\"DOG\",\"CAT\"]".to_string()
        );
        let field = Field::new("enum_col", dict_array.data_type().clone(), false)
            .with_metadata(md);
        let schema = ArrowSchema::new(vec![field]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(dict_array)],
        )?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        encoder.encode_row(&schema, &batch, 1, &mut out)?;
        let mut offset = 0;
        let (row0_val, used0) = decode_zigzag_long(&out[offset..]);
        offset += used0;
        assert_eq!(row0_val, 0, "Expected ordinal=0 in row 0");
        let (row1_val, used1) = decode_zigzag_long(&out[offset..]);
        offset += used1;
        assert_eq!(row1_val, 2, "Expected ordinal=2 in row 1");
        assert_eq!(offset, out.len(), "All encoded data should be consumed");
        Ok(())
    }

    #[test]
    fn test_map_field_encoder() {
        let key_builder = StringBuilder::new();
        let value_builder = Int32Builder::new();
        let mut map_builder = MapBuilder::new(None, key_builder, value_builder);
        map_builder.keys().append_value("apple");
        map_builder.values().append_value(10);
        map_builder.keys().append_value("banana");
        map_builder.values().append_value(20);
        let _ = map_builder.append(true);
        map_builder.keys().append_value("hello");
        map_builder.values().append_value(42);
        let _ = map_builder.append(true);
        let map_array = map_builder.finish();
        assert_eq!(map_array.len(), 2);
        let field = Field::new("my_map", map_array.data_type().clone(), true);
        let map_encoder = FieldEncoder::try_new(&field, false).expect("Failed to build FieldEncoder");
        let mut encoded = Vec::new();
        map_encoder.encode_one(&map_array, 0, &mut encoded).unwrap();
        map_encoder.encode_one(&map_array, 1, &mut encoded).unwrap();
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_map_encoder_null() {
        let key_builder = StringBuilder::new();
        let value_builder = Int32Builder::new();
        let mut map_builder = MapBuilder::new(None, key_builder, value_builder);
        let _ = map_builder.append(false);
        let map_array = map_builder.finish();
        assert_eq!(map_array.len(), 1, "Expected 1 row");
        assert!(map_array.is_null(0), "Row 0 should be null");
        let field = Field::new("nullable_map", map_array.data_type().clone(), true);
        let enc = FieldEncoder::try_new(&field, false).unwrap();
        let mut buf = Vec::new();
        enc.encode_one(&map_array, 0, &mut buf).unwrap();
        assert_eq!(buf, vec![0x00], "Expected union=0 => null for a null map");
    }

    #[test]
    fn test_encode_largelist_of_strings() -> Result<(), ArrowError> {
        let child_field = Field::new("str_item", DataType::Utf8, false);
        let ll_type = DataType::LargeList(Arc::new(child_field.clone()));
        let offsets = vec![0i64, 2, 3];
        let child_vals = StringArray::from(vec!["hello", "arrow", "avro"]);
        let ll_arr = LargeListArray::new(
            FieldRef::from(child_field),
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Arc::new(child_vals),
            None,
        );
        let schema = ArrowSchema::new(vec![Field::new("ll_col", ll_type, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(ll_arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        encoder.encode_row(&schema, &batch, 1, &mut out)?;
        let mut offset = 0;
        let (length0, used0) = decode_zigzag_long(&out[offset..]);
        offset += used0;
        assert_eq!(length0, 2);
        let (s0, used_s0) = decode_avro_string(&out[offset..]);
        offset += used_s0;
        assert_eq!(s0, "hello");
        let (s1, used_s1) = decode_avro_string(&out[offset..]);
        offset += used_s1;
        assert_eq!(s1, "arrow");
        let (block_term0, used_bt0) = decode_zigzag_long(&out[offset..]);
        offset += used_bt0;
        assert_eq!(block_term0, 0);
        let (length1, used1) = decode_zigzag_long(&out[offset..]);
        offset += used1;
        assert_eq!(length1, 1);
        let (s2, used_s2) = decode_avro_string(&out[offset..]);
        offset += used_s2;
        assert_eq!(s2, "avro");
        let (block_term1, used_bt1) = decode_zigzag_long(&out[offset..]);
        offset += used_bt1;
        assert_eq!(block_term1, 0);
        assert_eq!(offset, out.len());
        Ok(())
    }

    #[test]
    fn test_encode_list_of_int32() -> Result<(), ArrowError> {
        let child_field = Field::new("items", DataType::Int32, false);
        let list_type = DataType::List(Arc::new(child_field.clone()));
        let offsets = vec![0i32, 2, 2];
        let values = Int32Array::from(vec![10, 20]);
        let list_arr = ListArray::new(
            Arc::new(child_field),
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Arc::new(values),
            None,
        );
        let schema = ArrowSchema::new(vec![Field::new("list_col", list_type, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(list_arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        encoder.encode_row(&schema, &batch, 1, &mut out)?;
        let mut offset = 0;
        let (length0, used0) = decode_zigzag_long(&out[offset..]);
        offset += used0;
        assert_eq!(length0, 2);
        let (val0, used_v0) = decode_zigzag_long(&out[offset..]);
        offset += used_v0;
        assert_eq!(val0, 10);
        let (val1, used_v1) = decode_zigzag_long(&out[offset..]);
        offset += used_v1;
        assert_eq!(val1, 20);
        let (blk_term0, used_term0) = decode_zigzag_long(&out[offset..]);
        offset += used_term0;
        assert_eq!(blk_term0, 0);
        let (length1, used1) = decode_zigzag_long(&out[offset..]);
        offset += used1;
        assert_eq!(length1, 0);
        let (blk_term1, used_term1) = decode_zigzag_long(&out[offset..]);
        offset += used_term1;
        assert_eq!(blk_term1, 0);
        assert_eq!(offset, out.len());
        Ok(())
    }

    #[test]
    fn test_encode_fixedsizelist_of_bools() -> Result<(), ArrowError> {
        let size = 3;
        let child_data = BooleanArray::from(vec![
            Some(true),  Some(false), Some(true),
            Some(false), Some(false), Some(false),
        ]);
        let child_field = Arc::new(Field::new("fsl_item", DataType::Boolean, false));
        let fsl_arr = FixedSizeListArray::new(
            child_field.clone(),
            size,
            Arc::new(child_data),
            None,
        );
        let top_level_field = Field::new(
            "fsl_col",
            DataType::FixedSizeList(child_field.clone(), size),
            false,
        );
        let schema = ArrowSchema::new(vec![top_level_field]);
        let batch =
            RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(fsl_arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        encoder.encode_row(&schema, &batch, 1, &mut out)?;
        let mut offset = 0;
        let (arr_len0, used0) = decode_zigzag_long(&out[offset..]);
        offset += used0;
        assert_eq!(arr_len0, 3);
        assert_eq!(out[offset], 1);
        offset += 1;
        assert_eq!(out[offset], 0);
        offset += 1;
        assert_eq!(out[offset], 1);
        offset += 1;
        let (blk_term0, used_bt0) = decode_zigzag_long(&out[offset..]);
        offset += used_bt0;
        assert_eq!(blk_term0, 0);
        let (arr_len1, used1) = decode_zigzag_long(&out[offset..]);
        offset += used1;
        assert_eq!(arr_len1, 3);
        for _ in 0..3 {
            assert_eq!(out[offset], 0);
            offset += 1;
        }
        let (blk_term1, used_bt1) = decode_zigzag_long(&out[offset..]);
        offset += used_bt1;
        assert_eq!(blk_term1, 0);
        assert_eq!(offset, out.len(), "Consumed all bytes");
        Ok(())
    }

    #[test]
    fn test_encode_nested_struct() -> Result<(), ArrowError> {
        let child_a = Field::new("a", DataType::Int32, false);
        let child_b = Field::new("b", DataType::Boolean, true);
        let struct_type = DataType::Struct(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Boolean, true),
        ].into());
        let a_data = Arc::new(Int32Array::from(vec![100, 200])) as Arc<dyn Array>;
        let b_data = Arc::new(BooleanArray::from(vec![Some(true), None])) as Arc<dyn Array>;
        let struct_array = StructArray::new(
            Fields::from(vec![child_a, child_b]),
            vec![a_data, b_data],
            None,
        );
        let schema = ArrowSchema::new(vec![Field::new("nested_rec", struct_type, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(struct_array)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        encoder.encode_row(&schema, &batch, 1, &mut out)?;
        let mut offset = 0;
        let (val_a_0, consumed) = decode_zigzag_long(&out[offset..]);
        offset += consumed;
        assert_eq!(val_a_0, 100);
        let (branch, used) = decode_zigzag_long(&out[offset..]);
        offset += used;
        assert_eq!(branch, 1);
        let b0 = decode_avro_boolean(&out[offset..offset + 1]);
        offset += 1;
        assert_eq!(b0, true);
        let (val_a_1, consumed) = decode_zigzag_long(&out[offset..]);
        offset += consumed;
        assert_eq!(val_a_1, 200);
        let (branch1, used) = decode_zigzag_long(&out[offset..]);
        offset += used;
        assert_eq!(branch1, 0);
        assert_eq!(offset, out.len(), "All bytes should be consumed");
        Ok(())
    }

    #[test]
    fn test_encode_decimal128_fixed() -> Result<(), ArrowError> {
        let mut builder =  Decimal128Builder::new()
            .with_precision_and_scale(10, 2)?;
        builder.append_value(12345);
        let decimal_arr = builder.finish();
        let field = Field::new("dec128", DataType::Decimal128(10, 2), false);
        let schema = ArrowSchema::new(vec![field]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(decimal_arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        assert_eq!(out.len(), 16);
        Ok(())
    }

    #[test]
    fn test_encode_decimal256_fixed() -> Result<(), ArrowError> {
        let mut builder =  Decimal256Builder::new()
            .with_precision_and_scale(12, 2)?;
        builder.append_value(i256::from_i128(99900));
        let decimal_arr = builder.finish();
        let field = Field::new("dec256", DataType::Decimal256(12, 2), false);
        let schema = ArrowSchema::new(vec![field]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(decimal_arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        assert_eq!(out.len(), 32);
        Ok(())
    }

    #[test]
    fn test_encode_decimal128_as_bytes() -> Result<(), ArrowError> {
        let mut builder =  Decimal128Builder::new()
            .with_precision_and_scale(10, 2)?;
        builder.append_value(-250);
        let decimal_arr = builder.finish();
        let field = Field::new("dec_col", DataType::Decimal128(10, 2), false);
        let schema = ArrowSchema::new(vec![field]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(decimal_arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        if let FieldEncoder::Decimal128(_, _, ref mut size_opt) = encoder.fields[0] {
            *size_opt = None;
        }
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out)?;
        let (llen, used) = decode_zigzag_long(&out);
        let data = &out[used..];
        assert_eq!(llen as usize, data.len());
        Ok(())
    }

    #[test]
    fn test_encode_boolean() {
        let bool_arr = BooleanArray::from(vec![true]);
        let schema = ArrowSchema::new(vec![Field::new("bool_col", DataType::Boolean, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(bool_arr)])
            .unwrap();
        let mut encoder = RecordEncoder::try_new(&schema, false).unwrap();
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out).unwrap();
        let got = decode_avro_boolean(&out);
        assert_eq!(got, true);
    }

    #[test]
    fn test_encode_int32() {
        let int_arr = Int32Array::from(vec![42]);
        let schema = ArrowSchema::new(vec![Field::new("int_col", DataType::Int32, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(int_arr)])
            .unwrap();
        let mut encoder = RecordEncoder::try_new(&schema, false).unwrap();
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out).unwrap();
        let (decoded, consumed) = decode_zigzag_long(&out);
        assert_eq!(decoded, 42);
        assert_eq!(consumed, out.len());
    }

    #[test]
    fn test_encode_int64() {
        let long_arr = Int64Array::from(vec![-1_i64]);
        let schema = ArrowSchema::new(vec![Field::new("long_col", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(long_arr)])
            .unwrap();
        let mut encoder = RecordEncoder::try_new(&schema, false).unwrap();
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out).unwrap();
        let (decoded, consumed) = decode_zigzag_long(&out);
        assert_eq!(decoded, -1);
        assert_eq!(consumed, out.len());
    }

    #[test]
    fn test_encode_float32() {
        let float_arr = Float32Array::from(vec![3.14_f32]);
        let schema = ArrowSchema::new(vec![Field::new("float_col", DataType::Float32, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(float_arr)])
            .unwrap();
        let mut encoder = RecordEncoder::try_new(&schema, false).unwrap();
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out).unwrap();
        let got = decode_avro_f32(&out);
        assert!((got - 3.14).abs() < 1e-7);
    }

    #[test]
    fn test_encode_float64() {
        let double_arr = Float64Array::from(vec![std::f64::consts::E]);
        let schema = ArrowSchema::new(vec![Field::new("double_col", DataType::Float64, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(double_arr)])
            .unwrap();
        let mut encoder = RecordEncoder::try_new(&schema, false).unwrap();
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out).unwrap();
        let got = decode_avro_f64(&out);
        assert!((got - std::f64::consts::E).abs() < 1e-14);
    }

    #[test]
    fn test_encode_binary() {
        let bin_arr = BinaryArray::from(vec![Some(&b"hello"[..])]);
        let schema = ArrowSchema::new(vec![Field::new("bin_col", DataType::Binary, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(bin_arr)])
            .unwrap();
        let mut encoder = RecordEncoder::try_new(&schema, false).unwrap();
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out).unwrap();
        let got = decode_avro_bytes(&out);
        assert_eq!(got, b"hello");
    }

    #[test]
    fn test_encode_utf8() {
        let str_arr = StringArray::from(vec![Some("Avro!")]);
        let schema = ArrowSchema::new(vec![Field::new("str_col", DataType::Utf8, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(str_arr)])
            .unwrap();
        let mut encoder = RecordEncoder::try_new(&schema, false).unwrap();
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out).unwrap();
        let got = decode_avro_string(&out);
        assert_eq!(got.0, "Avro!");
    }

    #[test]
    fn test_encode_fixed() {
        let arr = FixedSizeBinaryArray::from(vec![
            Some(&b"ABCDE"[..]),
        ]);
        let schema = ArrowSchema::new(vec![Field::new(
            "fixed_col",
            DataType::FixedSizeBinary(5),
            false,
        )]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(arr)],
        ).unwrap();
        let mut encoder = RecordEncoder::try_new(&schema, false).unwrap();
        let mut out = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out).unwrap();
        assert_eq!(out.len(), 5);
        assert_eq!(&out, b"ABCDE");
    }

    #[test]
    fn test_encode_nullable() -> Result<(), ArrowError> {
        let str_arr = StringArray::from(vec![None, Some("non-null here")]);
        let field = Field::new("maybe_str", DataType::Utf8, true);
        let schema = ArrowSchema::new(vec![field]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(str_arr)])?;
        let mut encoder = RecordEncoder::try_new(&schema, false)?;
        let mut out0 = Vec::new();
        encoder.encode_row(&schema, &batch, 0, &mut out0)?;
        let (branch, consumed) = decode_zigzag_long(&out0);
        assert_eq!(branch, 0, "Expected union branch=0 => null");
        assert_eq!(consumed, out0.len(), "No payload after branch=0");
        let mut out1 = Vec::new();
        encoder.encode_row(&schema, &batch, 1, &mut out1)?;
        let (branch, used) = decode_zigzag_long(&out1);
        assert_eq!(branch, 1, "Expected branch=1 => string");
        let got_str = decode_avro_string(&out1[used..]);
        assert_eq!(got_str.0, "non-null here");
        Ok(())
    }
}
