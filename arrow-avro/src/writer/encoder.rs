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

use arrow_array::*;
use arrow_schema::{ArrowError, DataType, Field, Fields, TimeUnit, UnionMode};
use std::io::Write;

/// Encodes an `i64` value into Avro's variable-length format and writes it
/// to the provided writer.
///
/// This function uses ZigZag encoding to map signed integers to unsigned
/// integers, followed by a variable-length encoding scheme where each byte
/// uses 7 bits for data and the most significant bit to indicate whether
/// more bytes follow.
///
/// # Arguments
///
/// * `writer` - A mutable reference to a writer that implements `std::io::Write`.
/// * `value` - The `i64` value to encode.
///
/// # Errors
///
/// Returns an `ArrowError::IoError` if any I/O error occurs while writing
/// to the `writer`.
pub fn write_long<W: Write>(writer: &mut W, value: i64) -> Result<(), ArrowError> {
    let mut zz = ((value << 1) ^ (value >> 63)) as u64;
    while zz & !0x7F != 0 {
        writer
            .write_all(&[((zz & 0x7F) as u8 | 0x80)])
            .map_err(|e| ArrowError::IoError(format!("write long: {e}"), e))?;
        zz >>= 7;
    }
    writer
        .write_all(&[(zz & 0x7F) as u8])
        .map_err(|e| ArrowError::IoError(format!("write long: {e}"), e))
}

/// Encodes a `RecordBatch` into Avro's binary format.
///
/// This function iterates through each row of the `RecordBatch` and writes it to the
/// provided `out` writer. The encoding is performed in a row-oriented manner.
///
/// # Arguments
///
/// * `batch` - A reference to the `RecordBatch` to be encoded.
/// * `out` - A mutable reference to a writer that implements `std::io::Write`.
///
/// # Errors
///
/// This function will return an `ArrowError` if any I/O error occurs during
/// writing to `out` or if an unsupported Arrow data type is encountered.
pub fn encode_record_batch<W: Write>(batch: &RecordBatch, out: &mut W) -> Result<(), ArrowError> {
    let schema = batch.schema();
    let fields = schema.fields();
    let columns = batch.columns();
    for row in 0..batch.num_rows() {
        encode_row(fields, columns, row, out)?;
    }
    Ok(())
}

fn encode_row<W: Write>(
    fields: &Fields,
    cols: &[ArrayRef],
    row_idx: usize,
    out: &mut W,
) -> Result<(), ArrowError> {
    for (field, col) in fields.iter().zip(cols) {
        if field.is_nullable() {
            if col.is_null(row_idx) {
                // branch 0 = null
                write_long(out, 0)?;
                continue;
            } else {
                // branch 1 = actual value
                write_long(out, 1)?;
            }
        }
        encode_value(col.as_ref(), field.data_type(), row_idx, out)?;
    }
    Ok(())
}

fn encode_value<W: Write>(
    array: &dyn Array,
    dt: &DataType,
    index: usize,
    out: &mut W,
) -> Result<(), ArrowError> {
    use DataType::*;
    match dt {
        Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            out.write_all(&[if arr.value(index) { 1 } else { 0 }])?;
        }
        Int8 | Int16 | Int32 => {
            let v = match dt {
                Int8 => array
                    .as_any()
                    .downcast_ref::<Int8Array>()
                    .unwrap()
                    .value(index) as i64,
                Int16 => array
                    .as_any()
                    .downcast_ref::<Int16Array>()
                    .unwrap()
                    .value(index) as i64,
                Int32 => array
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .value(index) as i64,
                _ => unreachable!(),
            };
            write_long(out, v)?;
        }
        Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            write_long(out, arr.value(index))?;
        }
        UInt8 | UInt16 | UInt32 => {
            let v = match dt {
                UInt8 => array
                    .as_any()
                    .downcast_ref::<UInt8Array>()
                    .unwrap()
                    .value(index) as i64,
                UInt16 => array
                    .as_any()
                    .downcast_ref::<UInt16Array>()
                    .unwrap()
                    .value(index) as i64,
                UInt32 => array
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .unwrap()
                    .value(index) as i64,
                _ => unreachable!(),
            };
            write_long(out, v)?;
        }
        UInt64 => {
            let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            let v = arr.value(index);
            if v > i64::MAX as u64 {
                return Err(ArrowError::InvalidArgumentError(
                    "UInt64 value exceeds Avro long range".into(),
                ));
            }
            write_long(out, v as i64)?;
        }
        Float32 => {
            let bits = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(index)
                .to_bits();
            out.write_all(&bits.to_le_bytes())?;
        }
        Float64 => {
            let bits = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(index)
                .to_bits();
            out.write_all(&bits.to_le_bytes())?;
        }
        Utf8 => {
            let s = array
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(index);
            write_long(out, s.len() as i64)?;
            out.write_all(s.as_bytes())?;
        }
        LargeUtf8 => {
            let s = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .unwrap()
                .value(index);
            write_long(out, s.len() as i64)?;
            out.write_all(s.as_bytes())?;
        }
        Binary => {
            let b = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .value(index);
            write_long(out, b.len() as i64)?;
            out.write_all(b)?;
        }
        LargeBinary => {
            let b = array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .unwrap()
                .value(index);
            write_long(out, b.len() as i64)?;
            out.write_all(b)?;
        }
        FixedSizeBinary(n) => {
            let b = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap()
                .value(index);
            write_long(out, *n as i64)?;
            out.write_all(b)?;
        }
        Decimal128(_, _) => {
            let val = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .unwrap()
                .value(index);
            let mut bytes = val.to_be_bytes().to_vec();
            while bytes.len() > 1
                && ((bytes[0] == 0x00 && bytes[1] & 0x80 == 0)
                    || (bytes[0] == 0xFF && bytes[1] & 0x80 != 0))
            {
                bytes.remove(0);
            }
            write_long(out, bytes.len() as i64)?;
            out.write_all(&bytes)?;
        }
        Date32 => {
            let days = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .unwrap()
                .value(index);
            write_long(out, days as i64)?;
        }
        Date64 => {
            let ms = array
                .as_any()
                .downcast_ref::<Date64Array>()
                .unwrap()
                .value(index);
            write_long(out, ms)?;
        }
        Time32(_) => {
            let v = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .unwrap()
                .value(index);
            write_long(out, v as i64)?;
        }
        Time64(_) => {
            let v = array
                .as_any()
                .downcast_ref::<Date64Array>()
                .unwrap()
                .value(index);
            write_long(out, v)?;
        }
        Timestamp(unit, _) => {
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
        Struct(fields) => {
            let struct_arr = array.as_any().downcast_ref::<StructArray>().unwrap();
            for (child_idx, field) in fields.iter().enumerate() {
                let child_arr = struct_arr.column(child_idx);
                if field.is_nullable() && child_arr.is_null(index) {
                    write_long(out, 0)?; // union null branch
                } else {
                    if field.is_nullable() {
                        write_long(out, 1)?;
                    }
                    encode_value(child_arr.as_ref(), field.data_type(), index, out)?;
                }
            }
        }
        List(child) | LargeList(child) => {
            let (len, offset, values): (i64, i64, ArrayRef) = match dt {
                List(_) => {
                    let list_arr = array.as_any().downcast_ref::<ListArray>().unwrap();
                    (
                        list_arr.value_length(index) as i64,
                        list_arr.value_offsets()[index] as i64,
                        list_arr.values().clone(),
                    )
                }
                LargeList(_) => {
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
                if child.is_nullable() && values.is_null(elem_idx) {
                    write_long(out, 0)?;
                } else {
                    if child.is_nullable() {
                        write_long(out, 1)?;
                    }
                    encode_value(values.as_ref(), item_dt, elem_idx, out)?;
                }
            }
            write_long(out, 0)?;
        }
        Map(_, _) => {
            let map_arr = array.as_any().downcast_ref::<MapArray>().unwrap();
            let len = map_arr.value_length(index);
            write_long(out, len.into())?;
            let offset = map_arr.value_offsets()[index];
            let entries = map_arr.entries();
            let key_arr = entries.column(0);
            let val_arr = entries.column(1);
            let val_field = match dt {
                Map(f, _) => f,
                _ => unreachable!(),
            };
            for j in 0..len {
                let idx = (offset + j) as usize;
                // key (non‑nullable string)
                let key = key_arr
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .value(idx);
                write_long(out, key.len() as i64)?;
                out.write_all(key.as_bytes())?;
                // value
                if val_field.is_nullable() && val_arr.is_null(idx) {
                    write_long(out, 0)?;
                } else {
                    if val_field.is_nullable() {
                        write_long(out, 1)?;
                    }
                    encode_value(val_arr.as_ref(), val_field.data_type(), idx, out)?;
                }
            }
            write_long(out, 0)?; // end of map block
        }
        Union(field_set, mode) => {
            let union_arr = array.as_any().downcast_ref::<UnionArray>().unwrap();
            let type_id = union_arr.type_id(index);
            let (variant_idx, child_field) = field_set
                .iter()
                .enumerate()
                .find_map(|(i, (id, f))| if id == type_id { Some((i, f)) } else { None })
                .ok_or_else(|| ArrowError::InvalidArgumentError("union type id missing".into()))?;
            write_long(out, variant_idx as i64)?; // Avro branch index
            let child_arr = union_arr.child(variant_idx.try_into().unwrap());
            let child_idx = match mode {
                UnionMode::Sparse => index,
                UnionMode::Dense => union_arr.value_offset(index) as usize,
            };
            encode_value(child_arr.as_ref(), child_field.data_type(), child_idx, out)?;
        }
        _ => {
            return Err(ArrowError::NotYetImplemented(format!(
                "DataType {:?} not supported in Avro encoder",
                dt
            )))
        }
    }
    Ok(())
}
