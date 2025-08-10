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

use std::io::Write;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::OffsetSizeTrait;
use arrow_array::*;
use arrow_buffer::ArrowNativeType;
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{
    ArrowError, DataType, Field, FieldRef, Fields, IntervalUnit, TimeUnit, UnionFields, UnionMode,
};

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
pub fn write_long<W: Write + ?Sized>(writer: &mut W, value: i64) -> Result<(), ArrowError> {
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
fn write_int<W: Write + ?Sized>(writer: &mut W, value: i32) -> Result<(), ArrowError> {
    // Per spec, Avro `int` is ZigZag+varint, identical encoding shape to `long` for small values.
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
        .write_all(&[if v { 1 } else { 0 }])
        .map_err(|e| ArrowError::IoError(format!("write bool: {e}"), e))
}

#[inline]
fn write_optional_branch<W: Write + ?Sized>(
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
    write_int(writer, branch)
}

/// Public API: encode a `RecordBatch` in Avro binary format using **default options**.
pub fn encode_record_batch<W: Write>(batch: &RecordBatch, out: &mut W) -> Result<(), ArrowError> {
    encode_record_batch_with_options(batch, out, &EncoderOptions::default())
}

/// Encode a `RecordBatch` with explicit `EncoderOptions`.
pub fn encode_record_batch_with_options<W: Write>(
    batch: &RecordBatch,
    out: &mut W,
    opts: &EncoderOptions,
) -> Result<(), ArrowError> {
    // Build one encoder per column
    let schema = batch.schema(); // avoid borrowing from a temporary (E0716)
    let fields = schema.fields();
    let mut encoders = Vec::with_capacity(fields.len());
    for (field, array) in fields.iter().zip(batch.columns().iter()) {
        let enc = make_encoder(field, array.as_ref(), opts)?;
        encoders.push((field.clone(), enc));
    }
    for row in 0..batch.num_rows() {
        for (field, enc) in encoders.iter_mut() {
            if field.is_nullable() {
                let is_null = enc.is_null(row);
                write_optional_branch(out, is_null, opts.impala_mode)?;
                if is_null {
                    continue;
                }
            }
            enc.encode(row, out)?;
        }
    }
    Ok(())
}

/// Object-safe encoder of **non-null** values at a given row index.
pub trait Encoder {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError>;
}

/// An encoder + a null buffer.
pub struct NullableEncoder<'a> {
    encoder: Box<dyn Encoder + 'a>,
    nulls: Option<NullBuffer>,
}

impl<'a> NullableEncoder<'a> {
    pub fn new(encoder: Box<dyn Encoder + 'a>, nulls: Option<NullBuffer>) -> Self {
        Self { encoder, nulls }
    }

    pub fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        self.encoder.encode(idx, out)
    }

    pub fn is_null(&self, idx: usize) -> bool {
        self.nulls.as_ref().is_some_and(|nulls| nulls.is_null(idx))
    }
}

/// Creates an Avro encoder for the given `array` and `field`.
pub fn make_encoder<'a>(
    field: &'a FieldRef,
    array: &'a dyn Array,
    options: &'a EncoderOptions,
) -> Result<NullableEncoder<'a>, ArrowError> {
    let nulls = array.nulls().cloned();
    let enc = match array.data_type() {
        DataType::Boolean => {
            let arr = array.as_boolean();
            NullableEncoder::new(Box::new(BooleanEncoder(arr)), nulls)
        }
        DataType::Int8 => {
            let arr = array.as_primitive::<Int8Type>();
            NullableEncoder::new(Box::new(I8ToIntEncoder(arr)), nulls)
        }
        DataType::Int16 => {
            let arr = array.as_primitive::<Int16Type>();
            NullableEncoder::new(Box::new(I16ToIntEncoder(arr)), nulls)
        }
        DataType::Int32 => {
            let arr = array.as_primitive::<Int32Type>();
            NullableEncoder::new(Box::new(IntI32Encoder::new(arr)), nulls)
        }
        DataType::Int64 => {
            let arr = array.as_primitive::<Int64Type>();
            NullableEncoder::new(Box::new(IntI64Encoder::new(arr)), nulls)
        }
        DataType::UInt8 => {
            let arr = array.as_primitive::<UInt8Type>();
            NullableEncoder::new(Box::new(UInt8ToIntEncoder(arr)), nulls)
        }
        DataType::UInt16 => {
            let arr = array.as_primitive::<UInt16Type>();
            NullableEncoder::new(Box::new(UInt16ToIntEncoder(arr)), nulls)
        }
        DataType::UInt32 => {
            let arr = array.as_primitive::<UInt32Type>();
            NullableEncoder::new(Box::new(UInt32ToIntEncoder(arr)), nulls)
        }
        DataType::UInt64 => {
            let arr = array.as_primitive::<UInt64Type>();
            NullableEncoder::new(Box::new(UInt64ToLongEncoder(arr)), nulls)
        }
        DataType::Float32 => {
            let arr = array.as_primitive::<Float32Type>();
            NullableEncoder::new(Box::new(F32Encoder(arr)), nulls)
        }
        DataType::Float64 => {
            let arr = array.as_primitive::<Float64Type>();
            NullableEncoder::new(Box::new(F64Encoder(arr)), nulls)
        }
        DataType::Utf8 => {
            let arr = array.as_string::<i32>();
            NullableEncoder::new(Box::new(StringEncoder(arr)), nulls)
        }
        DataType::LargeUtf8 => {
            let arr = array.as_string::<i64>();
            // Use the tuple struct directly; type aliases aren't constructors
            NullableEncoder::new(Box::new(StringEncoder(arr)), nulls)
        }
        DataType::Binary => {
            let arr = array.as_binary::<i32>();
            NullableEncoder::new(Box::new(BinaryEncoder32(arr)), nulls)
        }
        DataType::LargeBinary => {
            let arr = array.as_binary::<i64>();
            NullableEncoder::new(Box::new(BinaryEncoder64(arr)), nulls)
        }
        DataType::FixedSizeBinary(_) => {
            let arr = array.as_fixed_size_binary();
            NullableEncoder::new(Box::new(FixedSizeBinaryEncoder(arr)), nulls)
        }
        DataType::Decimal128(_, _) => {
            // decimals are primitive arrays with Decimal128Type logical type
            let arr = array.as_primitive::<Decimal128Type>();
            NullableEncoder::new(Box::new(Decimal128Encoder(arr)), nulls)
        }
        DataType::Decimal256(_, _) => {
            let arr = array.as_primitive::<Decimal256Type>();
            NullableEncoder::new(Box::new(Decimal256Encoder(arr)), nulls)
        }
        DataType::Date32 => {
            let arr = array.as_primitive::<Date32Type>();
            NullableEncoder::new(Box::new(Date32Encoder(arr)), nulls)
        }
        DataType::Date64 => {
            let arr = array.as_primitive::<Date64Type>();
            NullableEncoder::new(Box::new(Date64Encoder(arr)), nulls)
        }
        DataType::Time32(TimeUnit::Second) => {
            let arr = array.as_primitive::<Time32SecondType>();
            NullableEncoder::new(Box::new(Time32SecondToMillisEncoder(arr)), nulls)
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            let arr = array.as_primitive::<Time32MillisecondType>();
            NullableEncoder::new(Box::new(Time32MillisEncoder(arr)), nulls)
        }
        DataType::Time32(other) => {
            return Err(ArrowError::NotYetImplemented(format!(
                "Time32 with unit {other:?} is not valid for Avro"
            )))
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            let arr = array.as_primitive::<Time64MicrosecondType>();
            NullableEncoder::new(Box::new(Time64MicrosEncoder(arr)), nulls)
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            let arr = array.as_primitive::<Time64NanosecondType>();
            NullableEncoder::new(Box::new(Time64NanoToMicrosEncoder(arr)), nulls)
        }
        DataType::Time64(other) => {
            return Err(ArrowError::NotYetImplemented(format!(
                "Time64 with unit {other:?} is not supported for Avro"
            )))
        }
        DataType::Timestamp(TimeUnit::Second, _) => {
            let arr = array.as_primitive::<TimestampSecondType>();
            NullableEncoder::new(Box::new(IntI64LikeEncoder(arr)), nulls)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            let arr = array.as_primitive::<TimestampMillisecondType>();
            NullableEncoder::new(Box::new(IntI64LikeEncoder(arr)), nulls)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let arr = array.as_primitive::<TimestampMicrosecondType>();
            NullableEncoder::new(Box::new(IntI64LikeEncoder(arr)), nulls)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            let arr = array.as_primitive::<TimestampNanosecondType>();
            NullableEncoder::new(Box::new(IntI64LikeEncoder(arr)), nulls)
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            // interval arrays are also primitive arrays
            let arr = array.as_primitive::<IntervalMonthDayNanoType>();
            NullableEncoder::new(Box::new(IntervalMonthDayNanoEncoder(arr)), nulls)
        }
        DataType::Struct(children) => {
            let arr = array.as_struct();
            let mut encs = Vec::with_capacity(children.len());
            for (field, child_arr) in children.iter().zip(arr.columns()) {
                let enc = make_encoder(field, child_arr.as_ref(), options)?;
                encs.push((field.clone(), enc));
            }
            NullableEncoder::new(
                Box::new(StructEncoder {
                    children: encs,
                    impala_mode: options.impala_mode,
                }),
                nulls,
            )
        }
        DataType::List(child) => {
            let arr = array.as_list::<i32>();
            let enc = make_encoder(child, arr.values().as_ref(), options)?;
            NullableEncoder::new(
                Box::new(ListEncoder::<i32>::new(
                    arr.offsets().clone(),
                    enc,
                    child.clone(),
                    options.impala_mode,
                )),
                nulls,
            )
        }
        DataType::LargeList(child) => {
            let arr = array.as_list::<i64>();
            let enc = make_encoder(child, arr.values().as_ref(), options)?;
            NullableEncoder::new(
                Box::new(ListEncoder::<i64>::new(
                    arr.offsets().clone(),
                    enc,
                    child.clone(),
                    options.impala_mode,
                )),
                nulls,
            )
        }
        DataType::FixedSizeList(child, _) => {
            let arr = array.as_fixed_size_list();
            let enc = make_encoder(child, arr.values().as_ref(), options)?;
            NullableEncoder::new(
                Box::new(FixedSizeListEncoder::new(
                    arr.value_length() as usize,
                    enc,
                    child.clone(),
                    options.impala_mode,
                )),
                nulls,
            )
        }
        DataType::Map(value_field, _keys_sorted) => {
            let arr = array.as_map();
            let keys_arr = arr.keys();
            // Enforce utf8 keys per Arrow spec and Avro map key requirements
            if !matches!(keys_arr.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Avro maps require Utf8 keys, got {:?}",
                    keys_arr.data_type()
                )));
            }
            let values_enc = make_encoder(value_field, arr.values().as_ref(), options)?;
            let keys = match keys_arr.data_type() {
                DataType::Utf8 => MapKeys::Utf8(keys_arr.as_string::<i32>()),
                DataType::LargeUtf8 => MapKeys::LargeUtf8(keys_arr.as_string::<i64>()),
                _ => unreachable!(),
            };
            NullableEncoder::new(
                Box::new(MapEncoder {
                    offsets: arr.offsets().clone(),
                    keys,
                    values: values_enc,
                    value_field: value_field.clone(),
                    impala_mode: options.impala_mode,
                }),
                nulls,
            )
        }
        DataType::Union(fields, mode) => {
            let arr = array.as_union();
            let enc = UnionEncoder::new(fields, *mode, arr, options)?;
            NullableEncoder::new(Box::new(enc), nulls)
        }
        DataType::Dictionary(key, value) => match (key.as_ref(), value.as_ref()) {
            (DataType::Int8, DataType::Utf8 | DataType::LargeUtf8) => {
                let arr = array.as_dictionary::<Int8Type>();
                NullableEncoder::new(Box::new(DictionaryIndexEncoderI8(arr)), nulls)
            }
            (DataType::Int16, DataType::Utf8 | DataType::LargeUtf8) => {
                let arr = array.as_dictionary::<Int16Type>();
                NullableEncoder::new(Box::new(DictionaryIndexEncoderI16(arr)), nulls)
            }
            (DataType::Int32, DataType::Utf8 | DataType::LargeUtf8) => {
                let arr = array.as_dictionary::<Int32Type>();
                NullableEncoder::new(Box::new(DictionaryIndexEncoderI32(arr)), nulls)
            }
            (DataType::Int64, DataType::Utf8 | DataType::LargeUtf8) => {
                let arr = array.as_dictionary::<Int64Type>();
                NullableEncoder::new(Box::new(DictionaryIndexEncoderI64(arr)), nulls)
            }
            (_, other) => {
                return Err(ArrowError::NotYetImplemented(format!(
                    "Dictionary of {other:?} not supported for Avro encoding (expect Utf8)"
                )))
            }
        },
        DataType::Null => NullableEncoder::new(Box::new(NullEncoder), nulls),
        other => {
            return Err(ArrowError::NotYetImplemented(format!(
                "Unsupported data type for Avro encoding: {other:?}"
            )))
        }
    };
    Ok(enc)
}

struct BooleanEncoder<'a>(&'a BooleanArray);
impl Encoder for BooleanEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_bool(out, self.0.value(idx))
    }
}

struct I8ToIntEncoder<'a>(&'a Int8Array);
impl Encoder for I8ToIntEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_int(out, self.0.value(idx) as i32)
    }
}

struct I16ToIntEncoder<'a>(&'a Int16Array);
impl Encoder for I16ToIntEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_int(out, self.0.value(idx) as i32)
    }
}

struct IntI32Encoder<'a> {
    arr: &'a Int32Array,
}
impl<'a> IntI32Encoder<'a> {
    fn new(arr: &'a Int32Array) -> Self {
        Self { arr }
    }
}
impl Encoder for IntI32Encoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_int(out, self.arr.value(idx))
    }
}

struct IntI64Encoder<'a> {
    arr: &'a Int64Array,
}
impl<'a> IntI64Encoder<'a> {
    fn new(arr: &'a Int64Array) -> Self {
        Self { arr }
    }
}
impl Encoder for IntI64Encoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_long(out, self.arr.value(idx))
    }
}

struct UInt8ToIntEncoder<'a>(&'a UInt8Array);
impl Encoder for UInt8ToIntEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_int(out, self.0.value(idx) as i32)
    }
}

struct UInt16ToIntEncoder<'a>(&'a UInt16Array);
impl Encoder for UInt16ToIntEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_int(out, self.0.value(idx) as i32)
    }
}

struct UInt32ToIntEncoder<'a>(&'a UInt32Array);
impl Encoder for UInt32ToIntEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let v = self.0.value(idx) as i64;
        if v > i32::MAX as i64 {
            return Err(ArrowError::InvalidArgumentError(
                "UInt32 value exceeds Avro int range".into(),
            ));
        }
        write_int(out, v as i32)
    }
}

struct UInt64ToLongEncoder<'a>(&'a UInt64Array);
impl Encoder for UInt64ToLongEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let v = self.0.value(idx);
        if v > i64::MAX as u64 {
            return Err(ArrowError::InvalidArgumentError(
                "UInt64 value exceeds Avro long range".into(),
            ));
        }
        write_long(out, v as i64)
    }
}

struct F32Encoder<'a>(&'a Float32Array);
impl Encoder for F32Encoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let bits = self.0.value(idx).to_bits();
        out.write_all(&bits.to_le_bytes())
            .map_err(|e| ArrowError::IoError(format!("write f32: {e}"), e))
    }
}

struct F64Encoder<'a>(&'a Float64Array);
impl Encoder for F64Encoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let bits = self.0.value(idx).to_bits();
        out.write_all(&bits.to_le_bytes())
            .map_err(|e| ArrowError::IoError(format!("write f64: {e}"), e))
    }
}

struct StringEncoder<'a, O: OffsetSizeTrait>(&'a GenericStringArray<O>);
impl<O: OffsetSizeTrait> Encoder for StringEncoder<'_, O> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_len_prefixed(out, self.0.value(idx).as_bytes())
    }
}

struct BinaryEncoder32<'a>(&'a BinaryArray);
impl Encoder for BinaryEncoder32<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_len_prefixed(out, self.0.value(idx))
    }
}

struct BinaryEncoder64<'a>(&'a LargeBinaryArray);
impl Encoder for BinaryEncoder64<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_len_prefixed(out, self.0.value(idx))
    }
}

struct FixedSizeBinaryEncoder<'a>(&'a FixedSizeBinaryArray);
impl Encoder for FixedSizeBinaryEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let v = self.0.value(idx);
        out.write_all(v)
            .map_err(|e| ArrowError::IoError(format!("write fixed: {e}"), e))
    }
}

struct Decimal128Encoder<'a>(&'a Decimal128Array);
impl Encoder for Decimal128Encoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let v = self.0.value(idx);
        let mut bytes = v.to_be_bytes().to_vec();
        strip_sign_extension(&mut bytes);
        write_len_prefixed(out, &bytes)
    }
}

struct Decimal256Encoder<'a>(&'a Decimal256Array);
impl Encoder for Decimal256Encoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let v = self.0.value(idx);
        let mut bytes = v.to_be_bytes().to_vec();
        strip_sign_extension(&mut bytes);
        write_len_prefixed(out, &bytes)
    }
}

struct Date32Encoder<'a>(&'a Date32Array);
impl Encoder for Date32Encoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_int(out, self.0.value(idx))
    }
}

struct Date64Encoder<'a>(&'a Date64Array);
impl Encoder for Date64Encoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_long(out, self.0.value(idx))
    }
}

struct Time32SecondToMillisEncoder<'a>(&'a Time32SecondArray);
impl Encoder for Time32SecondToMillisEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let secs = self.0.value(idx);
        let millis = secs
            .checked_mul(1000)
            .ok_or_else(|| ArrowError::ComputeError("time32(second) overflow".into()))?;
        write_int(out, millis)
    }
}

struct Time32MillisEncoder<'a>(&'a Time32MillisecondArray);
impl Encoder for Time32MillisEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_int(out, self.0.value(idx))
    }
}

struct Time64MicrosEncoder<'a>(&'a Time64MicrosecondArray);
impl Encoder for Time64MicrosEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_long(out, self.0.value(idx))
    }
}

struct Time64NanoToMicrosEncoder<'a>(&'a Time64NanosecondArray);
impl Encoder for Time64NanoToMicrosEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let nanos = self.0.value(idx);
        if nanos % 1_000 != 0 {
            return Err(ArrowError::InvalidArgumentError(
                "Cannot encode Time64(Nanosecond) exactly as Avro time-micros: \
                 value not divisible by 1,000"
                    .into(),
            ));
        }
        write_long(out, nanos / 1_000)
    }
}

/// Simple wrapper writing the raw `i64` of a primitive-like array (timestamps).
struct IntI64LikeEncoder<'a, P: ArrowPrimitiveType<Native = i64>>(&'a PrimitiveArray<P>);
impl<'a, P: ArrowPrimitiveType<Native = i64>> Encoder for IntI64LikeEncoder<'a, P> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_long(out, self.0.value(idx))
    }
}

struct IntervalMonthDayNanoEncoder<'a>(&'a IntervalMonthDayNanoArray);
impl Encoder for IntervalMonthDayNanoEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let v = self.0.value(idx);
        if v.nanoseconds % 1_000_000 != 0 {
            return Err(ArrowError::InvalidArgumentError(
                "IntervalMonthDayNano cannot be encoded as Avro duration: \
                 nanoseconds not divisible by 1,000,000"
                    .into(),
            ));
        }
        let months = v.months.to_le_bytes();
        let days = v.days.to_le_bytes();
        let millis = ((v.nanoseconds / 1_000_000) as i32).to_le_bytes();
        let mut buf = [0u8; 12];
        buf[0..4].copy_from_slice(&months);
        buf[4..8].copy_from_slice(&days);
        buf[8..12].copy_from_slice(&millis);
        out.write_all(&buf)
            .map_err(|e| ArrowError::IoError(format!("write duration: {e}"), e))
    }
}

struct StructEncoder<'a> {
    children: Vec<(FieldRef, NullableEncoder<'a>)>,
    impala_mode: bool,
}
impl Encoder for StructEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        for (field, enc) in self.children.iter_mut() {
            if field.is_nullable() {
                let is_null = enc.is_null(idx);
                write_optional_branch(out, is_null, self.impala_mode)?;
                if is_null {
                    continue;
                }
            }
            enc.encode(idx, out)?;
        }
        Ok(())
    }
}

struct FixedSizeListEncoder<'a> {
    value_length: usize,
    values: NullableEncoder<'a>,
    item_field: FieldRef,
    impala_mode: bool,
}
impl<'a> FixedSizeListEncoder<'a> {
    fn new(
        value_length: usize,
        values: NullableEncoder<'a>,
        item_field: FieldRef,
        impala_mode: bool,
    ) -> Self {
        Self {
            value_length,
            values,
            item_field,
            impala_mode,
        }
    }
}
impl Encoder for FixedSizeListEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_long(out, self.value_length as i64)?;
        let start = idx * self.value_length;
        let end = start + self.value_length;
        for i in start..end {
            if self.item_field.is_nullable() {
                let is_null = self.values.is_null(i);
                write_optional_branch(out, is_null, self.impala_mode)?;
                if is_null {
                    continue;
                }
            }
            self.values.encode(i, out)?;
        }
        write_long(out, 0)
    }
}

struct ListEncoder<'a, O: OffsetSizeTrait> {
    offsets: OffsetBuffer<O>,
    values: NullableEncoder<'a>,
    item_field: FieldRef,
    impala_mode: bool,
}
impl<'a, O: OffsetSizeTrait> ListEncoder<'a, O> {
    fn new(
        offsets: OffsetBuffer<O>,
        values: NullableEncoder<'a>,
        item_field: FieldRef,
        impala_mode: bool,
    ) -> Self {
        Self {
            offsets,
            values,
            item_field,
            impala_mode,
        }
    }
}
impl<O: OffsetSizeTrait> Encoder for ListEncoder<'_, O> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let start = self.offsets[idx].as_usize();
        let end = self.offsets[idx + 1].as_usize();
        let len = (end - start) as i64;
        write_long(out, len)?;
        for i in start..end {
            if self.item_field.is_nullable() {
                let is_null = self.values.is_null(i);
                write_optional_branch(out, is_null, self.impala_mode)?;
                if is_null {
                    continue;
                }
            }
            self.values.encode(i, out)?;
        }
        write_long(out, 0)
    }
}

enum MapKeys<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
}
struct MapEncoder<'a> {
    offsets: OffsetBuffer<i32>,
    keys: MapKeys<'a>,
    values: NullableEncoder<'a>,
    value_field: FieldRef,
    impala_mode: bool,
}
impl Encoder for MapEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let start = self.offsets[idx].as_usize();
        let end = self.offsets[idx + 1].as_usize();
        let len = (end - start) as i64;
        write_long(out, len)?;
        for i in start..end {
            match self.keys {
                MapKeys::Utf8(arr) => {
                    if arr.is_null(i) {
                        return Err(ArrowError::InvalidArgumentError(
                            "Avro map keys cannot be null".into(),
                        ));
                    }
                    write_len_prefixed(out, arr.value(i).as_bytes())?;
                }
                MapKeys::LargeUtf8(arr) => {
                    if arr.is_null(i) {
                        return Err(ArrowError::InvalidArgumentError(
                            "Avro map keys cannot be null".into(),
                        ));
                    }
                    write_len_prefixed(out, arr.value(i).as_bytes())?;
                }
            }
            if self.value_field.is_nullable() {
                let is_null = self.values.is_null(i);
                write_optional_branch(out, is_null, self.impala_mode)?;
                if is_null {
                    continue;
                }
            }
            self.values.encode(i, out)?;
        }
        write_long(out, 0)
    }
}

struct UnionEncoder<'a> {
    array: &'a UnionArray,
    mode: UnionMode,
    arms: Vec<(i8, i32, NullableEncoder<'a>)>,
}
impl<'a> UnionEncoder<'a> {
    fn new(
        fields: &'a UnionFields,
        mode: UnionMode,
        array: &'a UnionArray,
        options: &'a EncoderOptions,
    ) -> Result<Self, ArrowError> {
        let mut arms = Vec::with_capacity(fields.len());
        for (variant_idx, (type_id, field)) in fields.iter().enumerate() {
            let child = array.child(type_id);
            let enc = make_encoder(field, child.as_ref(), options)?;
            arms.push((type_id, variant_idx as i32, enc));
        }
        Ok(Self { array, mode, arms })
    }
}
impl Encoder for UnionEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let type_id = self.array.type_id(idx);
        let pos = self
            .arms
            .iter()
            .position(|(tid, _, _)| *tid == type_id)
            .ok_or_else(|| ArrowError::InvalidArgumentError("union type id missing".into()))?;
        let variant_idx = self.arms[pos].1;
        let child_idx = match self.mode {
            UnionMode::Sparse => idx,
            UnionMode::Dense => self.array.value_offset(idx) as usize,
        };
        write_int(out, variant_idx)?;
        self.arms[pos].2.encode(child_idx, out)
    }
}

struct DictionaryIndexEncoderI8<'a>(&'a DictionaryArray<Int8Type>);
impl Encoder for DictionaryIndexEncoderI8<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_int(out, self.0.keys().value(idx) as i32)
    }
}
struct DictionaryIndexEncoderI16<'a>(&'a DictionaryArray<Int16Type>);
impl Encoder for DictionaryIndexEncoderI16<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_int(out, self.0.keys().value(idx) as i32)
    }
}
struct DictionaryIndexEncoderI32<'a>(&'a DictionaryArray<Int32Type>);
impl Encoder for DictionaryIndexEncoderI32<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_int(out, self.0.keys().value(idx))
    }
}
struct DictionaryIndexEncoderI64<'a>(&'a DictionaryArray<Int64Type>);
impl Encoder for DictionaryIndexEncoderI64<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let k = self.0.keys().value(idx);
        if k > i32::MAX as i64 {
            return Err(ArrowError::InvalidArgumentError(
                "Dictionary index exceeds Avro enum index range".into(),
            ));
        }
        write_int(out, k as i32)
    }
}

struct NullEncoder;
impl Encoder for NullEncoder {
    fn encode(&mut self, _idx: usize, _out: &mut dyn Write) -> Result<(), ArrowError> {
        // should never be called for a non-null slot
        unreachable!("NullEncoder.encode called for a non-null slot")
    }
}

/// Strip redundant sign-extension bytes for Avro decimal (bytes form).
fn strip_sign_extension(bytes: &mut Vec<u8>) {
    while bytes.len() > 1
        && ((bytes[0] == 0x00 && (bytes[1] & 0x80) == 0)
            || (bytes[0] == 0xFF && (bytes[1] & 0x80) != 0))
    {
        bytes.remove(0);
    }
}

fn encode_field_value(
    field: &Field,
    array: &dyn Array,
    index: usize,
    out: &mut dyn Write,
    opts: &EncoderOptions,
) -> Result<(), ArrowError> {
    if field.is_nullable() {
        let is_null = array.is_null(index);
        write_optional_branch(out, is_null, opts.impala_mode)?;
        if is_null {
            return Ok(());
        }
    }
    encode_value(array, field.data_type(), index, out, opts)
}

fn encode_row<W: Write>(
    fields: &Fields,
    cols: &[ArrayRef],
    row_idx: usize,
    out: &mut W,
    opts: &EncoderOptions,
) -> Result<(), ArrowError> {
    for (field, col) in fields.iter().zip(cols) {
        encode_field_value(field, col.as_ref(), row_idx, out as &mut dyn Write, opts)?;
    }
    Ok(())
}

/// Fallback/value-based encoder used by older codepaths (kept for internal reuse).
fn encode_value(
    array: &dyn Array,
    dt: &DataType,
    index: usize,
    out: &mut dyn Write,
    opts: &EncoderOptions,
) -> Result<(), ArrowError> {
    let fake_field = Arc::new(Field::new("f", dt.clone(), array.is_nullable()));
    let mut enc = make_encoder(&fake_field, array, opts)?;
    enc.encode(index, out)
}
