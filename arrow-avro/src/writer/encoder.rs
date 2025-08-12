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
    ArrowPrimitiveType, Float32Type, Float64Type, Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, BinaryArray, BooleanArray, PrimitiveArray, RecordBatch};
use arrow_buffer::NullBuffer;
use arrow_schema::{ArrowError, DataType, FieldRef, TimeUnit};
use std::io::Write;

/// Behavior knobs for the Avro encoder.
///
/// Currently, only `impala_mode` is exposed. When `true`, optional/nullable
/// values are encoded as Avro unions with **null second** (Impala format),
/// i.e. `[T, "null"]`. When `false` (default), we use `["null", T]`.
#[derive(Debug, Clone, Copy, Default)]
pub struct EncoderOptions {
    /// If `true`, encode nullability as `[T, "null"]` (Impala / null-second).
    /// If `false` (default), encode as `["null", T]` (null-first).
    pub impala_mode: bool,
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
    // Per spec, Avro `int` is ZigZag + varint, identical encoding shape to `long` for small values.
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
    } else if is_null {
        0
    } else {
        1
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
    /// Encode the value at `idx` into `out`, assuming it is not-null.
    ///
    /// The caller is responsible for handling nullability, e.g. by writing
    /// a union branch index before calling this method.
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError>;
}

/// An encoder + a null buffer.
pub struct NullableEncoder<'a> {
    encoder: Box<dyn Encoder + 'a>,
    nulls: Option<NullBuffer>,
}

impl<'a> NullableEncoder<'a> {
    /// Create a new nullable encoder, wrapping a non-null encoder and a null buffer.
    pub fn new(encoder: Box<dyn Encoder + 'a>, nulls: Option<NullBuffer>) -> Self {
        Self { encoder, nulls }
    }

    /// Encode the value at `idx`, assuming it's not-null.
    ///
    /// This will be called only after the caller has checked `is_null` and handled
    /// the null case (by writing the union branch index and skipping the value).
    pub fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        self.encoder.encode(idx, out)
    }

    /// Check if the value at `idx` is null.
    pub fn is_null(&self, idx: usize) -> bool {
        self.nulls.as_ref().is_some_and(|nulls| nulls.is_null(idx))
    }
}

/// Creates an Avro encoder for the given `array` and `field`.
pub fn make_encoder<'a>(
    _field: &'a FieldRef,
    array: &'a dyn Array,
    _options: &'a EncoderOptions,
) -> Result<NullableEncoder<'a>, ArrowError> {
    let nulls = array.nulls().cloned();
    let enc = match array.data_type() {
        DataType::Boolean => {
            let arr = array.as_boolean();
            NullableEncoder::new(Box::new(BooleanEncoder(arr)), nulls)
        }
        DataType::Int32 => {
            let arr = array.as_primitive::<Int32Type>();
            NullableEncoder::new(Box::new(IntI32Encoder::new(arr)), nulls)
        }
        DataType::Int64 => {
            let arr = array.as_primitive::<Int64Type>();
            NullableEncoder::new(Box::new(IntI64Encoder::new(arr)), nulls)
        }
        DataType::Float32 => {
            let arr = array.as_primitive::<Float32Type>();
            NullableEncoder::new(Box::new(F32Encoder(arr)), nulls)
        }
        DataType::Float64 => {
            let arr = array.as_primitive::<Float64Type>();
            NullableEncoder::new(Box::new(F64Encoder(arr)), nulls)
        }
        DataType::Binary => {
            let arr = array.as_binary::<i32>();
            NullableEncoder::new(Box::new(BinaryEncoder32(arr)), nulls)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let arr = array.as_primitive::<TimestampMicrosecondType>();
            NullableEncoder::new(Box::new(IntI64LikeEncoder(arr)), nulls)
        }
        other => {
            return Err(ArrowError::NotYetImplemented(format!(
                "Unsupported data type for Avro encoding in slim build: {other:?}"
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

struct IntI32Encoder<'a> {
    arr: &'a arrow_array::Int32Array,
}
impl<'a> IntI32Encoder<'a> {
    fn new(arr: &'a arrow_array::Int32Array) -> Self {
        Self { arr }
    }
}
impl Encoder for IntI32Encoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_int(out, self.arr.value(idx))
    }
}

struct IntI64Encoder<'a> {
    arr: &'a arrow_array::Int64Array,
}
impl<'a> IntI64Encoder<'a> {
    fn new(arr: &'a arrow_array::Int64Array) -> Self {
        Self { arr }
    }
}
impl Encoder for IntI64Encoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_long(out, self.arr.value(idx))
    }
}

struct F32Encoder<'a>(&'a arrow_array::Float32Array);
impl Encoder for F32Encoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let bits = self.0.value(idx).to_bits();
        out.write_all(&bits.to_le_bytes())
            .map_err(|e| ArrowError::IoError(format!("write f32: {e}"), e))
    }
}

struct F64Encoder<'a>(&'a arrow_array::Float64Array);
impl Encoder for F64Encoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        let bits = self.0.value(idx).to_bits();
        out.write_all(&bits.to_le_bytes())
            .map_err(|e| ArrowError::IoError(format!("write f64: {e}"), e))
    }
}

struct BinaryEncoder32<'a>(&'a BinaryArray);
impl Encoder for BinaryEncoder32<'_> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_len_prefixed(out, self.0.value(idx))
    }
}

/// Simple wrapper writing the raw `i64` of a primitive-like array (timestamps).
struct IntI64LikeEncoder<'a, P: ArrowPrimitiveType<Native = i64>>(&'a PrimitiveArray<P>);
impl<'a, P: ArrowPrimitiveType<Native = i64>> Encoder for IntI64LikeEncoder<'a, P> {
    fn encode(&mut self, idx: usize, out: &mut dyn Write) -> Result<(), ArrowError> {
        write_long(out, self.0.value(idx))
    }
}
