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
    AvroDataType, AvroLiteral, Codec, EnumMapping, Nullability, Promotion, ResolutionInfo,
    ResolvedRecord,
};
use crate::reader::cursor::AvroCursor;
use arrow_array::builder::{
    ArrayBuilder, BooleanBufferBuilder, Decimal128Builder, Decimal256Builder,
    IntervalMonthDayNanoBuilder, NullBufferBuilder,
};
use arrow_array::types::*;
use arrow_array::*;
use arrow_buffer::{i256, Buffer, BufferBuilder, NullBuffer, OffsetBuffer, OffsetBufferBuilder};
use arrow_schema::{
    ArrowError, DataType, Field as ArrowField, FieldRef, Fields, Schema as ArrowSchema, SchemaRef,
};
use std::cmp::Ordering;
use std::sync::Arc;

const DEFAULT_CAPACITY: usize = 1024;

/// Builder for a [`RecordDecoder`]
#[derive(Debug)]
pub(crate) struct RecordDecoderBuilder<'a> {
    data_type: &'a AvroDataType,
    use_utf8view: bool,
}

impl<'a> RecordDecoderBuilder<'a> {
    pub(crate) fn new(data_type: &'a AvroDataType) -> Self {
        Self {
            data_type,
            use_utf8view: false,
        }
    }

    pub(crate) fn with_utf8_view(mut self, flag: bool) -> Self {
        self.use_utf8view = flag;
        self
    }

    /// Builds the `RecordDecoder`.
    pub(crate) fn build(self) -> Result<RecordDecoder, ArrowError> {
        RecordDecoder::try_new_with_options(self.data_type, self.use_utf8view)
    }
}

/// Converts Avro records into Arrow [`RecordBatch`]es
#[derive(Debug)]
pub(crate) struct RecordDecoder {
    schema: SchemaRef,
    fields: Vec<Decoder>,
}

impl RecordDecoder {
    /// Creates a new [`RecordDecoder`] from the provided [`AvroDataType`] with additional options.
    ///
    /// This method allows you to customize how the Avro data is decoded into Arrow arrays.
    ///
    /// # Arguments
    /// * `data_type` - The Avro data type to decode.
    /// * `use_utf8view` - A flag indicating whether to use `Utf8View` for string types.
    ///
    /// # Errors
    /// This function will return an error if the provided `data_type` is not a `Record`.
    pub(crate) fn try_new_with_options(
        data_type: &AvroDataType,
        use_utf8view: bool,
    ) -> Result<Self, ArrowError> {
        match Decoder::try_new_with_view(data_type, use_utf8view)? {
            Decoder::Record {
                arrow_fields,
                field_decoders,
                ..
            } => Ok(Self {
                schema: Arc::new(ArrowSchema::new(arrow_fields)),
                fields: field_decoders,
            }),
            _ => Err(ArrowError::ParseError(
                "Top-level Avro schema must be a record".into(),
            )),
        }
    }

    /// Create a new [`RecordDecoder`] from the provided [`AvroDataType`] with default options
    #[inline]
    pub(crate) fn try_new(data_type: &AvroDataType) -> Result<Self, ArrowError> {
        Self::try_new_with_options(data_type, false)
    }

    /// Returns the decoder's `SchemaRef`
    #[inline]
    pub(crate) fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Decodes `count` records from the provided buffer.
    pub(crate) fn decode(&mut self, buf: &[u8], count: usize) -> Result<usize, ArrowError> {
        for f in &mut self.fields {
            f.reserve(count);
        }
        let mut cursor = AvroCursor::new(buf);
        for _ in 0..count {
            for f in &mut self.fields {
                f.decode(&mut cursor)?;
            }
        }
        Ok(cursor.position())
    }

    /// Flushes the decoded records into a [`RecordBatch`].
    pub(crate) fn flush(&mut self) -> Result<RecordBatch, ArrowError> {
        let arrays = self
            .fields
            .iter_mut()
            .map(|d| d.flush(None))
            .collect::<Result<Vec<_>, _>>()?;
        RecordBatch::try_new(self.schema.clone(), arrays)
    }
}

#[derive(Debug)]
enum Decoder {
    Null(usize),
    Boolean(BooleanBufferBuilder),
    Int32(BufferBuilder<i32>),
    Int64(BufferBuilder<i64>),
    Float32(BufferBuilder<f32>),
    Float64(BufferBuilder<f64>),
    Int32ToInt64(BufferBuilder<i64>),
    Int32ToFloat32(BufferBuilder<f32>),
    Int32ToFloat64(BufferBuilder<f64>),
    Int64ToFloat32(BufferBuilder<f32>),
    Int64ToFloat64(BufferBuilder<f64>),
    Float32ToFloat64(BufferBuilder<f64>),
    BytesToString(OffsetBufferBuilder<i32>, BufferBuilder<u8>),
    StringToBytes(OffsetBufferBuilder<i32>, BufferBuilder<u8>),
    Binary(OffsetBufferBuilder<i32>, BufferBuilder<u8>),
    String(OffsetBufferBuilder<i32>, BufferBuilder<u8>),
    StringView(OffsetBufferBuilder<i32>, BufferBuilder<u8>),
    Date32(BufferBuilder<i32>),
    TimeMillis(BufferBuilder<i32>),
    TimeMicros(BufferBuilder<i64>),
    TimestampMillis(bool, BufferBuilder<i64>),
    TimestampMicros(bool, BufferBuilder<i64>),
    Fixed(i32, BufferBuilder<u8>),
    Decimal128(usize, Option<usize>, Option<usize>, Decimal128Builder),
    Decimal256(usize, Option<usize>, Option<usize>, Decimal256Builder),
    Uuid(BufferBuilder<u8>),
    Duration(IntervalMonthDayNanoBuilder),
    Enum(BufferBuilder<i32>, Arc<[String]>, Option<EnumMapping>),
    Array(FieldRef, OffsetBufferBuilder<i32>, Box<Decoder>),
    Map(
        FieldRef,
        OffsetBufferBuilder<i32>,
        OffsetBufferBuilder<i32>,
        BufferBuilder<u8>,
        Box<Decoder>,
    ),
    Record {
        arrow_fields: Fields,
        field_decoders: Vec<Decoder>,
        field_defaults: Vec<Option<AvroLiteral>>,
        mapping: Option<Arc<[Option<usize>]>>,
        skip_decoders: Option<Vec<Option<Decoder>>>,
    },
    Nullable(Nullability, NullBufferBuilder, Box<Decoder>),
}

impl Decoder {
    fn try_new_with_view(data_type: &AvroDataType, use_utf8view: bool) -> Result<Self, ArrowError> {
        use Decoder::*;
        let make_child = |dt: &AvroDataType| Decoder::try_new_with_view(dt, use_utf8view);
        let base = match data_type.codec() {
            Codec::Null => Null(0),
            Codec::Boolean => Boolean(BooleanBufferBuilder::new(DEFAULT_CAPACITY)),
            Codec::Int32 => Int32(BufferBuilder::new(DEFAULT_CAPACITY)),
            Codec::Int64 => Int64(BufferBuilder::new(DEFAULT_CAPACITY)),
            Codec::Float32 => Float32(BufferBuilder::new(DEFAULT_CAPACITY)),
            Codec::Float64 => Float64(BufferBuilder::new(DEFAULT_CAPACITY)),
            _ => match data_type.resolution.as_ref() {
                Some(ResolutionInfo::Promotion(p)) => match p {
                    Promotion::IntToLong => Int32ToInt64(BufferBuilder::new(DEFAULT_CAPACITY)),
                    Promotion::IntToFloat => Int32ToFloat32(BufferBuilder::new(DEFAULT_CAPACITY)),
                    Promotion::IntToDouble => Int32ToFloat64(BufferBuilder::new(DEFAULT_CAPACITY)),
                    Promotion::LongToFloat => Int64ToFloat32(BufferBuilder::new(DEFAULT_CAPACITY)),
                    Promotion::LongToDouble => Int64ToFloat64(BufferBuilder::new(DEFAULT_CAPACITY)),
                    Promotion::FloatToDouble => {
                        Float32ToFloat64(BufferBuilder::new(DEFAULT_CAPACITY))
                    }
                    Promotion::BytesToString => BytesToString(
                        OffsetBufferBuilder::new(DEFAULT_CAPACITY),
                        BufferBuilder::new(DEFAULT_CAPACITY),
                    ),
                    Promotion::StringToBytes => StringToBytes(
                        OffsetBufferBuilder::new(DEFAULT_CAPACITY),
                        BufferBuilder::new(DEFAULT_CAPACITY),
                    ),
                },
                _ => match data_type.codec() {
                    Codec::Binary => Binary(
                        OffsetBufferBuilder::new(DEFAULT_CAPACITY),
                        BufferBuilder::new(DEFAULT_CAPACITY),
                    ),
                    Codec::Utf8 => String(
                        OffsetBufferBuilder::new(DEFAULT_CAPACITY),
                        BufferBuilder::new(DEFAULT_CAPACITY),
                    ),
                    Codec::Utf8View if use_utf8view => StringView(
                        OffsetBufferBuilder::new(DEFAULT_CAPACITY),
                        BufferBuilder::new(DEFAULT_CAPACITY),
                    ),
                    Codec::Date32 => Date32(BufferBuilder::new(DEFAULT_CAPACITY)),
                    Codec::TimeMillis => TimeMillis(BufferBuilder::new(DEFAULT_CAPACITY)),
                    Codec::TimeMicros => TimeMicros(BufferBuilder::new(DEFAULT_CAPACITY)),
                    Codec::TimestampMillis(utc) => {
                        TimestampMillis(*utc, BufferBuilder::new(DEFAULT_CAPACITY))
                    }
                    Codec::TimestampMicros(utc) => {
                        TimestampMicros(*utc, BufferBuilder::new(DEFAULT_CAPACITY))
                    }
                    Codec::Fixed(sz) => Fixed(*sz, BufferBuilder::new(DEFAULT_CAPACITY)),
                    Codec::Decimal(p, s, size) => {
                        let prec = *p as u8;
                        let scale = s.unwrap_or(0) as i8;
                        match size {
                            Some(sz) if *sz > 16 => {
                                let b = Decimal256Builder::new()
                                    .with_precision_and_scale(prec, scale)?;
                                Decimal256(*p, *s, *size, b)
                            }
                            _ => {
                                let b = Decimal128Builder::new()
                                    .with_precision_and_scale(prec, scale)?;
                                Decimal128(*p, *s, *size, b)
                            }
                        }
                    }
                    Codec::Uuid => Uuid(BufferBuilder::new(DEFAULT_CAPACITY)),
                    Codec::Interval => Duration(IntervalMonthDayNanoBuilder::new()),
                    Codec::List(child) => Array(
                        Arc::new(child.field_with_name("item")),
                        OffsetBufferBuilder::new(DEFAULT_CAPACITY),
                        Box::new(make_child(child)?),
                    ),
                    Codec::Map(val) => {
                        let val_field = val.field_with_name("value").with_nullable(true);
                        let map_field = Arc::new(ArrowField::new(
                            "entries",
                            DataType::Struct(Fields::from(vec![
                                ArrowField::new("key", DataType::Utf8, false),
                                val_field,
                            ])),
                            false,
                        ));
                        Map(
                            map_field,
                            OffsetBufferBuilder::new(DEFAULT_CAPACITY),
                            OffsetBufferBuilder::new(DEFAULT_CAPACITY),
                            BufferBuilder::new(DEFAULT_CAPACITY),
                            Box::new(make_child(val)?),
                        )
                    }
                    Codec::Enum(symbols) => {
                        let mapping = data_type.resolution.as_ref().and_then(|r| match r {
                            ResolutionInfo::EnumMapping(m) => Some(m.clone()),
                            _ => None,
                        });
                        Enum(
                            BufferBuilder::new(DEFAULT_CAPACITY),
                            symbols.clone(),
                            mapping,
                        )
                    }
                    Codec::Struct(fields) => {
                        let mut arrow_fields = Vec::with_capacity(fields.len());
                        let mut field_decoders = Vec::with_capacity(fields.len());
                        let mut field_defaults = Vec::with_capacity(fields.len());
                        for f in fields.iter() {
                            arrow_fields.push(f.field());
                            field_decoders.push(make_child(f.data_type())?);
                            field_defaults.push(f.data_type().resolution.as_ref().and_then(|r| {
                                match r {
                                    ResolutionInfo::DefaultValue(l) => Some(l.clone()),
                                    _ => None,
                                }
                            }));
                        }
                        let (mapping, skip_decoders) = match &data_type.resolution {
                            Some(ResolutionInfo::Record(ResolvedRecord {
                                writer_to_reader,
                                skip_fields,
                                ..
                            })) => {
                                let skips = skip_fields
                                    .iter()
                                    .map(|opt| {
                                        opt.as_ref()
                                            .map(|dt| make_child(dt))
                                            .transpose()
                                            .unwrap_or(None)
                                    })
                                    .collect();
                                (Some(writer_to_reader.clone()), Some(skips))
                            }
                            _ => (None, None),
                        };
                        Record {
                            arrow_fields: arrow_fields.into(),
                            field_decoders,
                            field_defaults,
                            mapping,
                            skip_decoders,
                        }
                    }
                    Codec::Union(child) => make_child(child)?,
                    _ => unreachable!("primitive already matched"),
                },
            },
        };
        Ok(match data_type.nullability() {
            Some(n) => Nullable(n, NullBufferBuilder::new(DEFAULT_CAPACITY), Box::new(base)),
            None => base,
        })
    }

    /// Creates a new [`Decoder`] for the given [`AvroDataType`]
    pub(crate) fn try_new(data_type: &AvroDataType) -> Result<Self, ArrowError> {
        Self::try_new_with_view(data_type, false)
    }

    fn reserve(&mut self, count: usize) {
        use Decoder::*;
        match self {
            Boolean(b) => b.reserve(count),
            Int32(b) | Date32(b) | TimeMillis(b) => b.reserve(count),
            Int64(b)
            | Int32ToInt64(b)
            | TimeMicros(b)
            | TimestampMillis(_, b)
            | TimestampMicros(_, b) => b.reserve(count),
            Float32(b) | Int32ToFloat32(b) | Int64ToFloat32(b) => b.reserve(count),
            Float64(b) | Int32ToFloat64(b) | Int64ToFloat64(b) | Float32ToFloat64(b) => {
                b.reserve(count)
            }
            BytesToString(off, data)
            | StringToBytes(off, data)
            | Binary(off, data)
            | String(off, data)
            | StringView(off, data) => {
                off.reserve(count);
                data.reserve(count * 8);
            }
            Enum(keys, ..) => keys.reserve(count),
            Array(_, off, child) => {
                off.reserve(count);
                child.reserve(count);
            }
            Map(_, key_off, map_off, key_data, val_dec) => {
                key_off.reserve(count);
                map_off.reserve(count);
                key_data.reserve(count * 8);
                val_dec.reserve(count);
            }
            Record { field_decoders, .. } => {
                for d in field_decoders {
                    d.reserve(count);
                }
            }
            Nullable(_, _nb, inner) => inner.reserve(count),
            _ => {}
        }
    }

    /// Decode a single record from the cursor.
    fn decode(&mut self, buf: &mut AvroCursor<'_>) -> Result<(), ArrowError> {
        use Decoder::*;
        match self {
            Null(n) => *n += 1,
            Boolean(b) => b.append(buf.get_bool()?),
            Int32(b) => b.append(buf.get_int()?),
            Int64(b) => b.append(buf.get_long()?),
            Float32(b) => b.append(buf.get_float()?),
            Float64(b) => b.append(buf.get_double()?),
            Int32ToInt64(b) => b.append(buf.get_int()? as i64),
            Int32ToFloat32(b) => b.append(buf.get_int()? as f32),
            Int32ToFloat64(b) => b.append(buf.get_int()? as f64),
            Int64ToFloat32(b) => b.append(buf.get_long()? as f32),
            Int64ToFloat64(b) => b.append(buf.get_long()? as f64),
            Float32ToFloat64(b) => b.append(buf.get_float()? as f64),
            BytesToString(off, data)
            | String(off, data)
            | StringView(off, data)
            | Binary(off, data) => {
                let bytes = buf.get_bytes()?;
                off.push_length(bytes.len());
                data.append_slice(bytes);
            }
            StringToBytes(off, data) => {
                let bytes = buf.get_bytes()?;
                off.push_length(bytes.len());
                data.append_slice(bytes);
            }
            Uuid(data) => {
                let txt = std::str::from_utf8(buf.get_bytes()?)
                    .map_err(|e| ArrowError::ParseError(e.to_string()))?;
                data.append_slice(
                    uuid::Uuid::try_parse(txt)
                        .map_err(|e| ArrowError::ParseError(e.to_string()))?
                        .as_bytes(),
                );
            }
            Date32(b) | TimeMillis(b) => b.append(buf.get_int()?),
            TimeMicros(b) | TimestampMillis(_, b) | TimestampMicros(_, b) => {
                b.append(buf.get_long()?)
            }
            Fixed(len, b) => b.append_slice(buf.get_fixed(*len as usize)?),
            Decimal128(_, _, sz, builder) => {
                let raw = if let Some(s) = sz {
                    buf.get_fixed(*s)?
                } else {
                    buf.get_bytes()?
                };
                builder.append_value(i128::from_be_bytes(sign_extend_to::<16>(raw)?));
            }
            Decimal256(_, _, sz, builder) => {
                let raw = if let Some(s) = sz {
                    buf.get_fixed(*s)?
                } else {
                    buf.get_bytes()?
                };
                builder.append_value(i256::from_be_bytes(sign_extend_to::<32>(raw)?));
            }
            Duration(bldr) => {
                let raw = buf.get_fixed(12)?;
                let months = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as i32;
                let days = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as i32;
                let millis = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as i64;
                bldr.append_value(IntervalMonthDayNano::new(months, days, millis * 1_000_000));
            }
            Enum(keys, _symbols, mapping) => {
                let w = buf.get_int()?;
                let idx = mapping
                    .as_ref()
                    .and_then(|m| m.mapping.get(w as usize).copied())
                    .unwrap_or(w);
                keys.append(idx);
            }
            Array(_, off, item_dec) => {
                let n = read_items(buf, |c| item_dec.decode(c))?;
                off.push_length(n);
            }
            Map(_, key_off, map_off, key_data, val_dec) => {
                let n = read_items(buf, |c| {
                    let k = c.get_bytes()?;
                    key_off.push_length(k.len());
                    key_data.append_slice(k);
                    val_dec.decode(c)
                })?;
                map_off.push_length(n);
            }
            Record {
                mapping,
                skip_decoders,
                field_decoders,
                field_defaults,
                ..
            } => {
                if let Some(map) = mapping {
                    let mut present = vec![false; field_decoders.len()];
                    for (w_idx, rdr_idx) in map.iter().enumerate() {
                        match rdr_idx {
                            Some(pos) => {
                                present[*pos] = true;
                                field_decoders[*pos].decode(buf)?
                            }
                            None => {
                                if let Some(Some(dec)) =
                                    skip_decoders.as_mut().and_then(|v| v.get_mut(w_idx))
                                {
                                    dec.decode(buf)?
                                } else {
                                    skip_value(buf)?
                                }
                            }
                        }
                    }
                    // Append default literals for fields not seen
                    for (idx, default) in field_defaults.iter().enumerate() {
                        if !present[idx] {
                            if let Some(lit) = default {
                                field_decoders[idx].append_literal(lit);
                            }
                        }
                    }
                } else {
                    for d in field_decoders {
                        d.decode(buf)?;
                    }
                }
            }
            Nullable(order, nulls, inner) => {
                let branch = buf.read_vlq()?;
                let present = match order {
                    Nullability::NullFirst => branch != 0,
                    Nullability::NullSecond => branch == 0,
                };
                nulls.append(present);
                if present {
                    inner.decode(buf)?;
                } else {
                    inner.append_null();
                }
            }
        }
        Ok(())
    }
    /// Appends a null value to the decoder.
    fn append_null(&mut self) {
        use Decoder::*;
        match self {
            Null(n) => *n += 1,
            Boolean(b) => b.append(false),
            Int32(b) | Date32(b) | TimeMillis(b) => b.append(0),
            Int64(b)
            | Int32ToInt64(b)
            | TimeMicros(b)
            | TimestampMillis(_, b)
            | TimestampMicros(_, b) => b.append(0),
            Float32(b) | Int32ToFloat32(b) | Int64ToFloat32(b) => b.append(0.0),
            Float64(b) | Int32ToFloat64(b) | Int64ToFloat64(b) | Float32ToFloat64(b) => {
                b.append(0.0)
            }
            Binary(off, _)
            | String(off, _)
            | StringView(off, _)
            | BytesToString(off, _)
            | StringToBytes(off, _) => off.push_length(0),
            Uuid(b) => b.append_slice(&[0u8; 16]),
            Fixed(len, b) => b.append_slice(&vec![0u8; *len as usize]),
            Decimal128(.., bldr) => bldr.append_null(),
            Decimal256(.., bldr) => bldr.append_null(),
            Duration(bldr) => bldr.append_null(),
            Enum(keys, _, mapping) => {
                keys.append(mapping.as_ref().map(|m| m.default_index).unwrap_or(0));
            }
            Array(_, off, _) => off.push_length(0),
            Map(_, _, moff, _, _) => moff.push_length(0),
            Record { field_decoders, .. } => {
                for f in field_decoders {
                    f.append_null();
                }
            }
            Nullable(_, nb, inner) => {
                nb.append(false);
                inner.append_null();
            }
        }
    }

    /// Appends a literal value to the decoder.
    fn append_literal(&mut self, lit: &AvroLiteral) {
        use AvroLiteral::*;
        match lit {
            Null | Unsupported => self.append_null(),
            Boolean(bv) => match self {
                Decoder::Boolean(buf) => buf.append(*bv),
                _ => self.append_null(),
            },
            Int(i) => match self {
                Decoder::Int32(b) => b.append(*i),
                Decoder::Int32ToInt64(b) => b.append(*i as i64),
                Decoder::Int32ToFloat32(b) => b.append(*i as f32),
                Decoder::Int32ToFloat64(b) => b.append(*i as f64),
                _ => self.append_null(),
            },
            Long(l) => match self {
                Decoder::Int64(b) => b.append(*l),
                Decoder::Int64ToFloat32(b) => b.append(*l as f32),
                Decoder::Int64ToFloat64(b) => b.append(*l as f64),
                _ => self.append_null(),
            },
            Float(fv) => match self {
                Decoder::Float32(b) => b.append(*fv),
                Decoder::Float32ToFloat64(b) => b.append(*fv as f64),
                _ => self.append_null(),
            },
            Double(dv) => match self {
                Decoder::Float64(b) => b.append(*dv),
                _ => self.append_null(),
            },
            Bytes(bytes) => match self {
                Decoder::Binary(off, data) | Decoder::StringToBytes(off, data) => {
                    off.push_length(bytes.len());
                    data.append_slice(bytes);
                }
                _ => self.append_null(),
            },
            String(s) => match self {
                Decoder::String(off, data)
                | Decoder::StringView(off, data)
                | Decoder::BytesToString(off, data) => {
                    off.push_length(s.len());
                    data.append_slice(s.as_bytes());
                }
                _ => self.append_null(),
            },
            Enum(sym) => {
                if let Decoder::Enum(keys, symbols, _) = self {
                    let idx = symbols.iter().position(|x| x == sym).unwrap_or_default() as i32;
                    keys.append(idx);
                } else {
                    self.append_null()
                }
            }
        }
    }

    /// Flushes the decoded data into an [`ArrayRef`].
    fn flush(&mut self, nulls: Option<NullBuffer>) -> Result<ArrayRef, ArrowError> {
        use Decoder::*;
        Ok(match self {
            Int32ToInt64(b) => Arc::new(PrimitiveArray::<Int64Type>::new(b.finish().into(), nulls)),
            Int32ToFloat32(b) | Int64ToFloat32(b) => {
                Arc::new(PrimitiveArray::<Float32Type>::new(b.finish().into(), nulls))
            }
            Int32ToFloat64(b) | Int64ToFloat64(b) | Float32ToFloat64(b) => {
                Arc::new(PrimitiveArray::<Float64Type>::new(b.finish().into(), nulls))
            }
            BytesToString(off, data) | String(off, data) => Arc::new(StringArray::new(
                flush_offsets(off),
                flush_bytes(data),
                nulls,
            )),
            StringToBytes(off, data) | Binary(off, data) => Arc::new(BinaryArray::new(
                flush_offsets(off),
                flush_bytes(data),
                nulls,
            )),
            StringView(off, data) => {
                let offsets = flush_offsets(off);
                let bytes = flush_bytes(data);
                let base = StringArray::new(offsets.clone(), bytes.clone(), None);
                let vals: Vec<&str> = (0..base.len())
                    .map(|i| if base.is_valid(i) { base.value(i) } else { "" })
                    .collect();
                Arc::new(StringViewArray::from(vals))
            }
            Null(n) => Arc::new(NullArray::new(*n)),
            Boolean(b) => Arc::new(BooleanArray::new(b.finish(), nulls)),
            Int32(b) => Arc::new(PrimitiveArray::<Int32Type>::new(b.finish().into(), nulls)),
            Int64(b) => Arc::new(PrimitiveArray::<Int64Type>::new(b.finish().into(), nulls)),
            Float32(b) => Arc::new(PrimitiveArray::<Float32Type>::new(b.finish().into(), nulls)),
            Float64(b) => Arc::new(PrimitiveArray::<Float64Type>::new(b.finish().into(), nulls)),
            Date32(b) => Arc::new(PrimitiveArray::<Date32Type>::new(b.finish().into(), nulls)),
            TimeMillis(b) => Arc::new(PrimitiveArray::<Time32MillisecondType>::new(
                b.finish().into(),
                nulls,
            )),
            TimeMicros(b) => Arc::new(PrimitiveArray::<Time64MicrosecondType>::new(
                b.finish().into(),
                nulls,
            )),
            TimestampMillis(utc, b) => Arc::new(
                PrimitiveArray::<TimestampMillisecondType>::new(b.finish().into(), nulls)
                    .with_timezone_opt(utc.then(|| "+00:00")),
            ),
            TimestampMicros(utc, b) => Arc::new(
                PrimitiveArray::<TimestampMicrosecondType>::new(b.finish().into(), nulls)
                    .with_timezone_opt(utc.then(|| "+00:00")),
            ),
            Fixed(len, b) => Arc::new(FixedSizeBinaryArray::try_new(*len, b.finish(), nulls)?),
            Uuid(b) => Arc::new(FixedSizeBinaryArray::try_new(16, b.finish(), nulls)?),
            Decimal128(p, s, _, builder) => {
                let (_, vals, _) = builder.finish().into_parts();
                Arc::new(
                    Decimal128Array::new(vals, nulls)
                        .with_precision_and_scale(*p as u8, s.unwrap_or(0) as i8)?,
                )
            }
            Decimal256(p, s, _, builder) => {
                let (_, vals, _) = builder.finish().into_parts();
                Arc::new(
                    Decimal256Array::new(vals, nulls)
                        .with_precision_and_scale(*p as u8, s.unwrap_or(0) as i8)?,
                )
            }
            Duration(builder) => {
                let (_, vals, _) = builder.finish().into_parts();
                Arc::new(IntervalMonthDayNanoArray::try_new(vals, nulls)?)
            }
            Enum(keys, symbols, _) => {
                let dict_keys = PrimitiveArray::<Int32Type>::new(keys.finish().into(), nulls);
                let dict_vals = Arc::new(StringArray::from_iter_values(
                    symbols.iter().map(|s| s.as_str()),
                ));
                Arc::new(DictionaryArray::try_new(dict_keys, dict_vals)?)
            }
            Array(field, off, child) => {
                let values = child.flush(None)?;
                Arc::new(ListArray::new(
                    field.clone(),
                    flush_offsets(off),
                    values,
                    nulls,
                ))
            }
            Map(field, key_off, map_off, key_data, val_dec) => {
                let map_offsets = flush_offsets(map_off);
                let key_offsets = flush_offsets(key_off);
                let key_buffer = flush_bytes(key_data);
                let key_arr = StringArray::new(key_offsets, key_buffer, None);
                let val_arr = val_dec.flush(None)?;
                let entries = StructArray::new(
                    Fields::from(vec![
                        Arc::new(ArrowField::new("key", DataType::Utf8, false)),
                        Arc::new(ArrowField::new("value", val_arr.data_type().clone(), true)),
                    ]),
                    vec![Arc::new(key_arr), val_arr],
                    None,
                );
                Arc::new(MapArray::new(
                    field.clone(),
                    map_offsets,
                    entries,
                    nulls,
                    false,
                ))
            }
            Record {
                arrow_fields,
                field_decoders,
                ..
            } => {
                let arrays = field_decoders
                    .iter_mut()
                    .map(|d| d.flush(None))
                    .collect::<Result<Vec<_>, _>>()?;
                Arc::new(StructArray::new(arrow_fields.clone(), arrays, nulls))
            }
            Nullable(_, nb, inner) => inner.flush(nb.finish())?,
        })
    }
}
/// Skips a value by reading and discarding a byte slice.
fn skip_value(buf: &mut AvroCursor<'_>) -> Result<(), ArrowError> {
    let _ = buf.get_bytes()?;
    Ok(())
}
/// Reads blocks of items from the cursor.
fn read_items(
    buf: &mut AvroCursor,
    mut decode: impl FnMut(&mut AvroCursor) -> Result<(), ArrowError>,
) -> Result<usize, ArrowError> {
    let mut total = 0;
    loop {
        let cnt = buf.get_long()?;
        match cnt.cmp(&0) {
            Ordering::Equal => break,
            Ordering::Less => {
                let n = (-cnt) as usize;
                let _block_bytes = buf.get_long()?;
                for _ in 0..n {
                    decode(buf)?;
                }
                total += n;
            }
            Ordering::Greater => {
                let n = cnt as usize;
                for _ in 0..n {
                    decode(buf)?;
                }
                total += n;
            }
        }
    }
    Ok(total)
}

/// Swap-and-finish for variable-width data buffers
fn flush_bytes(b: &mut BufferBuilder<u8>) -> Buffer {
    let mut old = std::mem::replace(b, BufferBuilder::<u8>::new(DEFAULT_CAPACITY));
    old.finish()
}

/// Swap-and-finish for offset buffers
fn flush_offsets(b: &mut OffsetBufferBuilder<i32>) -> OffsetBuffer<i32> {
    let mut old = std::mem::replace(b, OffsetBufferBuilder::<i32>::new(DEFAULT_CAPACITY));
    old.finish()
}

/// Sign-extend an integer to `N` bytes
#[inline]
fn sign_extend_to<const N: usize>(raw: &[u8]) -> Result<[u8; N], ArrowError> {
    if raw.len() > N {
        return Err(ArrowError::ParseError(format!(
            "Cannot extend {} bytes to {N}-byte integer",
            raw.len()
        )));
    }
    let mut out = [0u8; N];
    let pad = if raw.first().map_or(false, |b| b & 0x80 != 0) {
        0xFF
    } else {
        0x00
    };
    out[..N - raw.len()].fill(pad);
    out[N - raw.len()..].copy_from_slice(raw);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::AvroField;
    use crate::schema::{ComplexType, Enum, Field, PrimitiveType, Record, Schema, TypeName};
    use arrow_array::{
        cast::AsArray, Array, Decimal128Array, DictionaryArray, FixedSizeBinaryArray,
        IntervalMonthDayNanoArray, ListArray, MapArray, StringArray, StructArray,
    };
    use serde_json::json;

    fn encode_avro_int(value: i32) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut v = (value << 1) ^ (value >> 31);
        while v & !0x7F != 0 {
            buf.push(((v & 0x7F) | 0x80) as u8);
            v >>= 7;
        }
        buf.push(v as u8);
        buf
    }

    fn encode_avro_long(value: i64) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut v = (value << 1) ^ (value >> 63);
        while v & !0x7F != 0 {
            buf.push(((v & 0x7F) | 0x80) as u8);
            v >>= 7;
        }
        buf.push(v as u8);
        buf
    }

    fn encode_avro_bytes(bytes: &[u8]) -> Vec<u8> {
        let mut buf = encode_avro_long(bytes.len() as i64);
        buf.extend_from_slice(bytes);
        buf
    }

    fn avro_from_codec(codec: Codec) -> AvroDataType {
        AvroDataType::new(codec, Default::default(), None)
    }

    #[test]
    fn test_map_decoding_one_entry() {
        let value_type = avro_from_codec(Codec::Utf8);
        let map_type = avro_from_codec(Codec::Map(Arc::new(value_type)));
        let mut decoder = Decoder::try_new(&map_type).unwrap();
        // Encode a single map with one entry: {"hello": "world"}
        let mut data = Vec::new();
        data.extend_from_slice(&encode_avro_long(1));
        data.extend_from_slice(&encode_avro_bytes(b"hello")); // key
        data.extend_from_slice(&encode_avro_bytes(b"world")); // value
        data.extend_from_slice(&encode_avro_long(0));
        let mut cursor = AvroCursor::new(&data);
        decoder.decode(&mut cursor).unwrap();
        let array = decoder.flush(None).unwrap();
        let map_arr = array.as_any().downcast_ref::<MapArray>().unwrap();
        assert_eq!(map_arr.len(), 1); // one map
        assert_eq!(map_arr.value_length(0), 1);
        let entries = map_arr.value(0);
        let struct_entries = entries.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(struct_entries.len(), 1);
        let key_arr = struct_entries
            .column_by_name("key")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let val_arr = struct_entries
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(key_arr.value(0), "hello");
        assert_eq!(val_arr.value(0), "world");
    }

    #[test]
    fn test_map_decoding_empty() {
        let value_type = avro_from_codec(Codec::Utf8);
        let map_type = avro_from_codec(Codec::Map(Arc::new(value_type)));
        let mut decoder = Decoder::try_new(&map_type).unwrap();
        let data = encode_avro_long(0);
        decoder.decode(&mut AvroCursor::new(&data)).unwrap();
        let array = decoder.flush(None).unwrap();
        let map_arr = array.as_any().downcast_ref::<MapArray>().unwrap();
        assert_eq!(map_arr.len(), 1);
        assert_eq!(map_arr.value_length(0), 0);
    }

    #[test]
    fn test_fixed_decoding() {
        let avro_type = avro_from_codec(Codec::Fixed(3));
        let mut decoder = Decoder::try_new(&avro_type).expect("Failed to create decoder");

        let data1 = [1u8, 2, 3];
        let mut cursor1 = AvroCursor::new(&data1);
        decoder
            .decode(&mut cursor1)
            .expect("Failed to decode data1");
        assert_eq!(cursor1.position(), 3, "Cursor should advance by fixed size");
        let data2 = [4u8, 5, 6];
        let mut cursor2 = AvroCursor::new(&data2);
        decoder
            .decode(&mut cursor2)
            .expect("Failed to decode data2");
        assert_eq!(cursor2.position(), 3, "Cursor should advance by fixed size");
        let array = decoder.flush(None).expect("Failed to flush decoder");
        assert_eq!(array.len(), 2, "Array should contain two items");
        let fixed_size_binary_array = array
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .expect("Failed to downcast to FixedSizeBinaryArray");
        assert_eq!(
            fixed_size_binary_array.value_length(),
            3,
            "Fixed size of binary values should be 3"
        );
        assert_eq!(
            fixed_size_binary_array.value(0),
            &[1, 2, 3],
            "First item mismatch"
        );
        assert_eq!(
            fixed_size_binary_array.value(1),
            &[4, 5, 6],
            "Second item mismatch"
        );
    }

    #[test]
    fn test_fixed_decoding_empty() {
        let avro_type = avro_from_codec(Codec::Fixed(5));
        let mut decoder = Decoder::try_new(&avro_type).expect("Failed to create decoder");

        let array = decoder
            .flush(None)
            .expect("Failed to flush decoder for empty input");

        assert_eq!(array.len(), 0, "Array should be empty");
        let fixed_size_binary_array = array
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .expect("Failed to downcast to FixedSizeBinaryArray for empty array");

        assert_eq!(
            fixed_size_binary_array.value_length(),
            5,
            "Fixed size of binary values should be 5 as per type"
        );
    }

    #[test]
    fn test_uuid_decoding() {
        let avro_type = avro_from_codec(Codec::Uuid);
        let mut decoder = Decoder::try_new(&avro_type).expect("Failed to create decoder");
        let uuid_str = "f81d4fae-7dec-11d0-a765-00a0c91e6bf6";
        let data = encode_avro_bytes(uuid_str.as_bytes());
        let mut cursor = AvroCursor::new(&data);
        decoder.decode(&mut cursor).expect("Failed to decode data");
        assert_eq!(
            cursor.position(),
            data.len(),
            "Cursor should advance by varint size + data size"
        );
        let array = decoder.flush(None).expect("Failed to flush decoder");
        let fixed_size_binary_array = array
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .expect("Array should be a FixedSizeBinaryArray");
        assert_eq!(fixed_size_binary_array.len(), 1);
        assert_eq!(fixed_size_binary_array.value_length(), 16);
        let expected_bytes = [
            0xf8, 0x1d, 0x4f, 0xae, 0x7d, 0xec, 0x11, 0xd0, 0xa7, 0x65, 0x00, 0xa0, 0xc9, 0x1e,
            0x6b, 0xf6,
        ];
        assert_eq!(fixed_size_binary_array.value(0), &expected_bytes);
    }

    #[test]
    fn test_array_decoding() {
        let item_dt = avro_from_codec(Codec::Int32);
        let list_dt = avro_from_codec(Codec::List(Arc::new(item_dt)));
        let mut decoder = Decoder::try_new(&list_dt).unwrap();
        let mut row1 = Vec::new();
        row1.extend_from_slice(&encode_avro_long(2));
        row1.extend_from_slice(&encode_avro_int(10));
        row1.extend_from_slice(&encode_avro_int(20));
        row1.extend_from_slice(&encode_avro_long(0));
        let row2 = encode_avro_long(0);
        let mut cursor = AvroCursor::new(&row1);
        decoder.decode(&mut cursor).unwrap();
        let mut cursor2 = AvroCursor::new(&row2);
        decoder.decode(&mut cursor2).unwrap();
        let array = decoder.flush(None).unwrap();
        let list_arr = array.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(list_arr.len(), 2);
        let offsets = list_arr.value_offsets();
        assert_eq!(offsets, &[0, 2, 2]);
        let values = list_arr.values();
        let int_arr = values.as_primitive::<Int32Type>();
        assert_eq!(int_arr.len(), 2);
        assert_eq!(int_arr.value(0), 10);
        assert_eq!(int_arr.value(1), 20);
    }

    #[test]
    fn test_array_decoding_with_negative_block_count() {
        let item_dt = avro_from_codec(Codec::Int32);
        let list_dt = avro_from_codec(Codec::List(Arc::new(item_dt)));
        let mut decoder = Decoder::try_new(&list_dt).unwrap();
        let mut data = encode_avro_long(-3);
        data.extend_from_slice(&encode_avro_long(12));
        data.extend_from_slice(&encode_avro_int(1));
        data.extend_from_slice(&encode_avro_int(2));
        data.extend_from_slice(&encode_avro_int(3));
        data.extend_from_slice(&encode_avro_long(0));
        let mut cursor = AvroCursor::new(&data);
        decoder.decode(&mut cursor).unwrap();
        let array = decoder.flush(None).unwrap();
        let list_arr = array.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(list_arr.len(), 1);
        assert_eq!(list_arr.value_length(0), 3);
        let values = list_arr.values().as_primitive::<Int32Type>();
        assert_eq!(values.len(), 3);
        assert_eq!(values.value(0), 1);
        assert_eq!(values.value(1), 2);
        assert_eq!(values.value(2), 3);
    }

    #[test]
    fn test_nested_array_decoding() {
        let inner_ty = avro_from_codec(Codec::List(Arc::new(avro_from_codec(Codec::Int32))));
        let nested_ty = avro_from_codec(Codec::List(Arc::new(inner_ty.clone())));
        let mut decoder = Decoder::try_new(&nested_ty).unwrap();
        let mut buf = Vec::new();
        buf.extend(encode_avro_long(1));
        buf.extend(encode_avro_long(2));
        buf.extend(encode_avro_int(5));
        buf.extend(encode_avro_int(6));
        buf.extend(encode_avro_long(0));
        buf.extend(encode_avro_long(0));
        let mut cursor = AvroCursor::new(&buf);
        decoder.decode(&mut cursor).unwrap();
        let arr = decoder.flush(None).unwrap();
        let outer = arr.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(outer.len(), 1);
        assert_eq!(outer.value_length(0), 1);
        let inner = outer.values().as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(inner.len(), 1);
        assert_eq!(inner.value_length(0), 2);
        let values = inner
            .values()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(values.values(), &[5, 6]);
    }

    #[test]
    fn test_array_decoding_empty_array() {
        let value_type = avro_from_codec(Codec::Utf8);
        let map_type = avro_from_codec(Codec::List(Arc::new(value_type)));
        let mut decoder = Decoder::try_new(&map_type).unwrap();
        let data = encode_avro_long(0);
        decoder.decode(&mut AvroCursor::new(&data)).unwrap();
        let array = decoder.flush(None).unwrap();
        let list_arr = array.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(list_arr.len(), 1);
        assert_eq!(list_arr.value_length(0), 0);
    }

    #[test]
    fn test_decimal_decoding_fixed256() {
        let dt = avro_from_codec(Codec::Decimal(5, Some(2), Some(32)));
        let mut decoder = Decoder::try_new(&dt).unwrap();
        let row1 = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x30, 0x39,
        ];
        let row2 = [
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0x85,
        ];
        let mut data = Vec::new();
        data.extend_from_slice(&row1);
        data.extend_from_slice(&row2);
        let mut cursor = AvroCursor::new(&data);
        decoder.decode(&mut cursor).unwrap();
        decoder.decode(&mut cursor).unwrap();
        let arr = decoder.flush(None).unwrap();
        let dec = arr.as_any().downcast_ref::<Decimal256Array>().unwrap();
        assert_eq!(dec.len(), 2);
        assert_eq!(dec.value_as_string(0), "123.45");
        assert_eq!(dec.value_as_string(1), "-1.23");
    }

    #[test]
    fn test_decimal_decoding_fixed128() {
        let dt = avro_from_codec(Codec::Decimal(5, Some(2), Some(16)));
        let mut decoder = Decoder::try_new(&dt).unwrap();
        let row1 = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x30, 0x39,
        ];
        let row2 = [
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0x85,
        ];
        let mut data = Vec::new();
        data.extend_from_slice(&row1);
        data.extend_from_slice(&row2);
        let mut cursor = AvroCursor::new(&data);
        decoder.decode(&mut cursor).unwrap();
        decoder.decode(&mut cursor).unwrap();
        let arr = decoder.flush(None).unwrap();
        let dec = arr.as_any().downcast_ref::<Decimal128Array>().unwrap();
        assert_eq!(dec.len(), 2);
        assert_eq!(dec.value_as_string(0), "123.45");
        assert_eq!(dec.value_as_string(1), "-1.23");
    }

    #[test]
    fn test_decimal_decoding_bytes_with_nulls() {
        let dt = avro_from_codec(Codec::Decimal(4, Some(1), None));
        let inner = Decoder::try_new(&dt).unwrap();
        let mut decoder = Decoder::Nullable(
            Nullability::NullSecond,
            NullBufferBuilder::new(DEFAULT_CAPACITY),
            Box::new(inner),
        );
        let mut data = Vec::new();
        data.extend_from_slice(&encode_avro_int(0));
        data.extend_from_slice(&encode_avro_bytes(&[0x04, 0xD2]));
        data.extend_from_slice(&encode_avro_int(1));
        data.extend_from_slice(&encode_avro_int(0));
        data.extend_from_slice(&encode_avro_bytes(&[0xFB, 0x2E]));
        let mut cursor = AvroCursor::new(&data);
        decoder.decode(&mut cursor).unwrap(); // row1
        decoder.decode(&mut cursor).unwrap(); // row2
        decoder.decode(&mut cursor).unwrap(); // row3
        let arr = decoder.flush(None).unwrap();
        let dec_arr = arr.as_any().downcast_ref::<Decimal128Array>().unwrap();
        assert_eq!(dec_arr.len(), 3);
        assert!(dec_arr.is_valid(0));
        assert!(!dec_arr.is_valid(1));
        assert!(dec_arr.is_valid(2));
        assert_eq!(dec_arr.value_as_string(0), "123.4");
        assert_eq!(dec_arr.value_as_string(2), "-123.4");
    }

    #[test]
    fn test_decimal_decoding_bytes_with_nulls_fixed_size() {
        let dt = avro_from_codec(Codec::Decimal(6, Some(2), Some(16)));
        let inner = Decoder::try_new(&dt).unwrap();
        let mut decoder = Decoder::Nullable(
            Nullability::NullSecond,
            NullBufferBuilder::new(DEFAULT_CAPACITY),
            Box::new(inner),
        );
        let row1 = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0xE2, 0x40,
        ];
        let row3 = [
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
            0x1D, 0xC0,
        ];
        let mut data = Vec::new();
        data.extend_from_slice(&encode_avro_int(0));
        data.extend_from_slice(&row1);
        data.extend_from_slice(&encode_avro_int(1));
        data.extend_from_slice(&encode_avro_int(0));
        data.extend_from_slice(&row3);
        let mut cursor = AvroCursor::new(&data);
        decoder.decode(&mut cursor).unwrap();
        decoder.decode(&mut cursor).unwrap();
        decoder.decode(&mut cursor).unwrap();
        let arr = decoder.flush(None).unwrap();
        let dec_arr = arr.as_any().downcast_ref::<Decimal128Array>().unwrap();
        assert_eq!(dec_arr.len(), 3);
        assert!(dec_arr.is_valid(0));
        assert!(!dec_arr.is_valid(1));
        assert!(dec_arr.is_valid(2));
        assert_eq!(dec_arr.value_as_string(0), "1234.56");
        assert_eq!(dec_arr.value_as_string(2), "-1234.56");
    }

    #[test]
    fn test_enum_decoding() {
        let symbols: Arc<[String]> = vec!["A", "B", "C"].into_iter().map(String::from).collect();
        let avro_type = avro_from_codec(Codec::Enum(symbols.clone()));
        let mut decoder = Decoder::try_new(&avro_type).unwrap();
        let mut data = Vec::new();
        data.extend_from_slice(&encode_avro_int(2));
        data.extend_from_slice(&encode_avro_int(0));
        data.extend_from_slice(&encode_avro_int(1));
        let mut cursor = AvroCursor::new(&data);
        decoder.decode(&mut cursor).unwrap();
        decoder.decode(&mut cursor).unwrap();
        decoder.decode(&mut cursor).unwrap();
        let array = decoder.flush(None).unwrap();
        let dict_array = array
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .unwrap();

        assert_eq!(dict_array.len(), 3);
        let values = dict_array
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(values.value(0), "A");
        assert_eq!(values.value(1), "B");
        assert_eq!(values.value(2), "C");
        assert_eq!(dict_array.keys().values(), &[2, 0, 1]);
    }

    #[test]
    fn test_enum_decoding_with_nulls() {
        let symbols: Arc<[String]> = vec!["X", "Y"].into_iter().map(String::from).collect();
        let enum_codec = Codec::Enum(symbols.clone());
        let avro_type =
            AvroDataType::new(enum_codec, Default::default(), Some(Nullability::NullFirst));
        let mut decoder = Decoder::try_new(&avro_type).unwrap();
        let mut data = Vec::new();
        data.extend_from_slice(&encode_avro_long(1));
        data.extend_from_slice(&encode_avro_int(1));
        data.extend_from_slice(&encode_avro_long(0));
        data.extend_from_slice(&encode_avro_long(1));
        data.extend_from_slice(&encode_avro_int(0));
        let mut cursor = AvroCursor::new(&data);
        decoder.decode(&mut cursor).unwrap();
        decoder.decode(&mut cursor).unwrap();
        decoder.decode(&mut cursor).unwrap();
        let array = decoder.flush(None).unwrap();
        let dict_array = array
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .unwrap();
        assert_eq!(dict_array.len(), 3);
        assert!(dict_array.is_valid(0));
        assert!(dict_array.is_null(1));
        assert!(dict_array.is_valid(2));
        let expected_keys = Int32Array::from(vec![Some(1), None, Some(0)]);
        assert_eq!(dict_array.keys(), &expected_keys);
        let values = dict_array
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(values.value(0), "X");
        assert_eq!(values.value(1), "Y");
    }

    #[test]
    fn test_duration_decoding_with_nulls() {
        let duration_codec = Codec::Interval;
        let avro_type = AvroDataType::new(
            duration_codec,
            Default::default(),
            Some(Nullability::NullFirst),
        );
        let mut decoder = Decoder::try_new(&avro_type).unwrap();
        let mut data = Vec::new();
        // First value: 1 month, 2 days, 3 millis
        data.extend_from_slice(&encode_avro_long(1)); // not null
        let mut duration1 = Vec::new();
        duration1.extend_from_slice(&1u32.to_le_bytes());
        duration1.extend_from_slice(&2u32.to_le_bytes());
        duration1.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&duration1);
        // Second value: null
        data.extend_from_slice(&encode_avro_long(0)); // null
        data.extend_from_slice(&encode_avro_long(1)); // not null
        let mut duration2 = Vec::new();
        duration2.extend_from_slice(&4u32.to_le_bytes());
        duration2.extend_from_slice(&5u32.to_le_bytes());
        duration2.extend_from_slice(&6u32.to_le_bytes());
        data.extend_from_slice(&duration2);
        let mut cursor = AvroCursor::new(&data);
        decoder.decode(&mut cursor).unwrap();
        decoder.decode(&mut cursor).unwrap();
        decoder.decode(&mut cursor).unwrap();
        let array = decoder.flush(None).unwrap();
        let interval_array = array
            .as_any()
            .downcast_ref::<IntervalMonthDayNanoArray>()
            .unwrap();
        assert_eq!(interval_array.len(), 3);
        assert!(interval_array.is_valid(0));
        assert!(interval_array.is_null(1));
        assert!(interval_array.is_valid(2));
        let expected = IntervalMonthDayNanoArray::from(vec![
            Some(IntervalMonthDayNano {
                months: 1,
                days: 2,
                nanoseconds: 3_000_000,
            }),
            None,
            Some(IntervalMonthDayNano {
                months: 4,
                days: 5,
                nanoseconds: 6_000_000,
            }),
        ]);
        assert_eq!(interval_array, &expected);
    }

    #[test]
    fn test_duration_decoding_empty() {
        let duration_codec = Codec::Interval;
        let avro_type = AvroDataType::new(duration_codec, Default::default(), None);
        let mut decoder = Decoder::try_new(&avro_type).unwrap();
        let array = decoder.flush(None).unwrap();
        assert_eq!(array.len(), 0);
    }

    #[test]
    fn test_schema_resolution_promotion_int_to_long() {
        let writer_schema = Schema::Complex(ComplexType::Record(Record {
            name: "R",
            namespace: None,
            doc: None,
            aliases: vec![],
            fields: vec![Field {
                name: "x",
                doc: None,
                r#type: Schema::TypeName(TypeName::Primitive(PrimitiveType::Int)),
                default: None,
            }],
            attributes: Default::default(),
        }));
        let reader_schema = Schema::Complex(ComplexType::Record(Record {
            name: "R",
            namespace: None,
            doc: None,
            aliases: vec![],
            fields: vec![Field {
                name: "x",
                doc: None,
                r#type: Schema::TypeName(TypeName::Primitive(PrimitiveType::Long)),
                default: None,
            }],
            attributes: Default::default(),
        }));
        let field =
            AvroField::resolve_from_writer_and_reader(&writer_schema, &reader_schema, false, false)
                .expect("resolution failed");
        let mut decoder = Decoder::try_new(field.data_type()).unwrap();
        let data = encode_avro_int(42);
        decoder.decode(&mut AvroCursor::new(&data)).unwrap();
        let array = decoder.flush(None).unwrap();
        let s = array.as_any().downcast_ref::<StructArray>().unwrap();
        let col = s.column(0).as_primitive::<Int64Type>();
        assert_eq!(col.value(0), 42);
    }

    #[test]
    fn test_schema_resolution_enum_mapping() {
        let writer_enum = Schema::Complex(ComplexType::Enum(Enum {
            name: "E",
            namespace: None,
            doc: None,
            aliases: vec![],
            symbols: vec!["A", "B", "C"],
            default: None,
            attributes: Default::default(),
        }));
        let reader_enum = Schema::Complex(ComplexType::Enum(Enum {
            name: "E",
            namespace: None,
            doc: None,
            aliases: vec![],
            symbols: vec!["C", "B", "A"],
            default: None,
            attributes: Default::default(),
        }));
        let field =
            AvroField::resolve_from_writer_and_reader(&writer_enum, &reader_enum, false, false)
                .expect("enum resolution failed");
        let mut decoder = Decoder::try_new(field.data_type()).unwrap();
        let mut buf = Vec::new();
        buf.extend(encode_avro_int(0));
        buf.extend(encode_avro_int(1));
        buf.extend(encode_avro_int(2));
        let mut cur = AvroCursor::new(&buf);
        decoder.decode(&mut cur).unwrap();
        decoder.decode(&mut cur).unwrap();
        decoder.decode(&mut cur).unwrap();
        let array = decoder.flush(None).unwrap();
        let dict = array
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .unwrap();
        assert_eq!(dict.keys().values(), &[2, 1, 0]);
        let values = dict
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(values.value(0), "C");
        assert_eq!(values.value(1), "B");
        assert_eq!(values.value(2), "A");
    }

    #[test]
    fn test_schema_resolution_nullable_union_order_change() {
        let writer_union = Schema::Union(vec![
            Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)),
            Schema::TypeName(TypeName::Primitive(PrimitiveType::String)),
        ]);
        let reader_union = Schema::Union(vec![
            Schema::TypeName(TypeName::Primitive(PrimitiveType::String)),
            Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)),
        ]);
        let field =
            AvroField::resolve_from_writer_and_reader(&writer_union, &reader_union, false, false)
                .expect("union resolution failed");
        let mut decoder = Decoder::try_new(field.data_type()).unwrap();
        let mut buf = Vec::new();
        buf.extend(encode_avro_int(1));
        buf.extend(encode_avro_bytes(b"hello"));
        buf.extend(encode_avro_int(0));
        let mut cur = AvroCursor::new(&buf);
        decoder.decode(&mut cur).unwrap();
        decoder.decode(&mut cur).unwrap();
        let array = decoder.flush(None).unwrap();
        let sa = array.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(sa.len(), 2);
        assert_eq!(sa.value(0), "hello");
        assert!(sa.is_null(1));
    }

    #[test]
    fn test_schema_resolution_default_value_int() {
        let writer_schema = Schema::Complex(ComplexType::Record(Record {
            name: "R",
            namespace: None,
            doc: None,
            aliases: vec![],
            fields: vec![Field {
                name: "x",
                doc: None,
                r#type: Schema::TypeName(TypeName::Primitive(PrimitiveType::Int)),
                default: None,
            }],
            attributes: Default::default(),
        }));
        let reader_schema = Schema::Complex(ComplexType::Record(Record {
            name: "R",
            namespace: None,
            doc: None,
            aliases: vec![],
            fields: vec![
                Field {
                    name: "x",
                    doc: None,
                    r#type: Schema::TypeName(TypeName::Primitive(PrimitiveType::Int)),
                    default: None,
                },
                Field {
                    name: "y",
                    doc: None,
                    r#type: Schema::TypeName(TypeName::Primitive(PrimitiveType::Int)),
                    default: Some(json!(9)),
                },
            ],
            attributes: Default::default(),
        }));

        let field =
            AvroField::resolve_from_writer_and_reader(&writer_schema, &reader_schema, false, false)
                .expect("resolution failed");
        let mut decoder = Decoder::try_new(field.data_type()).unwrap();
        let data = encode_avro_int(5);
        decoder.decode(&mut AvroCursor::new(&data)).unwrap();
        let array = decoder.flush(None).unwrap();
        let struct_arr = array.as_any().downcast_ref::<StructArray>().unwrap();
        let x_vals = struct_arr
            .column_by_name("x")
            .unwrap()
            .as_primitive::<Int32Type>();
        let y_vals = struct_arr
            .column_by_name("y")
            .unwrap()
            .as_primitive::<Int32Type>();
        assert_eq!(x_vals.value(0), 5);
        assert_eq!(y_vals.value(0), 9);
    }

    /// Reader adds `msg: string` with the default "hello"
    #[test]
    fn test_schema_resolution_default_value_string() {
        let writer_schema = Schema::Complex(ComplexType::Record(Record {
            name: "S",
            namespace: None,
            doc: None,
            aliases: vec![],
            fields: vec![Field {
                name: "id",
                doc: None,
                r#type: Schema::TypeName(TypeName::Primitive(PrimitiveType::Long)),
                default: None,
            }],
            attributes: Default::default(),
        }));
        let reader_schema = Schema::Complex(ComplexType::Record(Record {
            name: "S",
            namespace: None,
            doc: None,
            aliases: vec![],
            fields: vec![
                Field {
                    name: "id",
                    doc: None,
                    r#type: Schema::TypeName(TypeName::Primitive(PrimitiveType::Long)),
                    default: None,
                },
                Field {
                    name: "msg",
                    doc: None,
                    r#type: Schema::TypeName(TypeName::Primitive(PrimitiveType::String)),
                    default: Some(json!("hello")),
                },
            ],
            attributes: Default::default(),
        }));
        let field =
            AvroField::resolve_from_writer_and_reader(&writer_schema, &reader_schema, false, false)
                .unwrap();
        let mut decoder = Decoder::try_new(field.data_type()).unwrap();
        let data = encode_avro_long(123);
        decoder.decode(&mut AvroCursor::new(&data)).unwrap();
        let array = decoder.flush(None).unwrap();
        let struct_arr = array.as_any().downcast_ref::<StructArray>().unwrap();
        let id_vals = struct_arr
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int64Type>();
        let msg_vals = struct_arr
            .column_by_name("msg")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(id_vals.value(0), 123);
        assert_eq!(msg_vals.value(0), "hello");
    }

    /// Writer has an extra `y` field; reader ignores it but must still decode
    #[test]
    fn test_schema_resolution_skip_writer_only() {
        let writer_schema = Schema::Complex(ComplexType::Record(Record {
            name: "T",
            namespace: None,
            doc: None,
            aliases: vec![],
            fields: vec![
                Field {
                    name: "x",
                    doc: None,
                    r#type: Schema::TypeName(TypeName::Primitive(PrimitiveType::Int)),
                    default: None,
                },
                Field {
                    name: "y",
                    doc: None,
                    r#type: Schema::TypeName(TypeName::Primitive(PrimitiveType::Int)),
                    default: None,
                },
            ],
            attributes: Default::default(),
        }));
        let reader_schema = Schema::Complex(ComplexType::Record(Record {
            name: "T",
            namespace: None,
            doc: None,
            aliases: vec![],
            fields: vec![Field {
                name: "x",
                doc: None,
                r#type: Schema::TypeName(TypeName::Primitive(PrimitiveType::Int)),
                default: None,
            }],
            attributes: Default::default(),
        }));
        let field =
            AvroField::resolve_from_writer_and_reader(&writer_schema, &reader_schema, false, false)
                .unwrap();
        let mut decoder = Decoder::try_new(field.data_type()).unwrap();
        let mut data = encode_avro_int(1);
        data.extend(encode_avro_int(2));
        decoder.decode(&mut AvroCursor::new(&data)).unwrap();
        let array = decoder.flush(None).unwrap();
        let struct_arr = array.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(struct_arr.num_columns(), 1);
        let x_vals = struct_arr.column(0).as_primitive::<Int32Type>();
        assert_eq!(x_vals.value(0), 1);
    }
}
