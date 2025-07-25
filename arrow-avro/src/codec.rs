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

use crate::schema::{
    Array, Attributes, ComplexType, Enum, Field as SchemaField, Fixed, Map, PrimitiveType, Record,
    Schema, Type, TypeName,
};
use arrow_schema::{
    ArrowError, DataType, Field, Fields, IntervalUnit, TimeUnit, DECIMAL128_MAX_PRECISION,
    DECIMAL128_MAX_SCALE,
};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Avro types are not nullable, with nullability instead encoded as a union
/// where one of the variants is the null type.
///
/// To accommodate this, we have two-variant unions where one of the
/// variants is the null type, and use this to derive arrow's notion of nullability
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Nullability {
    /// The nulls are encoded as the first union variant
    NullFirst,
    /// The nulls are encoded as the second union variant
    NullSecond,
}

/// Defines the type of promotion to be applied during schema resolution.
///
/// Schema resolution may require promoting a writer's data type to a reader's data type.
/// For example, an `int` can be promoted to a `long`, `float`, or `double`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Promotion {
    /// Promotes an `int` to a `long`.
    IntToLong,
    /// Promotes an `int` to a `float`.
    IntToFloat,
    /// Promotes an `int` to a `double`.
    IntToDouble,
    /// Promotes a `long` to a `float`.
    LongToFloat,
    /// Promotes a `long` to a `double`.
    LongToDouble,
    /// Promotes a `float` to a `double`.
    FloatToDouble,
    /// Promotes a `string` to `bytes`.
    StringToBytes,
    /// Promotes `bytes` to a `string`.
    BytesToString,
}

/// Holds the mapping information for resolving Avro enums.
///
/// When resolving schemas, the writer's enum symbols must be mapped to the reader's symbols.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumMapping {
    /// A mapping from the writer's symbol index to the reader's symbol index.
    pub(crate) mapping: Arc<[i32]>,
    /// The index to use for a writer's symbol that is not present in the reader's enum
    /// and a default value is specified in the reader's schema.
    pub(crate) default_index: i32,
}

/// Represents a literal Avro value.
///
/// This is used to represent default values in an Avro schema.
#[derive(Debug, Clone, PartialEq)]
pub enum AvroLiteral {
    /// Represents a null value.
    Null,
    /// Represents a boolean value.
    Boolean(bool),
    /// Represents an integer value.
    Int(i32),
    /// Represents a long value.
    Long(i64),
    /// Represents a float value.
    Float(f32),
    /// Represents a double value.
    Double(f64),
    /// Represents a bytes value.
    Bytes(Vec<u8>),
    /// Represents a string value.
    String(String),
    /// Represents an enum symbol.
    Enum(String),
    /// Represents an unsupported literal type.
    Unsupported,
}

/// Contains information about how to resolve differences between a writer's and a reader's schema.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionInfo {
    /// Indicates that the writer's type should be promoted to the reader's type.
    Promotion(Promotion),
    /// Indicates that a default value should be used for a field.
    DefaultValue(AvroLiteral),
    /// Provides mapping information for resolving enums.
    EnumMapping(EnumMapping),
    /// Provides resolution information for record fields.
    Record(ResolvedRecord),
}

/// Contains the necessary information to resolve a writer's record against a reader's record schema.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedRecord {
    /// Maps a writer's field index to the corresponding reader's field index.
    /// `None` if the writer's field is not present in the reader's schema.
    pub writer_to_reader: Arc<[Option<usize>]>,
    /// A list of indices in the reader's schema for fields that have a default value.
    pub default_fields: Arc<[usize]>,
    /// For fields present in the writer's schema but not the reader's, this stores their data type.
    /// This is needed to correctly skip over these fields during deserialization.
    pub skip_fields: Arc<[Option<AvroDataType>]>,
}

/// An Avro datatype mapped to the arrow data model
#[derive(Debug, Clone, PartialEq)]
pub struct AvroDataType {
    pub(crate) codec: Codec,
    pub(crate) nullability: Option<Nullability>,
    pub(crate) metadata: HashMap<String, String>,
    pub(crate) resolution: Option<ResolutionInfo>,
}

impl AvroDataType {
    /// Create a new [`AvroDataType`] with the given parts.
    pub fn new(
        codec: Codec,
        metadata: HashMap<String, String>,
        nullability: Option<Nullability>,
    ) -> Self {
        Self {
            codec,
            metadata,
            nullability,
            resolution: None,
        }
    }

    #[inline]
    fn parsed(codec: Codec, metadata: HashMap<String, String>) -> Self {
        Self {
            codec,
            metadata,
            nullability: None,
            resolution: None,
        }
    }

    #[inline]
    fn resolved(
        codec: Codec,
        metadata: HashMap<String, String>,
        nullability: Option<Nullability>,
        resolution: Option<ResolutionInfo>,
    ) -> Self {
        Self {
            codec,
            metadata,
            nullability,
            resolution,
        }
    }

    /// Returns an arrow [`Field`] with the given name
    pub fn field_with_name(&self, name: &str) -> Field {
        let base = Field::new(name, self.codec.data_type(), self.nullability.is_some())
            .with_metadata(self.metadata.clone());
        #[cfg(feature = "canonical_extension_types")]
        {
            return match self.codec {
                Codec::Uuid => base.with_extension_type(arrow_schema::extension::Uuid),
                _ => base,
            };
        }
        #[cfg(not(feature = "canonical_extension_types"))]
        base
    }

    /// Returns a reference to the codec used by this data type
    ///
    /// The codec determines how Avro data is encoded and mapped to Arrow data types.
    /// This is useful when we need to inspect or use the specific encoding of a field.
    #[inline]
    pub fn codec(&self) -> &Codec {
        &self.codec
    }

    /// Returns the nullability status of this data type
    ///
    /// In Avro, nullability is represented through unions with null types.
    /// The returned value indicates how nulls are encoded in the Avro format:
    /// - `Some(Nullability::NullFirst)` - Nulls are encoded as the first union variant
    /// - `Some(Nullability::NullSecond)` - Nulls are encoded as the second union variant
    /// - `None` - The type is not nullable
    #[inline]
    pub fn nullability(&self) -> Option<Nullability> {
        self.nullability
    }
}

/// An Avro encoding
///
/// <https://avro.apache.org/docs/1.11.1/specification/#encodings>
#[derive(Debug, Clone, PartialEq)]
pub enum Codec {
    /// Represents Avro null type, maps to Arrow's Null data type
    Null,
    /// Represents Avro boolean type, maps to Arrow's Boolean data type
    Boolean,
    /// Represents Avro int type, maps to Arrow's Int32 data type
    Int32,
    /// Represents Avro long type, maps to Arrow's Int64 data type
    Int64,
    /// Represents Avro float type, maps to Arrow's Float32 data type
    Float32,
    /// Represents Avro double type, maps to Arrow's Float64 data type
    Float64,
    /// Represents Avro bytes type, maps to Arrow's Binary data type
    Binary,
    /// String data represented as UTF-8 encoded bytes, corresponding to Arrow's StringArray
    Utf8,
    /// String data represented as UTF-8 encoded bytes with an optimized view representation,
    /// corresponding to Arrow's StringViewArray which provides better performance for string operations
    ///
    /// The Utf8View option can be enabled via `ReadOptions::use_utf8view`.
    Utf8View,
    /// Represents Avro date logical type, maps to Arrow's Date32 data type
    Date32,
    /// Represents Avro time-millis logical type, maps to Arrow's Time32(TimeUnit::Millisecond) data type
    TimeMillis,
    /// Represents Avro time-micros logical type, maps to Arrow's Time64(TimeUnit::Microsecond) data type
    TimeMicros,
    /// Represents Avro timestamp-millis or local-timestamp-millis logical type
    ///
    /// Maps to Arrow's Timestamp(TimeUnit::Millisecond) data type
    /// The boolean parameter indicates whether the timestamp has a UTC timezone (true) or is local time (false)
    TimestampMillis(bool),
    /// Represents Avro timestamp-micros or local-timestamp-micros logical type
    ///
    /// Maps to Arrow's Timestamp(TimeUnit::Microsecond) data type
    /// The boolean parameter indicates whether the timestamp has a UTC timezone (true) or is local time (false)
    TimestampMicros(bool),
    /// Represents Avro fixed type, maps to Arrow's FixedSizeBinary data type
    /// The i32 parameter indicates the fixed binary size
    Fixed(i32),
    /// Represents Avro decimal type, maps to Arrow's Decimal128 or Decimal256 data types
    ///
    /// The fields are `(precision, scale, fixed_size)`.
    /// - `precision` (`usize`): Total number of digits.
    /// - `scale` (`Option<usize>`): Number of fractional digits.
    /// - `fixed_size` (`Option<usize>`): Size in bytes if backed by a `fixed` type, otherwise `None`.
    Decimal(usize, Option<usize>, Option<usize>),
    /// Represents Avro Uuid type, a FixedSizeBinary with a length of 16.
    Uuid,
    /// Represents an Avro enum, maps to Arrow's Dictionary(Int32, Utf8) type.
    ///
    /// The enclosed value contains the enum's symbols.
    Enum(Arc<[String]>),
    /// Represents Avro duration logical type, maps to Arrow's Interval(IntervalUnit::MonthDayNano) data type
    Interval,
    /// Represents Avro array type, maps to Arrow's List data type
    List(Arc<AvroDataType>),
    /// Represents Avro map type, maps to Arrow's Map data type
    Map(Arc<AvroDataType>),
    /// Represents Avro record type, maps to Arrow's Struct data type
    Struct(Arc<[AvroField]>),
    /// Represents a resolved union type.
    Union(Arc<AvroDataType>),
}

impl Codec {
    /// Converts a string codec to use Utf8View if requested
    ///
    /// The conversion only happens if both:
    /// 1. `use_utf8view` is true
    /// 2. The codec is currently `Utf8`
    #[inline]
    fn with_utf8view(self, use_utf8view: bool) -> Self {
        if use_utf8view && matches!(self, Self::Utf8) {
            Self::Utf8View
        } else {
            self
        }
    }

    fn data_type(&self) -> DataType {
        use Codec::*;
        match self {
            Null => DataType::Null,
            Boolean => DataType::Boolean,
            Int32 => DataType::Int32,
            Int64 => DataType::Int64,
            Float32 => DataType::Float32,
            Float64 => DataType::Float64,
            Binary => DataType::Binary,
            Utf8 => DataType::Utf8,
            Utf8View => DataType::Utf8View,
            Date32 => DataType::Date32,
            TimeMillis => DataType::Time32(TimeUnit::Millisecond),
            TimeMicros => DataType::Time64(TimeUnit::Microsecond),
            TimestampMillis(utc) => {
                DataType::Timestamp(TimeUnit::Millisecond, utc.then(|| "+00:00".into()))
            }
            TimestampMicros(utc) => {
                DataType::Timestamp(TimeUnit::Microsecond, utc.then(|| "+00:00".into()))
            }
            Interval => DataType::Interval(IntervalUnit::MonthDayNano),
            Fixed(n) => DataType::FixedSizeBinary(*n),
            Decimal(p, s, size_b) => {
                let p8 = *p as u8;
                let s8 = s.unwrap_or(0) as i8;
                let needs256 = size_b.map_or_else(
                    || {
                        (*p > DECIMAL128_MAX_PRECISION as usize)
                            || (s.unwrap_or(0) > DECIMAL128_MAX_SCALE as usize)
                    },
                    |b| b > 16,
                );
                if needs256 {
                    DataType::Decimal256(p8, s8)
                } else {
                    DataType::Decimal128(p8, s8)
                }
            }
            Uuid => DataType::FixedSizeBinary(16),
            Enum(_) => DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            List(child) => DataType::List(Arc::new(
                child.field_with_name(Field::LIST_FIELD_DEFAULT_NAME),
            )),
            Map(val) => {
                let value_field =
                    Field::new("value", val.codec.data_type(), val.nullability.is_some())
                        .with_metadata(val.metadata.clone());
                DataType::Map(
                    Arc::new(Field::new(
                        "entries",
                        DataType::Struct(Fields::from(vec![
                            Field::new("key", DataType::Utf8, false),
                            value_field,
                        ])),
                        false,
                    )),
                    false,
                )
            }
            Struct(flds) => DataType::Struct(flds.iter().map(|f| f.field()).collect()),
            Union(child) => child.codec.data_type(),
        }
    }
}

impl From<PrimitiveType> for Codec {
    fn from(pt: PrimitiveType) -> Self {
        use PrimitiveType::*;
        match pt {
            Null => Codec::Null,
            Boolean => Codec::Boolean,
            Int => Codec::Int32,
            Long => Codec::Int64,
            Float => Codec::Float32,
            Double => Codec::Float64,
            Bytes => Codec::Binary,
            String => Codec::Utf8,
        }
    }
}

/// Builder for an [`AvroField`]
#[derive(Debug)]
pub struct AvroFieldBuilder<'a> {
    writer_schema: &'a Schema<'a>,
    reader_schema: Option<&'a Schema<'a>>,
    use_utf8view: bool,
    strict_mode: bool,
}

impl<'a> AvroFieldBuilder<'a> {
    /// Creates a new [`AvroFieldBuilder`] for a given writer schema.
    pub fn new(writer_schema: &'a Schema<'a>) -> Self {
        Self {
            writer_schema,
            reader_schema: None,
            use_utf8view: false,
            strict_mode: false,
        }
    }

    /// Sets the reader schema for schema resolution.
    ///
    /// If a reader schema is provided, the builder will produce a resolved `AvroField`
    /// that can handle differences between the writer's and reader's schemas.
    #[inline]
    pub fn with_reader_schema(mut self, reader_schema: &'a Schema<'a>) -> Self {
        self.reader_schema = Some(reader_schema);
        self
    }

    /// Enable or disable Utf8View support
    #[inline]
    pub fn with_utf8view(mut self, enabled: bool) -> Self {
        self.use_utf8view = enabled;
        self
    }

    /// Enable or disable strict mode.
    #[inline]
    pub fn with_strict_mode(mut self, enabled: bool) -> Self {
        self.strict_mode = enabled;
        self
    }

    /// Build an [`AvroField`] from the builder
    pub fn build(self) -> Result<AvroField, ArrowError> {
        let mut resolver = SchemaResolver::with_config(self.use_utf8view, self.strict_mode);
        let dt = resolver.visit(self.writer_schema, self.reader_schema, None)?;
        let top_schema = self.reader_schema.unwrap_or(self.writer_schema);
        let top_name = match top_schema {
            Schema::Complex(ComplexType::Record(rec)) => rec.name.to_string(),
            _ => "root".to_string(),
        };
        Ok(AvroField {
            name: top_name,
            data_type: dt,
        })
    }
}

/// A named [`AvroDataType`]
#[derive(Debug, Clone, PartialEq)]
pub struct AvroField {
    name: String,
    data_type: AvroDataType,
}

impl AvroField {
    /// Returns the arrow [`Field`]
    #[inline]
    pub fn field(&self) -> Field {
        self.data_type.field_with_name(&self.name)
    }

    /// Returns the [`AvroDataType`]
    #[inline]
    pub fn data_type(&self) -> &AvroDataType {
        &self.data_type
    }

    /// Returns the name of this Avro field
    ///
    /// This is the field name as defined in the Avro schema.
    /// It's used to identify fields within a record structure.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a new [`AvroField`] with Utf8View support enabled
    ///
    /// This will convert any Utf8 codecs to Utf8View codecs. This method is used to
    /// enable potential performance optimizations in string-heavy workloads by using
    /// Arrow's StringViewArray data structure.
    ///
    /// Returns a new `AvroField` with the same structure, but with string types
    /// converted to use `Utf8View` instead of `Utf8`.
    pub fn with_utf8view(&self) -> Self {
        let mut cloned = self.clone();
        if let Codec::Utf8 = cloned.data_type.codec {
            cloned.data_type.codec = Codec::Utf8View;
        }
        cloned
    }

    /// Performs schema resolution between a writer and reader schema.
    ///
    /// This is the primary entry point for handling schema evolution. It produces an
    /// `AvroField` that contains all the necessary information to read data written
    /// with the `writer` schema as if it were written with the `reader` schema.
    pub fn resolve_from_writer_and_reader<'a>(
        writer: &'a Schema<'a>,
        reader: &'a Schema<'a>,
        use_utf8view: bool,
        strict_mode: bool,
    ) -> Result<Self, ArrowError> {
        let mut resolver = SchemaResolver::new(use_utf8view, strict_mode);
        let dt = resolver.visit(writer, Some(reader), None)?;
        let top_name = match reader {
            Schema::Complex(ComplexType::Record(r)) => r.name.to_string(),
            _ => "root".to_string(),
        };
        Ok(Self {
            name: top_name,
            data_type: dt,
        })
    }
}

impl<'a> TryFrom<&'a Schema<'a>> for AvroField {
    type Error = ArrowError;
    fn try_from(schema: &'a Schema<'a>) -> Result<Self, Self::Error> {
        if let Schema::Complex(ComplexType::Record(r)) = schema {
            let mut resolver = SchemaResolver::new(false, false);
            let dt = resolver.visit(schema, None, None)?;
            Ok(Self {
                name: r.name.to_string(),
                data_type: dt,
            })
        } else {
            Err(ArrowError::ParseError("Expected top‑level record".into()))
        }
    }
}

/// Checks if a writer schema is compatible with a reader schema.
///
/// This is a convenience function that performs schema resolution and returns `Ok(())`
/// if the schemas are compatible, or an `ArrowError` otherwise.
#[inline]
pub fn check_schema_compatibility<'a>(
    writer: &'a Schema<'a>,
    reader: &'a Schema<'a>,
) -> Result<(), ArrowError> {
    AvroField::resolve_from_writer_and_reader(writer, reader, false, false).map(|_| ())
}

#[derive(Default)]
struct NameCache<'a> {
    named: HashMap<(&'a str, &'a str), AvroDataType>,
    resolved: HashMap<(&'a str, &'a str), AvroDataType>,
    in_progress: HashSet<(&'a str, &'a str)>,
}

impl<'a> NameCache<'a> {
    #[inline]
    fn lookup_named(&self, name: &str, ns_hint: Option<&'a str>) -> Option<AvroDataType> {
        let (ns, nm) = name
            .rsplit_once('.')
            .unwrap_or_else(|| (ns_hint.unwrap_or(""), name));
        let key = (ns, nm);
        if self.in_progress.contains(&key) {
            return None;
        }
        self.named.get(&key).cloned()
    }
}

#[inline]
fn split_null_union<'u, 'a>(branches: &'u [Schema<'a>]) -> Option<(Nullability, &'u Schema<'a>)> {
    match branches {
        [Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)), nonnull] => {
            Some((Nullability::NullFirst, nonnull))
        }
        [nonnull, Schema::TypeName(TypeName::Primitive(PrimitiveType::Null))] => {
            Some((Nullability::NullSecond, nonnull))
        }
        _ => None,
    }
}

/// Resolves Avro type names to [`AvroDataType`]
///
/// See <https://avro.apache.org/docs/1.11.1/specification/#names>
struct SchemaResolver<'a> {
    cache: NameCache<'a>,
    use_utf8view: bool,
    strict_mode: bool,
}

impl<'a> SchemaResolver<'a> {
    fn new(use_utf8view: bool, strict_mode: bool) -> Self {
        Self {
            cache: Default::default(),
            use_utf8view,
            strict_mode,
        }
    }

    #[inline]
    pub fn with_config(use_utf8view: bool, strict_mode: bool) -> Self {
        Self::new(use_utf8view, strict_mode)
    }

    fn visit<'s>(
        &mut self,
        writer: &'s Schema<'a>,
        reader: Option<&'s Schema<'a>>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        match reader {
            None => self.parse(writer, namespace),
            Some(r) => self.resolve(writer, r, namespace),
        }
    }

    fn parse<'s>(
        &mut self,
        schema: &'s Schema<'a>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        match schema {
            Schema::TypeName(TypeName::Primitive(p)) => Ok(AvroDataType::parsed(
                Codec::from(*p).with_utf8view(self.use_utf8view),
                Default::default(),
            )),
            Schema::TypeName(TypeName::Ref(name)) => self
                .cache
                .lookup_named(name, namespace)
                .ok_or_else(|| ArrowError::ParseError(format!("Failed to resolve .{name}")))
                .and_then(|dt| {
                    if matches!(&dt.codec, Codec::Struct(flds) if flds.is_empty()) {
                        Err(ArrowError::ParseError(format!("Failed to resolve .{name}")))
                    } else {
                        Ok(dt)
                    }
                }),
            Schema::Union(branches) => self.parse_nullable_union(branches, namespace),
            Schema::Complex(ct) => self.parse_complex(ct, namespace),
            Schema::Type(t) => {
                let mut dt = self.parse(&Schema::TypeName(t.r#type.clone()), namespace)?;
                self.apply_logical_type(&mut dt, &t.attributes, None)?;
                Ok(dt)
            }
        }
    }

    fn parse_nullable_union<'u>(
        &mut self,
        branches: &'u [Schema<'a>],
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        if let Some((null_pos, nonnull)) = split_null_union(branches) {
            if self.strict_mode && matches!(null_pos, Nullability::NullSecond) {
                return Err(ArrowError::SchemaError(
                    "Found Avro union of the form ['T','null'], which is disallowed in strict_mode"
                        .into(),
                ));
            }
            let mut dt = self.parse(nonnull, namespace)?;
            dt.nullability = Some(null_pos);
            Ok(dt)
        } else {
            Err(ArrowError::NotYetImplemented(
                "Non‑nullable unions are not yet supported".into(),
            ))
        }
    }

    fn parse_complex(
        &mut self,
        ct: &ComplexType<'a>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        use ComplexType::*;
        match ct {
            Record(r) => self.parse_record(r, namespace),
            Enum(e) => self.parse_enum(e, namespace),
            Fixed(f) => self.parse_fixed(f, namespace),
            Array(arr) => {
                let item = self.parse(&arr.items, namespace)?;
                Ok(AvroDataType::parsed(
                    Codec::List(Arc::new(item)),
                    arr.attributes.field_metadata(),
                ))
            }
            Map(mp) => {
                let val = self.parse(&mp.values, namespace)?;
                Ok(AvroDataType::parsed(
                    Codec::Map(Arc::new(val)),
                    mp.attributes.field_metadata(),
                ))
            }
        }
    }

    fn parse_record(
        &mut self,
        r: &Record<'a>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        let ns = r.namespace.or(namespace);
        let ns_str = ns.unwrap_or("");
        let key = (ns_str, r.name);
        self.cache.named.insert(
            key,
            AvroDataType::parsed(Codec::Struct(Arc::new([])), r.attributes.field_metadata()),
        );
        self.cache.in_progress.insert(key);
        let inner = (|| {
            let mut fields = Vec::with_capacity(r.fields.len());
            for f in &r.fields {
                let dt = self.parse(&f.r#type, ns)?;
                fields.push(AvroField {
                    name: f.name.to_string(),
                    data_type: dt,
                });
            }
            let final_dt = AvroDataType::parsed(
                Codec::Struct(Arc::from(fields)),
                r.attributes.field_metadata(),
            );
            self.cache.named.insert(key, final_dt.clone());
            Ok(final_dt)
        })();
        self.cache.in_progress.remove(&key);
        inner
    }

    fn parse_enum(
        &mut self,
        e: &Enum<'a>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        let symbols: Arc<[String]> = e
            .symbols
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .into();
        let mut md = e.attributes.field_metadata();
        let symbols_json =
            serde_json::to_string(&e.symbols).map_err(|e| ArrowError::JsonError(e.to_string()))?;
        md.insert("avro.enum.symbols".into(), symbols_json);
        let dt = AvroDataType::parsed(Codec::Enum(symbols), md);
        self.cache.named.insert(
            (e.namespace.or(namespace).unwrap_or(""), e.name),
            dt.clone(),
        );
        Ok(dt)
    }

    fn parse_fixed(
        &mut self,
        f: &Fixed<'a>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        let md = f.attributes.field_metadata();
        let codec = match f.attributes.logical_type {
            Some("decimal") => {
                let (p, s, _) = parse_decimal_attrs(&f.attributes, Some(f.size), true)?;
                Codec::Decimal(p, Some(s), Some(f.size))
            }
            Some("duration") => {
                if f.size != 12 {
                    return Err(ArrowError::ParseError(
                        "Duration logical type requires 12‑byte fixed".into(),
                    ));
                }
                Codec::Interval
            }
            _ => Codec::Fixed(f.size as i32),
        };
        let dt = AvroDataType::parsed(codec, md);
        self.cache.named.insert(
            (f.namespace.or(namespace).unwrap_or(""), f.name),
            dt.clone(),
        );
        Ok(dt)
    }

    fn resolve<'s>(
        &mut self,
        writer: &'s Schema<'a>,
        reader: &'s Schema<'a>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        use ComplexType::*;
        match (writer, reader) {
            (
                Schema::TypeName(TypeName::Primitive(wp)),
                Schema::TypeName(TypeName::Primitive(rp)),
            )
            | (
                Schema::Type(Type {
                    r#type: TypeName::Primitive(wp),
                    ..
                }),
                Schema::Type(Type {
                    r#type: TypeName::Primitive(rp),
                    ..
                }),
            ) => self.resolve_primitives(*wp, *rp, reader),
            (Schema::TypeName(TypeName::Primitive(wp)), Schema::Type(r_t))
                if matches!(r_t.r#type, TypeName::Primitive(_)) =>
            {
                if let TypeName::Primitive(rp) = r_t.r#type {
                    self.resolve_primitives(*wp, rp, reader)
                } else {
                    unreachable!()
                }
            }
            (Schema::Complex(Record(wr)), Schema::Complex(Record(rr))) => {
                self.resolve_records(wr, rr, namespace)
            }
            (Schema::Complex(Enum(we)), Schema::Complex(Enum(re))) => self.resolve_enums(we, re),
            (Schema::Complex(Fixed(wf)), Schema::Complex(Fixed(rf))) if wf.size == rf.size => {
                self.parse(reader, namespace)
            }
            (Schema::Complex(Array(wa)), Schema::Complex(Array(ra))) => {
                let child = self.visit(&wa.items, Some(&ra.items), namespace)?;
                Ok(AvroDataType::resolved(
                    Codec::List(Arc::new(child)),
                    ra.attributes.field_metadata(),
                    None,
                    None,
                ))
            }
            (Schema::Complex(Map(wm)), Schema::Complex(Map(rm))) => {
                let val = self.visit(&wm.values, Some(&rm.values), namespace)?;
                Ok(AvroDataType::resolved(
                    Codec::Map(Arc::new(val)),
                    rm.attributes.field_metadata(),
                    None,
                    None,
                ))
            }
            (Schema::Union(wu), Schema::Union(ru)) => {
                self.resolve_nullable_union(wu, ru, namespace)
            }
            (w_nonunion, Schema::Union(ru)) if Self::is_nullable_union(ru) => {
                let (null_pos, r_nonnull) = split_null_union(ru).ok_or_else(|| {
                    ArrowError::SchemaError("Reader schema should be a nullable union".to_string())
                })?;
                let mut dt = self.visit(w_nonunion, Some(r_nonnull), namespace)?;
                dt.nullability = Some(null_pos);
                Ok(dt)
            }
            (Schema::Union(wu), r_nonunion) if Self::is_nullable_union(wu) => {
                let (_, w_nonnull) = split_null_union(wu).ok_or_else(|| {
                    ArrowError::SchemaError("Writer schema should be a nullable union".to_string())
                })?;
                self.visit(w_nonnull, Some(r_nonunion), namespace)
            }
            (w_nonunion, Schema::Union(ru)) => {
                for branch in ru {
                    if let Ok(dt) = self.visit(w_nonunion, Some(branch), namespace) {
                        return Ok(AvroDataType::resolved(
                            Codec::Union(Arc::new(dt)),
                            Default::default(),
                            None,
                            None,
                        ));
                    }
                }
                Err(ArrowError::ParseError(
                    "Writer type not found in reader union".into(),
                ))
            }
            (Schema::Union(wu), r_nonunion) => {
                for branch in wu {
                    if let Ok(dt) = self.visit(branch, Some(r_nonunion), namespace) {
                        return Ok(dt);
                    }
                }
                Err(ArrowError::ParseError(
                    "Reader type not found in writer union".into(),
                ))
            }
            _ => Err(ArrowError::ParseError(format!(
                "Incompatible schemas\nwriter: {writer:?}\nreader: {reader:?}"
            ))),
        }
    }

    fn resolve_primitives(
        &mut self,
        wp: PrimitiveType,
        rp: PrimitiveType,
        reader_schema: &Schema<'a>,
    ) -> Result<AvroDataType, ArrowError> {
        if wp == rp {
            return self.parse(reader_schema, None);
        }
        use PrimitiveType::*;
        let promotion = match (wp, rp) {
            (Int, Long) => Promotion::IntToLong,
            (Int, Float) => Promotion::IntToFloat,
            (Int, Double) => Promotion::IntToDouble,
            (Long, Float) => Promotion::LongToFloat,
            (Long, Double) => Promotion::LongToDouble,
            (Float, Double) => Promotion::FloatToDouble,
            (String, Bytes) => Promotion::StringToBytes,
            (Bytes, String) => Promotion::BytesToString,
            _ => {
                return Err(ArrowError::ParseError(format!(
                    "Illegal promotion {wp:?}→{rp:?}"
                )))
            }
        };
        let mut dt = self.parse(reader_schema, None)?;
        dt.resolution = Some(ResolutionInfo::Promotion(promotion));
        Ok(dt)
    }

    fn resolve_enums(&mut self, w: &Enum<'a>, r: &Enum<'a>) -> Result<AvroDataType, ArrowError> {
        let name_ok = w.name == r.name || r.aliases.iter().any(|&a| a == w.name);
        if !name_ok {
            return Err(ArrowError::ParseError(format!(
                "Enum name mismatch writer={}, reader={}",
                w.name, r.name
            )));
        }
        let mut mapping = vec![0_i32; w.symbols.len()];
        let mut def_idx = None;
        for (i, &wsym) in w.symbols.iter().enumerate() {
            if let Some(p) = r.symbols.iter().position(|&rs| rs == wsym) {
                mapping[i] = p as i32;
            } else if let Some(def) = r.default {
                let p = r.symbols.iter().position(|&rs| rs == def).ok_or_else(|| {
                    ArrowError::ParseError("Reader enum default not in symbols".into())
                })?;
                mapping[i] = p as i32;
            } else {
                return Err(ArrowError::ParseError(format!(
                    "Writer enum symbol '{wsym}' not in reader"
                )));
            }
            def_idx = def_idx.or_else(|| {
                r.default
                    .and_then(|d| r.symbols.iter().position(|&s| s == d))
                    .map(|p| p as i32)
            });
        }
        let enum_codec = Codec::Enum(Arc::from(
            r.symbols.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        ));
        Ok(AvroDataType::resolved(
            enum_codec,
            r.attributes.field_metadata(),
            None,
            Some(ResolutionInfo::EnumMapping(EnumMapping {
                mapping: Arc::from(mapping),
                default_index: def_idx.unwrap_or(0),
            })),
        ))
    }

    fn resolve_records(
        &mut self,
        w: &Record<'a>,
        r: &Record<'a>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        let names_match = w.name == r.name
            || r.aliases.iter().any(|&a| a == w.name)
            || w.aliases.iter().any(|&a| a == r.name);
        if !names_match {
            return Err(ArrowError::ParseError(format!(
                "Record name mismatch writer={}, reader={}",
                w.name, r.name
            )));
        }
        if let Some(prev) = self.cache.resolved.get(&(w.name, r.name)).cloned() {
            return Ok(prev);
        }
        let ns = r.namespace.or(namespace);
        let mut w_index_map = HashMap::<&str, usize>::with_capacity(w.fields.len());
        for (idx, wf) in w.fields.iter().enumerate() {
            w_index_map.insert(wf.name, idx);
        }
        let mut reader_fields = Vec::with_capacity(r.fields.len());
        let mut w_to_r = vec![None; w.fields.len()];
        let mut default_indices = Vec::new();
        for (r_idx, rf) in r.fields.iter().enumerate() {
            if let Some(&w_idx) = w_index_map.get(rf.name) {
                let wf = &w.fields[w_idx];
                let child = self.visit(&wf.r#type, Some(&rf.r#type), ns)?;
                reader_fields.push(AvroField {
                    name: rf.name.to_string(),
                    data_type: child,
                });
                w_to_r[w_idx] = Some(r_idx);
            } else if let Some(def_val) = rf.default.as_ref() {
                let lit = parse_default_literal(def_val, &rf.r#type)?;
                let mut child = self.parse(&rf.r#type, ns)?;
                child.resolution = Some(ResolutionInfo::DefaultValue(lit));
                reader_fields.push(AvroField {
                    name: rf.name.to_string(),
                    data_type: child,
                });
                default_indices.push(r_idx);
            } else {
                return Err(ArrowError::ParseError(format!(
                    "Field '{0}' missing in writer and no default",
                    rf.name
                )));
            }
        }
        let mut skip_fields = vec![None; w.fields.len()];
        for (w_idx, wf) in w.fields.iter().enumerate() {
            if w_to_r[w_idx].is_none() {
                skip_fields[w_idx] = Some(self.parse(&wf.r#type, ns)?);
            }
        }
        let mut md = r.attributes.field_metadata();
        if !default_indices.is_empty() {
            let defaults_json = serde_json::to_string(&default_indices)
                .map_err(|e| ArrowError::JsonError(e.to_string()))?;
            md.insert("avro.resolution.defaults".into(), defaults_json);
        }
        let resolved = AvroDataType::resolved(
            Codec::Struct(Arc::from(reader_fields)),
            md,
            None,
            Some(ResolutionInfo::Record(ResolvedRecord {
                writer_to_reader: Arc::from(w_to_r),
                default_fields: Arc::from(default_indices),
                skip_fields: Arc::from(skip_fields),
            })),
        );
        self.cache
            .resolved
            .insert((w.name, r.name), resolved.clone());
        Ok(resolved)
    }

    fn resolve_nullable_union<'u>(
        &mut self,
        wu: &'u [Schema<'a>],
        ru: &'u [Schema<'a>],
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        if !(Self::is_nullable_union(wu) && Self::is_nullable_union(ru)) {
            return Err(ArrowError::NotYetImplemented(
                "Full union resolution not implemented".into(),
            ));
        }
        let (null_pos, w_nonnull) = split_null_union(wu).ok_or_else(|| {
            ArrowError::SchemaError("Writer schema should be a nullable union".to_string())
        })?;
        let (_, r_nonnull) = split_null_union(ru).ok_or_else(|| {
            ArrowError::SchemaError("Reader schema should be a nullable union".to_string())
        })?;
        let mut dt = self.visit(w_nonnull, Some(r_nonnull), namespace)?;
        dt.nullability = Some(null_pos);
        Ok(dt)
    }

    #[inline]
    fn is_nullable_union(branches: &[Schema<'_>]) -> bool {
        split_null_union(branches).is_some()
    }

    fn apply_logical_type(
        &self,
        dt: &mut AvroDataType,
        attrs: &Attributes,
        fallback_size: Option<usize>,
    ) -> Result<(), ArrowError> {
        if let Some(logical_type) = attrs.logical_type.as_deref() {
            use Codec::*;
            match (logical_type, &mut dt.codec) {
                ("decimal", c @ Binary) => {
                    let (p, s, _) = parse_decimal_attrs(attrs, fallback_size, false)?;
                    *c = Decimal(p, Some(s), fallback_size);
                }
                ("date", c @ Int32) => *c = Date32,
                ("time-millis", c @ Int32) => *c = TimeMillis,
                ("time-micros", c @ Int64) => *c = TimeMicros,
                ("timestamp-millis", c @ Int64) => *c = TimestampMillis(true),
                ("timestamp-micros", c @ Int64) => *c = TimestampMicros(true),
                ("local-timestamp-millis", c @ Int64) => *c = TimestampMillis(false),
                ("local-timestamp-micros", c @ Int64) => *c = TimestampMicros(false),
                ("uuid", c @ Utf8) => *c = Uuid,
                (other, _) => {
                    dt.metadata.insert("logicalType".into(), other.to_string());
                }
            }
        }
        for (k, v) in &attrs.additional {
            dt.metadata.insert(k.to_string(), v.to_string());
        }
        Ok(())
    }
}

fn parse_default_literal(v: &Value, ty: &Schema) -> Result<AvroLiteral, ArrowError> {
    Ok(match (ty, v) {
        (_, Value::Null) => AvroLiteral::Null,
        (Schema::TypeName(TypeName::Primitive(PrimitiveType::Boolean)), Value::Bool(b)) => {
            AvroLiteral::Boolean(*b)
        }
        (Schema::TypeName(TypeName::Primitive(PrimitiveType::Int)), Value::Number(n))
            if n.is_i64() =>
        {
            AvroLiteral::Int(n.as_i64().unwrap() as i32)
        }
        (Schema::TypeName(TypeName::Primitive(PrimitiveType::Long)), Value::Number(n))
            if n.is_i64() =>
        {
            AvroLiteral::Long(n.as_i64().unwrap())
        }
        (Schema::TypeName(TypeName::Primitive(PrimitiveType::Float)), Value::Number(n))
            if n.is_f64() =>
        {
            AvroLiteral::Float(n.as_f64().unwrap() as f32)
        }
        (Schema::TypeName(TypeName::Primitive(PrimitiveType::Double)), Value::Number(n))
            if n.is_f64() =>
        {
            AvroLiteral::Double(n.as_f64().unwrap())
        }
        (Schema::TypeName(TypeName::Primitive(PrimitiveType::String)), Value::String(s)) => {
            AvroLiteral::String(s.clone())
        }
        (Schema::TypeName(TypeName::Primitive(PrimitiveType::Bytes)), Value::String(s)) => {
            AvroLiteral::Bytes(s.clone().into_bytes())
        }
        (Schema::Complex(ComplexType::Enum(_)), Value::String(s)) => AvroLiteral::Enum(s.clone()),
        _ => AvroLiteral::Unsupported,
    })
}

fn parse_decimal_attrs(
    attrs: &Attributes,
    fallback_size: Option<usize>,
    precision_required: bool,
) -> Result<(usize, usize, Option<usize>), ArrowError> {
    let precision = attrs
        .additional
        .get("precision")
        .and_then(|v| v.as_u64())
        .or(if precision_required { None } else { Some(10) })
        .ok_or_else(|| ArrowError::ParseError("Decimal missing precision".into()))?
        as usize;
    let scale = attrs
        .additional
        .get("scale")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let size = attrs
        .additional
        .get("size")
        .and_then(|v| v.as_u64())
        .map(|s| s as usize)
        .or(fallback_size);
    Ok((precision, scale, size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{
        Attributes, ComplexType, Fixed, PrimitiveType, Record, Schema, Type, TypeName,
    };
    use serde_json;
    use std::collections::HashMap;

    fn parse_schema(json: &str) -> Schema<'_> {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn int_to_long_promotion() {
        let w = parse_schema(r#""int""#);
        let r = parse_schema(r#""long""#);
        let field = AvroField::resolve_from_writer_and_reader(&w, &r, false, false).unwrap();
        assert!(matches!(field.data_type.codec(), Codec::Int64));
        matches!(
            &field.data_type.resolution,
            Some(ResolutionInfo::Promotion(Promotion::IntToLong))
        );
    }

    #[test]
    fn enum_symbol_mapping() {
        let w = parse_schema(r#"{"type":"enum","name":"Color","symbols":["RED","GREEN"]}"#);
        let r = parse_schema(
            r#"{
                "type":"enum",
                "name":"Color",
                "symbols":["RED","BLUE"],
                "default":"BLUE"
            }"#,
        );
        let field = AvroField::resolve_from_writer_and_reader(&w, &r, false, false).unwrap();
        if let Codec::Enum(_) = field.data_type.codec() {
            assert!(matches!(
                &field.data_type.resolution,
                Some(ResolutionInfo::EnumMapping(_))
            ));
        } else {
            panic!("expected enum codec");
        }
    }

    fn create_schema_with_logical_type(
        primitive_type: PrimitiveType,
        logical_type: &'static str,
    ) -> Schema<'static> {
        let attributes = Attributes {
            logical_type: Some(logical_type),
            additional: Default::default(),
        };

        Schema::Type(Type {
            r#type: TypeName::Primitive(primitive_type),
            attributes,
        })
    }

    fn create_fixed_schema(size: usize, logical_type: &'static str) -> Schema<'static> {
        let attributes = Attributes {
            logical_type: Some(logical_type),
            additional: Default::default(),
        };

        Schema::Complex(ComplexType::Fixed(Fixed {
            name: "fixed_type",
            namespace: None,
            aliases: Vec::new(),
            size,
            attributes,
        }))
    }

    macro_rules! assert_codec {
        ($schema:expr, $codec_pat:pat, $utf8view:expr) => {{
            let mut parser = SchemaResolver::new(false, false);
            let result = parser.parse(&$schema, None).unwrap();
            assert!(matches!(result.codec, $codec_pat));
        }};
    }

    #[test]
    fn test_date_logical_type() {
        let schema = create_schema_with_logical_type(PrimitiveType::Int, "date");
        assert_codec!(schema, Codec::Date32, false);
    }

    #[test]
    fn test_time_millis_logical_type() {
        let schema = create_schema_with_logical_type(PrimitiveType::Int, "time-millis");
        assert_codec!(schema, Codec::TimeMillis, false);
    }

    #[test]
    fn test_time_micros_logical_type() {
        let schema = create_schema_with_logical_type(PrimitiveType::Long, "time-micros");
        assert_codec!(schema, Codec::TimeMicros, false);
    }

    #[test]
    fn test_timestamp_millis_logical_type() {
        let schema = create_schema_with_logical_type(PrimitiveType::Long, "timestamp-millis");
        assert_codec!(schema, Codec::TimestampMillis(true), false);
    }

    #[test]
    fn test_timestamp_micros_logical_type() {
        let schema = create_schema_with_logical_type(PrimitiveType::Long, "timestamp-micros");
        assert_codec!(schema, Codec::TimestampMicros(true), false);
    }

    #[test]
    fn test_local_timestamp_millis_logical_type() {
        let schema = create_schema_with_logical_type(PrimitiveType::Long, "local-timestamp-millis");
        assert_codec!(schema, Codec::TimestampMillis(false), false);
    }

    #[test]
    fn test_local_timestamp_micros_logical_type() {
        let schema = create_schema_with_logical_type(PrimitiveType::Long, "local-timestamp-micros");
        assert_codec!(schema, Codec::TimestampMicros(false), false);
    }

    #[test]
    fn test_uuid_type() {
        let mut codec = Codec::Fixed(16);
        if let c @ Codec::Fixed(16) = &mut codec {
            *c = Codec::Uuid;
        }
        assert!(matches!(codec, Codec::Uuid));
    }

    #[test]
    fn test_duration_logical_type() {
        let mut codec = Codec::Fixed(12);
        if let c @ Codec::Fixed(12) = &mut codec {
            *c = Codec::Interval;
        }
        assert!(matches!(codec, Codec::Interval));
    }

    #[test]
    fn test_decimal_logical_type_not_implemented() {
        let mut codec = Codec::Fixed(16);
        let process_decimal = || -> Result<(), ArrowError> {
            if let Codec::Fixed(_) = codec {
                return Err(ArrowError::NotYetImplemented(
                    "Decimals are not currently supported".to_string(),
                ));
            }
            Ok(())
        };
        let result = process_decimal();
        assert!(result.is_err());
        if let Err(ArrowError::NotYetImplemented(msg)) = result {
            assert!(msg.contains("Decimals are not currently supported"));
        } else {
            panic!("Expected NotYetImplemented error");
        }
    }

    #[test]
    fn test_unknown_logical_type_added_to_metadata() {
        let schema = create_schema_with_logical_type(PrimitiveType::Int, "custom-type");
        let mut parser = SchemaResolver::new(false, false);
        let result = parser.parse(&schema, None).unwrap();
        assert_eq!(
            result.metadata.get("logicalType"),
            Some(&"custom-type".to_string())
        );
    }

    #[test]
    fn test_string_with_utf8view_enabled() {
        let schema = Schema::TypeName(TypeName::Primitive(PrimitiveType::String));
        let mut parser = SchemaResolver::new(true, false);
        let result = parser.parse(&schema, None).unwrap();
        assert!(matches!(result.codec, Codec::Utf8View));
    }

    #[test]
    fn test_string_without_utf8view_enabled() {
        let schema = Schema::TypeName(TypeName::Primitive(PrimitiveType::String));
        let mut parser = SchemaResolver::new(false, false);
        let result = parser.parse(&schema, None).unwrap();
        assert!(matches!(result.codec, Codec::Utf8));
    }

    #[test]
    fn test_record_with_string_and_utf8view_enabled() {
        let field_schema = Schema::TypeName(TypeName::Primitive(PrimitiveType::String));
        let avro_field = crate::schema::Field {
            name: "string_field",
            r#type: field_schema,
            default: None,
            doc: None,
        };
        let record = Record {
            name: "test_record",
            namespace: None,
            aliases: vec![],
            doc: None,
            fields: vec![avro_field],
            attributes: Attributes::default(),
        };
        let schema = Schema::Complex(ComplexType::Record(record));
        let mut parser = SchemaResolver::new(true, false);
        let result = parser.parse(&schema, None).unwrap();

        if let Codec::Struct(fields) = &result.codec {
            assert!(matches!(fields[0].data_type().codec, Codec::Utf8View));
        } else {
            panic!("Expected Struct codec");
        }
    }
}
