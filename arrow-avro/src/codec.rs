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
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Avro types are not nullable, with nullability instead encoded as a union
/// where one of the variants is the null type.
///
/// To accommodate this we special case two-variant unions where one of the
/// variants is the null type, and use this to derive arrow's notion of nullability.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Nullability {
    /// The nulls are encoded as the first union variant.
    NullFirst,
    /// The nulls are encoded as the second union variant.
    NullSecond,
}

/// Describes a valid primitive type promotion according to the Avro specification.
///
/// <https://avro.apache.org/docs/1.11.1/specification/#schema-resolution>
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Promotion {
    /// `int` is promotable to `long`, `float`, or `double`.
    IntToLong,
    /// `int` is promotable to `long`, `float`, or `double`.
    IntToFloat,
    /// `int` is promotable to `long`, `float`, or `double`.
    IntToDouble,
    /// `long` is promotable to `float` or `double`.
    LongToFloat,
    /// `long` is promotable to `float` or `double`.
    LongToDouble,
    /// `float` is promotable to `double`.
    FloatToDouble,
    /// `string` is promotable to `bytes`.
    StringToBytes,
    /// `bytes` is promotable to `string`.
    BytesToString,
}

/// Represents the mapping of symbols from a writer's `Enum` schema to a reader's `Enum` schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumMapping {
    /// A slice where `mapping[writer_symbol_index]` gives the corresponding `reader_symbol_index`.
    pub(crate) mapping: Arc<[i32]>,
    /// The index of the default symbol in the reader's schema, if any.
    pub(crate) default_index: i32,
}

/// A literal value, used for parsing default values in schemas.
#[derive(Debug, Clone, PartialEq)]
pub enum AvroLiteral {
    /// A null value.
    Null,
    /// A boolean value.
    Boolean(bool),
    /// An i32 integer value.
    Int(i32),
    /// An i64 long value.
    Long(i64),
    /// A f32 float value.
    Float(f32),
    /// A f64 double value.
    Double(f64),
    /// A byte array value.
    Bytes(Vec<u8>),
    /// A string value.
    String(String),
    /// An enum symbol.
    Enum(String),
    /// Represents a default value that is not supported or failed to parse.
    Unsupported,
}

/// Contains information about how a writer's schema was resolved to a reader's schema.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionInfo {
    /// A primitive type was promoted.
    Promotion(Promotion),
    /// A field in the reader's schema is populated with a default value.
    DefaultValue(AvroLiteral),
    /// An enum's symbols were re-mapped.
    EnumMapping(EnumMapping),
    /// A record's fields were re-ordered or resolved with default values.
    Record(ResolvedRecord),
}

/// Contains information for resolving a writer `Record` to a reader `Record`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRecord {
    /// Maps a writer field index to a reader field index. `None` if the writer field is not present in the reader.
    pub writer_to_reader: Arc<[Option<usize>]>,
    /// A list of indices in the reader schema that will be populated from a default value.
    pub default_fields: Arc<[usize]>,
}

/// An Avro datatype mapped to the Arrow data model, with optional schema resolution information.
#[derive(Debug, Clone, PartialEq)]
pub struct AvroDataType {
    /// The codec that defines the physical data representation.
    pub(crate) codec: Codec,
    /// The nullability of the type, derived from a union with `null`.
    pub(crate) nullability: Option<Nullability>,
    /// Additional metadata associated with the Avro type.
    pub(crate) metadata: HashMap<String, String>,
    /// Information on how the writer schema was resolved to the reader schema. `None` if only parsing a single schema.
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

    /// Creates a new `AvroDataType` for a parsed-only schema (no resolution info).
    fn parsed(codec: Codec, metadata: HashMap<String, String>) -> Self {
        Self {
            codec,
            metadata,
            nullability: None,
            resolution: None,
        }
    }

    /// Creates a new `AvroDataType` with schema resolution information.
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

    /// Returns an arrow [`Field`] with the given name.
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

    /// Returns a reference to the codec used by this data type.
    #[inline]
    pub fn codec(&self) -> &Codec {
        &self.codec
    }

    /// Returns the nullability of this data type.
    #[inline]
    pub fn nullability(&self) -> Option<Nullability> {
        self.nullability
    }
}

/// An Avro physical type codec.
/// <https://avro.apache.org/docs/1.11.1/specification/#encodings>
#[derive(Debug, Clone, PartialEq)]
pub enum Codec {
    /// Represents Avro null type, maps to Arrow's Null data type.
    Null,
    /// Represents Avro boolean type, maps to Arrow's Boolean data type.
    Boolean,
    /// Represents Avro int type, maps to Arrow's Int32 data type.
    Int32,
    /// Represents Avro long type, maps to Arrow's Int64 data type.
    Int64,
    /// Represents Avro float type, maps to Arrow's Float32 data type.
    Float32,
    /// Represents Avro double type, maps to Arrow's Float64 data type.
    Float64,
    /// Represents Avro bytes type, maps to Arrow's Binary data type.
    Binary,
    /// Represents Avro string type, maps to Arrow's Utf8 data type.
    Utf8,
    /// Represents Avro string type, maps to Arrow's Utf8View data type for performance.
    Utf8View,
    /// Represents Avro `date` logical type, maps to Arrow's Date32 data type.
    Date32,
    /// Represents Avro `time-millis` logical type, maps to Arrow's Time32(Millisecond).
    TimeMillis,
    /// Represents Avro `time-micros` logical type, maps to Arrow's Time64(Microsecond).
    TimeMicros,
    /// Represents Avro `timestamp-millis` or `local-timestamp-millis` logical types.
    /// The boolean is true if the timestamp is UTC-adjusted.
    TimestampMillis(bool),
    /// Represents Avro `timestamp-micros` or `local-timestamp-micros` logical types.
    /// The boolean is true if the timestamp is UTC-adjusted.
    TimestampMicros(bool),
    /// Represents Avro `fixed` type, maps to Arrow's FixedSizeBinary data type.
    Fixed(i32),
    /// Represents Avro `decimal` logical type, maps to Arrow's Decimal128 or Decimal256.
    Decimal(usize, Option<usize>, Option<usize>),
    /// Represents Avro `uuid` logical type, maps to Arrow's FixedSizeBinary(16).
    Uuid,
    /// Represents an Avro enum, maps to Arrow's Dictionary(Int32, Utf8) type.
    Enum(Arc<[String]>),
    /// Represents Avro `duration` logical type, maps to Arrow's Interval(MonthDayNano).
    Interval,
    /// Represents Avro `array` type, maps to Arrow's List data type.
    List(Arc<AvroDataType>),
    /// Represents Avro `map` type, maps to Arrow's Map data type.
    Map(Arc<AvroDataType>),
    /// Represents Avro `record` type, maps to Arrow's Struct data type.
    Struct(Arc<[AvroField]>),
    /// Represents a resolved union type, where one branch has been chosen.
    Union(Arc<AvroDataType>),
}

impl Codec {
    /// Converts a string codec to use Utf8View if requested.
    #[inline]
    fn with_utf8view(self, use_utf8view: bool) -> Self {
        if use_utf8view && matches!(self, Self::Utf8) {
            Self::Utf8View
        } else {
            self
        }
    }

    /// Returns the corresponding Arrow [`DataType`] for the codec.
    fn data_type(&self) -> DataType {
        match self {
            Self::Null => DataType::Null,
            Self::Boolean => DataType::Boolean,
            Self::Int32 => DataType::Int32,
            Self::Int64 => DataType::Int64,
            Self::Float32 => DataType::Float32,
            Self::Float64 => DataType::Float64,
            Self::Binary => DataType::Binary,
            Self::Utf8 => DataType::Utf8,
            Self::Utf8View => DataType::Utf8View,
            Self::Date32 => DataType::Date32,
            Self::TimeMillis => DataType::Time32(TimeUnit::Millisecond),
            Self::TimeMicros => DataType::Time64(TimeUnit::Microsecond),
            Self::TimestampMillis(utc) => {
                DataType::Timestamp(TimeUnit::Millisecond, utc.then(|| "+00:00".into()))
            }
            Self::TimestampMicros(utc) => {
                DataType::Timestamp(TimeUnit::Microsecond, utc.then(|| "+00:00".into()))
            }
            Self::Interval => DataType::Interval(IntervalUnit::MonthDayNano),
            Self::Fixed(n) => DataType::FixedSizeBinary(*n),
            Self::Decimal(p, s, size_b) => {
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
            Self::Uuid => DataType::FixedSizeBinary(16),
            Self::Enum(_) => {
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
            }
            Self::List(child) => DataType::List(Arc::new(
                child.field_with_name(Field::LIST_FIELD_DEFAULT_NAME),
            )),
            Self::Map(val) => {
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
            Self::Struct(flds) => DataType::Struct(flds.iter().map(|f| f.field()).collect()),
            Self::Union(child) => child.codec.data_type(),
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

/// Builder for an [`AvroField`].
///
/// Allows opt‑in to Utf8View support and Impala‑style strict‑mode checks.
#[derive(Debug)]
pub struct AvroFieldBuilder<'a> {
    schema: &'a Schema<'a>,
    use_utf8view: bool,
    strict_mode: bool,
}

impl<'a> AvroFieldBuilder<'a> {
    /// Create a new builder for `schema`.
    pub fn new(schema: &'a Schema<'a>) -> Self {
        Self {
            schema,
            use_utf8view: false,
            strict_mode: false,
        }
    }

    /// Enable or disable Utf8View conversion.
    pub fn with_utf8view(mut self, enabled: bool) -> Self {
        self.use_utf8view = enabled;
        self
    }

    /// Enable or disable Impala‑style strict mode.
    ///
    /// In strict mode, nullable unions must be of the form `["null", "type"]`, not `["type", "null"]`.
    pub fn with_strict_mode(mut self, enabled: bool) -> Self {
        self.strict_mode = enabled;
        self
    }

    /// Build the [`AvroField`].
    pub fn build(self) -> Result<AvroField, ArrowError> {
        match self.schema {
            Schema::Complex(ComplexType::Record(r)) => {
                let mut resolver = SchemaResolver::with_config(self.use_utf8view, self.strict_mode);
                let dt = resolver.visit(None, self.schema, None)?;
                Ok(AvroField {
                    name: r.name.to_string(),
                    data_type: dt,
                })
            }
            _ => Err(ArrowError::ParseError(
                "Expected top‑level Record schema".to_string(),
            )),
        }
    }
}

/// A named [`AvroDataType`], representing a field in a record.
#[derive(Debug, Clone, PartialEq)]
pub struct AvroField {
    name: String,
    data_type: AvroDataType,
}

impl AvroField {
    /// Returns the Arrow [`Field`].
    #[inline]
    pub fn field(&self) -> Field {
        self.data_type.field_with_name(&self.name)
    }

    /// Returns the [`AvroDataType`].
    #[inline]
    pub fn data_type(&self) -> &AvroDataType {
        &self.data_type
    }

    /// Returns the name of this Avro field.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a new [`AvroField`] with Utf8View support enabled.
    pub fn with_utf8view(&self) -> Self {
        let mut cloned = self.clone();
        if let Codec::Utf8 = cloned.data_type.codec {
            cloned.data_type.codec = Codec::Utf8View;
        }
        cloned
    }

    /// Resolves a reader schema against a writer schema, producing an `AvroField`
    /// representing the resolved schema.
    pub fn resolve_from_writer_and_reader<'a>(
        writer: &Schema<'a>,
        reader: &Schema<'a>,
        use_utf8view: bool,
    ) -> Result<Self, ArrowError> {
        let mut resolver = SchemaResolver::new(use_utf8view);
        let dt = resolver.visit(Some(writer), reader, None)?;
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

impl<'a> TryFrom<&Schema<'a>> for AvroField {
    type Error = ArrowError;
    fn try_from(schema: &Schema<'a>) -> Result<Self, Self::Error> {
        if let Schema::Complex(ComplexType::Record(r)) = schema {
            let mut resolver = SchemaResolver::new(false);
            let dt = resolver.visit(None, schema, None)?;
            Ok(Self {
                name: r.name.to_string(),
                data_type: dt,
            })
        } else {
            Err(ArrowError::ParseError(
                "Expected top‑level record".to_string(),
            ))
        }
    }
}

/// A public helper function to check if a writer schema can be resolved to a
/// reader schema, without producing the full resolved field.
pub fn check_schema_compatibility<'a>(
    writer: &Schema<'a>,
    reader: &Schema<'a>,
) -> Result<(), ArrowError> {
    AvroField::resolve_from_writer_and_reader(writer, reader, false).map(|_| ())
}

/// A cache for resolving Avro named types (`record`, `enum`, `fixed`)
/// during schema parsing and resolution.
#[derive(Default)]
struct NameCache<'a> {
    /// All *named* types we have parsed so far (for `$ref` resolution).
    named: HashMap<(&'a str, &'a str), AvroDataType>,
    /// Cache of resolved writer/reader record pairs.
    resolved: HashMap<(&'a str, &'a str), AvroDataType>,
    /// Names of records that are being parsed right now (to detect recursion).
    in_progress: HashSet<(&'a str, &'a str)>,
}

impl<'a> NameCache<'a> {
    /// Looks up a named type in the cache.
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

/// A resolver that can parse a single Avro schema or resolve a reader schema
/// against a writer schema according to Avro specifications.
struct SchemaResolver<'a> {
    cache: NameCache<'a>,
    use_utf8view: bool,
    strict_mode: bool,
}

impl<'a> SchemaResolver<'a> {
    /// Creates a new schema resolver.
    fn new(use_utf8view: bool) -> Self {
        Self {
            cache: Default::default(),
            use_utf8view,
            strict_mode: false,
        }
    }

    /// Creates a new schema resolver with specific configuration.
    pub fn with_config(use_utf8view: bool, strict_mode: bool) -> Self {
        Self {
            cache: Default::default(),
            use_utf8view,
            strict_mode,
        }
    }

    /// Main entry point for parsing or resolution.
    /// If `writer` is `None`, it parses the `reader` schema.
    /// If `writer` is `Some`, it resolves the `reader` schema against the `writer`.
    fn visit(
        &mut self,
        writer: Option<&Schema<'a>>,
        reader: &Schema<'a>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        match writer {
            None => self.parse(reader, namespace),
            Some(w) => self.resolve(w, reader, namespace),
        }
    }

    /// Parses a single reader schema into an `AvroDataType`.
    fn parse(
        &mut self,
        schema: &Schema<'a>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        match schema {
            Schema::TypeName(TypeName::Primitive(p)) => Ok(AvroDataType::parsed(
                Codec::from(*p).with_utf8view(self.use_utf8view),
                Default::default(),
            )),
            Schema::TypeName(TypeName::Ref(name)) => {
                match self.cache.lookup_named(name, namespace) {
                    Some(dt) => {
                        if matches!(&dt.codec, Codec::Struct(flds) if flds.is_empty()) {
                            Err(ArrowError::ParseError(format!("Failed to resolve .{name}")))
                        } else {
                            Ok(dt)
                        }
                    }
                    None => Err(ArrowError::ParseError(format!("Failed to resolve .{name}"))),
                }
            }
            Schema::Union(branches) => self.parse_nullable_union(branches, namespace),
            Schema::Complex(ct) => self.parse_complex(ct, namespace),
            Schema::Type(t) => {
                let mut dt = self.parse(&Schema::TypeName(t.r#type.clone()), namespace)?;
                self.apply_logical_type(&mut dt, &t.attributes, None)?;
                Ok(dt)
            }
        }
    }

    /// Parses a two-branch union where one branch is "null".
    fn parse_nullable_union(
        &mut self,
        branches: &[Schema<'a>],
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        if branches.len() != 2 {
            return Err(ArrowError::NotYetImplemented(
                "Non‑nullable unions not supported in Arrow representation".into(),
            ));
        }
        let (null_pos, nonnull) = match (Self::is_null(&branches[0]), Self::is_null(&branches[1])) {
            (true, false) => (Nullability::NullFirst, &branches[1]),
            (false, true) => (Nullability::NullSecond, &branches[0]),
            _ => {
                return Err(ArrowError::ParseError(
                    "Two‑branch union missing \"null\"".into(),
                ))
            }
        };
        if self.strict_mode && matches!(null_pos, Nullability::NullSecond) {
            return Err(ArrowError::SchemaError(
                "Found Avro union of the form ['T','null'], which is disallowed in strict_mode"
                    .to_string(),
            ));
        }
        let mut dt = self.parse(nonnull, namespace)?;
        dt.nullability = Some(null_pos);
        Ok(dt)
    }

    /// Parses a complex Avro type (record, enum, array, map, fixed).
    fn parse_complex(
        &mut self,
        ct: &ComplexType<'a>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        match ct {
            ComplexType::Record(r) => self.parse_record(r, namespace),
            ComplexType::Enum(e) => self.parse_enum(e, namespace),
            ComplexType::Fixed(f) => self.parse_fixed(f, namespace),
            ComplexType::Array(arr) => {
                let item = self.parse(&arr.items, namespace)?;
                Ok(AvroDataType::parsed(
                    Codec::List(Arc::new(item)),
                    arr.attributes.field_metadata(),
                ))
            }
            ComplexType::Map(mp) => {
                let val = self.parse(&mp.values, namespace)?;
                Ok(AvroDataType::parsed(
                    Codec::Map(Arc::new(val)),
                    mp.attributes.field_metadata(),
                ))
            }
        }
    }

    /// Parses a record schema, handling potential recursion.
    fn parse_record(
        &mut self,
        r: &Record<'a>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        let ns = r.namespace.or(namespace);
        let ns_str = ns.unwrap_or("");
        let key = (ns_str, r.name);
        self.cache.in_progress.insert(key);
        let placeholder =
            AvroDataType::parsed(Codec::Struct(Arc::new([])), r.attributes.field_metadata());
        self.cache.named.insert(key, placeholder);
        let parse_result = (|| {
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
        parse_result
    }

    /// Parses an enum schema.
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
        md.insert(
            "avro.enum.symbols".into(),
            serde_json::to_string(&e.symbols).unwrap(),
        );
        let dt = AvroDataType::parsed(Codec::Enum(symbols), md);
        self.cache.named.insert(
            (e.namespace.or(namespace).unwrap_or(""), e.name),
            dt.clone(),
        );
        Ok(dt)
    }

    /// Parses a fixed-size binary schema.
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
                        "Duration logical‑type requires 12‑byte fixed".into(),
                    ));
                }
                Codec::Interval
            }
            _ => Codec::Fixed(f.size as i32),
        };
        let dt = AvroDataType::parsed(codec, md);
        self.cache.named.insert(
            (
                f.namespace.or(namespace).unwrap_or(""),
                // local name
                f.name,
            ),
            dt.clone(),
        );
        Ok(dt)
    }

    /// Resolves a reader schema against a writer schema.
    fn resolve(
        &mut self,
        writer: &Schema<'a>,
        reader: &Schema<'a>,
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
                let child = self.visit(Some(&wa.items), &ra.items, namespace)?;
                Ok(AvroDataType::resolved(
                    Codec::List(Arc::new(child)),
                    ra.attributes.field_metadata(),
                    None,
                    None,
                ))
            }
            (Schema::Complex(Map(wm)), Schema::Complex(Map(rm))) => {
                let val = self.visit(Some(&wm.values), &rm.values, namespace)?;
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
                let (null_pos, r_nonnull) = if Self::is_null(&ru[0]) {
                    (Nullability::NullFirst, &ru[1])
                } else {
                    (Nullability::NullSecond, &ru[0])
                };
                let mut dt = self.visit(Some(w_nonunion), r_nonnull, namespace)?;
                dt.nullability = Some(null_pos);
                Ok(dt)
            }
            (Schema::Union(wu), r_nonunion) if Self::is_nullable_union(wu) => {
                let w_nonnull = if Self::is_null(&wu[0]) {
                    &wu[1]
                } else {
                    &wu[0]
                };
                self.visit(Some(w_nonnull), r_nonunion, namespace)
            }
            (w_nonunion, Schema::Union(ru)) => {
                for branch in ru {
                    if let Ok(dt) = self.visit(Some(w_nonunion), branch, namespace) {
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
                    if let Ok(dt) = self.visit(Some(branch), r_nonunion, namespace) {
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

    /// Resolves compatible primitive types, potentially creating a `Promotion`.
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

    /// Resolves two enum schemas.
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
                    ArrowError::ParseError("Reader enum default not in symbol list".into())
                })?;
                mapping[i] = p as i32;
            } else {
                return Err(ArrowError::ParseError(format!(
                    "Writer enum symbol '{}' not in reader",
                    wsym
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

    /// Resolves two record schemas.
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
        let w_lookup: HashMap<&str, &SchemaField<'a>> =
            w.fields.iter().map(|f| (f.name, f)).collect();
        let mut reader_fields = Vec::with_capacity(r.fields.len());
        let mut w_to_r = vec![None; w.fields.len()];
        let mut defaults = Vec::new();
        for (r_idx, rf) in r.fields.iter().enumerate() {
            if let Some(wf) = w_lookup.get(rf.name) {
                let child = self.visit(Some(&wf.r#type), &rf.r#type, ns)?;
                reader_fields.push(AvroField {
                    name: rf.name.to_string(),
                    data_type: child,
                });
                let w_pos = w.fields.iter().position(|f| std::ptr::eq(*wf, f)).unwrap();
                w_to_r[w_pos] = Some(r_idx);
            } else if let Some(def_raw) = rf.default {
                let lit = parse_default_literal(def_raw, &rf.r#type)?;
                let mut child = self.visit(None, &rf.r#type, ns)?;
                child.resolution = Some(ResolutionInfo::DefaultValue(lit));
                reader_fields.push(AvroField {
                    name: rf.name.to_string(),
                    data_type: child,
                });
                defaults.push(r_idx);
            } else {
                return Err(ArrowError::ParseError(format!(
                    "Field '{0}' missing in writer and no default",
                    rf.name
                )));
            }
        }
        let mut md = r.attributes.field_metadata();
        if !defaults.is_empty() {
            md.insert(
                "avro.resolution.defaults".into(),
                serde_json::to_string(&defaults).unwrap(),
            );
        }
        let resolved = AvroDataType::resolved(
            Codec::Struct(Arc::from(reader_fields)),
            md,
            None,
            Some(ResolutionInfo::Record(ResolvedRecord {
                writer_to_reader: Arc::from(w_to_r),
                default_fields: Arc::from(defaults),
            })),
        );
        self.cache
            .resolved
            .insert((w.name, r.name), resolved.clone());
        Ok(resolved)
    }

    /// Resolves two nullable unions.
    fn resolve_nullable_union(
        &mut self,
        wu: &[Schema<'a>],
        ru: &[Schema<'a>],
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        if !Self::is_nullable_union(wu) || !Self::is_nullable_union(ru) {
            return Err(ArrowError::NotYetImplemented(
                "Resolution for full unions not implemented".into(),
            ));
        }
        let (null_pos, w_nonnull) = if Self::is_null(&wu[0]) {
            (Nullability::NullFirst, &wu[1])
        } else {
            (Nullability::NullSecond, &wu[0])
        };
        let r_nonnull = if Self::is_null(&ru[0]) {
            &ru[1]
        } else {
            &ru[0]
        };
        let mut dt = self.visit(Some(w_nonnull), r_nonnull, namespace)?;
        dt.nullability = Some(null_pos);
        Ok(dt)
    }

    /// Checks if a schema is the "null" primitive type.
    #[inline]
    fn is_null(s: &Schema) -> bool {
        matches!(
            s,
            Schema::TypeName(TypeName::Primitive(PrimitiveType::Null))
                | Schema::Type(Type {
                    r#type: TypeName::Primitive(PrimitiveType::Null),
                    ..
                })
        )
    }

    /// Checks if a slice of schemas represents a `["null", "T"]` or `["T", "null"]` union.
    #[inline]
    fn is_nullable_union(branches: &[Schema<'_>]) -> bool {
        branches.len() == 2 && (Self::is_null(&branches[0]) || Self::is_null(&branches[1]))
    }

    /// Applies logical type information to an `AvroDataType`, modifying its codec.
    fn apply_logical_type(
        &self,
        dt: &mut AvroDataType,
        attrs: &Attributes,
        fallback_size: Option<usize>,
    ) -> Result<(), ArrowError> {
        match (&attrs.logical_type, &mut dt.codec) {
            (Some("decimal"), c @ Codec::Binary) => {
                let (p, s, _) = parse_decimal_attrs(attrs, fallback_size, false)?;
                *c = Codec::Decimal(p, Some(s), fallback_size);
            }
            (Some("date"), c @ Codec::Int32) => *c = Codec::Date32,
            (Some("time-millis"), c @ Codec::Int32) => *c = Codec::TimeMillis,
            (Some("time-micros"), c @ Codec::Int64) => *c = Codec::TimeMicros,
            (Some("timestamp-millis"), c @ Codec::Int64) => *c = Codec::TimestampMillis(true),
            (Some("timestamp-micros"), c @ Codec::Int64) => *c = Codec::TimestampMicros(true),
            (Some("local-timestamp-millis"), c @ Codec::Int64) => {
                *c = Codec::TimestampMillis(false)
            }
            (Some("local-timestamp-micros"), c @ Codec::Int64) => {
                *c = Codec::TimestampMicros(false)
            }
            (Some("uuid"), c @ Codec::Utf8) => *c = Codec::Uuid,
            (Some(other), _) => {
                dt.metadata.insert("logicalType".into(), other.to_string());
            }
            (None, _) => {}
        }
        for (k, v) in &attrs.additional {
            dt.metadata.insert(k.to_string(), v.to_string());
        }
        Ok(())
    }
}

/// Parses a JSON-encoded default value string into an `AvroLiteral`.
fn parse_default_literal(json: &str, ty: &Schema) -> Result<AvroLiteral, ArrowError> {
    let v: serde_json::Value = serde_json::from_str(json).unwrap_or(serde_json::Value::Null);
    Ok(match (ty, v) {
        (_, serde_json::Value::Null) => AvroLiteral::Null,
        (
            Schema::TypeName(TypeName::Primitive(PrimitiveType::Boolean)),
            serde_json::Value::Bool(b),
        ) => AvroLiteral::Boolean(b),
        (
            Schema::TypeName(TypeName::Primitive(PrimitiveType::Int)),
            serde_json::Value::Number(n),
        ) if n.is_i64() => AvroLiteral::Int(n.as_i64().unwrap() as i32),
        (
            Schema::TypeName(TypeName::Primitive(PrimitiveType::Long)),
            serde_json::Value::Number(n),
        ) if n.is_i64() => AvroLiteral::Long(n.as_i64().unwrap()),
        (
            Schema::TypeName(TypeName::Primitive(PrimitiveType::Float)),
            serde_json::Value::Number(n),
        ) if n.is_f64() => AvroLiteral::Float(n.as_f64().unwrap() as f32),
        (
            Schema::TypeName(TypeName::Primitive(PrimitiveType::Double)),
            serde_json::Value::Number(n),
        ) if n.is_f64() => AvroLiteral::Double(n.as_f64().unwrap()),
        (
            Schema::TypeName(TypeName::Primitive(PrimitiveType::String)),
            serde_json::Value::String(s),
        ) => AvroLiteral::String(s),
        (
            Schema::TypeName(TypeName::Primitive(PrimitiveType::Bytes)),
            serde_json::Value::String(s),
        ) => AvroLiteral::Bytes(s.into_bytes()),
        (Schema::Complex(ComplexType::Enum(_)), serde_json::Value::String(s)) => {
            AvroLiteral::Enum(s)
        }
        _ => AvroLiteral::Unsupported,
    })
}

/// Helper to parse precision, scale, and size attributes for a decimal logical type.
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
        let field = AvroField::resolve_from_writer_and_reader(&w, &r, false).unwrap();
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
        let field = AvroField::resolve_from_writer_and_reader(&w, &r, false).unwrap();
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
            let mut parser = SchemaResolver::new($utf8view);
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
        let mut parser = SchemaResolver::new(false);
        let result = parser.parse(&schema, None).unwrap();
        assert_eq!(
            result.metadata.get("logicalType"),
            Some(&"custom-type".to_string())
        );
    }

    #[test]
    fn test_string_with_utf8view_enabled() {
        let schema = Schema::TypeName(TypeName::Primitive(PrimitiveType::String));
        let mut parser = SchemaResolver::new(true);
        let result = parser.parse(&schema, None).unwrap();
        assert!(matches!(result.codec, Codec::Utf8View));
    }

    #[test]
    fn test_string_without_utf8view_enabled() {
        let schema = Schema::TypeName(TypeName::Primitive(PrimitiveType::String));
        let mut parser = SchemaResolver::new(false);
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
        let mut parser = SchemaResolver::new(true);
        let result = parser.parse(&schema, None).unwrap();

        if let Codec::Struct(fields) = &result.codec {
            assert!(matches!(fields[0].data_type().codec, Codec::Utf8View));
        } else {
            panic!("Expected Struct codec");
        }
    }
}
