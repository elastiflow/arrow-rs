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
    Array, Attributes, ComplexType, Enum, Fixed, Map, PrimitiveType, Record, RecordField, Schema,
    Type, TypeName,
};
use arrow_schema::{
    ArrowError, DataType, Field, Fields, IntervalUnit, TimeUnit, DECIMAL128_MAX_PRECISION,
    DECIMAL128_MAX_SCALE,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Avro types are not nullable, with nullability instead encoded as a union
/// where one of the variants is the null type.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Nullability {
    /// The nulls are encoded as the first union variant => `[ "null", T ]`
    NullFirst,
    /// The nulls are encoded as the second union variant => `[ T, "null" ]`
    NullSecond,
}

/// An Avro datatype mapped to the arrow data model
#[derive(Debug, Clone)]
pub struct AvroDataType {
    pub nullability: Option<Nullability>,
    pub metadata: Arc<HashMap<String, String>>,
    pub codec: Codec,
}

impl AvroDataType {
    /// Create a new AvroDataType with the given parts.
    pub fn new(
        codec: Codec,
        nullability: Option<Nullability>,
        metadata: HashMap<String, String>,
    ) -> Self {
        AvroDataType {
            codec,
            nullability,
            metadata: Arc::new(metadata),
        }
    }

    /// Create a new AvroDataType from a `Codec`, with default (no) nullability and empty metadata.
    pub fn from_codec(codec: Codec) -> Self {
        Self::new(codec, None, Default::default())
    }

    /// Returns the name of this field
    fn to_schema<'a>(&self) -> Schema<'a> {
        let metadata = Arc::try_unwrap(self.metadata.clone())
            .unwrap_or_else(|arc| (*arc).clone());
        self.codec.schema(metadata, self.nullability).unwrap()
    }

    /// Returns an arrow [`Field`] with the given name, applying `nullability` if present.
    pub fn field_with_name(&self, name: &str) -> Field {
        let is_nullable = self.nullability.is_some();
        let metadata = Arc::try_unwrap(self.metadata.clone())
            .unwrap_or_else(|arc| (*arc).clone());
        Field::new(name, self.codec.data_type(), is_nullable).with_metadata(metadata)
    }
}

/// A named [`AvroDataType`]
#[derive(Debug, Clone)]
pub struct AvroField {
    name: String,
    data_type: AvroDataType,
    default: Option<serde_json::Value>,
}

impl AvroField {
    /// Returns the arrow [`Field`]
    pub fn field(&self) -> Field {
        let mut fld = self.data_type.field_with_name(&self.name);
        if let Some(def_val) = &self.default {
            if !def_val.is_null() {
                let mut md = fld.metadata().clone();
                md.insert("avro.default".to_string(), def_val.to_string());
                fld = fld.with_metadata(md);
            }
        }
        fld
    }

    /// Returns the name of this field
    pub fn to_schema<'a>(&self) -> Schema<'a> {
        self.data_type.to_schema()
    }

    /// Returns the [`AvroDataType`]
    pub fn data_type(&self) -> &AvroDataType {
        &self.data_type
    }

    /// Returns the name of this field
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<'a> TryFrom<&Schema<'a>> for AvroField {
    type Error = ArrowError;

    fn try_from(schema: &Schema<'a>) -> Result<Self, Self::Error> {
        match schema {
            Schema::Complex(ComplexType::Record(r)) => {
                let mut resolver = Resolver::default();
                let data_type = make_data_type(schema, None, &mut resolver)?;
                Ok(Self {
                    data_type,
                    name: r.name.to_string(),
                    default: None,
                })
            }
            _ => Err(ArrowError::ParseError(format!(
                "Expected record got {schema:?}"
            ))),
        }
    }
}

/// Convert an Arrow schema to an Avro [Schema](Schema),
/// optionally using `impala=true` to produce `[ T, "null" ]` unions.
pub fn make_schema<'a>(
    arrow_schema: &'a arrow_schema::Schema,
    impala_mode: &'a bool,
) -> Result<Schema<'a>, ArrowError> {
    let record_fields = arrow_schema
        .fields()
        .iter()
        .map(|fref| make_record_field(fref, impala_mode))
        .collect::<Result<Vec<_>, _>>()?;
    let record_name = arrow_schema
        .metadata()
        .get("avro.record.name")
        .cloned()
        .unwrap_or_else(|| "topLevelRecord".to_string());
    let record_namespace = arrow_schema
        .metadata()
        .get("avro.record.namespace")
        .cloned();
    let record = Record {
        name: Box::leak(record_name.into_boxed_str()),
        namespace: record_namespace.map(|ns| {
            let leaked = Box::leak(ns.into_boxed_str());
            leaked as &'static str
        }),
        doc: None,
        aliases: vec![],
        fields: record_fields,
        attributes: Default::default(),
    };
    Ok(Schema::Complex(ComplexType::Record(record)))
}

/// Convert a single Arrow `Field` into an Avro `RecordField`, respecting `impala` union ordering.
fn make_record_field<'a>(
    field: &'a Field,
    impala_mode: &'a bool,
) -> Result<RecordField<'a>, ArrowError> {
    let nullability = if *impala_mode {
        Nullability::NullSecond
    } else {
        Nullability::NullFirst
    };
    let codec = Codec::from_field(field, nullability)?;
    let nullable = if field.is_nullable() {
        Some(nullability)
    } else {
        None
    };
    let avro_data_type = AvroDataType::new(codec, nullable, field.metadata().clone());
    // let field_name = Box::leak(field.name().clone().into_boxed_str());
    let default_val = field
        .metadata()
        .get("avro.default")
        .and_then(|s| serde_json::from_str(s).ok());
    Ok(RecordField {
        name: field.name(),
        doc: None,
        aliases: vec![],
        r#type: avro_data_type.to_schema(),
        default: default_val,
    })
}


/// An Avro encoding
#[derive(Debug, Clone)]
pub enum Codec {
    /// Primitive
    Null,
    Boolean,
    Int32,
    Int64,
    Float32,
    Float64,
    Binary,
    String,
    /// Complex
    Record(Arc<[AvroField]>),
    Enum(Arc<[String]>, Arc<[i32]>),
    Array(Arc<AvroDataType>),
    Map(Arc<AvroDataType>),
    Fixed(i32),
    /// Logical
    Decimal(usize, Option<usize>, Option<usize>),
    Uuid,
    Date32,
    TimeMillis,
    TimeMicros,
    TimestampMillis(bool),
    TimestampMicros(bool),
    Duration,
}

impl Codec {

    fn from_field(field: &Field, nullability_type: Nullability) -> Result<Self, ArrowError> {
        let metadata = field.metadata().clone();
        match field.data_type() {
            // Primitive Types
            DataType::Null => Ok(Self::Null),
            DataType::Boolean => Ok(Self::Boolean),
            DataType::Int8 | DataType::Int16 | DataType::Int32 => Ok(Self::Int32),
            DataType::Int64 => Ok(Self::Int64),
            DataType::Float32 => Ok(Self::Float32),
            DataType::Float64 => Ok(Self::Float64),
            DataType::Binary | DataType::LargeBinary => Ok(Self::Binary),
            DataType::Utf8 | DataType::LargeUtf8 => Ok(Self::String),
            // Complex Types
            DataType::Struct(fields) => {
                let avro_fields: Vec<AvroField> = fields
                    .iter()
                    .map(|fref| {
                        let child_codec = Codec::from_field(fref.as_ref(), nullability_type)?;
                        let default_val = fref
                            .metadata()
                            .get("avro.default")
                            .and_then(|s| serde_json::from_str(s).ok());
                        let nullability = if fref.is_nullable() {
                            Some(nullability_type)
                        } else {
                            None
                        };
                        Ok(AvroField {
                            name: fref.name().clone(),
                            data_type: AvroDataType::new(child_codec, nullability, fref.metadata().clone()),
                            default: default_val,
                        })
                    })
                    .collect::<Result<_, ArrowError>>()?;
                Ok(Self::Record(Arc::from(avro_fields)))
            }
            DataType::Dictionary(key_type, value_type) => {
                let valid_key = matches!(
                key_type.as_ref(),
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
            );
                let valid_val = matches!(
                value_type.as_ref(),
                DataType::Utf8 | DataType::LargeUtf8
            );
                match (valid_key && valid_val, metadata.get("avro.enum.symbols")) {
                    (false, _) => Ok(Self::String),
                    (true, None) => Ok(Self::String),
                    (true, Some(sym_json_str)) => {
                        let parsed: serde_json::Value = serde_json::from_str(sym_json_str)
                            .map_err(|e| ArrowError::ParseError(format!(
                                "Invalid JSON in avro.enum.symbols: {e}"
                            )))?;
                        if let Some(arr) = parsed.as_array() {
                            let symbols: Vec<String> = arr
                                .iter()
                                .filter_map(|v| v.as_str())
                                .map(|s| s.to_string())
                                .collect();
                            Ok(Self::Enum(Arc::from(symbols), Arc::from(vec![])))
                        } else {
                            Err(ArrowError::ParseError(
                                "Expected JSON array for avro.enum.symbols".to_string(),
                            ))
                        }
                    }
                }
            }
            DataType::List(child_field) | DataType::LargeList(child_field) => {
                let nullability = if child_field.is_nullable() {
                    Some(nullability_type)
                } else {
                    None
                };
                let child_codec = Codec::from_field(child_field.as_ref(), nullability_type)?;
                Ok(Self::Array(
                    Arc::new(AvroDataType::new(
                        child_codec,
                        nullability,
                        child_field.metadata().clone()),
                    )
                ))
            }
            DataType::FixedSizeList(child_field, _sz) => {
                let nullability = if child_field.is_nullable() {
                    Some(nullability_type)
                } else {
                    None
                };
                let child_codec = Codec::from_field(child_field.as_ref(), nullability_type)?;
                Ok(Self::Array(
                    Arc::new(AvroDataType::new(
                        child_codec,
                        nullability,
                        child_field.metadata().clone()),
                    )
                ))
            }
            DataType::Map(entry_field, _keys_sorted) => match entry_field.data_type() {
                DataType::Struct(children) if children.len() == 2 => {
                    let value_field = &children[1];
                    let nullability = if value_field.is_nullable() {
                        Some(nullability_type)
                    } else {
                        None
                    };
                    let val_codec = Codec::from_field(value_field, nullability_type)?;
                    Ok(Self::Map(Arc::new(AvroDataType::new(
                        val_codec,
                        nullability,
                        value_field.metadata().clone()),
                    )))
                }
                _ => Ok(Self::String),
            },
            DataType::FixedSizeBinary(n) => {
                let logical_type = metadata.get("logicalType").map(|s| s.as_str());
                match (*n, logical_type) {
                    (16, Some("uuid")) => Ok(Self::Uuid),
                    (12, Some("duration")) => Ok(Self::Duration),
                    _ => Ok(Self::Fixed(*n)),
                }
            }
            // Logical Types
            DataType::Interval(IntervalUnit::MonthDayNano) => Ok(Self::Duration),
            DataType::Decimal128(p, s) => {
                Ok(Self::Decimal(*p as usize, Some(*s as usize), Some(16)))
            }
            DataType::Decimal256(p, s) => {
                Ok(Self::Decimal(*p as usize, Some(*s as usize), Some(32)))
            }
            DataType::Date32 => Ok(Self::Date32),
            DataType::Time32(TimeUnit::Millisecond) => Ok(Self::TimeMillis),
            DataType::Time64(TimeUnit::Microsecond) => Ok(Self::TimeMicros),
            DataType::Timestamp(TimeUnit::Millisecond, tz_opt) => {
                let is_utc = tz_opt.as_deref() == Some("+00:00");
                Ok(Self::TimestampMillis(is_utc))
            }
            DataType::Timestamp(TimeUnit::Microsecond, tz_opt) => {
                let is_utc = tz_opt.as_deref() == Some("+00:00");
                Ok(Self::TimestampMicros(is_utc))
            }
            other => {
                Err(ArrowError::AvroError(format!(
                    "Unrecognized Avro logicalType={other}")
                ))
            }
        }
    }
    
    /// Convert this to an Arrow `DataType`
    pub(crate) fn data_type(&self) -> DataType {
        match self {
            Self::Null => DataType::Null,
            Self::Boolean => DataType::Boolean,
            Self::Int32 => DataType::Int32,
            Self::Int64 => DataType::Int64,
            Self::Float32 => DataType::Float32,
            Self::Float64 => DataType::Float64,
            Self::Binary => DataType::Binary,
            Self::String => DataType::Utf8,
            Self::Record(fields) => {
                let arrow_fields: Vec<Field> = fields.iter().map(|f| f.field()).collect();
                DataType::Struct(arrow_fields.into())
            }
            Self::Enum(_, _) => {
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
            }
            Self::Array(child_type) => {
                let child_dt = child_type.codec.data_type();
                let child_md = Arc::try_unwrap(child_type.metadata.clone())
                    .unwrap_or_else(|arc| (*arc).clone());
                let child_field =
                    Field::new(Field::LIST_FIELD_DEFAULT_NAME, child_dt, true).with_metadata(child_md);
                DataType::List(Arc::new(child_field))
            }
            Self::Map(value_type) => {
                let val_dt = value_type.codec.data_type();
                let val_md = Arc::try_unwrap(value_type.metadata.clone())
                    .unwrap_or_else(|arc| (*arc).clone());
                let val_field = Field::new("value", val_dt, true).with_metadata(val_md);
                DataType::Map(
                    Arc::new(Field::new(
                        "entries",
                        DataType::Struct(Fields::from(vec![
                            Field::new("key", DataType::Utf8, false),
                            val_field,
                        ])),
                        false,
                    )),
                    false,
                )
            }
            Self::Fixed(sz) => DataType::FixedSizeBinary(*sz),
            Self::Decimal(precision, scale, size_opt) => {
                let p = *precision as u8;
                let s = scale.unwrap_or(0) as i8;
                let too_large_for_128 = match *size_opt {
                    Some(sz) => sz > 16,
                    None => {
                        (p as usize) > DECIMAL128_MAX_PRECISION as usize
                            || (s as usize) > DECIMAL128_MAX_SCALE as usize
                    }
                };
                if too_large_for_128 {
                    DataType::Decimal256(p, s)
                } else {
                    DataType::Decimal128(p, s)
                }
            }
            Self::Uuid => DataType::FixedSizeBinary(16),
            Self::Date32 => DataType::Date32,
            Self::TimeMillis => DataType::Time32(TimeUnit::Millisecond),
            Self::TimeMicros => DataType::Time64(TimeUnit::Microsecond),
            Self::TimestampMillis(is_utc) => DataType::Timestamp(
                TimeUnit::Millisecond,
                is_utc.then(|| "+00:00".into()),
            ),
            Self::TimestampMicros(is_utc) => DataType::Timestamp(
                TimeUnit::Microsecond,
                is_utc.then(|| "+00:00".into()),
            ),
            Self::Duration => DataType::Interval(IntervalUnit::MonthDayNano),
        }
    }
    pub(crate) fn schema<'a>(
        &self,
        metadata: HashMap<String, String>,
        nullability: Option<Nullability>,
    ) -> Result<Schema<'a>, ArrowError> {
        let base = match self {
            Self::Null => Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)),
            Self::Boolean => Schema::TypeName(TypeName::Primitive(PrimitiveType::Boolean)),
            Self::Int32 => Schema::TypeName(TypeName::Primitive(PrimitiveType::Int)),
            Self::Int64 => Schema::TypeName(TypeName::Primitive(PrimitiveType::Long)),
            Self::Float32 => Schema::TypeName(TypeName::Primitive(PrimitiveType::Float)),
            Self::Float64 => Schema::TypeName(TypeName::Primitive(PrimitiveType::Double)),
            Self::Binary => Schema::TypeName(TypeName::Primitive(PrimitiveType::Bytes)),
            Self::String => Schema::TypeName(TypeName::Primitive(PrimitiveType::String)),
            Self::Record(fields) => {
                let record_fields = fields
                    .iter()
                    .map(|field| RecordField {
                        name: Box::leak(field.name().to_string().into_boxed_str()),
                        doc: None,
                        aliases: vec![],
                        r#type: field.data_type.to_schema(),
                        default: field.default.clone(),
                    })
                    .collect::<Vec<_>>();
                // TODO: Make metadata part of lifetime
                let record_name = metadata
                    .get("avro.record.name")
                    .cloned()
                    .unwrap_or_else(|| "record".to_string());
                let record_namespace = metadata
                    .get("avro.record.namespace")
                    .cloned();
                let mut attributes = Attributes::default();
                copy_metadata_to_attributes(&metadata, &mut attributes);
                Schema::Complex(ComplexType::Record(Record {
                    name: Box::leak(record_name.into_boxed_str()),
                    namespace: record_namespace.map(|ns| {
                        let leaked = Box::leak(ns.into_boxed_str());
                        leaked as &'a str
                    }),
                    doc: None,
                    aliases: vec![],
                    fields: record_fields,
                    attributes,
                }))
            }
            Self::Enum(symbols, _ordinals) => {
                let enum_name = metadata
                    .get("avro.enum.name")
                    .cloned()
                    .unwrap_or_else(|| "enum".to_string());
                let enum_namespace = metadata
                    .get("avro.enum.namespace")
                    .cloned();
                let mut attributes = Attributes::default();
                copy_metadata_to_attributes(&metadata, &mut attributes);
                let mut leaked_syms = Vec::with_capacity(symbols.len());
                for sym in symbols.iter() {
                    let leaked: &'a str = Box::leak(sym.clone().into_boxed_str());
                    leaked_syms.push(leaked);
                }
                Schema::Complex(ComplexType::Enum(Enum {
                    name: Box::leak(enum_name.into_boxed_str()),
                    namespace: enum_namespace.map(|ns| {
                        let leaked = Box::leak(ns.into_boxed_str());
                        leaked as &'a str
                    }),
                    doc: None,
                    aliases: vec![],
                    symbols: leaked_syms,
                    default: None,
                    attributes,
                }))
            }
            Self::Array(child) => {
                let items_schema = child.to_schema();
                let mut attributes = Attributes::default();
                copy_metadata_to_attributes(&metadata, &mut attributes);
                Schema::Complex(ComplexType::Array(Array {
                    items: Box::new(items_schema),
                    attributes,
                }))
            }
            Self::Map(value_type) => {
                let value_schema = value_type.to_schema();
                let mut attributes = Attributes::default();
                copy_metadata_to_attributes(&metadata, &mut attributes);
                Schema::Complex(ComplexType::Map(Map {
                    values: Box::new(value_schema),
                    attributes,
                }))
            }
            Self::Fixed(size) => {
                let fixed_name = metadata
                    .get("avro.fixed.name")
                    .cloned()
                    .unwrap_or_else(|| format!("fixed_{size}"));
                let fixed_namespace = metadata
                    .get("avro.fixed.namespace")
                    .cloned();
                let mut attributes = Attributes::default();
                copy_metadata_to_attributes(&metadata, &mut attributes);
                Schema::Complex(ComplexType::Fixed(Fixed {
                    name: Box::leak(fixed_name.into_boxed_str()),
                    namespace: fixed_namespace.map(|ns| {
                        let leaked = Box::leak(ns.into_boxed_str());
                        leaked as &'a str
                    }),
                    aliases: vec![],
                    size: *size as usize,
                    attributes,
                }))
            }
            Self::Decimal(precision, scale, size_opt) => {
                let p = *precision;
                let s = scale.unwrap_or(0);
                let mut attrs = Attributes {
                    logical_type: Some("decimal"),
                    additional: HashMap::from([
                        ("precision", serde_json::Value::Number(p.into())),
                        ("scale", serde_json::Value::Number(s.into())),
                    ]),
                };
                copy_metadata_to_attributes(&metadata, &mut attrs);
                if let Some(size) = size_opt {
                    let fixed_name = metadata
                        .get("avro.fixed.name")
                        .cloned()
                        .unwrap_or_else(|| format!("decimal_fixed_{size}_{p}_{s}"));
                    let fixed_namespace = metadata
                        .get("avro.fixed.namespace")
                        .cloned();
                    Schema::Complex(ComplexType::Fixed(Fixed {
                        name: Box::leak(fixed_name.into_boxed_str()),
                        namespace: fixed_namespace.map(|ns| {
                            let leaked = Box::leak(ns.into_boxed_str());
                            leaked as &'a str
                        }),
                        aliases: vec![],
                        size: *size,
                        attributes: attrs,
                    }))
                } else {
                    Schema::Type(Type {
                        r#type: TypeName::Primitive(PrimitiveType::Bytes),
                        attributes: attrs,
                    })
                }
            }
            Self::Uuid => {
                let mut attrs = Attributes::default();
                attrs.logical_type = Some("uuid");
                copy_metadata_to_attributes(&metadata, &mut attrs);
                let fixed_name = metadata
                    .get("avro.fixed.name")
                    .cloned()
                    .unwrap_or_else(|| "fixed_16_uuid".to_string());
                let fixed_namespace = metadata
                    .get("avro.fixed.namespace")
                    .cloned();
                Schema::Complex(ComplexType::Fixed(Fixed {
                    name: Box::leak(fixed_name.into_boxed_str()),
                    namespace: fixed_namespace.map(|ns| {
                        let leaked = Box::leak(ns.into_boxed_str());
                        leaked as &'a str
                    }),
                    aliases: vec![],
                    size: 16,
                    attributes: attrs,
                }))
            }
            Self::Date32 => {
                let mut attrs = Attributes::default();
                attrs.logical_type = Some("date");
                copy_metadata_to_attributes(&metadata, &mut attrs);
                Schema::Type(Type {
                    r#type: TypeName::Primitive(PrimitiveType::Int),
                    attributes: attrs,
                })
            }
            Self::TimeMillis => {
                let mut attrs = Attributes::default();
                attrs.logical_type = Some("time-millis");
                copy_metadata_to_attributes(&metadata, &mut attrs);
                Schema::Type(Type {
                    r#type: TypeName::Primitive(PrimitiveType::Int),
                    attributes: attrs,
                })
            }
            Self::TimeMicros => {
                let mut attrs = Attributes::default();
                attrs.logical_type = Some("time-micros");
                copy_metadata_to_attributes(&metadata, &mut attrs);
                Schema::Type(Type {
                    r#type: TypeName::Primitive(PrimitiveType::Long),
                    attributes: attrs,
                })
            }
            Self::TimestampMillis(is_utc) => {
                let mut attrs = Attributes::default();
                let lt = if *is_utc {
                    "timestamp-millis"
                } else {
                    "local-timestamp-millis"
                };
                attrs.logical_type = Some(lt);
                copy_metadata_to_attributes(&metadata, &mut attrs);
                Schema::Type(Type {
                    r#type: TypeName::Primitive(PrimitiveType::Long),
                    attributes: attrs,
                })
            }
            Self::TimestampMicros(is_utc) => {
                let mut attrs = Attributes::default();
                let lt = if *is_utc {
                    "timestamp-micros"
                } else {
                    "local-timestamp-micros"
                };
                attrs.logical_type = Some(lt);
                copy_metadata_to_attributes(&metadata, &mut attrs);
                Schema::Type(Type {
                    r#type: TypeName::Primitive(PrimitiveType::Long),
                    attributes: attrs,
                })
            }
            Self::Duration => {
                let mut attrs = Attributes::default();
                attrs.logical_type = Some("duration");
                copy_metadata_to_attributes(&metadata, &mut attrs);
                let fixed_name = metadata
                    .get("avro.fixed.name")
                    .cloned()
                    .unwrap_or_else(|| "fixed_12_duration".to_string());
                let fixed_namespace = metadata
                    .get("avro.fixed.namespace")
                    .cloned();
                Schema::Complex(ComplexType::Fixed(Fixed {
                    name: Box::leak(fixed_name.into_boxed_str()),
                    namespace: fixed_namespace.map(|ns| {
                        let leaked = Box::leak(ns.into_boxed_str());
                        leaked as &'a str
                    }),
                    aliases: vec![],
                    size: 12,
                    attributes: attrs,
                }))
            }
        };
        if let Some(nul) = nullability {
            let union = match nul {
                Nullability::NullFirst => vec![
                    Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)),
                    base,
                ],
                Nullability::NullSecond => vec![
                    base,
                    Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)),
                ],
            };
            Ok(Schema::Union(union))
        } else {
            Ok(base)
        }
    }
}

impl From<PrimitiveType> for Codec {
    fn from(value: PrimitiveType) -> Self {
        match value {
            PrimitiveType::Null => Self::Null,
            PrimitiveType::Boolean => Self::Boolean,
            PrimitiveType::Int => Self::Int32,
            PrimitiveType::Long => Self::Int64,
            PrimitiveType::Float => Self::Float32,
            PrimitiveType::Double => Self::Float64,
            PrimitiveType::Bytes => Self::Binary,
            PrimitiveType::String => Self::String,
        }
    }
}

/// Resolves Avro type names to [`AvroDataType`]
#[derive(Default, Debug)]
struct Resolver<'a> {
    map: HashMap<(&'a str, &'a str), AvroDataType>,
}

impl<'a> Resolver<'a> {
    fn register(&mut self, name: &'a str, namespace: Option<&'a str>, dt: AvroDataType) {
        let ns = namespace.unwrap_or("");
        self.map.insert((name, ns), dt);
    }

    fn resolve(
        &self,
        full_name: &str,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        let (ns, nm) = match full_name.rsplit_once('.') {
            Some((a, b)) => (a, b),
            None => (namespace.unwrap_or(""), full_name),
        };
        self.map
            .get(&(nm, ns))
            .cloned()
            .ok_or_else(|| ArrowError::ParseError(format!("Failed to resolve {ns}.{nm}")))
    }
}

/// Parses a [`AvroDataType`] from the provided [`Schema`], plus optional `namespace`.
fn make_data_type<'a>(
    schema: &Schema<'a>,
    namespace: Option<&'a str>,
    resolver: &mut Resolver<'a>,
) -> Result<AvroDataType, ArrowError> {
    match schema {
        Schema::TypeName(TypeName::Primitive(p)) => Ok(AvroDataType {
            nullability: None,
            metadata: Arc::new(Default::default()),
            codec: (*p).into(),
        }),
        Schema::TypeName(TypeName::Ref(name)) => resolver.resolve(name, namespace),
        Schema::Union(u) => {
            let null_count = u
                .iter()
                .filter(|x| *x == &Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)))
                .count();
            if null_count == 1 && u.len() == 2 {
                let null_idx = u
                    .iter()
                    .position(|x| x == &Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)))
                    .unwrap();
                let other_idx = if null_idx == 0 { 1 } else { 0 };
                let mut dt = make_data_type(&u[other_idx], namespace, resolver)?;
                dt.nullability = if null_idx == 0 {
                    Some(Nullability::NullFirst)
                } else {
                    Some(Nullability::NullSecond)
                };
                Ok(dt)
            } else {
                Err(ArrowError::NotYetImplemented(format!(
                    "Union of {u:?} not currently supported"
                )))
            }
        }
        Schema::Complex(c) => match c {
            ComplexType::Record(r) => {
                let ns = r.namespace.or(namespace);
                let fields = r
                    .fields
                    .iter()
                    .map(|f| {
                        let data_type = make_data_type(&f.r#type, ns, resolver)?;
                        Ok::<AvroField, ArrowError>(AvroField {
                            name: f.name.to_string(),
                            data_type,
                            default: f.default.clone(),
                        })
                    })
                    .collect::<Result<Vec<AvroField>, ArrowError>>()?;
                let rec_dt = AvroDataType {
                    nullability: None,
                    metadata: Arc::new(r.attributes.field_metadata()),
                    codec: Codec::Record(Arc::from(fields)),
                };
                resolver.register(r.name, ns, rec_dt.clone());
                Ok(rec_dt)
            }
            ComplexType::Enum(e) => {
                // Insert "avro.enum.symbols" into metadata so we can preserve it.
                let mut md = e.attributes.field_metadata();
                if let Ok(symbols_json) = serde_json::to_string(&e.symbols) {
                    md.insert("avro.enum.symbols".to_string(), symbols_json);
                }
                let en = AvroDataType {
                    nullability: None,
                    metadata: Arc::new(md),
                    codec: Codec::Enum(
                        Arc::from(
                            e.symbols
                                .iter()
                                .map(|s| s.to_string())
                                .collect::<Vec<_>>(),
                        ),
                        Arc::from(vec![]),
                    ),
                };
                resolver.register(e.name, namespace, en.clone());
                Ok(en)
            }
            ComplexType::Array(a) => {
                let child = make_data_type(&a.items, namespace, resolver)?;
                Ok(AvroDataType {
                    nullability: None,
                    metadata: Arc::new(a.attributes.field_metadata()),
                    codec: Codec::Array(Arc::new(child)),
                })
            }
            ComplexType::Map(m) => {
                let val = make_data_type(&m.values, namespace, resolver)?;
                Ok(AvroDataType {
                    nullability: None,
                    metadata: Arc::new(m.attributes.field_metadata()),
                    codec: Codec::Map(Arc::new(val)),
                })
            }
            ComplexType::Fixed(fx) => {
                let size = fx.size as i32;
                let md = Arc::new(fx.attributes.field_metadata());
                let dt = match fx.attributes.logical_type.as_deref() {
                    Some("decimal") => {
                        let (precision, scale, _) =
                            parse_decimal_attributes(&fx.attributes, Some(size as usize), true)?;
                        AvroDataType {
                            nullability: None,
                            metadata: md,
                            codec: Codec::Decimal(precision, Some(scale), Some(size as usize)),
                        }
                    }
                    Some("duration") if fx.size == 12 => AvroDataType {
                        nullability: None,
                        metadata: md,
                        codec: Codec::Duration,
                    },
                    Some("uuid") if fx.size == 16 => AvroDataType {
                        nullability: None,
                        metadata: md,
                        codec: Codec::Uuid,
                    },
                    _ => fixed_fallback(md, size),
                };
                resolver.register(fx.name, namespace, dt.clone());
                Ok(dt)
            }
        },
        Schema::Type(t) => {
            let mut dt =
                make_data_type(&Schema::TypeName(t.r#type.clone()), namespace, resolver)?;
            match (t.attributes.logical_type, &mut dt.codec) {
                (Some("decimal"), Codec::Fixed(size)) => {
                    let (precision, scale, size_opt) =
                        parse_decimal_attributes(&t.attributes, Some(*size as usize), false)?;
                    if let Some(sz_actual) = size_opt {
                        *size = sz_actual as i32;
                    }
                    dt.codec = Codec::Decimal(precision, Some(scale), Some(*size as usize));
                }
                (Some("decimal"), Codec::Binary) => {
                    let (precision, scale, _) = parse_decimal_attributes(&t.attributes, None, false)?;
                    dt.codec = Codec::Decimal(precision, Some(scale), None);
                }
                (Some("uuid"), Codec::String) => {
                    dt.codec = Codec::Uuid;
                }
                (Some("date"), Codec::Int32) => {
                    dt.codec = Codec::Date32;
                }
                (Some("time-millis"), Codec::Int32) => {
                    dt.codec = Codec::TimeMillis;
                }
                (Some("time-micros"), Codec::Int64) => {
                    dt.codec = Codec::TimeMicros;
                }
                (Some("timestamp-millis"), Codec::Int64) => {
                    dt.codec = Codec::TimestampMillis(true);
                }
                (Some("timestamp-micros"), Codec::Int64) => {
                    dt.codec = Codec::TimestampMicros(true);
                }
                (Some("local-timestamp-millis"), Codec::Int64) => {
                    dt.codec = Codec::TimestampMillis(false);
                }
                (Some("local-timestamp-micros"), Codec::Int64) => {
                    dt.codec = Codec::TimestampMicros(false);
                }
                (Some("duration"), Codec::Fixed(12)) => {
                    dt.codec = Codec::Duration;
                }
                (Some(other), _) => {
                    if !dt.metadata.contains_key("logicalType") {
                        let mut map = (*dt.metadata).clone();
                        map.insert("logicalType".into(), other.into());
                        dt.metadata = Arc::new(map);
                    }
                }
                (None, _) => {}
            }
            for (k, v) in &t.attributes.additional {
                let mut map = (*dt.metadata).clone();
                map.insert(k.to_string(), v.to_string());
                dt.metadata = Arc::new(map);
            }
            Ok(dt)
        }
    }
}

fn fixed_fallback(md: Arc<HashMap<String, String>>, size: i32) -> AvroDataType {
    AvroDataType {
        nullability: None,
        metadata: md,
        codec: Codec::Fixed(size),
    }
}

fn copy_metadata_to_attributes(
    source: &HashMap<String, String>,
    target: &mut Attributes,
) {
    for (k, v) in source {
        if k == "type"
            || k == "name"
            || k == "fields"
            || k == "aliases"
            || k == "namespace"
            || k == "doc"
            || k.starts_with("avro.")
        {
            continue;
        }
        // For "precision" or "scale", try parsing as an integer.
        let maybe_parsed_value = if k == "precision" || k == "scale" {
            match v.parse::<i64>() {
                Ok(parsed_int) => serde_json::Value::Number(parsed_int.into()),
                Err(_) => serde_json::Value::String(v.clone()),
            }
        } else {
            serde_json::Value::String(v.clone())
        };
        target.additional.insert(
            Box::leak(k.clone().into_boxed_str()),
            maybe_parsed_value,
        );
    }
}

fn parse_decimal_attributes(
    attributes: &Attributes,
    fallback_size: Option<usize>,
    precision_required: bool,
) -> Result<(usize, usize, Option<usize>), ArrowError> {
    let precision = attributes
        .additional
        .get("precision")
        .and_then(|v| v.as_u64())
        .or(if precision_required { None } else { Some(10) })
        .ok_or_else(|| ArrowError::ParseError("Decimal requires precision".to_string()))?
        as usize;
    let scale = attributes
        .additional
        .get("scale")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let size = attributes
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
    use crate::schema::{Schema, ComplexType};
    use arrow_schema::{ArrowError, DataType, Field, TimeUnit, Schema as ArrowSchema};
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn arrow_schema_round_trip(schema: &ArrowSchema) -> Result<AvroDataType, ArrowError> {
        let avro_schema = make_schema(schema, &false)?;
        let mut resolver = Resolver::default();
        let avro_dt = make_data_type(&avro_schema, None, &mut resolver)?;
        Ok(avro_dt)
    }

    fn single_field_codec(avro_dt: &AvroDataType) -> &Codec {
        match &avro_dt.codec {
            Codec::Record(fields) => {
                if fields.len() != 1 {
                    panic!("Expected exactly 1 field in record, got {}", fields.len());
                }
                &fields[0].data_type().codec
            }
            other => panic!("Expected top-level record, got {other:?}"),
        }
    }

    #[test]
    fn test_field_to_schema_uuid() {
        let mut md = HashMap::new();
        md.insert("logicalType".to_string(), "uuid".to_string());
        let arrow_field =
            Field::new("uuid_col", DataType::FixedSizeBinary(16), false).with_metadata(md);
        let arrow_schema = ArrowSchema::new(vec![arrow_field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema).unwrap();
        match &avro_dt.codec {
            Codec::Record(fields) => {
                assert_eq!(fields.len(), 1);
                let col = &fields[0];
                assert_eq!(col.name(), "uuid_col");
                match col.data_type().codec {
                    Codec::Uuid => {}
                    ref other => panic!("Expected Codec::Uuid, got {other:?}"),
                }
            }
            ref other => panic!("Expected top-level record, got {other:?}"),
        }
    }

    #[test]
    fn test_field_to_schema_duration() -> Result<(), ArrowError> {
        let mut md = HashMap::new();
        md.insert("logicalType".to_string(), "duration".to_string());
        let arrow_field = Field::new("duration_col", DataType::FixedSizeBinary(12), true)
            .with_metadata(md);
        let arrow_schema = ArrowSchema::new(vec![arrow_field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let f0 = match &avro_dt.codec {
            Codec::Record(fields) => &fields[0],
            other => panic!("Expected record, got {other:?}"),
        };
        assert_eq!(f0.name(), "duration_col");
        match f0.data_type().codec {
            Codec::Duration => {}
            ref other => panic!("Expected Codec::Duration, got {other:?}"),
        };
        assert_eq!(f0.data_type().nullability, Some(Nullability::NullFirst));
        Ok(())
    }

    #[test]
    fn test_field_to_schema_enum_dictionary_with_symbols() -> Result<(), ArrowError> {
        let mut md = HashMap::new();
        md.insert("avro.enum.symbols".to_string(), r#"["RED","GREEN","BLUE"]"#.to_string());
        let dict_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let arrow_field = Field::new("enum_col", dict_type, false).with_metadata(md);
        let arrow_schema = ArrowSchema::new(vec![arrow_field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let codec = single_field_codec(&avro_dt);
        match codec {
            Codec::Enum(symbols, _defaults) => {
                assert_eq!(symbols.len(), 3);
                assert_eq!(symbols[0], "RED");
                assert_eq!(symbols[1], "GREEN");
                assert_eq!(symbols[2], "BLUE");
            }
            other => panic!("Expected Codec::Enum, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_field_to_schema_enum_dictionary_no_symbols() -> Result<(), ArrowError> {
        let dict_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let arrow_field = Field::new("maybe_enum_col", dict_type, true);
        let arrow_schema = ArrowSchema::new(vec![arrow_field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let codec = single_field_codec(&avro_dt);
        assert!(matches!(codec, Codec::String));
        Ok(())
    }

    #[test]
    fn test_field_to_schema_date32() -> Result<(), ArrowError> {
        let arrow_field = Field::new("d32", DataType::Date32, false);
        let arrow_schema = ArrowSchema::new(vec![arrow_field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let codec = single_field_codec(&avro_dt);
        match codec {
            Codec::Date32 => {}
            other => panic!("Expected Codec::Date32, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_field_to_schema_time_millis() -> Result<(), ArrowError> {
        let arrow_field = Field::new("tmillis", DataType::Time32(TimeUnit::Millisecond), true);
        let arrow_schema = ArrowSchema::new(vec![arrow_field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let f0_codec = single_field_codec(&avro_dt);
        match f0_codec {
            Codec::TimeMillis => {}
            other => panic!("Expected Codec::TimeMillis, got {other:?}"),
        }
        let f0 = match &avro_dt.codec {
            Codec::Record(fields) => &fields[0],
            other => panic!("Expected record, got {other:?}"),
        };
        assert_eq!(f0.data_type().nullability, Some(Nullability::NullFirst));
        Ok(())
    }

    #[test]
    fn test_field_to_schema_time_micros() -> Result<(), ArrowError> {
        let arrow_field = Field::new("tmicros", DataType::Time64(TimeUnit::Microsecond), false);
        let arrow_schema = ArrowSchema::new(vec![arrow_field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let codec = single_field_codec(&avro_dt);
        match codec {
            Codec::TimeMicros => {}
            other => panic!("Expected Codec::TimeMicros, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_field_to_schema_timestamp_millis_utc() -> Result<(), ArrowError> {
        let arrow_field = Field::new(
            "tsmillis_utc",
            DataType::Timestamp(TimeUnit::Millisecond, Some(Arc::from("+00:00"))),
            true,
        );
        let arrow_schema = ArrowSchema::new(vec![arrow_field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let f0_codec = single_field_codec(&avro_dt);
        match f0_codec {
            Codec::TimestampMillis(is_utc) => assert!(*is_utc),
            other => panic!("Expected Codec::TimestampMillis(true), got {other:?}"),
        }
        let f0 = match &avro_dt.codec {
            Codec::Record(fields) => &fields[0],
            _ => panic!("Expected record"),
        };
        assert_eq!(f0.data_type().nullability, Some(Nullability::NullFirst));
        Ok(())
    }

    #[test]
    fn test_field_to_schema_timestamp_micros_local() -> Result<(), ArrowError> {
        let arrow_field = Field::new(
            "tsmicros_local",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        );
        let arrow_schema = ArrowSchema::new(vec![arrow_field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let codec = single_field_codec(&avro_dt);
        match codec {
            Codec::TimestampMicros(is_utc) => assert!(!*is_utc),
            other => panic!("Expected Codec::TimestampMicros(false), got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_arrow_map_to_avro_schema() -> Result<(), ArrowError> {
        let key_field = Field::new("key", DataType::Utf8, false);
        let value_field = Field::new("value", DataType::Int32, true);
        let entries_struct_field = Field::new(
            "entries",
            DataType::Struct(vec![key_field.clone(), value_field.clone()].into()),
            false,
        );
        let map_field = Field::new("my_map", DataType::Map(Arc::new(entries_struct_field), false), true);
        let arrow_schema = ArrowSchema::new(vec![map_field.clone()]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let f0 = match &avro_dt.codec {
            Codec::Record(fields) => &fields[0],
            other => panic!("Expected a record, got {other:?}"),
        };
        match f0.data_type().codec {
            Codec::Map(ref val_type) => {
                match val_type.codec {
                    Codec::Int32 => {}
                    ref other => panic!("Expected int or union, got {other:?}"),
                }
            }
            ref other => panic!("Expected a Map codec, got {other:?}"),
        }
        let avro_sch = make_schema(&arrow_schema, &false)?;
        if let Schema::Complex(ComplexType::Record(r)) = avro_sch {
            assert_eq!(r.fields.len(), 1);
            let top_f = &r.fields[0];
            if let Schema::Union(u) = &top_f.r#type {
                assert_eq!(u.len(), 2);
            } else {
                panic!("Expected union for a nullable field");
            }
        } else {
            panic!("Expected record");
        }
        Ok(())
    }

    #[test]
    fn test_avro_map_round_trip() -> Result<(), ArrowError> {
        let key_field = Field::new("key", DataType::Utf8, false);
        let value_field = Field::new("value", DataType::Int32, false);
        let entries_struct = Field::new(
            "entries",
            DataType::Struct(vec![key_field, value_field].into()),
            false,
        );
        let map_field = Field::new(
            "example_map",
            DataType::Map(Arc::new(entries_struct), false),
            false,
        );
        let arrow_schema = ArrowSchema::new(vec![map_field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let f0 = match &avro_dt.codec {
            Codec::Record(fields) => &fields[0],
            other => panic!("Expected a record, got {other:?}"),
        };
        match &f0.data_type().codec {
            Codec::Map(val_type) => match val_type.codec {
                Codec::Int32 => { /* as expected */ }
                ref other => panic!("Unexpected map value type: {:?}", other),
            },
            other => panic!("Expected map codec, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_field_to_schema_list_of_int() -> Result<(), ArrowError> {
        let item_field = Field::new("item", DataType::Int32, false);
        let list_field = Field::new("list_col", DataType::List(Arc::new(item_field)), false);
        let arrow_schema = ArrowSchema::new(vec![list_field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let child_codec = match &avro_dt.codec {
            Codec::Record(fields) => &fields[0].data_type().codec,
            other => panic!("Expected record, got {other:?}"),
        };
        match child_codec {
            Codec::Array(child_at) => match child_at.codec {
                Codec::Int32 => {}
                ref other => panic!("Expected child=Int32, got {other:?}"),
            },
            other => panic!("Expected Codec::Array(...), got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_field_to_schema_fixedsizelist_of_strings_nullable() {
        let item_field = Field::new("sub", DataType::Utf8, true);
        let fsl_field = Field::new("fsl_col", DataType::FixedSizeList(Arc::new(item_field), 2), true);
        let arrow_schema = ArrowSchema::new(vec![fsl_field]);
        let avro_sch = make_schema(&arrow_schema, &false).unwrap();
        let mut resolver = Resolver::default();
        let dt = make_data_type(&avro_sch, None, &mut resolver).unwrap();
        match &dt.codec {
            Codec::Record(fields) => {
                assert_eq!(fields.len(), 1);
                let child_dt = &fields[0].data_type().codec;
                match child_dt {
                    Codec::Array(child2) => {
                        assert!(matches!(child2.codec, Codec::String | Codec::Record(_)));
                    }
                    other => panic!("Expected array for fixedSizeList => {other:?}"),
                }
            }
            other => panic!("Expected record => {other:?}"),
        }
    }

    #[test]
    fn test_field_to_schema_record_simple() {
        let child_a = Field::new("child_a", DataType::Int32, false);
        let mut md_b = HashMap::new();
        md_b.insert("avro.default".to_string(), "true".to_string());
        let child_b = Field::new("child_b", DataType::Boolean, false).with_metadata(md_b);
        let struct_type = DataType::Struct(vec![child_a.clone(), child_b.clone()].into());
        let top_field = Field::new("my_struct", struct_type, false);
        let arrow_schema = ArrowSchema::new(vec![top_field]);
        let avro_sch = make_schema(&arrow_schema, &false).unwrap();
        let mut resolver = Resolver::default();
        let dt = make_data_type(&avro_sch, None, &mut resolver).unwrap();
        match &dt.codec {
            Codec::Record(fields) => {
                assert_eq!(fields.len(), 1);
                let struct_avro = &fields[0];
                assert_eq!(struct_avro.name(), "my_struct");
                match &struct_avro.data_type().codec {
                    Codec::Record(child_fields) => {
                        assert_eq!(child_fields.len(), 2);
                        assert_eq!(child_fields[0].name(), "child_a");
                        match child_fields[0].data_type().codec {
                            Codec::Int32 => {}
                            ref other => panic!("Expected Int32 for child_a => {other:?}"),
                        }
                        assert_eq!(child_fields[1].name(), "child_b");
                        match child_fields[1].data_type().codec {
                            Codec::Boolean => {}
                            ref other => panic!("Expected Boolean for child_b => {other:?}"),
                        }
                        if let Some(def_val) = &child_fields[1].default {
                            assert_eq!(def_val, &json!(true));
                        } else {
                            panic!("Expected default=true for child_b");
                        }
                    }
                    ref other => panic!("Expected inner Codec::Record => {other:?}"),
                }
            }
            other => panic!("Expected top-level record => {other:?}"),
        }
    }

    #[test]
    fn test_decimal_arrow_field_to_schema() -> Result<(), ArrowError> {
        let field = Field::new("decimal_col", DataType::Decimal128(10, 2), false);
        let arrow_schema = ArrowSchema::new(vec![field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let f0 = match &avro_dt.codec {
            Codec::Record(fields) => &fields[0],
            other => panic!("Expected record, got {other:?}"),
        };
        match f0.data_type().codec {
            Codec::Decimal(prec, sc, sz) => {
                assert_eq!(prec, 10);
                assert_eq!(sc, Some(2));
                assert_eq!(sz, Some(16), "Default for decimal128 => 16 bytes");
            }
            ref other => panic!("Expected decimal, got {other:?}"),
        }
        let avro_sch = make_schema(&arrow_schema, &false)?;
        match avro_sch {
            Schema::Complex(ComplexType::Record(r)) => {
                assert_eq!(r.fields.len(), 1);
                let df = &r.fields[0];
                match &df.r#type {
                    Schema::Complex(ComplexType::Fixed(fx)) => {
                        assert_eq!(fx.size, 16);
                        let lt = fx.attributes.logical_type;
                        assert_eq!(lt, Some("decimal"));
                        let extra = &fx.attributes.additional;
                        let prec_val = extra.get("precision").unwrap();
                        let scale_val = extra.get("scale").unwrap();
                        assert_eq!(prec_val, &serde_json::Value::Number(10.into()));
                        assert_eq!(scale_val, &serde_json::Value::Number(2.into()));
                    }
                    _ => panic!("Expected a fixed decimal schema"),
                }
            }
            other => panic!("Expected a record, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_field_to_schema_boolean() -> Result<(), ArrowError> {
        let field = Field::new("bool_col", DataType::Boolean, false);
        let arrow_schema = ArrowSchema::new(vec![field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let f0_codec = single_field_codec(&avro_dt);
        assert!(matches!(f0_codec, Codec::Boolean));
        Ok(())
    }

    #[test]
    fn test_field_to_schema_int32() -> Result<(), ArrowError> {
        let field = Field::new("int_col", DataType::Int32, true);
        let arrow_schema = ArrowSchema::new(vec![field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let f0 = match &avro_dt.codec {
            Codec::Record(fields) => &fields[0],
            _ => panic!("Expected record"),
        };
        match f0.data_type().codec {
            Codec::Int32 => {}
            ref other => panic!("Expected Codec::Int32, got {other:?}"),
        }
        assert_eq!(f0.data_type().nullability, Some(Nullability::NullFirst));
        Ok(())
    }

    #[test]
    fn test_field_to_schema_int64() -> Result<(), ArrowError> {
        let field = Field::new("long_col", DataType::Int64, false);
        let arrow_schema = ArrowSchema::new(vec![field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let c0 = single_field_codec(&avro_dt);
        assert!(matches!(c0, Codec::Int64));
        Ok(())
    }

    #[test]
    fn test_field_to_schema_float32() -> Result<(), ArrowError> {
        let field = Field::new("float_col", DataType::Float32, false);
        let arrow_schema = ArrowSchema::new(vec![field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        match single_field_codec(&avro_dt) {
            Codec::Float32 => {}
            ref other => panic!("Expected float32, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_field_to_schema_float64() -> Result<(), ArrowError> {
        let field = Field::new("double_col", DataType::Float64, false);
        let arrow_schema = ArrowSchema::new(vec![field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        match single_field_codec(&avro_dt) {
            Codec::Float64 => {}
            ref other => panic!("Expected float64, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_field_to_schema_binary() -> Result<(), ArrowError> {
        let field = Field::new("bin_col", DataType::Binary, true);
        let arrow_schema = ArrowSchema::new(vec![field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        let f0 = match &avro_dt.codec {
            Codec::Record(fields) => &fields[0],
            other => panic!("Expected record, got {other:?}"),
        };
        match f0.data_type().codec {
            Codec::Binary => {}
            ref other => panic!("Expected Codec::Binary, got {other:?}"),
        }
        assert_eq!(f0.data_type().nullability, Some(Nullability::NullFirst));
        Ok(())
    }

    #[test]
    fn test_field_to_schema_string() -> Result<(), ArrowError> {
        let field = Field::new("str_col", DataType::Utf8, false);
        let arrow_schema = ArrowSchema::new(vec![field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        match single_field_codec(&avro_dt) {
            Codec::String => {}
            ref other => panic!("Expected string, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_field_to_schema_fixedsizebinary() -> Result<(), ArrowError> {
        let field = Field::new("fixed8_col", DataType::FixedSizeBinary(8), false);
        let arrow_schema = ArrowSchema::new(vec![field]);
        let avro_dt = arrow_schema_round_trip(&arrow_schema)?;
        match single_field_codec(&avro_dt) {
            Codec::Fixed(sz) => {
                assert_eq!(*sz, 8);
            }
            other => panic!("Expected Codec::Fixed(8), got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_arrow_schema_to_avro_schema_all_supported() -> Result<(), ArrowError> {
        let arrow_schema = ArrowSchema::new(vec![
            Field::new("bool_col", DataType::Boolean, false),
            Field::new("int_col", DataType::Int32, true),
            Field::new("long_col", DataType::Int64, false),
            Field::new("float_col", DataType::Float32, false),
            Field::new("double_col", DataType::Float64, true),
            Field::new("bin_col", DataType::Binary, true),
            Field::new("str_col", DataType::Utf8, false),
            Field::new("fixed4_col", DataType::FixedSizeBinary(4), true),
        ]);
        let avro_sch = make_schema(&arrow_schema, &false)?;
        let mut resolver = Resolver::default();
        let top_dt = make_data_type(&avro_sch, None, &mut resolver)?;
        match &top_dt.codec {
            Codec::Record(fields) => {
                assert_eq!(fields.len(), 8);
                match fields[0].data_type().codec {
                    Codec::Boolean => {}
                    ref other => panic!("Expected bool => {other:?}"),
                }
                match fields[1].data_type().codec {
                    Codec::Int32 => {}
                    ref other => panic!("Expected int => {other:?}"),
                }
                assert_eq!(fields[1].data_type().nullability, Some(Nullability::NullFirst));
                match fields[2].data_type().codec {
                    Codec::Int64 => {}
                    ref other => panic!("Expected long => {other:?}"),
                }
                match fields[3].data_type().codec {
                    Codec::Float32 => {}
                    ref other => panic!("Expected float => {other:?}"),
                }
                match fields[4].data_type().codec {
                    Codec::Float64 => {}
                    ref other => panic!("Expected double => {other:?}"),
                }
                assert_eq!(fields[4].data_type().nullability, Some(Nullability::NullFirst));
                match fields[5].data_type().codec {
                    Codec::Binary => {}
                    ref other => panic!("Expected bytes => {other:?}"),
                }
                match fields[6].data_type().codec {
                    Codec::String => {}
                    ref other => panic!("Expected string => {other:?}"),
                }
                match fields[7].data_type().codec {
                    Codec::Fixed(sz) => {
                        assert_eq!(sz, 4);
                    }
                    ref other => panic!("Expected fixed => {other:?}"),
                }
                assert_eq!(
                    fields[7].data_type().nullability,
                    Some(Nullability::NullFirst)
                );
            }
            ref other => panic!("Expected top-level record => {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_skip_avro_default_null_in_metadata() {
        let dt = AvroDataType::from_codec(Codec::Int32);
        let field = AvroField {
            name: "test_col".into(),
            data_type: dt,
            default: Some(json!(null)),
        };
        let arrow_field = field.field();
        assert!(arrow_field.metadata().get("avro.default").is_none());
    }

    #[test]
    fn test_store_avro_default_nonnull_in_metadata() {
        let dt = AvroDataType::from_codec(Codec::Int32);
        let field = AvroField {
            name: "test_col".into(),
            data_type: dt,
            default: Some(json!(42)),
        };
        let arrow_field = field.field();
        let metadata = arrow_field.metadata();
        let got = metadata.get("avro.default").cloned();
        assert_eq!(got, Some("42".to_string()));
    }

    #[test]
    fn test_no_default_metadata_if_none() {
        let dt = AvroDataType::from_codec(Codec::String);
        let field = AvroField {
            name: "col".to_string(),
            data_type: dt,
            default: None,
        };
        let arrow_field = field.field();
        assert!(arrow_field.metadata().get("avro.default").is_none());
    }

    #[test]
    fn test_avro_field() {
        let field_codec = AvroDataType::from_codec(Codec::Int64);
        let avro_field = AvroField {
            name: "long_col".to_string(),
            data_type: field_codec.clone(),
            default: None,
        };
        assert_eq!(avro_field.name(), "long_col");
        let arrow_field = avro_field.field();
        assert_eq!(arrow_field.name(), "long_col");
        assert_eq!(arrow_field.data_type(), &DataType::Int64);
        assert!(!arrow_field.is_nullable());
    }

    #[test]
    fn test_avro_field_with_default() {
        let field_codec = AvroDataType::from_codec(Codec::Int32);
        let default_value = json!(123);
        let avro_field = AvroField {
            name: "int_col".to_string(),
            data_type: field_codec.clone(),
            default: Some(default_value.clone()),
        };
        let arrow_field = avro_field.field();
        let metadata = arrow_field.metadata();
        assert_eq!(
            metadata.get("avro.default").unwrap(),
            &default_value.to_string()
        );
    }

    #[test]
    fn test_codec_fixedsizebinary() {
        let codec = Codec::Fixed(12);
        let dt = codec.data_type();
        match dt {
            DataType::FixedSizeBinary(n) => assert_eq!(n, 12),
            _ => panic!("Expected FixedSizeBinary(12)"),
        }
    }

    #[test]
    fn test_union_long_null() -> Result<(), ArrowError> {
        let json_schema = r#"
        {
            "type": "record",
            "name": "test_long_null",
            "fields": [
                {"name": "f0", "type": ["long", "null"]}
            ]
        }
        "#;
        let schema: Schema = serde_json::from_str(json_schema).unwrap();
        let avro_field = AvroField::try_from(&schema)?;
        match &avro_field.data_type().codec {
            Codec::Record(fields) => {
                assert_eq!(fields.len(), 1);
                assert_eq!(fields[0].name(), "f0");
                let child_dt = fields[0].data_type();
                assert_eq!(child_dt.nullability, Some(Nullability::NullSecond));
                assert!(matches!(child_dt.codec, Codec::Int64));
            }
            _ => panic!("Expected record with a single [long,null] field"),
        }
        Ok(())
    }

    #[test]
    fn test_union_array_of_int_null() -> Result<(), ArrowError> {
        let json_schema = r#"
        {
            "type":"record",
            "name":"test_array_int_null",
            "fields":[
                {"name":"arr","type":[{"type":"array","items":["int","null"]},"null"]}
            ]
        }
        "#;
        let schema: Schema = serde_json::from_str(json_schema).unwrap();
        let avro_field = AvroField::try_from(&schema)?;
        match &avro_field.data_type().codec {
            Codec::Record(fields) => {
                assert_eq!(fields.len(), 1);
                let arr_dt = fields[0].data_type();
                assert_eq!(arr_dt.nullability, Some(Nullability::NullSecond));
                match &arr_dt.codec {
                    Codec::Array(child_dt) => {
                        assert_eq!(child_dt.nullability, Some(Nullability::NullSecond));
                        assert!(matches!(child_dt.codec, Codec::Int32));
                    }
                    other => panic!("Expected Array, got {other:?}"),
                }
            }
            other => panic!("Expected record, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_union_nested_array_of_int_null() -> Result<(), ArrowError> {
        let json_schema = r#"
        {
            "type":"record",
            "name":"test_nested_array_int_null",
            "fields":[
                {
                    "name":"nested_arr",
                    "type":[
                        {
                            "type":"array",
                            "items":[
                                {
                                    "type":"array",
                                    "items":["int","null"]
                                },
                                "null"
                            ]
                        },
                        "null"
                    ]
                }
            ]
        }
        "#;
        let schema: Schema = serde_json::from_str(json_schema).unwrap();
        let avro_field = AvroField::try_from(&schema)?;
        match &avro_field.data_type().codec {
            Codec::Record(fields) => {
                assert_eq!(fields.len(), 1);
                let outer = fields[0].data_type();
                assert_eq!(outer.nullability, Some(Nullability::NullSecond));
                match &outer.codec {
                    Codec::Array(mid) => {
                        assert_eq!(mid.nullability, Some(Nullability::NullSecond));
                        match &mid.codec {
                            Codec::Array(inner) => {
                                assert_eq!(inner.nullability, Some(Nullability::NullSecond));
                                assert!(matches!(inner.codec, Codec::Int32));
                            }
                            other => panic!("Expected inner array => {other:?}"),
                        }
                    }
                    other => panic!("Expected outer array => {other:?}"),
                }
            }
            other => panic!("Expected record => {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_union_map_of_int_null() -> Result<(), ArrowError> {
        let json_schema = r#"
        {
            "type":"record",
            "name":"test_map_int_null",
            "fields":[
                {"name":"map_field","type":[{"type":"map","values":["int","null"]},"null"]}
            ]
        }
        "#;
        let schema: Schema = serde_json::from_str(json_schema).unwrap();
        let avro_field = AvroField::try_from(&schema)?;
        match &avro_field.data_type().codec {
            Codec::Record(fields) => {
                let map_dt = fields[0].data_type();
                assert_eq!(map_dt.nullability, Some(Nullability::NullSecond));
                match map_dt.codec {
                    Codec::Map(ref val_dt) => {
                        assert_eq!(val_dt.nullability, Some(Nullability::NullSecond));
                        assert!(matches!(val_dt.codec, Codec::Int32));
                    }
                    ref other => panic!("Expected Map => {other:?}"),
                }
            }
            other => panic!("Expected record => {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_union_map_array_of_int_null() -> Result<(), ArrowError> {
        let json_schema = r#"
        {
            "type":"record",
            "name":"test_map_array_int_null",
            "fields":[
                {
                   "name":"map_arr",
                   "type":[
                      {
                         "type":"array",
                         "items":[
                            {
                               "type":"map",
                               "values":["int","null"]
                            },
                            "null"
                         ]
                      },
                      "null"
                   ]
                }
            ]
        }
        "#;
        let schema: Schema = serde_json::from_str(json_schema).unwrap();
        let avro_field = AvroField::try_from(&schema)?;
        match &avro_field.data_type().codec {
            Codec::Record(fields) => {
                let outer_dt = fields[0].data_type();
                assert_eq!(outer_dt.nullability, Some(Nullability::NullSecond));
                match &outer_dt.codec {
                    Codec::Array(map_dt) => {
                        assert_eq!(map_dt.nullability, Some(Nullability::NullSecond));
                        match &map_dt.codec {
                            Codec::Map(val_dt) => {
                                assert_eq!(val_dt.nullability, Some(Nullability::NullSecond));
                                assert!(matches!(val_dt.codec, Codec::Int32));
                            }
                            other => panic!("Expected map => {other:?}"),
                        }
                    }
                    other => panic!("Expected array => {other:?}"),
                }
            }
            other => panic!("Expected record => {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_union_nested_struct_out_of_spec() -> Result<(), ArrowError> {
        let json_schema = r#"
        {
            "type":"record","name":"topLevelRecord","fields":[
                {"name":"nested_struct","type":[
                    {
                        "type":"record",
                        "name":"nested_struct",
                        "namespace":"topLevelRecord",
                        "fields":[
                            {"name":"A","type":["int","null"]},
                            {"name":"b","type":[{"type":"array","items":["int","null"]},"null"]}
                        ]
                    },
                    "null"
                ]}
            ]
        }
        "#;
        let schema: Schema = serde_json::from_str(json_schema).unwrap();
        let avro_field = AvroField::try_from(&schema)?;
        match &avro_field.data_type().codec {
            Codec::Record(fields) => {
                assert_eq!(fields.len(), 1);
                let nested_dt = fields[0].data_type();
                assert_eq!(nested_dt.nullability, Some(Nullability::NullSecond));
                match nested_dt.codec {
                    Codec::Record(ref subfields) => {
                        assert_eq!(subfields.len(), 2);
                        let f_a = &subfields[0];
                        assert_eq!(f_a.data_type().nullability, Some(Nullability::NullSecond));
                        assert!(matches!(f_a.data_type().codec, Codec::Int32));
                    }
                    ref other => panic!("Expected record => {other:?}"),
                }
            }
            other => panic!("Expected top-level record => {other:?}"),
        }
        Ok(())
    }
}
