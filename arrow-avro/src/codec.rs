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
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Nullability {
    NullFirst,
    NullSecond,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Promotion {
    IntToLong,
    IntToFloat,
    IntToDouble,
    LongToFloat,
    LongToDouble,
    FloatToDouble,
    StringToBytes,
    BytesToString,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumMapping {
    pub(crate) mapping: Arc<[i32]>,
    pub(crate) default_index: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionInfo {
    Promotion(Promotion),
    Default(serde_json::Value),
    EnumMapping(EnumMapping),
    Record(ResolvedRecord),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRecord {
    pub writer_to_reader: Arc<[Option<usize>]>,
    pub default_fields: Arc<[usize]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AvroDataType {
    pub(crate) codec: Codec,
    pub(crate) nullability: Option<Nullability>,
    pub(crate) metadata: HashMap<String, String>,
    pub(crate) resolution: Option<ResolutionInfo>,
}

impl AvroDataType {
    /// **Public ctor kept for backward‑compatibility** (reader/record.rs depends on it)
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

    fn parsed(codec: Codec, metadata: HashMap<String, String>) -> Self {
        Self {
            codec,
            metadata,
            nullability: None,
            resolution: None,
        }
    }

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

    pub fn field_with_name(&self, name: &str) -> Field {
        let f = Field::new(name, self.codec.data_type(), self.nullability.is_some())
            .with_metadata(self.metadata.clone());
        #[cfg(feature = "canonical_extension_types")]
        return match self.codec {
            Codec::Uuid => f.with_extension_type(arrow_schema::extension::Uuid),
            _ => f,
        };
        #[cfg(not(feature = "canonical_extension_types"))]
        f
    }

    pub fn codec(&self) -> &Codec {
        &self.codec
    }
    pub fn nullability(&self) -> Option<Nullability> {
        self.nullability
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Codec {
    Null,
    Boolean,
    Int32,
    Int64,
    Float32,
    Float64,
    Binary,
    Utf8,
    Utf8View,
    Date32,
    TimeMillis,
    TimeMicros,
    TimestampMillis(bool),
    TimestampMicros(bool),
    Fixed(i32),
    Decimal(usize, Option<usize>, Option<usize>),
    Uuid,
    Enum(Arc<[String]>),
    List(Arc<AvroDataType>),
    Struct(Arc<[AvroField]>),
    Map(Arc<AvroDataType>),
    Interval,
}

impl Codec {
    fn with_utf8view(self, use_utf8view: bool) -> Self {
        if use_utf8view && matches!(self, Self::Utf8) {
            Self::Utf8View
        } else {
            self
        }
    }

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
            Self::Fixed(sz) => DataType::FixedSizeBinary(*sz),
            Self::Decimal(p, s, sz) => {
                let p8 = *p as u8;
                let s8 = s.unwrap_or(0) as i8;
                let needs256 = sz.map_or_else(
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
            Self::Enum(_) => DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            Self::List(child) => DataType::List(Arc::new(
                child.field_with_name(Field::LIST_FIELD_DEFAULT_NAME),
            )),
            Self::Struct(flds) => DataType::Struct(flds.iter().map(|f| f.field()).collect()),
            Self::Map(val) => {
                let vfield = Field::new(
                    "value",
                    val.codec.data_type(),
                    val.nullability.is_some(),
                )
                    .with_metadata(val.metadata.clone());
                DataType::Map(
                    Arc::new(Field::new(
                        "entries",
                        DataType::Struct(Fields::from(vec![
                            Field::new("key", DataType::Utf8, false),
                            vfield,
                        ])),
                        false,
                    )),
                    false,
                )
            }
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

#[derive(Debug, Clone, PartialEq)]
pub struct AvroField {
    name: String,
    data_type: AvroDataType,
}

impl AvroField {
    pub fn field(&self) -> Field {
        self.data_type.field_with_name(&self.name)
    }
    pub fn data_type(&self) -> &AvroDataType {
        &self.data_type
    }
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn resolve_from_writer_and_reader<'a>(
        writer: &Schema<'a>,
        reader: &Schema<'a>,
        use_utf8view: bool,
    ) -> Result<Self, ArrowError> {
        let mut res = SchemaResolver::new(use_utf8view);
        let dt = res.visit(Some(writer), reader, None)?;
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
            let mut res = SchemaResolver::new(false);
            let dt = res.visit(None, schema, None)?;
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

#[derive(Default)]
struct NameCache<'a> {
    named: HashMap<(&'a str, &'a str), AvroDataType>,
    resolved: HashMap<(&'a str, &'a str), AvroDataType>,
}

impl<'a> NameCache<'a> {
    fn lookup_named(&self, name: &str, ns: Option<&'a str>) -> Option<AvroDataType> {
        let (nsp, nm) = name
            .rsplit_once('.')
            .unwrap_or_else(|| (ns.unwrap_or(""), name));
        self.named.get(&(nsp, nm)).cloned()
    }
    fn insert_named(
        &mut self,
        name: &'a str,
        ns: Option<&'a str>,
        dt: AvroDataType,
    ) -> AvroDataType {
        self.named.insert((name, ns.unwrap_or("")), dt.clone());
        dt
    }
}

struct SchemaResolver<'a> {
    cache: NameCache<'a>,
    use_utf8view: bool,
}

impl<'a> SchemaResolver<'a> {
    fn new(use_utf8view: bool) -> Self {
        Self {
            cache: Default::default(),
            use_utf8view,
        }
    }

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
            Schema::TypeName(TypeName::Ref(name)) => self
                .cache
                .lookup_named(name, namespace)
                .ok_or_else(|| ArrowError::ParseError(format!("Failed to resolve .{name}"))),
            Schema::Union(branches) => self.parse_nullable_union(branches, namespace),
            Schema::Complex(ct) => self.parse_complex(ct, namespace),
            Schema::Type(t) => {
                let mut dt = self.parse(&Schema::TypeName(t.r#type.clone()), namespace)?;
                self.apply_logical_type(&mut dt, &t.attributes, None)?;
                Ok(dt)
            }
        }
    }

    fn parse_nullable_union(
        &mut self,
        branches: &[Schema<'a>],
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        if branches.len() != 2 {
            return Err(ArrowError::NotYetImplemented(
                "Full unions not implemented".into(),
            ));
        }
        let (null_pos, nonnull) = if Self::is_null(&branches[0]) {
            (Nullability::NullFirst, &branches[1])
        } else if Self::is_null(&branches[1]) {
            (Nullability::NullSecond, &branches[0])
        } else {
            return Err(ArrowError::ParseError(
                "Two‑branch union missing \"null\"".into(),
            ));
        };
        let mut dt = self.parse(nonnull, namespace)?;
        dt.nullability = Some(null_pos);
        Ok(dt)
    }

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
                let v = self.parse(&mp.values, namespace)?;
                Ok(AvroDataType::parsed(
                    Codec::Map(Arc::new(v)),
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
        // placeholder to allow recursion
        let placeholder = AvroDataType::parsed(
            Codec::Struct(Arc::new([])),
            r.attributes.field_metadata(),
        );
        self.cache.insert_named(r.name, ns, placeholder);
        let mut fields = Vec::with_capacity(r.fields.len());
        for f in &r.fields {
            let dt = self.parse(&f.r#type, ns)?;
            fields.push(AvroField {
                name: f.name.to_string(),
                data_type: dt,
            });
        }
        let struct_dt = AvroDataType::parsed(
            Codec::Struct(Arc::from(fields)),
            r.attributes.field_metadata(),
        );
        self.cache.insert_named(r.name, ns, struct_dt.clone());
        Ok(struct_dt)
    }

    fn parse_enum(
        &mut self,
        e: &Enum<'a>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        let symbols = Arc::<[String]>::from(e.symbols.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let mut md = e.attributes.field_metadata();
        md.insert(
            "avro.enum.symbols".into(),
            serde_json::to_string(&e.symbols).unwrap(),
        );
        let dt = AvroDataType::parsed(Codec::Enum(symbols), md);
        self.cache.insert_named(e.name, e.namespace.or(namespace), dt.clone());
        Ok(dt)
    }

    fn parse_fixed(
        &mut self,
        f: &Fixed<'a>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        let size_i32 = i32::try_from(f.size)
            .map_err(|e| ArrowError::ParseError(format!("Fixed size overflow: {e}")))?;
        let mut md = f.attributes.field_metadata();
        let codec = match f.attributes.logical_type {
            Some("decimal") => {
                let (p, s, _) = parse_decimal_attrs(&f.attributes, Some(f.size), true)?;
                Codec::Decimal(p, Some(s), Some(f.size))
            }
            Some("duration") => {
                if f.size != 12 {
                    return Err(ArrowError::ParseError(
                        "Duration fixed must have size 12".into(),
                    ));
                }
                Codec::Interval
            }
            _ => Codec::Fixed(size_i32),
        };
        let dt = AvroDataType::parsed(codec, md);
        self.cache.insert_named(f.name, f.namespace.or(namespace), dt.clone());
        Ok(dt)
    }

    fn resolve(
        &mut self,
        writer: &Schema<'a>,
        reader: &Schema<'a>,
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        match (writer, reader) {
            (Schema::TypeName(TypeName::Primitive(wp)), Schema::TypeName(TypeName::Primitive(rp)))
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
            (
                Schema::TypeName(TypeName::Primitive(wp)),
                Schema::Type(r_t),
            ) if matches!(r_t.r#type, TypeName::Primitive(_)) => {
                if let TypeName::Primitive(rp) = r_t.r#type {
                    self.resolve_primitives(*wp, rp, reader)
                } else {
                    unreachable!()
                }
            }
            (
                Schema::Complex(ComplexType::Record(wr)),
                Schema::Complex(ComplexType::Record(rr)),
            ) => self.resolve_records(wr, rr, namespace),
            (
                Schema::Complex(ComplexType::Enum(we)),
                Schema::Complex(ComplexType::Enum(re)),
            ) => self.resolve_enums(we, re),
            (
                Schema::Complex(ComplexType::Fixed(wf)),
                Schema::Complex(ComplexType::Fixed(rf)),
            ) if wf.size == rf.size => self.parse(reader, namespace),
            (
                Schema::Complex(ComplexType::Array(wa)),
                Schema::Complex(ComplexType::Array(ra)),
            ) => {
                let child = self.visit(Some(&wa.items), &ra.items, namespace)?;
                Ok(AvroDataType::resolved(
                    Codec::List(Arc::new(child)),
                    ra.attributes.field_metadata(),
                    None,
                    None,
                ))
            }
            (
                Schema::Complex(ComplexType::Map(wm)),
                Schema::Complex(ComplexType::Map(rm)),
            ) => {
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
                let w_nonnull = if Self::is_null(&wu[0]) { &wu[1] } else { &wu[0] };
                self.visit(Some(w_nonnull), r_nonunion, namespace)
            }
            _ => Err(ArrowError::ParseError(format!(
                "Schemas incompatible\nwriter: {writer:?}\nreader: {reader:?}"
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
        let prom = match (wp, rp) {
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
        dt.resolution = Some(ResolutionInfo::Promotion(prom));
        Ok(dt)
    }

    fn resolve_enums(
        &mut self,
        w: &Enum<'a>,
        r: &Enum<'a>,
    ) -> Result<AvroDataType, ArrowError> {
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
                let p = r
                    .symbols
                    .iter()
                    .position(|&rs| rs == def)
                    .ok_or_else(|| {
                        ArrowError::ParseError("Reader enum default not in symbol list".into())
                    })?;
                mapping[i] = p as i32;
            } else {
                return Err(ArrowError::ParseError(format!(
                    "Writer enum symbol '{}' not in reader enum",
                    wsym
                )));
            }
            def_idx = def_idx.or_else(|| {
                r.default
                    .and_then(|d| r.symbols.iter().position(|&s| s == d))
                    .map(|p| p as i32)
            });
        }
        let enum_codec = Codec::Enum(
            Arc::<[String]>::from(r.symbols.iter().map(|s| s.to_string()).collect::<Vec<_>>()),
        );
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
        let same = w.name == r.name
            || r.aliases.iter().any(|&a| a == w.name)
            || w.aliases.iter().any(|&a| a == r.name);
        if !same {
            return Err(ArrowError::ParseError(format!(
                "Record name mismatch writer={}, reader={}",
                w.name, r.name
            )));
        }
        if let Some(prev) = self.cache.resolved.get(&(w.name, r.name)).cloned() {
            return Ok(prev);
        }
        let ns = r.namespace.or(namespace);
        let mut w_lookup: HashMap<&str, &SchemaField<'a>> = w.fields.iter().map(|f| (f.name, f)).collect();
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
            } else {
                if let Some(def_raw) = rf.default {
                    let def_val: serde_json::Value =
                        serde_json::from_str(def_raw).unwrap_or_else(|_| def_raw.into());
                    let mut child = self.visit(None, &rf.r#type, ns)?;
                    child.resolution = Some(ResolutionInfo::Default(def_val));
                    reader_fields.push(AvroField {
                        name: rf.name.to_string(),
                        data_type: child,
                    });
                    defaults.push(r_idx);
                } else {
                    return Err(ArrowError::ParseError(format!(
                        "Field '{}' missing in writer and no default",
                        rf.name
                    )));
                }
            }
        }
        let mut md = r.attributes.field_metadata();
        if !defaults.is_empty() {
            md.insert(
                "avro.resolution.defaults".into(),
                serde_json::to_string(&defaults).unwrap(),
            );
        }
        let dt = AvroDataType::resolved(
            Codec::Struct(Arc::from(reader_fields)),
            md,
            None,
            Some(ResolutionInfo::Record(ResolvedRecord {
                writer_to_reader: Arc::from(w_to_r),
                default_fields: Arc::from(defaults),
            })),
        );
        self.cache.resolved.insert((w.name, r.name), dt.clone());
        Ok(dt)
    }

    fn resolve_nullable_union(
        &mut self,
        wu: &[Schema<'a>],
        ru: &[Schema<'a>],
        namespace: Option<&'a str>,
    ) -> Result<AvroDataType, ArrowError> {
        if !Self::is_nullable_union(wu) || !Self::is_nullable_union(ru) {
            return Err(ArrowError::NotYetImplemented(
                "Full union resolution not implemented".into(),
            ));
        }
        let (null_pos, w_nonnull) = if Self::is_null(&wu[0]) {
            (Nullability::NullFirst, &wu[1])
        } else {
            (Nullability::NullSecond, &wu[0])
        };
        let r_nonnull = if Self::is_null(&ru[0]) { &ru[1] } else { &ru[0] };
        let mut dt = self.visit(Some(w_nonnull), r_nonnull, namespace)?;
        dt.nullability = Some(null_pos);
        Ok(dt)
    }

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
    #[inline]
    fn is_nullable_union(branches: &[Schema<'_>]) -> bool {
        branches.len() == 2 && (Self::is_null(&branches[0]) || Self::is_null(&branches[1]))
    }

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
        .ok_or_else(|| ArrowError::ParseError("Decimal requires precision".into()))? as usize;
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
            Some(ResolutionInfo::Promotion(
                Promotion::IntToLong
            ))
        );
    }

    #[test]
    fn added_field_with_default() {
        let w = parse_schema(
            r#"{"type":"record","name":"R","fields":[{"name":"id","type":"int"}]}"#,
        );
        let r = parse_schema(
            r#"{"type":"record","name":"R","fields":[
                   {"name":"id","type":"int"},
                   {"name":"country","type":"string","default":"US"}
              ]}"#,
        );
        let root = AvroField::resolve_from_writer_and_reader(&w, &r, false).unwrap();
        if let Codec::Struct(fields) = root.data_type.codec() {
            assert_eq!(fields.len(), 2);
            // Ensure the second field carries default info
            assert!(matches!(
                &fields[1].data_type.resolution,
                Some(ResolutionInfo::Default(_))
            ));
        } else {
            panic!("expected struct codec");
        }
    }

    #[test]
    fn enum_symbol_mapping() {
        let w = parse_schema(
            r#"{"type":"enum","name":"Color","symbols":["RED","GREEN"]}"#,
        );
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
        let schema =
            create_schema_with_logical_type(PrimitiveType::Long, "local-timestamp-millis");
        assert_codec!(schema, Codec::TimestampMillis(false), false);
    }

    #[test]
    fn test_local_timestamp_micros_logical_type() {
        let schema =
            create_schema_with_logical_type(PrimitiveType::Long, "local-timestamp-micros");
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
