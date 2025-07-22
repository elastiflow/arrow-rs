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

use arrow_schema::ArrowError;
use digest::Digest;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::Sha256;
use std::collections::HashMap;
use std::sync::OnceLock;

/// The metadata key used for storing the JSON encoded [`Schema`]
pub const SCHEMA_METADATA_KEY: &str = "avro.schema";

/// The Avro single‑object encoding “magic” bytes (`0xC3 0x01`)
pub const SINGLE_OBJECT_MAGIC: [u8; 2] = [0xC3, 0x01];

/// Compare two Avro schemas for equality (identical schemas).
/// Returns true if the schemas have the same parsing canonical form (i.e., logically identical).
pub fn compare_schemas(writer: &Schema, reader: &Schema) -> bool {
    let canon_writer = generate_canonical_form(writer);
    let canon_reader = generate_canonical_form(reader);
    canon_writer == canon_reader
}

/// Either a [`PrimitiveType`] or a reference to a previously defined named type
///
/// <https://avro.apache.org/docs/1.11.1/specification/#names>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
/// A type name in an Avro schema
///
/// This represents the different ways a type can be referenced in an Avro schema.
pub enum TypeName<'a> {
    /// A primitive type like null, boolean, int, etc.
    Primitive(PrimitiveType),
    /// A reference to another named type
    Ref(&'a str),
}

/// A primitive type
///
/// <https://avro.apache.org/docs/1.11.1/specification/#primitive-types>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PrimitiveType {
    /// null: no value
    Null,
    /// boolean: a binary value
    Boolean,
    /// int: 32-bit signed integer
    Int,
    /// long: 64-bit signed integer
    Long,
    /// float: single precision (32-bit) IEEE 754 floating-point number
    Float,
    /// double: double precision (64-bit) IEEE 754 floating-point number
    Double,
    /// bytes: sequence of 8-bit unsigned bytes
    Bytes,
    /// string: Unicode character sequence
    String,
}

/// Additional attributes within a [`Schema`]
///
/// <https://avro.apache.org/docs/1.11.1/specification/#schema-declaration>
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Attributes<'a> {
    /// A logical type name
    ///
    /// <https://avro.apache.org/docs/1.11.1/specification/#logical-types>
    #[serde(default)]
    pub logical_type: Option<&'a str>,

    /// Additional JSON attributes
    #[serde(flatten)]
    pub additional: HashMap<&'a str, serde_json::Value>,
}

impl Attributes<'_> {
    /// Returns the field metadata for this [`Attributes`]
    pub(crate) fn field_metadata(&self) -> HashMap<String, String> {
        self.additional
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }
}

/// A type definition that is not a variant of [`ComplexType`]
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Type<'a> {
    /// The type of this Avro data structure
    #[serde(borrow)]
    pub r#type: TypeName<'a>,
    /// Additional attributes associated with this type
    #[serde(flatten)]
    pub attributes: Attributes<'a>,
}

/// An Avro schema
///
/// This represents the different shapes of Avro schemas as defined in the specification.
/// See <https://avro.apache.org/docs/1.11.1/specification/#schemas> for more details.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Schema<'a> {
    /// A direct type name (primitive or reference)
    #[serde(borrow)]
    TypeName(TypeName<'a>),
    /// A union of multiple schemas (e.g., ["null", "string"])
    #[serde(borrow)]
    Union(Vec<Schema<'a>>),
    /// A complex type such as record, array, map, etc.
    #[serde(borrow)]
    Complex(ComplexType<'a>),
    /// A type with attributes
    #[serde(borrow)]
    Type(Type<'a>),
}

/// A complex type
///
/// <https://avro.apache.org/docs/1.11.1/specification/#complex-types>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ComplexType<'a> {
    /// Record type: a sequence of fields with names and types
    #[serde(borrow)]
    Record(Record<'a>),
    /// Enum type: a set of named values
    #[serde(borrow)]
    Enum(Enum<'a>),
    /// Array type: a sequence of values of the same type
    #[serde(borrow)]
    Array(Array<'a>),
    /// Map type: a mapping from strings to values of the same type
    #[serde(borrow)]
    Map(Map<'a>),
    /// Fixed type: a fixed-size byte array
    #[serde(borrow)]
    Fixed(Fixed<'a>),
}

/// A record
///
/// <https://avro.apache.org/docs/1.11.1/specification/#schema-record>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Record<'a> {
    /// Name of the record
    #[serde(borrow)]
    pub name: &'a str,
    /// Optional namespace for the record, provides a way to organize names
    #[serde(borrow, default)]
    pub namespace: Option<&'a str>,
    /// Optional documentation string for the record
    #[serde(borrow, default)]
    pub doc: Option<&'a str>,
    /// Alternative names for this record
    #[serde(borrow, default)]
    pub aliases: Vec<&'a str>,
    /// The fields contained in this record
    #[serde(borrow)]
    pub fields: Vec<Field<'a>>,
    /// Additional attributes for this record
    #[serde(flatten)]
    pub attributes: Attributes<'a>,
}

/// A field within a [`Record`]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Field<'a> {
    /// Name of the field within the record
    #[serde(borrow)]
    pub name: &'a str,
    /// Optional documentation for this field
    #[serde(borrow, default)]
    pub doc: Option<&'a str>,
    /// The field's type definition
    #[serde(borrow)]
    pub r#type: Schema<'a>,
    /// Optional default value for this field
    #[serde(borrow, default)]
    pub default: Option<&'a str>,
}

/// An enumeration
///
/// <https://avro.apache.org/docs/1.11.1/specification/#enums>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Enum<'a> {
    /// Name of the enum
    #[serde(borrow)]
    pub name: &'a str,
    /// Optional namespace for the enum, provides organizational structure
    #[serde(borrow, default)]
    pub namespace: Option<&'a str>,
    /// Optional documentation string describing the enum
    #[serde(borrow, default)]
    pub doc: Option<&'a str>,
    /// Alternative names for this enum
    #[serde(borrow, default)]
    pub aliases: Vec<&'a str>,
    /// The symbols (values) that this enum can have
    #[serde(borrow)]
    pub symbols: Vec<&'a str>,
    /// Optional default value for this enum
    #[serde(borrow, default)]
    pub default: Option<&'a str>,
    /// Additional attributes for this enum
    #[serde(flatten)]
    pub attributes: Attributes<'a>,
}

/// An array
///
/// <https://avro.apache.org/docs/1.11.1/specification/#arrays>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Array<'a> {
    /// The schema for items in this array
    #[serde(borrow)]
    pub items: Box<Schema<'a>>,
    /// Additional attributes for this array
    #[serde(flatten)]
    pub attributes: Attributes<'a>,
}

/// A map
///
/// <https://avro.apache.org/docs/1.11.1/specification/#maps>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Map<'a> {
    /// The schema for values in this map
    #[serde(borrow)]
    pub values: Box<Schema<'a>>,
    /// Additional attributes for this map
    #[serde(flatten)]
    pub attributes: Attributes<'a>,
}

/// A fixed length binary array
///
/// <https://avro.apache.org/docs/1.11.1/specification/#fixed>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fixed<'a> {
    /// Name of the fixed type
    #[serde(borrow)]
    pub name: &'a str,
    /// Optional namespace for the fixed type
    #[serde(borrow, default)]
    pub namespace: Option<&'a str>,
    /// Alternative names for this fixed type
    #[serde(borrow, default)]
    pub aliases: Vec<&'a str>,
    /// The number of bytes in this fixed type
    pub size: usize,
    /// Additional attributes for this fixed type
    #[serde(flatten)]
    pub attributes: Attributes<'a>,
}

/// Supported fingerprint algorithms for Avro schema identification.
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HashType {
    /// 64-bit CRC-64-AVRO Rabin fingerprint.
    Rabin,
    /// 128-bit MD5 message digest.
    MD5,
    /// 256-bit SHA-256 digest.
    SHA256,
}

/// A schema fingerprint in one of the supported formats.
///
/// This is used as the *key* inside `SchemaStore`’s `HashMap`.  Each `SchemaStore`
/// instance always stores only one variant, matching its configured
/// `HashType`, but the enum makes the API uniform.
///
/// <https://avro.apache.org/docs/1.11.1/specification/#schema-fingerprints>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fingerprint {
    /// A 64-bit Rabin fingerprint.
    Rabin(u64),
    /// A 128-bit MD5 fingerprint.
    MD5([u8; 16]),
    /// A 256-bit SHA-256 fingerprint.
    SHA256([u8; 32]),
}

/// Generates a fingerprint for the given `Schema` using the specified `HashType`.
///
/// The fingerprint is computed from the canonical form of the schema. This allows
/// generating any of the supported fingerprint types: Rabin, MD5, or SHA-256.
///
#[inline]
pub fn generate_fingerprint(schema: &Schema, hash_type: HashType) -> Fingerprint {
    let canonical = generate_canonical_form(&schema);
    match hash_type {
        HashType::Rabin => Fingerprint::Rabin(compute_fingerprint_rabin(&canonical)),
        HashType::MD5 => Fingerprint::MD5(compute_fingerprint_md5(&canonical)),
        HashType::SHA256 => Fingerprint::SHA256(compute_fingerprint_sha256(&canonical)),
    }
}

/// Generates the 64-bit Rabin fingerprint for the given `Schema`.
///
/// The fingerprint is computed from the canonical form of the schema.
/// This is also known as `CRC-64-AVRO`.
///
#[inline]
pub fn generate_fingerprint_rabin(schema: &Schema) -> Fingerprint {
    generate_fingerprint(schema, HashType::Rabin)
}

/// Generates the MD5 fingerprint for the given `Schema`.
///
/// The fingerprint is computed from the canonical form of the schema.
/// The result is a 128-bit (16-byte) hash.
///
#[inline]
pub fn generate_fingerprint_md5(schema: &Schema) -> Fingerprint {
    generate_fingerprint(schema, HashType::MD5)
}

/// Generates the SHA-256 fingerprint for the given `Schema`.
///
/// The fingerprint is computed from the canonical form of the schema.
/// The result is a 256-bit (32-byte) hash.
///
#[inline]
pub fn generate_fingerprint_sha256(schema: &Schema) -> Fingerprint {
    generate_fingerprint(schema, HashType::SHA256)
}

/// Generates the Parsing Canonical Form for the given [`Schema`].
///
/// The canonical form is a standardized JSON representation of the schema,
/// primarily used for generating a schema fingerprint for equality checking.
///
/// This form strips attributes that do not affect the schema's identity,
/// such as `doc` fields, `aliases`, and any properties not defined in the
/// Avro specification.
///
/// <https://avro.apache.org/docs/1.11.1/specification/#parsing-canonical-form-for-schemas>
#[inline]
pub fn generate_canonical_form(schema: &Schema) -> String {
    serde_json::to_string(&parse_canonical_json(schema)).unwrap()
}

/// Builder for configuring and constructing a [`SchemaStore`].
///
/// ```rust
/// use arrow_avro::schema::{HashType, SchemaStoreBuilder};
/// let store = SchemaStoreBuilder::new()
///     .with_fingerprint_type(HashType::MD5)
///     .build();
/// ```
pub struct SchemaStoreBuilder {
    hash_type: HashType,
}

impl SchemaStoreBuilder {
    /// Start a new builder with default settings (Rabin fingerprint).
    pub fn new() -> Self {
        Self {
            hash_type: HashType::Rabin,
        }
    }

    /// Specify the fingerprint algorithm to use.
    pub fn with_fingerprint_type(mut self, hash_type: HashType) -> Self {
        self.hash_type = hash_type;
        self
    }

    /// Finish the builder and create an *empty* [`SchemaStore`].
    pub fn build<'a>(self) -> SchemaStore<'a> {
        SchemaStore {
            hash_type: self.hash_type,
            schemas: HashMap::new(),
        }
    }

    /// Convenience helper: create a store and immediately register a slice of
    /// schemas with it.
    pub fn build_with_schemas<'a>(
        self,
        schemas: &'a [Schema<'a>],
    ) -> Result<SchemaStore<'a>, ArrowError> {
        let mut store = self.build();
        for s in schemas {
            store.register(s.clone())?;
        }
        Ok(store)
    }
}

/// An in‑memory cache of Avro schemas indexed by their fingerprint.
///
/// A store is *configured* for exactly **one** [`HashType`].
/// * All registrations compute the fingerprint using that algorithm.
/// * All look‑ups must use the same fingerprint type.
#[derive(Debug, Clone)]
pub struct SchemaStore<'a> {
    hash_type: HashType,
    schemas: HashMap<Fingerprint, Schema<'a>>,
}

impl<'a> TryFrom<&'a [Schema<'a>]> for SchemaStore<'a> {
    type Error = ArrowError;

    /// Creates a `SchemaStore` from a slice of schemas.
    ///
    /// Each schema in the slice is registered with the new store.
    ///
    fn try_from(schemas: &'a [Schema<'a>]) -> Result<Self, Self::Error> {
        let mut store = SchemaStore::new();
        for schema in schemas {
            store.register(schema.clone())?;
        }
        Ok(store)
    }
}

impl<'a> SchemaStore<'a> {
    /// Create an *empty* store using the **default** fingerprint (Rabin 64‑bit).
    pub fn new() -> Self {
        Self {
            hash_type: HashType::Rabin,
            schemas: HashMap::new(),
        }
    }

    /// Convenience constructor for a specific hash type.
    pub fn with_hash_type(hash_type: HashType) -> Self {
        Self {
            hash_type,
            schemas: HashMap::new(),
        }
    }

    /// Register a schema, returning its fingerprint.
    ///
    /// If the fingerprint already exists, the schema is **not** overwritten.
    pub fn register(&mut self, schema: Schema<'a>) -> Result<Fingerprint, ArrowError> {
        let fp = generate_fingerprint(&schema, self.hash_type);
        self.schemas.entry(fp).or_insert(schema);
        Ok(fp)
    }

    /// Generic lookup by reference to a [`Fingerprint`].
    pub fn lookup(&self, fp: &Fingerprint) -> Option<Schema<'a>> {
        self.schemas.get(fp).cloned()
    }

    /// Returns the `HashType` used by the `SchemaStore`
    pub fn lookup_keys_hash_type(&self) -> Option<HashType> {
        self.schemas.keys().next().map(|fp| match fp {
            Fingerprint::Rabin(_) => HashType::Rabin,
            Fingerprint::MD5(_) => HashType::MD5,
            Fingerprint::SHA256(_) => HashType::SHA256,
        })
    }
}

fn parse_canonical_json(schema: &Schema) -> Value {
    match schema {
        Schema::TypeName(tn) => match tn {
            TypeName::Primitive(pt) => serde_json::to_value(pt).unwrap(),
            TypeName::Ref(name) => serde_json::to_value(name).unwrap(),
        },
        Schema::Union(schemas) => Value::Array(schemas.iter().map(parse_canonical_json).collect()),
        Schema::Complex(ct) => match ct {
            ComplexType::Record(r) => {
                let full_name = r
                    .namespace
                    .map_or_else(|| r.name.to_string(), |ns| format!("{ns}.{}", r.name));
                let fields: Vec<Value> = r
                    .fields
                    .iter()
                    .map(|f| json!({ "name": f.name, "type": parse_canonical_json(&f.r#type) }))
                    .collect();
                json!({ "type": "record", "name": full_name, "fields": fields })
            }
            ComplexType::Enum(e) => {
                let full_name = e
                    .namespace
                    .map_or_else(|| e.name.to_string(), |ns| format!("{ns}.{}", e.name));
                json!({ "type": "enum", "name": full_name, "symbols": e.symbols })
            }
            ComplexType::Array(a) => {
                json!({ "type": "array", "items": parse_canonical_json(&a.items) })
            }
            ComplexType::Map(m) => {
                json!({ "type": "map", "values": parse_canonical_json(&m.values) })
            }
            ComplexType::Fixed(f) => {
                let full_name = f
                    .namespace
                    .map_or_else(|| f.name.to_string(), |ns| format!("{ns}.{}", f.name));
                json!({ "type": "fixed", "name": full_name, "size": f.size })
            }
        },
        Schema::Type(t) => match &t.r#type {
            TypeName::Primitive(pt) => serde_json::to_value(pt).unwrap(),
            TypeName::Ref(name) => serde_json::to_value(name).unwrap(),
        },
    }
}

static FINGERPRINT_TABLE: OnceLock<[u64; 256]> = OnceLock::new();

/// Compute the 64‑bit Rabin fingerprint described in the Avro spec.
///
#[inline]
pub(crate) fn compute_fingerprint_rabin(canonical_form: &str) -> u64 {
    let buf = canonical_form.as_bytes();
    const EMPTY: u64 = 0xc15d213aa4d7a795;
    // The lookup table is computed once and cached in a thread-safe manner.
    let table = FINGERPRINT_TABLE.get_or_init(|| {
        let mut table = [0u64; 256];
        for i in 0..256 {
            let mut fp = i as u64;
            for _ in 0..8 {
                fp = (fp >> 1) ^ (EMPTY & (0u64.wrapping_sub(fp & 1)));
            }
            table[i] = fp;
        }
        table
    });
    let mut fp = EMPTY;
    for &b in buf {
        fp = (fp >> 8) ^ table[((fp ^ b as u64) & 0xff) as usize];
    }
    fp
}

/// Compute the **128‑bit MD5** fingerprint of the canonical form.
///
/// Returns a 16‑byte array (`[u8; 16]`) containing the full MD5 digest,
/// exactly as required by the Avro specification.
#[inline]
pub(crate) fn compute_fingerprint_md5(canonical_form: &str) -> [u8; 16] {
    let digest = md5::compute(canonical_form.as_bytes());
    digest.0
}

/// Compute the **256‑bit SHA‑256** fingerprint of the canonical form.
///
/// Returns a 32‑byte array (`[u8; 32]`) containing the full SHA‑256 digest.
#[inline]
pub(crate) fn compute_fingerprint_sha256(canonical_form: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(canonical_form.as_bytes());
    let digest = hasher.finalize();
    digest.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{AvroDataType, AvroField};
    use arrow_schema::{DataType, Fields, TimeUnit};
    use serde_json::json;

    fn int_schema() -> Schema<'static> {
        Schema::TypeName(TypeName::Primitive(PrimitiveType::Int))
    }

    fn record_schema() -> Schema<'static> {
        Schema::Complex(ComplexType::Record(Record {
            name: "record1",
            namespace: Some("test.namespace"),
            doc: Some("A test record"),
            aliases: vec![],
            fields: vec![
                Field {
                    name: "field1",
                    doc: Some("An integer field"),
                    r#type: int_schema(),
                    default: None,
                },
                Field {
                    name: "field2",
                    doc: None,
                    r#type: Schema::TypeName(TypeName::Primitive(PrimitiveType::String)),
                    default: None,
                },
            ],
            attributes: Attributes::default(),
        }))
    }

    #[test]
    fn test_md5_and_sha256_fingerprints() {
        // Simple primitive schema "int"
        let schema = Schema::TypeName(TypeName::Primitive(PrimitiveType::Int));
        let canonical = generate_canonical_form(&schema);
        assert_eq!(canonical, "\"int\"");

        let md5_fp = compute_fingerprint_md5(&canonical);
        let sha_fp = compute_fingerprint_sha256(&canonical);

        let expected_md5: [u8; 16] = [
            0xef, 0x52, 0x4e, 0xa1, 0xb9, 0x1e, 0x73, 0x17, 0x3d, 0x93, 0x8a, 0xde, 0x36, 0xc1,
            0xdb, 0x32,
        ];
        let expected_sha: [u8; 32] = [
            0x3f, 0x2b, 0x87, 0xa9, 0xfe, 0x7c, 0xc9, 0xb1, 0x38, 0x35, 0x59, 0x8c, 0x39, 0x81,
            0xcd, 0x45, 0xe3, 0xe3, 0x55, 0x30, 0x9e, 0x50, 0x90, 0xaa, 0x09, 0x33, 0xd7, 0xbe,
            0xcb, 0x6f, 0xba, 0x45,
        ];

        assert_eq!(md5_fp, expected_md5);
        assert_eq!(sha_fp, expected_sha);
    }

    #[test]
    fn test_deserialize() {
        let t: Schema = serde_json::from_str("\"string\"").unwrap();
        assert_eq!(
            t,
            Schema::TypeName(TypeName::Primitive(PrimitiveType::String))
        );

        let t: Schema = serde_json::from_str("[\"int\", \"null\"]").unwrap();
        assert_eq!(
            t,
            Schema::Union(vec![
                Schema::TypeName(TypeName::Primitive(PrimitiveType::Int)),
                Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)),
            ])
        );

        let t: Type = serde_json::from_str(
            r#"{
                   "type":"long",
                   "logicalType":"timestamp-micros"
                }"#,
        )
        .unwrap();

        let timestamp = Type {
            r#type: TypeName::Primitive(PrimitiveType::Long),
            attributes: Attributes {
                logical_type: Some("timestamp-micros"),
                additional: Default::default(),
            },
        };

        assert_eq!(t, timestamp);

        let t: ComplexType = serde_json::from_str(
            r#"{
                   "type":"fixed",
                   "name":"fixed",
                   "namespace":"topLevelRecord.value",
                   "size":11,
                   "logicalType":"decimal",
                   "precision":25,
                   "scale":2
                }"#,
        )
        .unwrap();

        let decimal = ComplexType::Fixed(Fixed {
            name: "fixed",
            namespace: Some("topLevelRecord.value"),
            aliases: vec![],
            size: 11,
            attributes: Attributes {
                logical_type: Some("decimal"),
                additional: vec![("precision", json!(25)), ("scale", json!(2))]
                    .into_iter()
                    .collect(),
            },
        });

        assert_eq!(t, decimal);

        let schema: Schema = serde_json::from_str(
            r#"{
               "type":"record",
               "name":"topLevelRecord",
               "fields":[
                  {
                     "name":"value",
                     "type":[
                        {
                           "type":"fixed",
                           "name":"fixed",
                           "namespace":"topLevelRecord.value",
                           "size":11,
                           "logicalType":"decimal",
                           "precision":25,
                           "scale":2
                        },
                        "null"
                     ]
                  }
               ]
            }"#,
        )
        .unwrap();

        assert_eq!(
            schema,
            Schema::Complex(ComplexType::Record(Record {
                name: "topLevelRecord",
                namespace: None,
                doc: None,
                aliases: vec![],
                fields: vec![Field {
                    name: "value",
                    doc: None,
                    r#type: Schema::Union(vec![
                        Schema::Complex(decimal),
                        Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)),
                    ]),
                    default: None,
                },],
                attributes: Default::default(),
            }))
        );

        let schema: Schema = serde_json::from_str(
            r#"{
                  "type": "record",
                  "name": "LongList",
                  "aliases": ["LinkedLongs"],
                  "fields" : [
                    {"name": "value", "type": "long"},
                    {"name": "next", "type": ["null", "LongList"]}
                  ]
                }"#,
        )
        .unwrap();

        assert_eq!(
            schema,
            Schema::Complex(ComplexType::Record(Record {
                name: "LongList",
                namespace: None,
                doc: None,
                aliases: vec!["LinkedLongs"],
                fields: vec![
                    Field {
                        name: "value",
                        doc: None,
                        r#type: Schema::TypeName(TypeName::Primitive(PrimitiveType::Long)),
                        default: None,
                    },
                    Field {
                        name: "next",
                        doc: None,
                        r#type: Schema::Union(vec![
                            Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)),
                            Schema::TypeName(TypeName::Ref("LongList")),
                        ]),
                        default: None,
                    }
                ],
                attributes: Attributes::default(),
            }))
        );

        // Recursive schema are not supported
        let err = AvroField::try_from(&schema).unwrap_err().to_string();
        assert_eq!(err, "Parser error: Failed to resolve .LongList");

        let schema: Schema = serde_json::from_str(
            r#"{
               "type":"record",
               "name":"topLevelRecord",
               "fields":[
                  {
                     "name":"id",
                     "type":[
                        "int",
                        "null"
                     ]
                  },
                  {
                     "name":"timestamp_col",
                     "type":[
                        {
                           "type":"long",
                           "logicalType":"timestamp-micros"
                        },
                        "null"
                     ]
                  }
               ]
            }"#,
        )
        .unwrap();

        assert_eq!(
            schema,
            Schema::Complex(ComplexType::Record(Record {
                name: "topLevelRecord",
                namespace: None,
                doc: None,
                aliases: vec![],
                fields: vec![
                    Field {
                        name: "id",
                        doc: None,
                        r#type: Schema::Union(vec![
                            Schema::TypeName(TypeName::Primitive(PrimitiveType::Int)),
                            Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)),
                        ]),
                        default: None,
                    },
                    Field {
                        name: "timestamp_col",
                        doc: None,
                        r#type: Schema::Union(vec![
                            Schema::Type(timestamp),
                            Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)),
                        ]),
                        default: None,
                    }
                ],
                attributes: Default::default(),
            }))
        );
        let codec = AvroField::try_from(&schema).unwrap();
        assert_eq!(
            codec.field(),
            arrow_schema::Field::new(
                "topLevelRecord",
                DataType::Struct(Fields::from(vec![
                    arrow_schema::Field::new("id", DataType::Int32, true),
                    arrow_schema::Field::new(
                        "timestamp_col",
                        DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into())),
                        true
                    ),
                ])),
                false
            )
        );

        let schema: Schema = serde_json::from_str(
            r#"{
                  "type": "record",
                  "name": "HandshakeRequest", "namespace":"org.apache.avro.ipc",
                  "fields": [
                    {"name": "clientHash", "type": {"type": "fixed", "name": "MD5", "size": 16}},
                    {"name": "clientProtocol", "type": ["null", "string"]},
                    {"name": "serverHash", "type": "MD5"},
                    {"name": "meta", "type": ["null", {"type": "map", "values": "bytes"}]}
                  ]
            }"#,
        )
        .unwrap();

        assert_eq!(
            schema,
            Schema::Complex(ComplexType::Record(Record {
                name: "HandshakeRequest",
                namespace: Some("org.apache.avro.ipc"),
                doc: None,
                aliases: vec![],
                fields: vec![
                    Field {
                        name: "clientHash",
                        doc: None,
                        r#type: Schema::Complex(ComplexType::Fixed(Fixed {
                            name: "MD5",
                            namespace: None,
                            aliases: vec![],
                            size: 16,
                            attributes: Default::default(),
                        })),
                        default: None,
                    },
                    Field {
                        name: "clientProtocol",
                        doc: None,
                        r#type: Schema::Union(vec![
                            Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)),
                            Schema::TypeName(TypeName::Primitive(PrimitiveType::String)),
                        ]),
                        default: None,
                    },
                    Field {
                        name: "serverHash",
                        doc: None,
                        r#type: Schema::TypeName(TypeName::Ref("MD5")),
                        default: None,
                    },
                    Field {
                        name: "meta",
                        doc: None,
                        r#type: Schema::Union(vec![
                            Schema::TypeName(TypeName::Primitive(PrimitiveType::Null)),
                            Schema::Complex(ComplexType::Map(Map {
                                values: Box::new(Schema::TypeName(TypeName::Primitive(
                                    PrimitiveType::Bytes
                                ))),
                                attributes: Default::default(),
                            })),
                        ]),
                        default: None,
                    }
                ],
                attributes: Default::default(),
            }))
        );
    }

    #[test]
    fn test_new_schema_store() {
        let store = SchemaStore::new();
        assert!(store.schemas.is_empty());
    }

    #[test]
    fn test_try_from_schemas_rabin() {
        let schemas = vec![int_schema(), record_schema()];
        let store = SchemaStore::try_from(schemas.as_slice()).unwrap();

        let record_fp = Fingerprint::Rabin(compute_fingerprint_rabin("\"int\""));
        assert_eq!(store.lookup(&record_fp), Some(int_schema()));

        let canonical = generate_canonical_form(&record_schema());
        let rec_fp = Fingerprint::Rabin(compute_fingerprint_rabin(&canonical));
        assert_eq!(store.lookup(&rec_fp), Some(record_schema()));
    }

    #[test]
    fn test_try_from_with_duplicates() {
        let schemas = vec![int_schema(), record_schema(), int_schema()];
        let store = SchemaStore::try_from(schemas.as_slice()).unwrap();
        assert_eq!(store.schemas.len(), 2);
        let int_canonical = r#""int""#;
        let int_fp = compute_fingerprint_rabin(int_canonical);
        assert_eq!(
            store.lookup(&Fingerprint::Rabin(int_fp)),
            Some(int_schema())
        );
    }

    #[test]
    fn test_register_and_lookup_rabin() {
        let mut store = SchemaStore::new();
        let schema = int_schema();
        let fp_enum = store.register(schema.clone()).unwrap();
        let fp_val = match fp_enum {
            Fingerprint::Rabin(v) => v,
            _ => panic!("expected Rabin fingerprint"),
        };

        assert_eq!(
            store.lookup(&Fingerprint::Rabin(fp_val)),
            Some(schema.clone())
        );
        assert!(store
            .lookup(&Fingerprint::Rabin(fp_val.wrapping_add(1)))
            .is_none());
    }

    #[test]
    fn test_register_and_lookup_md5() {
        let mut store = SchemaStore::with_hash_type(HashType::MD5);
        let schema = int_schema();
        let fp_enum = store.register(schema.clone()).unwrap();
        let mut arr = match fp_enum {
            Fingerprint::MD5(a) => a,
            _ => panic!("expected MD5 fingerprint"),
        };
        assert_eq!(store.lookup(&Fingerprint::MD5(arr)), Some(schema.clone()));

        arr[0] ^= 0xFF;
        assert!(store.lookup(&Fingerprint::MD5(arr)).is_none());
    }

    #[test]
    fn test_register_and_lookup_sha256() {
        let mut store = SchemaStore::with_hash_type(HashType::SHA256);
        let schema = int_schema();
        let fp_enum = store.register(schema.clone()).unwrap();
        let mut arr = match fp_enum {
            Fingerprint::SHA256(a) => a,
            _ => panic!("expected SHA256 fingerprint"),
        };
        assert_eq!(
            store.lookup(&Fingerprint::SHA256(arr)),
            Some(schema.clone())
        );

        arr[31] ^= 0xAA;
        assert!(store.lookup(&Fingerprint::SHA256(arr)).is_none());
    }

    #[test]
    fn test_register_duplicate_schema() {
        let mut store = SchemaStore::new();
        let schema1 = int_schema();
        let schema2 = int_schema();
        let fingerprint1 = store.register(schema1).unwrap();
        let fingerprint2 = store.register(schema2).unwrap();
        assert_eq!(fingerprint1, fingerprint2);
        assert_eq!(store.schemas.len(), 1);
    }

    #[test]
    fn test_canonical_form_generation_primitive() {
        let schema = int_schema();
        let canonical_form = generate_canonical_form(&schema);
        assert_eq!(canonical_form, r#""int""#);
    }

    #[test]
    fn test_canonical_form_generation_record() {
        let schema = record_schema();
        let expected_canonical_form = r#"{"fields":[{"name":"field1","type":"int"},{"name":"field2","type":"string"}],"name":"test.namespace.record1","type":"record"}"#;
        let canonical_form = generate_canonical_form(&schema);
        assert_eq!(canonical_form, expected_canonical_form);
    }

    #[test]
    fn test_fingerprint_calculation() {
        let canonical_form = r#"{"fields":[{"name":"a","type":"long"},{"name":"b","type":"string"}],"name":"test","type":"record"}"#;
        let expected_fingerprint = 10505236152925314060;
        let fingerprint = compute_fingerprint_rabin(canonical_form);
        assert_eq!(fingerprint, expected_fingerprint);
    }

    #[test]
    fn test_register_and_lookup_complex_schema() {
        let mut store = SchemaStore::new();
        let schema = record_schema();
        let canonical_form = r#"{"fields":[{"name":"field1","type":"int"},{"name":"field2","type":"string"}],"name":"test.namespace.record1","type":"record"}"#;
        let expected_fingerprint = Fingerprint::Rabin(compute_fingerprint_rabin(canonical_form));
        let fingerprint = store.register(schema.clone()).unwrap();
        assert_eq!(fingerprint, expected_fingerprint);
        let looked_up = store.lookup(&fingerprint);
        assert_eq!(looked_up, Some(schema));
    }

    #[test]
    fn test_canonical_form_strips_attributes() {
        let schema_with_attrs = Schema::Complex(ComplexType::Record(Record {
            name: "record_with_attrs",
            namespace: None,
            doc: Some("This doc should be stripped"),
            aliases: vec!["alias1", "alias2"],
            fields: vec![Field {
                name: "f1",
                doc: Some("field doc"),
                r#type: Schema::Type(Type {
                    r#type: TypeName::Primitive(PrimitiveType::Bytes),
                    attributes: Attributes {
                        logical_type: Some("decimal"),
                        additional: HashMap::from([("precision", json!(4))]),
                    },
                }),
                default: None,
            }],
            attributes: Attributes {
                logical_type: None,
                additional: HashMap::from([("custom_attr", json!("value"))]),
            },
        }));
        let expected_canonical_form = r#"{"fields":[{"name":"f1","type":"bytes"}],"name":"record_with_attrs","type":"record"}"#;
        let canonical_form = generate_canonical_form(&schema_with_attrs);
        assert_eq!(canonical_form, expected_canonical_form);
    }

    #[test]
    fn test_lookup_keys_hash_type() {
        // Test with an empty store
        let store = SchemaStore::new();
        assert_eq!(store.lookup_keys_hash_type(), None);
        // Test with Rabin
        let mut store_rabin = SchemaStore::with_hash_type(HashType::Rabin);
        store_rabin.register(int_schema()).unwrap();
        assert_eq!(store_rabin.lookup_keys_hash_type(), Some(HashType::Rabin));
        // Test with MD5
        let mut store_md5 = SchemaStore::with_hash_type(HashType::MD5);
        store_md5.register(int_schema()).unwrap();
        assert_eq!(store_md5.lookup_keys_hash_type(), Some(HashType::MD5));
        // Test with SHA256
        let mut store_sha256 = SchemaStore::with_hash_type(HashType::SHA256);
        store_sha256.register(int_schema()).unwrap();
        assert_eq!(store_sha256.lookup_keys_hash_type(), Some(HashType::SHA256));
    }
}
