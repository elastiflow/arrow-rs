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

use arrow_schema::{ArrowError, DataType, Field, Schema, TimeUnit};
use serde_json::{json, Map, Number, Value};

/// Convert an Arrow [`Schema`] to a compact Avro **record** schema in JSON form.
///
/// The returned string is suitable for the mandatory `"avro.schema"` file
/// metadata entry (OCF) or for use with a schema registry.  All field and
/// record names are sanitized to conform to the Avro naming rules:
/// `[A‑Za‑z_][A‑Za‑z0‑9_]*`.
///
/// *No* compression‑related information is embedded in the schema; the
/// `codec` parameter is accepted only because the surrounding API passes it.
pub fn to_avro_schema_json(
    schema: &Schema,
    _codec: Option<crate::writer::CompressionCodec>,
) -> Result<String, ArrowError> {
    fn sanitize(name: &str) -> String {
        let mut out = String::with_capacity(name.len() + 1);
        let mut chars = name.chars();
        // first char
        match chars.next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => out.push(c),
            Some(c) => {
                // prepend underscore if the first char is invalid
                if c.is_ascii_digit() {
                    out.push('_');
                    out.push(c);
                } else {
                    out.push('_');
                }
            }
            None => out.push('_'),
        }
        // rest
        for c in chars {
            if c.is_ascii_alphanumeric() || c == '_' {
                out.push(c);
            } else {
                out.push('_');
            }
        }
        // never let the name be empty
        if out.is_empty() {
            "_".into()
        } else {
            out
        }
    }

    /// Convert a *data type* (not nullable) to an Avro `Value`.
    fn data_type_to_avro_type(dt: &DataType, ctx_name: &str) -> Result<Value, ArrowError> {
        Ok(match dt {
            DataType::Boolean => json!("boolean"),
            DataType::Int8 | DataType::Int16 | DataType::Int32 => json!("int"),
            DataType::Int64 => json!("long"),
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                json!("long")
            }
            DataType::Float32 => json!("float"),
            DataType::Float64 => json!("double"),
            DataType::Utf8 | DataType::LargeUtf8 => json!("string"),
            DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_) => {
                json!("bytes")
            }
            DataType::Decimal128(precision, scale) => json!({
                "type": "bytes",
                "logicalType": "decimal",
                "precision": precision,
                "scale": scale,
            }),
            DataType::Date32 => json!({
                "type": "int",
                "logicalType": "date"
            }),
            DataType::Date64 => json!({
                "type": "long",
                "logicalType": "timestamp-millis"
            }),
            DataType::Time32(_) => json!({
                "type": "int",
                "logicalType": "time-millis"
            }),
            DataType::Time64(TimeUnit::Microsecond) => json!({
                "type": "long",
                "logicalType": "time-micros"
            }),
            DataType::Time64(_) => json!("long"),
            DataType::Timestamp(TimeUnit::Second, _)
            | DataType::Timestamp(TimeUnit::Millisecond, _) => json!({
                "type": "long",
                "logicalType": "timestamp-millis"
            }),
            DataType::Timestamp(TimeUnit::Microsecond, _) => json!({
                "type": "long",
                "logicalType": "timestamp-micros"
            }),
            DataType::Timestamp(_, _) => json!("long"),
            DataType::Struct(fields) => {
                let rec_name = sanitize(ctx_name);
                let mut field_vec = Vec::<Value>::with_capacity(fields.len());
                for f in fields {
                    field_vec.push(field_to_avro_field(f, &rec_name)?);
                }
                json!({
                    "type": "record",
                    "name": rec_name,
                    "fields": field_vec,
                })
            }
            DataType::List(child) | DataType::LargeList(child) => json!({
                "type": "array",
                "items": data_type_to_avro_type(
                    child.data_type(),
                    &format!("{}_item", ctx_name)
                )?
            }),
            DataType::Map(field, _) => {
                // Arrow Map = list of struct<key, value>
                let struct_dt = field.data_type();
                let DataType::Struct(ref kv_fields) = struct_dt else {
                    return Err(ArrowError::InvalidArgumentError(
                        "Map child is not Struct".into(),
                    ));
                };
                if kv_fields.len() != 2 {
                    return Err(ArrowError::InvalidArgumentError(
                        "Map struct must have exactly 2 fields".into(),
                    ));
                }
                let key_field = &kv_fields[0];
                let val_field = &kv_fields[1];
                match key_field.data_type() {
                    DataType::Utf8 | DataType::LargeUtf8 => (),
                    _ => {
                        return Err(ArrowError::InvalidArgumentError(
                            "Avro only supports string keys in map".into(),
                        ))
                    }
                }
                if key_field.is_nullable() {
                    return Err(ArrowError::InvalidArgumentError(
                        "Map keys must be non-nullable".into(),
                    ));
                }
                json!({
                    "type": "map",
                    "values": data_type_to_avro_type(
                        val_field.data_type(),
                        &format!("{}_value", ctx_name)
                    )?
                })
            }
            DataType::Union(field_set, _mode) => {
                let variants: Vec<Value> = field_set
                    .iter()
                    .map(|(_id, f)| {
                        data_type_to_avro_type(
                            f.data_type(),
                            &format!("{}_{}", ctx_name, sanitize(f.name())),
                        )
                    })
                    .collect::<Result<_, _>>()?;
                Value::Array(variants)
            }
            _ => {
                return Err(ArrowError::NotYetImplemented(format!(
                    "DataType {:?} not supported for Avro writing",
                    dt
                )))
            }
        })
    }

    /// Convert an Arrow *field* to an Avro field object with `"name"` + `"type"`.
    fn field_to_avro_field(field: &Field, ctx_name: &str) -> Result<Value, ArrowError> {
        let avro_name = sanitize(field.name());
        let base_type = data_type_to_avro_type(field.data_type(), ctx_name)?;

        let avro_type = if field.is_nullable() {
            // wrap in ["null", <type>] union
            Value::Array(vec![json!("null"), base_type])
        } else {
            base_type
        };
        Ok(json!({
            "name": avro_name,
            "type": avro_type
        }))
    }
    let mut top_fields = Vec::<Value>::with_capacity(schema.fields().len());
    for f in schema.fields() {
        top_fields.push(field_to_avro_field(f, "arrow_schema")?);
    }
    let top_record = json!({
        "type": "record",
        "name": "arrow_schema",
        "fields": top_fields,
    });
    serde_json::to_string(&top_record)
        .map_err(|e| ArrowError::JsonError(format!("serialising Avro schema: {e}")))
}
