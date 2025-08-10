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
use crate::compression::CompressionCodec;
use crate::schema::AvroSchema;
use crate::writer::encoder::{write_long, EncoderOptions};
use arrow_schema::{ArrowError, Schema};
use rand::RngCore;
use serde_json::{Map as JsonMap, Value as JsonValue};
use std::fmt::Debug;
use std::io::Write;

/// Format abstraction implemented by each container‐level writer.
///
/// Only **stream‑start** handling is delegated here; record/batch encoding and
/// block framing are handled by the top‑level [`Writer`](crate::writer::Writer).
pub trait AvroFormat: Debug + Default {
    /// Write any bytes required at the very beginning of the output stream
    /// (file header, etc.).
    /// Implementations **must not** write any record data.
    fn start_stream<W: Write>(
        &mut self,
        writer: &mut W,
        schema: &Schema,
        compression: Option<CompressionCodec>,
    ) -> Result<(), ArrowError>;

    /// Return the 16‑byte sync marker (OCF) or `None` (binary stream).
    fn sync_marker(&self) -> Option<&[u8; 16]>;
}

/// Avro Object Container File (OCF) format writer.
///
/// Writes the OCF header:
/// - Magic `Obj\x01`
/// - Metadata map with `avro.schema` and `avro.codec`
/// - 16‑byte sync marker
///
/// See: Avro 1.11.1 Spec – Object Container Files, header layout and metadata.
/// The schema JSON is optionally post-processed to flip union null ordering
/// for Impala null-second style when `encoder_options.impala_mode = true`.
#[derive(Debug)]
pub struct AvroOcfFormat {
    sync_marker: [u8; 16],
    /// Optional encoder behavior hints to keep file header schema ordering
    /// consistent with value encoding (e.g. Impala null-second).
    encoder_options: EncoderOptions,
}

impl Default for AvroOcfFormat {
    fn default() -> Self {
        Self {
            sync_marker: [0u8; 16],
            encoder_options: EncoderOptions::default(),
        }
    }
}

impl AvroOcfFormat {
    /// Optional helper to attach encoder options (e.g. Impala null-second) to the format.
    /// This does **not** change the `AvroFormat` trait or `Writer` public API.
    #[allow(dead_code)]
    pub fn with_encoder_options(mut self, opts: EncoderOptions) -> Self {
        self.encoder_options = opts;
        self
    }

    /// Access the options used by this format.
    #[allow(dead_code)]
    pub fn encoder_options(&self) -> &EncoderOptions {
        &self.encoder_options
    }
}

impl AvroFormat for AvroOcfFormat {
    fn start_stream<W: Write>(
        &mut self,
        writer: &mut W,
        schema: &Schema,
        compression: Option<CompressionCodec>,
    ) -> Result<(), ArrowError> {
        // Generate a fresh 16‑byte sync marker
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut self.sync_marker);

        // Build the Avro schema JSON from the Arrow schema
        let avro_schema = AvroSchema::try_from(schema)?;

        // Optionally rewrite union ordering for nullable fields to match encoding behavior.
        // When `impala_mode` is true: ensure ["<type>", "null"] ordering everywhere.
        // Otherwise we leave the JSON as produced (typically ["null", "<type>"]).
        let schema_json = if self.encoder_options.impala_mode {
            maybe_rewrite_nullable_union_order(
                &avro_schema.json_string,
                /*null_second=*/ true,
            )?
        } else {
            avro_schema.json_string
        };

        // 1) Magic: "Obj\x01"
        writer
            .write_all(b"Obj\x01")
            .map_err(|e| ArrowError::IoError(format!("write OCF magic: {e}"), e))?;

        // 2) Metadata map (encoded as Avro map: one positive-length block then 0)
        // Keys are strings; values are bytes. We write exactly two entries:
        // "avro.schema" -> <schema json utf-8>
        // "avro.codec"  -> <codec name utf-8>
        // See Avro Spec: OCF header metadata keys.
        let codec_str = match compression {
            Some(CompressionCodec::Deflate) => "deflate",
            Some(CompressionCodec::Snappy) => "snappy",
            Some(CompressionCodec::ZStandard) => "zstandard",
            Some(CompressionCodec::Bzip2) => "bzip2",
            Some(CompressionCodec::Xz) => "xz",
            None => "null",
        };

        // Map block count = 2
        write_long(writer, 2)?;
        write_string(writer, "avro.schema")?;
        write_bytes(writer, schema_json.as_bytes())?;
        write_string(writer, "avro.codec")?;
        write_bytes(writer, codec_str.as_bytes())?;
        // Map terminator
        write_long(writer, 0)?;

        // 3) Sync marker (16 bytes)
        writer
            .write_all(&self.sync_marker)
            .map_err(|e| ArrowError::IoError(format!("write OCF sync marker: {e}"), e))?;

        Ok(())
    }

    fn sync_marker(&self) -> Option<&[u8; 16]> {
        Some(&self.sync_marker)
    }
}

/// Raw Avro binary streaming format (no header or footer).
#[derive(Debug, Default)]
pub struct AvroBinaryFormat;

impl AvroFormat for AvroBinaryFormat {
    fn start_stream<W: Write>(
        &mut self,
        _writer: &mut W,
        _schema: &Schema,
        _compression: Option<CompressionCodec>,
    ) -> Result<(), ArrowError> {
        // Nothing to do for raw stream.
        Ok(())
    }

    fn sync_marker(&self) -> Option<&[u8; 16]> {
        None
    }
}

#[inline]
fn write_string<W: Write>(writer: &mut W, s: &str) -> Result<(), ArrowError> {
    write_bytes(writer, s.as_bytes())
}

#[inline]
fn write_bytes<W: Write>(writer: &mut W, bytes: &[u8]) -> Result<(), ArrowError> {
    write_long(writer, bytes.len() as i64)?;
    writer
        .write_all(bytes)
        .map_err(|e| ArrowError::IoError(format!("write bytes: {e}"), e))
}

/// If `null_second` is true, traverse the Avro schema JSON and force all
/// nullable unions to the form `[T, "null"]`. If false, no changes are made.
///
/// This is a **best-effort** transformation intended to keep header schema
/// ordering in sync with encoder behavior when Impala-style unions are desired.
/// It only reorders unions that contain `null` (either as the string `"null"`
/// or as an object with `"type":"null"`), preserving the relative order of
/// non-null entries.
///
/// The traversal visits:
/// - object `"type"` values
/// - array union type nodes
/// - nested `"items"`, `"values"` (for arrays/maps)
/// - nested record `"fields"[i].type`
///
/// On JSON parse errors we surface an Arrow error rather than silently
/// proceeding with a mismatched schema.
fn maybe_rewrite_nullable_union_order(json: &str, null_second: bool) -> Result<String, ArrowError> {
    let mut v: JsonValue = serde_json::from_str(json)
        .map_err(|e| ArrowError::ParseError(format!("parse Avro schema JSON: {e}")))?;
    visit_schema_type(&mut v, null_second);
    serde_json::to_string(&v)
        .map_err(|e| ArrowError::ParseError(format!("serialize Avro schema JSON: {e}")))
}

fn visit_schema_type(node: &mut JsonValue, null_second: bool) {
    match node {
        JsonValue::Array(alts) => {
            reorder_null_in_union(alts, null_second);
            // Also visit nested union members (they can be objects with types)
            for m in alts.iter_mut() {
                visit_schema_type(m, null_second);
            }
        }
        JsonValue::Object(obj) => visit_schema_object(obj, null_second),
        _ => {
            // primitives and strings: nothing to do
        }
    }
}

fn visit_schema_object(obj: &mut JsonMap<String, JsonValue>, null_second: bool) {
    // Recurse into "type"
    if let Some(t) = obj.get_mut("type") {
        visit_schema_type(t, null_second);
    }
    // Recurse into array/map element/value types
    if let Some(items) = obj.get_mut("items") {
        visit_schema_type(items, null_second);
    }
    if let Some(values) = obj.get_mut("values") {
        visit_schema_type(values, null_second);
    }
    // Recurse into record fields
    if let Some(JsonValue::Array(fields)) = obj.get_mut("fields") {
        for f in fields.iter_mut() {
            if let JsonValue::Object(field_obj) = f {
                if let Some(t) = field_obj.get_mut("type") {
                    visit_schema_type(t, null_second);
                }
            }
        }
    }
}

fn is_json_null_type(v: &JsonValue) -> bool {
    match v {
        JsonValue::String(s) => s == "null",
        JsonValue::Object(o) => o.get("type").and_then(|t| t.as_str()) == Some("null"),
        _ => false,
    }
}

fn reorder_null_in_union(alts: &mut Vec<JsonValue>, null_second: bool) {
    if !alts.iter().any(is_json_null_type) {
        return;
    }
    let mut nulls = Vec::new();
    let mut others = Vec::new();
    for v in alts.drain(..) {
        if is_json_null_type(&v) {
            nulls.push(v);
        } else {
            others.push(v);
        }
    }
    if null_second {
        alts.extend(others);
        alts.extend(nulls);
    } else {
        alts.extend(nulls);
        alts.extend(others);
    }
}
