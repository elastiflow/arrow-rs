// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

//! Avro file header logic, including magic bytes, metadata map, sync marker.

use std::io::Write;

use arrow_schema::ArrowError;

use crate::compression::CompressionCodec;
use crate::schema::Schema as AvroSchema;
use crate::writer::utils::{
    to_arrow_io_err, write_bytes, write_string,
};
use crate::writer::zigzag::write_zigzag_long;

/// Holds information needed to write the Avro container-file header.
#[derive(Debug)]
pub struct AvroHeader {
    /// The Avro schema (JSON-serialized into metadata)
    pub avro_schema: AvroSchema<'static>,
    /// Optional compression codec
    pub compression: Option<CompressionCodec>,
    /// Additional metadata key-value pairs
    pub extra_meta: Vec<(String, Vec<u8>)>,
    /// The 16-byte sync marker used to separate file blocks
    pub sync_marker: [u8; 16],
}

impl AvroHeader {
    /// Writes the Avro container file header
    pub fn write_header(&self, sink: &mut dyn Write) -> Result<(), ArrowError> {
        sink.write_all(b"Obj\x01")
            .map_err(|e| to_arrow_io_err(e, "Writing Avro magic"))?;
        let mut meta_entries = self.extra_meta.len() + 1; // for avro.schema
        if self.compression.is_some() {
            meta_entries += 1; // for avro.codec
        }
        write_zigzag_long(meta_entries as i64, sink)?;
        let schema_json = serde_json::to_vec(&self.avro_schema)
            .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
        write_string("avro.schema", sink)?;
        write_bytes(&schema_json, sink)?;
        if let Some(codec) = self.compression {
            write_string(crate::compression::CODEC_METADATA_KEY, sink)?;
            let codec_str: &'static [u8] = match codec {
                CompressionCodec::Snappy => b"snappy",
                CompressionCodec::Deflate => b"deflate",
                CompressionCodec::ZStandard => b"zstandard",
                CompressionCodec::Bzip2 => b"bzip2",
                CompressionCodec::Xz => b"xz",
            };
            write_bytes(codec_str, sink)?;
        }
        for (k, v) in &self.extra_meta {
            write_string(k, sink)?;
            write_bytes(v, sink)?;
        }
        write_zigzag_long(0, sink)?;
        sink.write_all(&self.sync_marker)
            .map_err(|e| to_arrow_io_err(e, "Writing sync marker"))?;
        Ok(())
    }
}
