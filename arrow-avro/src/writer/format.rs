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

use crate::compression::{CompressionCodec, CODEC_METADATA_KEY};
use crate::schema::{AvroSchema, Fingerprint, SCHEMA_METADATA_KEY, SINGLE_OBJECT_MAGIC};
use crate::writer::encoder::{write_long, EncoderOptions};
use arrow_schema::{ArrowError, Schema};
use rand::RngCore;
use serde_json::{Map as JsonMap, Value as JsonValue};
use std::fmt::Debug;
use std::io::Write;

/// Format abstraction implemented by each container‐level writer.
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

    /// Return the 10‑byte **Avro single‑object** prefix (`C3 01` magic + 8‑byte
    /// little‑endian schema fingerprint) to be written **before each record**,
    /// or `None` if the format does not use single‑object encoding.
    ///
    /// The default implementation returns `None`. `AvroBinaryFormat` overrides this
    /// to return `Some(&[u8; 10])`.
    #[inline]
    fn single_object_prefix(&self) -> Option<&[u8; 10]> {
        None
    }
}

/// Avro Object Container File (OCF) format writer.
#[derive(Debug, Default)]
pub struct AvroOcfFormat {
    sync_marker: [u8; 16],
    /// Optional encoder behavior hints to keep file header schema ordering
    /// consistent with value encoding (e.g. Impala null-second).
    encoder_options: EncoderOptions,
}

impl AvroOcfFormat {
    /// Optional helper to attach encoder options (i.e., Impala null-second) to the format.
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
        let mut rng = rand::rng();
        rng.fill_bytes(&mut self.sync_marker);
        let avro_schema = AvroSchema::try_from(schema)?;
        writer
            .write_all(b"Obj\x01")
            .map_err(|e| ArrowError::IoError(format!("write OCF magic: {e}"), e))?;
        let codec_str = match compression {
            Some(CompressionCodec::Deflate) => "deflate",
            Some(CompressionCodec::Snappy) => "snappy",
            Some(CompressionCodec::ZStandard) => "zstandard",
            Some(CompressionCodec::Bzip2) => "bzip2",
            Some(CompressionCodec::Xz) => "xz",
            None => "null",
        };
        write_long(writer, 2)?;
        write_string(writer, SCHEMA_METADATA_KEY)?;
        write_bytes(writer, avro_schema.json_string.as_bytes())?;
        write_string(writer, CODEC_METADATA_KEY)?;
        write_bytes(writer, codec_str.as_bytes())?;
        write_long(writer, 0)?;
        // Sync marker (16 bytes)
        writer
            .write_all(&self.sync_marker)
            .map_err(|e| ArrowError::IoError(format!("write OCF sync marker: {e}"), e))?;

        Ok(())
    }

    fn sync_marker(&self) -> Option<&[u8; 16]> {
        Some(&self.sync_marker)
    }
}

/// Raw Avro binary streaming format using **Single-Object Encoding** per record.
///
/// Each record written by the stream writer is framed as:
///   * 2-byte magic: `0xC3, 0x01` (Avro single-object version 1)
///   * 8-byte little-endian CRC-64-AVRO fingerprint of the Avro schema
///   * Avro-encoded record bytes
///
/// See: <https://avro.apache.org/docs/1.11.1/specification/#single-object-encoding>
#[derive(Debug, Default)]
pub struct AvroBinaryFormat {
    /// Pre-built 10-byte single-object prefix written before each record.
    /// [0..2) = magic, [2..10) = schema fingerprint (little endian)
    prefix: [u8; 10],
    /// Encoder behavior knobs (currently unused by the format layer, but kept
    /// for symmetry with OCF and potential future use).
    encoder_options: EncoderOptions,
}

impl AvroBinaryFormat {
    /// Optional helper to attach encoder options (i.e., Impala null-second) to the format.
    #[allow(dead_code)]
    pub fn with_encoder_options(mut self, opts: EncoderOptions) -> Self {
        self.encoder_options = opts;
        self
    }

    /// Returns the 10-byte prefix (`C3 01` + fingerprint) to be written before each record.
    #[inline]
    pub fn prefix(&self) -> &[u8; 10] {
        &self.prefix
    }

    /// Returns the 8-byte little-endian schema fingerprint portion of the prefix.
    #[allow(dead_code)]
    #[inline]
    pub fn schema_fingerprint(&self) -> [u8; 8] {
        let mut fp = [0u8; 8];
        fp.copy_from_slice(&self.prefix[2..10]);
        fp
    }

    /// Access the encoder options used by this format.
    #[allow(dead_code)]
    #[inline]
    pub fn encoder_options(&self) -> &EncoderOptions {
        &self.encoder_options
    }
}

impl AvroFormat for AvroBinaryFormat {
    fn start_stream<W: Write>(
        &mut self,
        _writer: &mut W,
        schema: &Schema,
        compression: Option<CompressionCodec>,
    ) -> Result<(), ArrowError> {
        // Avro single-object encoding does **not** support per-record compression.
        if compression.is_some() {
            return Err(ArrowError::InvalidArgumentError(
                "Compression not supported for Avro binary streaming (single-object encoding)".to_string(),
            ));
        }
        // Compute and stash the schema fingerprint (CRC-64-AVRO) once.
        let avro_schema = AvroSchema::try_from(schema)?;
        let fp = match avro_schema.fingerprint()? {
            Fingerprint::Rabin(v) => v.to_le_bytes(),
        };
        // Build the 10-byte prefix: magic (2) + fingerprint (8)
        self.prefix[..2].copy_from_slice(&SINGLE_OBJECT_MAGIC);
        self.prefix[2..].copy_from_slice(&fp);
        Ok(())
    }

    fn sync_marker(&self) -> Option<&[u8; 16]> {
        None
    }

    #[inline]
    fn single_object_prefix(&self) -> Option<&[u8; 10]> {
        Some(&self.prefix)
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
