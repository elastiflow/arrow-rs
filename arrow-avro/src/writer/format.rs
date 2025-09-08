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
use crate::schema::{
    AvroSchema, Fingerprint, CONFLUENT_MAGIC, SCHEMA_METADATA_KEY, SINGLE_OBJECT_MAGIC,
};
use crate::writer::encoder::write_long;
use arrow_schema::{ArrowError, Schema};
use rand::RngCore;
use std::fmt::Debug;
use std::io::Write;

/// Format abstraction implemented by each container‐level writer.
pub trait AvroFormat: Debug + Default {
    /// Write any bytes required at the very beginning of the output stream
    /// Implementations **must not** write any record data.
    fn start_stream<W: Write>(
        &mut self,
        writer: &mut W,
        schema: &Schema,
        compression: Option<CompressionCodec>,
    ) -> Result<(), ArrowError>;

    /// Return the 16‑byte sync marker (OCF) or `None` (binary stream).
    fn sync_marker(&self) -> Option<&[u8; 16]>;

    /// Called before writing each row for streaming formats.
    /// Default: no‑op.
    fn write_record_prefix<W: Write>(&mut self, _writer: &mut W) -> Result<(), ArrowError> {
        Ok(())
    }
}

/// Avro Object Container File (OCF) format writer.
#[derive(Debug, Default)]
pub struct AvroOcfFormat {
    sync_marker: [u8; 16],
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

/// Raw Avro binary streaming format (no header or footer).
#[derive(Debug, Default)]
pub struct AvroBinaryFormat;

impl AvroFormat for AvroBinaryFormat {
    fn start_stream<W: Write>(
        &mut self,
        _writer: &mut W,
        _schema: &Schema,
        compression: Option<CompressionCodec>,
    ) -> Result<(), ArrowError> {
        // Binary streams have no header/sync; compression isn't supported in this mode.
        if compression.is_some() {
            return Err(ArrowError::InvalidArgumentError(
                "AvroBinaryFormat does not support compression".to_string(),
            ));
        }
        Ok(())
    }

    fn sync_marker(&self) -> Option<&[u8; 16]> {
        None
    }
}

/// Avro **Single‑Object Encoding** writer.
///
/// Per Avro 1.11.1, each datum is preceded by:
///  1. Two‑byte marker `C3 01`,
///  2. The 8‑byte **little‑endian** CRC‑64‑AVRO fingerprint of the writer schema,
///  3. The datum in Avro **binary encoding**.
///
/// This implementation also mirrors reader support for:
///  • **Confluent wire format** when given `Fingerprint::Id(id)` - `0x00` + 4‑bytes **big‑endian** ID and payload.
///  • Feature‑gated MD5 / SHA‑256 variants (still prefixed by `C3 01`).
#[derive(Debug, Default)]
pub struct AvroSingleObjectFormat {
    /// The header written before each datum (magic and fingerprint bytes).
    header: Vec<u8>,
    /// The writer‑schema fingerprint this stream uses (exposed for tests).
    fingerprint: Option<Fingerprint>,
}

impl AvroSingleObjectFormat {
    fn build_header(fp: &Fingerprint) -> Vec<u8> {
        let mut out = Vec::new();
        match fp {
            Fingerprint::Rabin(u) => {
                out.extend_from_slice(&SINGLE_OBJECT_MAGIC);
                out.extend_from_slice(&u.to_le_bytes());
            }
            Fingerprint::Id(id) => {
                out.extend_from_slice(&CONFLUENT_MAGIC);
                out.extend_from_slice(&id.to_be_bytes());
            }
            #[cfg(feature = "md5")]
            Fingerprint::MD5(bytes) => {
                out.extend_from_slice(&SINGLE_OBJECT_MAGIC);
                out.extend_from_slice(bytes);
            }
            #[cfg(feature = "sha256")]
            Fingerprint::SHA256(bytes) => {
                out.extend_from_slice(&SINGLE_OBJECT_MAGIC);
                out.extend_from_slice(bytes);
            }
        }
        out
    }

    /// Returns the computed writer‑schema fingerprint (if available).
    pub fn fingerprint(&self) -> Option<Fingerprint> {
        self.fingerprint
    }
}

impl AvroFormat for AvroSingleObjectFormat {
    fn start_stream<W: Write>(
        &mut self,
        _writer: &mut W,
        schema: &Schema,
        compression: Option<CompressionCodec>,
    ) -> Result<(), ArrowError> {
        if compression.is_some() {
            return Err(ArrowError::InvalidArgumentError(
                "AvroSingleObjectFormat does not support compression".to_string(),
            ));
        }
        let avro_schema = AvroSchema::try_from(schema)?;
        let fp = avro_schema.fingerprint()?;
        self.header = Self::build_header(&fp);
        self.fingerprint = Some(fp);
        Ok(())
    }

    fn sync_marker(&self) -> Option<&[u8; 16]> {
        None
    }

    fn write_record_prefix<W: Write>(&mut self, writer: &mut W) -> Result<(), ArrowError> {
        writer
            .write_all(&self.header)
            .map_err(|e| ArrowError::IoError(format!("write single-object prefix: {e}"), e))
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
