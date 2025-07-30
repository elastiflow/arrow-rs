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

//! Avro writer implementation for the `arrow-avro` crate.
//!
//! # Overview
//!
//! *   Use **[`AvroWriter`]** (Object Container File) when you want a
//!     self‑contained Avro file with header, schema JSON, optional compression,
//!     blocks, and sync markers.
//! *   Use **[`AvroStreamWriter`]** (raw binary stream) when you already know the
//!     schema out‑of‑band (i.e., via a schema registry) and need a stream
//!     of Avro‑encoded records with minimal framing.
//!
//! ```no_run
//! # use std::{fs::File, sync::Arc};
//! # use arrow_array::{Int32Array, RecordBatch};
//! # use arrow_avro::compression::CompressionCodec;
//! # use arrow_schema::{Schema, Field, DataType};
//! # use arrow_avro::writer::{AvroWriter, AvroStreamWriter};
//!
//! // ---------- write an Avro OCF file ----------
//! let schema = Arc::new(
//!     Schema::new(vec![Field::new("a", DataType::Int32, /*nullable=*/ false)])
//! );
//!
//! // some example data
//! let col = Int32Array::from(vec![1, 2, 3]);
//! let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(col)]).unwrap();
//!
//! let file = File::create("data.avro")?;
//! let mut writer = AvroWriter::new(file, schema.clone())?          // default = no compression
//!                     .with_compression(Option::from(CompressionCodec::Deflate)); // enable deflate
//! writer.write(&batch)?;
//! writer.finish()?; // flushes the last block and closes the file
//!
//! // ---------- write an Avro binary *stream* ----------
//! let mut out = Vec::new();
//! let mut stream_writer = AvroStreamWriter::new(&mut out, schema.clone())?;
//! stream_writer.write(&batch)?; // streaming mode: no header, no blocks
//! stream_writer.finish()?;
//! assert!(!out.is_empty());
//! # Ok::<(), arrow_schema::ArrowError>(())
//! ```
//!

/// Encodes `RecordBatch`es into the Avro binary format.
pub mod encoder;
/// Logic for different Avro container file formats.
pub mod format;
/// Converts Arrow schemas to Avro schemas.
pub mod schema;

use std::io::{self, Write};
use std::sync::Arc;

use crate::compression::CompressionCodec;
use crate::writer::encoder::{encode_record_batch, write_long};
use crate::writer::format::{AvroBinaryFormat, AvroFormat, AvroOcfFormat};
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, Schema};

/// Builder to configure and create [`Writer`]s.
#[derive(Debug, Clone)]
pub struct WriterBuilder {
    codec: Option<CompressionCodec>,
}

impl WriterBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self { codec: None }
    }

    /// Change the compression codec.
    pub fn compression(mut self, codec: Option<CompressionCodec>) -> Self {
        self.codec = codec;
        self
    }

    /// Build an **Avro Object Container File** writer.
    pub fn build_ocf<W: Write>(
        self,
        writer: W,
        schema: Arc<Schema>,
    ) -> Result<Writer<W, AvroOcfFormat>, ArrowError> {
        Ok(Writer {
            writer,
            schema,
            format: AvroOcfFormat::default(),
            compression: self.codec,
            started: false,
        })
    }

    /// Build a **raw Avro binary stream** writer.
    pub fn build_stream<W: Write>(
        self,
        writer: W,
        schema: Arc<Schema>,
    ) -> Result<Writer<W, AvroBinaryFormat>, ArrowError> {
        Ok(Writer {
            writer,
            schema,
            format: AvroBinaryFormat::default(),
            compression: self.codec, // ignored for stream mode
            started: false,
        })
    }
}

/// Generic Avro writer.
///
/// The generic parameter **`F`** controls the output format
/// (OCF vs raw stream).  Use the convenient type aliases
/// [`AvroWriter`] and [`AvroStreamWriter`] for common cases.
#[derive(Debug)]
pub struct Writer<W: Write, F: AvroFormat> {
    writer: W,
    schema: Arc<Schema>,
    format: F,
    compression: Option<CompressionCodec>,
    started: bool,
}

/// Alias for an Avro **Object Container File** writer.
pub type AvroWriter<W> = Writer<W, AvroOcfFormat>;
/// Alias for a raw Avro **binary stream** writer.
pub type AvroStreamWriter<W> = Writer<W, AvroBinaryFormat>;

impl<W: Write> Writer<W, AvroOcfFormat> {
    /// Convenience constructor – same as
    pub fn new(writer: W, schema: Arc<Schema>) -> Result<Self, ArrowError> {
        WriterBuilder::new().build_ocf(writer, schema)
    }

    /// Change the compression codec **after** construction.
    /// (mainly for the convenience constructor – the builder already
    /// has `compression()`).
    pub fn with_compression(mut self, codec: Option<CompressionCodec>) -> Self {
        self.compression = codec;
        self
    }

    /// Return a reference to the 16‑byte sync marker generated for this file.
    pub fn sync_marker(&self) -> Option<&[u8; 16]> {
        self.format.sync_marker()
    }
}

impl<W: Write> Writer<W, AvroBinaryFormat> {
    pub fn new(writer: W, schema: Arc<Schema>) -> Result<Self, ArrowError> {
        WriterBuilder::new().build_stream(writer, schema)
    }
}

impl<W: Write, F: AvroFormat> Writer<W, F> {
    /// Serialize one [`RecordBatch`] to the output.
    pub fn write(&mut self, batch: &RecordBatch) -> Result<(), ArrowError> {
        if !self.started {
            self.format
                .start_stream(&mut self.writer, &self.schema, self.compression)?;
            self.started = true;
        }
        if batch.schema() != self.schema {
            return Err(ArrowError::SchemaError(
                "Schema of RecordBatch differs from Writer schema".to_string(),
            ));
        }
        match self.format.sync_marker() {
            Some(&sync) => self.write_ocf_block(batch, &sync),
            None => self.write_stream(batch),
        }
    }

    /// A convenience method to write a slice of [`RecordBatch`].
    ///
    /// This is equivalent to calling [`write`](Self::write) for each batch in the slice.
    pub fn write_batches(&mut self, batches: &[&RecordBatch]) -> Result<(), ArrowError> {
        for b in batches {
            self.write(b)?;
        }
        Ok(())
    }

    /// Flush remaining buffered data and (for OCF) ensure the header is present.
    pub fn finish(&mut self) -> Result<(), ArrowError> {
        if !self.started {
            self.format
                .start_stream(&mut self.writer, &self.schema, self.compression)?;
            self.started = true;
        }
        self.writer
            .flush()
            .map_err(|e| ArrowError::IoError(format!("Error flushing writer: {e}"), e))
    }

    /// Consume the writer, returning the underlying output object.
    ///
    /// **NOTE**: Call [`finish`](Self::finish) first; otherwise the output may
    /// be incomplete or invalid.
    pub fn into_inner(self) -> W {
        self.writer
    }

    fn write_ocf_block(&mut self, batch: &RecordBatch, sync: &[u8; 16]) -> Result<(), ArrowError> {
        let mut buf = Vec::<u8>::with_capacity(1024);
        encode_record_batch(batch, &mut buf)?;
        let (encoded, codec_name) = match self.compression {
            Some(CompressionCodec::Deflate) => (compress_deflate(&buf)?, "deflate"),
            Some(CompressionCodec::Snappy) => (compress_snappy(&buf)?, "snappy"),
            Some(CompressionCodec::Xz) => (compress_zstd(&buf)?, "zstandard"),
            None => (buf, "null"),
            Some(CompressionCodec::ZStandard) | Some(CompressionCodec::Bzip2) => todo!(),
        };
        let count = batch.num_rows() as i64;
        write_long(&mut self.writer, count)?;
        write_long(&mut self.writer, encoded.len() as i64)?;
        self.writer
            .write_all(&encoded)
            .map_err(|e| ArrowError::IoError(format!("Error writing Avro block: {e}"), e))?;
        self.writer
            .write_all(sync)
            .map_err(|e| ArrowError::IoError(format!("Error writing Avro sync: {e}"), e))?;
        Ok(())
    }

    fn write_stream(&mut self, batch: &RecordBatch) -> Result<(), ArrowError> {
        encode_record_batch(batch, &mut self.writer)
    }
}

fn compress_deflate(input: &[u8]) -> Result<Vec<u8>, ArrowError> {
    #[cfg(feature = "deflate")]
    {
        use flate2::{write::DeflateEncoder, Compression};
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(input)
            .map_err(|e| ArrowError::IoError(format!("deflate write error: {e}"), e))?;
        enc.finish()
            .map_err(|e| ArrowError::IoError(format!("deflate finish: {e}"), e))
    }
    #[cfg(not(feature = "deflate"))]
    {
        Err(ArrowError::InvalidArgument(
            "deflate codec support not enabled – activate the `deflate` \
             Cargo feature"
                .into(),
        ))
    }
}

fn compress_snappy(input: &[u8]) -> Result<Vec<u8>, ArrowError> {
    #[cfg(feature = "snappy")]
    {
        use crc32fast::Hasher;
        use snap::raw::{max_compress_len, Encoder};

        let mut enc = Encoder::new();
        let mut out = Vec::with_capacity(max_compress_len(input.len()) + 4);
        enc.compress_vec(input)
            .map_err(|e| ArrowError::ExternalError(format!("snappy compression: {e}").into()))?;

        // Avro spec: append 4‑byte big‑endian CRC32 of *uncompressed* data
        let mut hasher = Hasher::new();
        hasher.update(input);
        let crc = hasher.finalize();
        out.extend_from_slice(&crc.to_be_bytes());

        Ok(out)
    }
    #[cfg(not(feature = "snappy"))]
    {
        Err(ArrowError::InvalidArgument(
            "snappy codec support not enabled – activate the `snappy` \
             Cargo feature"
                .into(),
        ))
    }
}

fn compress_zstd(input: &[u8]) -> Result<Vec<u8>, ArrowError> {
    #[cfg(feature = "zstd")]
    {
        zstd::stream::encode_all(input, /*level*/ 0)
            .map_err(|e| ArrowError::ExternalError(format!("zstd compression: {e}").into()))
    }
    #[cfg(not(feature = "zstd"))]
    {
        Err(ArrowError::InvalidArgument(
            "zstd codec support not enabled – activate the `zstd` \
             Cargo feature"
                .into(),
        ))
    }
}
