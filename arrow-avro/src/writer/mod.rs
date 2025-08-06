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
//! # use arrow_schema::{Schema, Field, DataType, SchemaRef};
//! # use arrow_avro::writer::{AvroWriter, AvroStreamWriter};
//!
//! let schema =
//!     Schema::new(vec![Field::new("a", DataType::Int32, /*nullable=*/ false)]);
//!
//! // some example data
//! let col = Int32Array::from(vec![1, 2, 3]);
//! let batch = RecordBatch::try_new(SchemaRef::from(schema.clone()), vec![Arc::new(col)]).unwrap();
//!
//! let file = File::create("data.avro")?;
//! let mut writer = AvroWriter::new(file, schema.clone())?          // default = no compression
//!                     .with_compression(Option::from(CompressionCodec::Deflate)); // enable deflate
//! writer.write(&batch)?;
//! writer.finish()?; // flushes the last block and closes the file
//!
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
use crate::compression::CompressionCodec;
use crate::schema::AvroSchema;
use crate::writer::encoder::{encode_record_batch, write_long};
use crate::writer::format::{AvroBinaryFormat, AvroFormat, AvroOcfFormat};
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, Schema};
use std::io::{self, Write};
use std::sync::Arc;

/// Builder to configure and create [`Writer`]s.
#[derive(Debug, Clone)]
pub struct WriterBuilder {
    schema: Schema,
    codec: Option<CompressionCodec>,
}

impl WriterBuilder {
    /// Create a new builder with default settings.
    pub fn new(schema: Schema) -> Self {
        Self {
            schema,
            codec: None,
        }
    }

    /// Change the compression codec.
    pub fn with_compression(mut self, codec: Option<CompressionCodec>) -> Self {
        self.codec = codec;
        self
    }

    /// Create a new `Writer` with specified `JsonFormat` and builder options.
    pub fn build<W, F>(self, writer: W) -> Writer<W, F>
    where
        W: Write,
        F: AvroFormat,
    {
        Writer {
            writer,
            schema: Arc::from(self.schema),
            format: F::default(),
            compression: self.codec,
            started: false,
        }
    }
}

/// Generic Avro writer.
///
/// The generic parameter **`F`** controls the output format
/// (OCF vs. raw stream).  Use the convenient type aliases
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
    pub fn new(writer: W, schema: Schema) -> Result<Self, ArrowError> {
        Ok(WriterBuilder::new(schema).build::<W, AvroOcfFormat>(writer))
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
    /// Convenience constructor to create a new [`AvroStreamWriter`].
    pub fn new(writer: W, schema: Schema) -> Result<Self, ArrowError> {
        Ok(WriterBuilder::new(schema).build::<W, AvroBinaryFormat>(writer))
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
        let encoded = match self.compression {
            Some(codec) => codec.compress(&buf)?,
            None => buf,
        };
        write_long(&mut self.writer, batch.num_rows() as i64)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    fn make_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ])
    }

    fn make_batch() -> RecordBatch {
        let ids = Int32Array::from(vec![1, 2, 3]);
        let names = StringArray::from(vec!["a", "b", "c"]);
        RecordBatch::try_new(
            Arc::new(make_schema()),
            vec![Arc::new(ids) as ArrayRef, Arc::new(names)],
        )
        .expect("failed to build test RecordBatch")
    }

    fn contains_ascii(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }

    #[test]
    fn test_ocf_writer_generates_header_and_sync() -> Result<(), ArrowError> {
        let batch = make_batch();
        let buffer: Vec<u8> = Vec::new();
        let mut writer = AvroWriter::new(buffer, make_schema())?;
        writer.write(&batch)?;
        writer.finish()?; // flush
        let out = writer.into_inner();
        assert_eq!(&out[..4], b"Obj\x01", "OCF magic bytes missing/incorrect");
        let sync = AvroWriter::new(Vec::new(), make_schema())?
            .sync_marker()
            .cloned();
        let trailer = &out[out.len() - 16..];
        assert_eq!(trailer.len(), 16, "expected 16‑byte sync marker");
        let _ = sync;
        Ok(())
    }

    #[test]
    fn test_stream_writer_has_no_header() -> Result<(), ArrowError> {
        let batch = make_batch();
        let buffer = Vec::<u8>::new();
        let mut writer = AvroStreamWriter::new(buffer, make_schema())?;
        writer.write(&batch)?;
        writer.finish()?;
        let out = writer.into_inner();
        assert_ne!(&out[..4], b"Obj\x01", "stream must not contain OCF header");
        assert!(!out.is_empty(), "stream output unexpectedly empty");
        Ok(())
    }

    #[test]
    fn test_schema_mismatch_yields_error() {
        let batch = make_batch();
        let alt_schema = Schema::new(vec![Field::new("x", DataType::Int32, false)]);
        let buffer = Vec::<u8>::new();
        let mut writer = AvroWriter::new(buffer, alt_schema).unwrap();
        let err = writer.write(&batch).unwrap_err();
        assert!(matches!(err, ArrowError::SchemaError(_)));
    }

    #[test]
    fn test_write_batches_accumulates_multiple() -> Result<(), ArrowError> {
        let batch1 = make_batch();
        let batch2 = make_batch();
        let buffer = Vec::<u8>::new();
        let mut writer = AvroWriter::new(buffer, make_schema())?;
        writer.write_batches(&[&batch1, &batch2])?;
        writer.finish()?;
        let out = writer.into_inner();
        assert!(out.len() > 4, "combined batches produced tiny file");
        Ok(())
    }

    #[test]
    fn test_finish_without_write_adds_header() -> Result<(), ArrowError> {
        let buffer = Vec::<u8>::new();
        let mut writer = AvroWriter::new(buffer, make_schema())?;
        writer.finish()?;
        let out = writer.into_inner();
        assert_eq!(&out[..4], b"Obj\x01", "finish() should emit OCF header");
        Ok(())
    }

    #[test]
    fn test_into_inner_returns_underlying_writer() -> Result<(), ArrowError> {
        let batch = make_batch();
        let buffer = Vec::<u8>::new();
        let mut writer = AvroWriter::new(buffer, make_schema())?;
        writer.write(&batch)?;
        writer.finish()?;
        let buf = writer.into_inner();
        assert!(!buf.is_empty(), "into_inner returned empty buffer");
        Ok(())
    }

    #[cfg(all(feature = "deflate"))]
    #[test]
    fn test_deflate_compression_codec() -> Result<(), ArrowError> {
        let batch = make_batch();
        let buffer = Vec::<u8>::new();
        let mut writer = AvroWriter::new(buffer, make_schema())?
            .with_compression(Some(CompressionCodec::Deflate));
        writer.write(&batch)?;
        writer.finish()?;
        let out = writer.into_inner();
        assert!(
            contains_ascii(&out, b"deflate"),
            "OCF header must advertise deflate codec"
        );
        Ok(())
    }

    #[cfg(all(feature = "snappy"))]
    #[test]
    fn test_snappy_compression_codec() -> Result<(), ArrowError> {
        let batch = make_batch();
        let buffer = Vec::<u8>::new();
        let mut writer = AvroWriter::new(buffer, make_schema())?
            .with_compression(Some(CompressionCodec::Snappy));
        writer.write(&batch)?;
        writer.finish()?;
        let out = writer.into_inner();
        assert!(
            contains_ascii(&out, b"snappy"),
            "OCF header must advertise snappy codec"
        );
        Ok(())
    }

    #[cfg(all(feature = "zstd"))]
    #[test]
    fn test_zstd_compression_codec() -> Result<(), ArrowError> {
        let batch = make_batch();
        let buffer = Vec::<u8>::new();
        let mut writer = AvroWriter::new(buffer, make_schema())?
            .with_compression(Some(CompressionCodec::ZStandard));
        writer.write(&batch)?;
        writer.finish()?;
        let out = writer.into_inner();
        assert!(
            contains_ascii(&out, b"zstandard"),
            "OCF header must advertise zstandard codec"
        );
        Ok(())
    }

    #[test]
    fn test_write_long_encodes_zigzag_varint() -> Result<(), ArrowError> {
        let mut buf = Vec::new();
        write_long(&mut buf, 0)?;
        write_long(&mut buf, -1)?;
        write_long(&mut buf, 1)?;
        write_long(&mut buf, -2)?;
        write_long(&mut buf, 2147483647)?;
        assert!(
            buf.starts_with(&[0x00, 0x01, 0x02, 0x03]),
            "zig‑zag varint encodings incorrect: {buf:?}"
        );
        Ok(())
    }
}
