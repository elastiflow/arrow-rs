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


//! This module provides writing capabilities for Avro container files,

mod block;
mod header;
pub mod encoder;
pub mod zigzag;
mod utils;

use std::io::Write;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{ArrowError, SchemaRef};

use crate::codec::arrow_schema_to_avro_schema;
use crate::compression::CompressionCodec;
use crate::schema::Schema as AvroSchema;

use block::BlockEncoder;
use encoder::RecordEncoder;
use header::AvroHeader;

/// A builder for creating an Avro [`Writer`] for Arrow data.
///
/// Use this builder to configure options like compression, max block size,
/// custom Avro schema, etc., before creating the writer.
pub struct WriterBuilder<W: Write> {
    /// The underlying output [`Write`] to which Avro data is written.
    writer: W,

    /// The Arrow schema used for converting columns to Avro.
    arrow_schema: SchemaRef,

    /// Optional user-supplied Avro schema that overrides the Arrow-generated schema.
    avro_schema: Option<AvroSchema<'static>>,

    /// If present, enables block-level compression (e.g. Snappy).
    compression: Option<CompressionCodec>,

    /// The threshold in bytes at which the writer flushes a data block.
    max_block_size: usize,

    /// Arbitrary additional metadata key-value pairs stored in the Avro file header.
    extra_meta: Vec<(String, Vec<u8>)>,

    /// If provided, the 16-byte sync marker used in the file; otherwise generated.
    sync_marker: Option<[u8; 16]>,
}

impl<W: Write> WriterBuilder<W> {
    /// Create a new `WriterBuilder` with the specified `writer` and Arrow schema.
    pub fn new(writer: W, arrow_schema: SchemaRef) -> Self {
        Self {
            writer,
            arrow_schema,
            avro_schema: None,
            compression: None,
            max_block_size: 16 * 1024 * 1024,
            extra_meta: vec![],
            sync_marker: None,
        }
    }

    /// Provide a custom Avro schema, overriding the derived Arrow-to-Avro conversion.
    pub fn with_avro_schema(mut self, avro_schema: AvroSchema<'static>) -> Self {
        self.avro_schema = Some(avro_schema);
        self
    }

    /// Enable block-level compression (e.g. Snappy, Deflate, etc.).
    pub fn with_compression(mut self, codec: CompressionCodec) -> Self {
        self.compression = Some(codec);
        self
    }

    /// Set the maximum in-memory block size (in bytes). Once the block buffer
    /// exceeds this size, the writer flushes the block to disk.
    pub fn with_max_block_size(mut self, size: usize) -> Self {
        self.max_block_size = size;
        self
    }

    /// Add a key-value pair to the Avro file header's metadata map.
    pub fn with_metadata(mut self, key: &str, value: &[u8]) -> Self {
        self.extra_meta.push((key.to_string(), value.to_vec()));
        self
    }

    /// Specify a sync marker to be used in the file. If not set, a default
    /// or random marker is used instead.
    pub fn with_sync_marker(mut self, marker: [u8; 16]) -> Self {
        self.sync_marker = Some(marker);
        self
    }

    /// Finalize the configuration and construct the [`Writer`], immediately
    /// writing the Avro file header to the underlying output.
    ///
    /// # Errors
    ///
    /// Returns an error if the Arrow schema cannot be converted to Avro
    /// or if writing the header fails.
    pub fn build(mut self) -> Result<Writer<W>, ArrowError> {
        let avro_schema = match self.avro_schema.take() {
            Some(sch) => sch,
            None => arrow_schema_to_avro_schema(&self.arrow_schema)?,
        };
        let sync_marker = self.sync_marker.unwrap_or([0xAA; 16]);
        let header = AvroHeader {
            avro_schema,
            compression: self.compression,
            extra_meta: self.extra_meta,
            sync_marker,
        };
        header.write_header(&mut self.writer)?;
        let block_encoder = BlockEncoder::new(
            self.compression,
            sync_marker,
            self.max_block_size,
        );
        let record_encoder = RecordEncoder::try_new(self.arrow_schema.as_ref())?;
        Ok(Writer {
            sink: self.writer,
            header_written: true,
            header,
            block_encoder,
            record_encoder,
            arrow_schema: self.arrow_schema,
            finished: false,
        })
    }
}

/// An Avro writer that produces container files from Arrow batches or single rows.
///
/// Use [`WriterBuilder`] to construct a writer and then call:
/// * [`write`](Self::write) or [`write_batches`](Self::write_batches) to write entire [`RecordBatch`]es.
/// * [`finish`](Self::finish) to finalize the file.
pub struct Writer<W: Write> {
    /// The output sink for all Avro bytes.
    pub(crate) sink: W,

    #[allow(dead_code)]
    /// Indicates we wrote the Avro file header already (unused in logic).
    pub(crate) header_written: bool,

    #[allow(dead_code)]
    /// The Avro file header data, including schema and compression.
    pub(crate) header: AvroHeader,

    /// Handles buffering rows into blocks, compression, etc.
    pub(crate) block_encoder: BlockEncoder,

    /// Encodes a single row from an Arrow `RecordBatch` into Avro bytes.
    pub(crate) record_encoder: RecordEncoder,

    /// The Arrow schema used by this writer.
    pub(crate) arrow_schema: SchemaRef,

    /// Whether this writer has been finished (no more data allowed).
    pub(crate) finished: bool,
}

impl<W: Write> Writer<W> {
    /// Write all rows in `batch` to the Avro file.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns do not match the schema or if an
    /// underlying I/O or encoding error occurs.
    pub fn write(&mut self, batch: &RecordBatch) -> Result<(), ArrowError> {
        if batch.num_columns() != self.arrow_schema.fields().len() {
            return Err(ArrowError::InvalidArgumentError(
                "Number of columns mismatch".into(),
            ));
        }
        let row_count = batch.num_rows();
        for row_idx in 0..row_count {
            let encoded_row = self
                .record_encoder
                .encode_row_to_vec(self.arrow_schema.as_ref(), batch, row_idx)?;
            self.block_encoder.append_encoded(&encoded_row);
            self.block_encoder.inc_count();
            self.block_encoder.maybe_flush(&mut self.sink)?;
        }
        Ok(())
    }

    /// Writes multiple [`RecordBatch`]es in succession.
    ///
    /// # Errors
    ///
    /// Returns an error if any batch fails to encode/write.
    pub fn write_batches(&mut self, batches: &[&RecordBatch]) -> Result<(), ArrowError> {
        for b in batches {
            self.write(b)?;
        }
        Ok(())
    }

    /// Finish the file, flushing the final block if needed.
    ///
    /// After calling `finish()`, no further data can be written.
    ///
    /// # Errors
    ///
    /// If flushing fails, returns an error.
    pub fn finish(&mut self) -> Result<(), ArrowError> {
        if self.finished {
            return Ok(());
        }
        self.block_encoder.close(&mut self.sink)?;
        self.finished = true;
        Ok(())
    }

    /// Consume this writer and return the underlying output.
    pub fn into_inner(self) -> W {
        self.sink
    }
}

/// A convenience trait that some Arrow writers implement for a standard
/// interface to write [`RecordBatch`]es or close the writer.
pub trait BatchWriter {
    /// Write a single [`RecordBatch`] to the writer.
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<(), ArrowError>;

    /// Close or finalize the writer.
    fn close(&mut self) -> Result<(), ArrowError>;
}

impl<W: Write> BatchWriter for Writer<W> {
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<(), ArrowError> {
        self.write(batch)
    }

    fn close(&mut self) -> Result<(), ArrowError> {
        self.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        Array, Int32Array, StringArray
    };
    use arrow_schema::{DataType, Field, Schema};
    use crate::reader::{ReaderBuilder};
    use std::io::Cursor;

    fn round_trip(
        batches: &[RecordBatch],
        compression: Option<CompressionCodec>,
    ) -> Result<Vec<RecordBatch>, ArrowError> {
        if batches.is_empty() {
            return Ok(vec![]);
        }
        let schema = batches[0].schema();
        let mut buffer = Vec::new();
        {
            let mut writer = WriterBuilder::new(&mut buffer, schema.clone());
            if let Some(codec) = compression {
                writer = writer.with_compression(codec);
            }
            let mut writer = writer.build()?;
            for b in batches {
                writer.write(b)?;
            }
            writer.finish()?;
        }
        let mut reader = ReaderBuilder::new()
            .with_batch_size(64)
            .build(Cursor::new(buffer))?;
        let mut out = Vec::new();
        while let Some(batch) = reader.next() {
            let batch = batch?;
            out.push(batch);
        }
        Ok(out)
    }

    #[test]
    fn test_round_trip_simple() -> Result<(), ArrowError> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("int_field", DataType::Int32, true),
            Field::new("str_field", DataType::Utf8, true),
        ]));
        let ints = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])) as ArrayRef;
        let strs = Arc::new(StringArray::from(vec![None, Some("x"), Some("y")])) as ArrayRef;
        let batch = RecordBatch::try_new(schema.clone(), vec![ints, strs])?;
        let result = round_trip(&[batch.clone()], None)?;
        assert_eq!(result.len(), 1);
        let out_batch = &result[0];
        assert_eq!(out_batch.num_rows(), 3);
        assert_eq!(out_batch.num_columns(), 2);
        let out_ints = out_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let out_strs = out_batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(out_ints.is_null(1), true);
        assert_eq!(out_ints.value(0), 1);
        assert_eq!(out_ints.value(2), 3);
        assert_eq!(out_strs.is_null(0), true);
        assert_eq!(out_strs.value(1), "x");
        assert_eq!(out_strs.value(2), "y");
        Ok(())
    }
}