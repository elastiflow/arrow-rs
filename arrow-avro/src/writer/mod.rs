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
//! *   Use **`AvroWriter`** (Object Container File) when you want a
//!     self‑contained Avro file with header, schema JSON, optional compression,
//!     blocks, and sync markers.
//! *   Use **`AvroStreamWriter`** (raw binary stream) when you already know the
//!     schema out‑of‑band (i.e., via a schema registry) and need a stream
//!     of Avro‑encoded records with minimal framing (Avro Single‑Object encoding
//!     header per record).

/// Encodes `RecordBatch` into the Avro binary format.
pub mod encoder;
/// Logic for different Avro container file formats.
pub mod format;

use crate::compression::CompressionCodec;
use crate::schema::{AvroSchema, SchemaGenOptions, SCHEMA_METADATA_KEY};
use crate::writer::encoder::{
    build_write_plan_from_avro_root, encode_record_batch_single_object_with_plan,
    encode_record_batch_with_plan, write_long, WritePlan,
};
use crate::writer::format::{AvroBinaryFormat, AvroFormat, AvroOcfFormat};

use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, Schema};
use std::io::{self, Write};
use std::sync::Arc;

// Plan construction: parse the Avro JSON we are going to advertise and build a
// schema‑driven, per‑site union ordering for the encoder.
use crate::codec::AvroFieldBuilder;

/// Builder to configure and create a `Writer`.
#[derive(Debug, Clone)]
pub struct WriterBuilder {
    schema: Schema,
    codec: Option<CompressionCodec>,
    /// If set, this exact Avro JSON will be embedded in the header and used
    /// to drive union ordering for the encoder.
    avro_json_override: Option<String>,
    /// Options that influence **generation** of Avro JSON from an Arrow schema
    /// (only used if `avro_json_override` is `None` and the Arrow schema metadata
    /// does not already contain `avro.schema`).
    schema_gen_options: Option<SchemaGenOptions>,
}

impl WriterBuilder {
    /// Create a new builder with default settings.
    pub fn new(schema: Schema) -> Self {
        Self {
            schema,
            codec: None,
            avro_json_override: None,
            schema_gen_options: None,
        }
    }

    /// Change the compression codec.
    pub fn with_compression(mut self, codec: Option<CompressionCodec>) -> Self {
        self.codec = codec;
        self
    }

    /// Provide an explicit Avro **JSON schema** to embed in the header and to
    /// drive the encoder's union ordering. This takes precedence over any
    /// `avro.schema` present in the Arrow schema metadata.
    pub fn with_avro_json_schema(mut self, json: String) -> Self {
        self.avro_json_override = Some(json);
        self
    }

    /// Provide options to control **generation** of Avro JSON from the Arrow
    /// schema when no explicit `avro.schema` is provided.
    pub fn with_schema_gen_options(mut self, opts: SchemaGenOptions) -> Self {
        self.schema_gen_options = Some(opts);
        self
    }

    /// Create a new `Writer` with specified `AvroFormat` and builder options.
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
            avro_json_override: self.avro_json_override,
            schema_gen_options: self.schema_gen_options,
            plan: None,
        }
    }
}

/// Generic Avro writer.
#[derive(Debug)]
pub struct Writer<W: Write, F: AvroFormat> {
    writer: W,
    schema: Arc<Schema>,
    format: F,
    compression: Option<CompressionCodec>,
    started: bool,
    /// Builder‑provided schema JSON override (if any).
    avro_json_override: Option<String>,
    /// Options controlling Arrow→Avro JSON generation if needed.
    schema_gen_options: Option<SchemaGenOptions>,
    /// Prepared, schema‑driven write plan computed on first `write(...)`.
    plan: Option<WritePlan>,
}

/// Alias for an Avro **Object Container File** writer.
pub type AvroWriter<W> = Writer<W, AvroOcfFormat>;
/// Alias for a raw Avro **binary stream** writer.
pub type AvroStreamWriter<W> = Writer<W, AvroBinaryFormat>;

impl<W: Write> Writer<W, AvroOcfFormat> {
    /// Convenience constructor – same as [`WriterBuilder::build`] with `AvroOcfFormat`.
    pub fn new(writer: W, schema: Schema) -> Result<Self, ArrowError> {
        Ok(WriterBuilder::new(schema).build::<W, AvroOcfFormat>(writer))
    }

    /// Change the compression codec after construction.
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
        // Lazily initialize the stream and prepare the per‑field write plan
        // upon the first `write(...)`.
        if !self.started {
            // 1) Choose the **exact Avro JSON** we will advertise
            let chosen_json = if let Some(s) = &self.avro_json_override {
                s.clone()
            } else if let Some(json) = self.schema.metadata.get(SCHEMA_METADATA_KEY) {
                json.clone()
            } else if let Some(opts) = &self.schema_gen_options {
                // Generate JSON from Arrow using requested options
                let avro = AvroSchema::from_arrow_with_options(self.schema.as_ref(), opts.clone())?;
                avro.json_string
            } else {
                // Default generation (typically null‑first)
                AvroSchema::try_from(self.schema.as_ref())?.json_string
            };

            // 2) Inject the chosen JSON into the **writer schema's metadata**
            //    so the header and the encoder share the same single source of truth.
            let mut md = self.schema.metadata().clone();
            md.insert(SCHEMA_METADATA_KEY.to_string(), chosen_json.clone());
            let updated = Schema::new_with_metadata(self.schema.fields().clone(), md);
            self.schema = Arc::new(updated);

            // 3) Start the stream (writes header / single‑object prefix)
            self.format
                .start_stream(&mut self.writer, &self.schema, self.compression)?;

            // 4) Build the **schema‑driven write plan** from the chosen Avro JSON
            //    NOTE: Keep the AvroSchema owner alive while borrowing from it
            //    to avoid E0716 (temporary value dropped while borrowed).
            let avro_holder = AvroSchema::new(chosen_json);
            let avro_ast = avro_holder.schema()?;
            let avro_root = AvroFieldBuilder::new(&avro_ast).build()?;
            let plan = build_write_plan_from_avro_root(&avro_root, self.schema.as_ref())?;
            self.plan = Some(plan);

            self.started = true;
        }

        // Validate incoming batch matches the writer's Arrow **fields** (ignore metadata)
        if batch.schema().fields() != self.schema.fields() {
            return Err(ArrowError::SchemaError(
                "Schema of RecordBatch differs from Writer schema (fields/types)".to_string(),
            ));
        }

        // Encode according to container format
        match self.format.sync_marker() {
            Some(&sync) => self.write_ocf_block(batch, &sync),
            None => self.write_stream(batch),
        }
    }

    /// A convenience method to write a slice of [`RecordBatch`].
    ///
    /// This is equivalent to calling `write` for each batch in the slice.
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
    pub fn into_inner(self) -> W {
        self.writer
    }

    /// Write a single OCF data block: row count, block length, block bytes, sync.
    fn write_ocf_block(&mut self, batch: &RecordBatch, sync: &[u8; 16]) -> Result<(), ArrowError> {
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| ArrowError::ComputeError("internal: write plan missing".into()))?;
        let mut buf = Vec::<u8>::with_capacity(1024);
        // Encode with the **plan** (schema‑driven union ordering)
        encode_record_batch_with_plan(batch, &mut buf, plan)?;
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

    /// Write in streaming/binary mode, using Single‑Object framing if available.
    fn write_stream(&mut self, batch: &RecordBatch) -> Result<(), ArrowError> {
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| ArrowError::ComputeError("internal: write plan missing".into()))?;
        if let Some(prefix) = self.format.single_object_prefix() {
            // Avro Single‑Object encoding per record: magic + fingerprint + record
            encode_record_batch_single_object_with_plan(batch, &mut self.writer, prefix, plan)
        } else {
            // Raw Avro binary (no header/prefix)
            encode_record_batch_with_plan(batch, &mut self.writer, plan)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::CompressionCodec;
    use crate::reader::ReaderBuilder;
    use crate::schema::{AvroSchema, SchemaStore};
    use crate::test_util::arrow_test_data;
    use arrow_array::{ArrayRef, BinaryArray, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, IntervalUnit, Schema};
    use std::fs::File;
    use std::io::{BufReader, Cursor};
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    fn make_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Binary, false),
        ])
    }

    fn make_batch() -> RecordBatch {
        let ids = Int32Array::from(vec![1, 2, 3]);
        let names = BinaryArray::from_vec(vec![b"a".as_ref(), b"b".as_ref(), b"c".as_ref()]);
        RecordBatch::try_new(
            Arc::new(make_schema()),
            vec![Arc::new(ids) as ArrayRef, Arc::new(names) as ArrayRef],
        )
            .expect("failed to build test RecordBatch")
    }

    #[test]
    fn test_ocf_writer_generates_header_and_sync() -> Result<(), ArrowError> {
        let batch = make_batch();
        let buffer: Vec<u8> = Vec::new();
        let mut writer = AvroWriter::new(buffer, make_schema())?;
        writer.write(&batch)?;
        writer.finish()?;
        let out = writer.into_inner();
        assert_eq!(&out[..4], b"Obj\x01", "OCF magic bytes missing/incorrect");
        let trailer = &out[out.len() - 16..];
        assert_eq!(trailer.len(), 16, "expected 16‑byte sync marker");
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

    #[test]
    fn test_roundtrip_alltypes_roundtrip_writer() -> Result<(), ArrowError> {
        let files = [
            "avro/alltypes_plain.avro",
            "avro/alltypes_plain.snappy.avro",
            "avro/alltypes_plain.zstandard.avro",
            "avro/alltypes_plain.bzip2.avro",
            "avro/alltypes_plain.xz.avro",
        ];
        for rel in files {
            let path = arrow_test_data(rel);
            let rdr_file = File::open(&path).expect("open input avro");
            let mut reader = ReaderBuilder::new()
                .build(BufReader::new(rdr_file))
                .expect("build reader");
            let schema = reader.schema();
            let input_batches = reader.collect::<Result<Vec<_>, _>>()?;
            let original =
                arrow::compute::concat_batches(&schema, &input_batches).expect("concat input");
            let tmp = NamedTempFile::new().expect("create temp file");
            let out_path = tmp.into_temp_path();
            let out_file = File::create(&out_path).expect("create temp avro");
            let mut writer = AvroWriter::new(out_file, original.schema().as_ref().clone())?;
            if rel.contains(".snappy.") {
                writer = writer.with_compression(Some(CompressionCodec::Snappy));
            } else if rel.contains(".zstandard.") {
                writer = writer.with_compression(Some(CompressionCodec::ZStandard));
            } else if rel.contains(".bzip2.") {
                writer = writer.with_compression(Some(CompressionCodec::Bzip2));
            } else if rel.contains(".xz.") {
                writer = writer.with_compression(Some(CompressionCodec::Xz));
            }
            writer.write(&original)?;
            writer.finish()?;
            drop(writer);
            let rt_file = File::open(&out_path).expect("open roundtrip avro");
            let mut rt_reader = ReaderBuilder::new()
                .build(BufReader::new(rt_file))
                .expect("build roundtrip reader");
            let rt_schema = rt_reader.schema();
            let rt_batches = rt_reader.collect::<Result<Vec<_>, _>>()?;
            let roundtrip =
                arrow::compute::concat_batches(&rt_schema, &rt_batches).expect("concat roundtrip");
            assert_eq!(
                roundtrip, original,
                "Round-trip batch mismatch for file: {}",
                rel
            );
        }
        Ok(())
    }

    #[test]
    fn test_roundtrip_nested_records_writer() -> Result<(), ArrowError> {
        let path = arrow_test_data("avro/nested_records.avro");
        let rdr_file = File::open(&path).expect("open nested_records.avro");
        let mut reader = ReaderBuilder::new()
            .build(BufReader::new(rdr_file))
            .expect("build reader for nested_records.avro");
        let schema = reader.schema();
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        let original = arrow::compute::concat_batches(&schema, &batches).expect("concat original");
        let tmp = NamedTempFile::new().expect("create temp file");
        let out_path = tmp.into_temp_path();
        {
            let out_file = File::create(&out_path).expect("create output avro");
            let mut writer = AvroWriter::new(out_file, original.schema().as_ref().clone())?;
            writer.write(&original)?;
            writer.finish()?;
        }
        let rt_file = File::open(&out_path).expect("open round_trip avro");
        let mut rt_reader = ReaderBuilder::new()
            .build(BufReader::new(rt_file))
            .expect("build round_trip reader");
        let rt_schema = rt_reader.schema();
        let rt_batches = rt_reader.collect::<Result<Vec<_>, _>>()?;
        let round_trip =
            arrow::compute::concat_batches(&rt_schema, &rt_batches).expect("concat round_trip");
        assert_eq!(
            round_trip, original,
            "Round-trip batch mismatch for nested_records.avro"
        );
        Ok(())
    }

    #[test]
    fn test_roundtrip_nested_lists_writer() -> Result<(), ArrowError> {
        let path = arrow_test_data("avro/nested_lists.snappy.avro");
        let rdr_file = File::open(&path).expect("open nested_lists.snappy.avro");
        let mut reader = ReaderBuilder::new()
            .build(BufReader::new(rdr_file))
            .expect("build reader for nested_lists.snappy.avro");
        let schema = reader.schema();
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        let original = arrow::compute::concat_batches(&schema, &batches).expect("concat original");
        let tmp = NamedTempFile::new().expect("create temp file");
        let out_path = tmp.into_temp_path();
        {
            let out_file = File::create(&out_path).expect("create output avro");
            let mut writer = AvroWriter::new(out_file, original.schema().as_ref().clone())?
                .with_compression(Some(CompressionCodec::Snappy));
            writer.write(&original)?;
            writer.finish()?;
        }
        let rt_file = File::open(&out_path).expect("open round_trip avro");
        let mut rt_reader = ReaderBuilder::new()
            .build(BufReader::new(rt_file))
            .expect("build round_trip reader");
        let rt_schema = rt_reader.schema();
        let rt_batches = rt_reader.collect::<Result<Vec<_>, _>>()?;
        let round_trip =
            arrow::compute::concat_batches(&rt_schema, &rt_batches).expect("concat round_trip");
        assert_eq!(
            round_trip, original,
            "Round-trip batch mismatch for nested_lists.snappy.avro"
        );
        Ok(())
    }

    #[test]
    fn test_round_trip_simple_fixed_ocf() -> Result<(), ArrowError> {
        let path = arrow_test_data("avro/simple_fixed.avro");
        let rdr_file = File::open(&path).expect("open avro/simple_fixed.avro");
        let mut reader = ReaderBuilder::new()
            .build(BufReader::new(rdr_file))
            .expect("build avro reader");
        let schema = reader.schema();
        let input_batches = reader.collect::<Result<Vec<_>, _>>()?;
        let original =
            arrow::compute::concat_batches(&schema, &input_batches).expect("concat input");
        let tmp = NamedTempFile::new().expect("create temp file");
        let out_file = File::create(tmp.path()).expect("create temp avro");
        let mut writer = AvroWriter::new(out_file, original.schema().as_ref().clone())?;
        writer.write(&original)?;
        writer.finish()?;
        drop(writer);
        let rt_file = File::open(tmp.path()).expect("open round_trip avro");
        let mut rt_reader = ReaderBuilder::new()
            .build(BufReader::new(rt_file))
            .expect("build round_trip reader");
        let rt_schema = rt_reader.schema();
        let rt_batches = rt_reader.collect::<Result<Vec<_>, _>>()?;
        let round_trip =
            arrow::compute::concat_batches(&rt_schema, &rt_batches).expect("concat round_trip");
        assert_eq!(round_trip, original);
        Ok(())
    }

    #[test]
    fn test_round_trip_duration_and_uuid_ocf() -> Result<(), ArrowError> {
        let in_file =
            File::open("test/data/duration_uuid.avro").expect("open test/data/duration_uuid.avro");
        let mut reader = ReaderBuilder::new()
            .build(BufReader::new(in_file))
            .expect("build reader for duration_uuid.avro");
        let in_schema = reader.schema();
        let has_mdn = in_schema.fields().iter().any(|f| {
            matches!(
                f.data_type(),
                DataType::Interval(IntervalUnit::MonthDayNano)
            )
        });
        assert!(
            has_mdn,
            "expected at least one Interval(MonthDayNano) field in duration_uuid.avro"
        );
        let has_uuid_fixed =
            in_schema
                .fields()
                .iter()
                .any(|f| matches!(f.data_type(), DataType::FixedSizeBinary(16)));
        assert!(
            has_uuid_fixed,
            "expected at least one FixedSizeBinary(16) (uuid) field in duration_uuid.avro"
        );
        let input_batches = reader.collect::<Result<Vec<_>, _>>()?;
        let input =
            arrow::compute::concat_batches(&in_schema, &input_batches).expect("concat input");
        let tmp = NamedTempFile::new().expect("create temp file");
        {
            let out_file = File::create(tmp.path()).expect("create temp avro");
            let mut writer = AvroWriter::new(out_file, in_schema.as_ref().clone())?;
            writer.write(&input)?;
            writer.finish()?;
        }
        let rt_file = File::open(tmp.path()).expect("open round_trip avro");
        let mut rt_reader = ReaderBuilder::new()
            .build(BufReader::new(rt_file))
            .expect("build round_trip reader");
        let rt_schema = rt_reader.schema();
        let rt_batches = rt_reader.collect::<Result<Vec<_>, _>>()?;
        let round_trip =
            arrow::compute::concat_batches(&rt_schema, &rt_batches).expect("concat round_trip");
        assert_eq!(round_trip, input);
        Ok(())
    }

    #[test]
    fn test_stream_writer_single_object_roundtrip_with_decoder() -> Result<(), ArrowError> {
        // Prepare a simple schema and a small batch
        let schema = make_schema();
        let batch = make_batch();
        // Write using the streaming writer (Avro Single-Object per record)
        let buf = Vec::<u8>::new();
        let mut writer = AvroStreamWriter::new(buf, schema.clone())?;
        writer.write(&batch)?;
        writer.finish()?;
        let out = writer.into_inner();
        assert!(
            out.len() > 10,
            "streaming output should contain at least one single-object record"
        );
        // Build a SchemaStore and register the writer Avro schema to obtain the fingerprint
        let avro_schema = AvroSchema::try_from(&schema)?;
        let mut store = SchemaStore::new();
        let fp = store.register(avro_schema.clone()).expect("register schema");
        // Decode the streaming bytes with the low-level Decoder
        let mut decoder = ReaderBuilder::new()
            .with_batch_size(1024)
            .with_writer_schema_store(store)
            .with_active_fingerprint(fp)
            .with_reader_schema(avro_schema)
            .build_decoder()?;
        let consumed = decoder.decode(&out)?;
        assert_eq!(
            consumed,
            out.len(),
            "decoder should consume entire single-object stream"
        );
        let decoded = decoder
            .flush()?
            .expect("decoder should yield a RecordBatch after flush");
        assert_eq!(
            decoded, batch,
            "decoded RecordBatch should match the original input batch"
        );
        Ok(())
    }

    // === New round-trip test that exercises Map support ===
    //
    // This test reads the same 'nonnullable.impala.avro' used by the reader tests,
    // writes it back out with the writer (hitting Map encoding paths), then reads it
    // again and asserts exact Arrow equivalence.
    #[test]
    fn test_nonnullable_impala_roundtrip_writer() -> Result<(), ArrowError> {
        // Load source Avro with Map fields
        let path = arrow_test_data("avro/nonnullable.impala.avro");
        let rdr_file = File::open(&path).expect("open avro/nonnullable.impala.avro");
        let mut reader = ReaderBuilder::new()
            .build(BufReader::new(rdr_file))
            .expect("build reader for nonnullable.impala.avro");

        // Collect all input batches and concatenate to a single RecordBatch
        let in_schema = reader.schema();
        // Sanity: ensure the file actually contains at least one Map field
        let has_map = in_schema
            .fields()
            .iter()
            .any(|f| matches!(f.data_type(), DataType::Map(_, _)));
        assert!(
            has_map,
            "expected at least one Map field in avro/nonnullable.impala.avro"
        );

        let input_batches = reader.collect::<Result<Vec<_>, _>>()?;
        let original =
            arrow::compute::concat_batches(&in_schema, &input_batches).expect("concat input");

        // Write out using the OCF writer into an in-memory Vec<u8>
        let buffer = Vec::<u8>::new();
        let mut writer = AvroWriter::new(buffer, in_schema.as_ref().clone())?;
        writer.write(&original)?;
        writer.finish()?;
        let out_bytes = writer.into_inner();

        // Read the produced bytes back with the Reader
        let mut rt_reader = ReaderBuilder::new()
            .build(Cursor::new(out_bytes))
            .expect("build reader for round-tripped in-memory OCF");
        let rt_schema = rt_reader.schema();
        let rt_batches = rt_reader.collect::<Result<Vec<_>, _>>()?;
        let roundtrip =
            arrow::compute::concat_batches(&rt_schema, &rt_batches).expect("concat roundtrip");

        // Exact value fidelity (schema + data)
        assert_eq!(
            roundtrip, original,
            "Round-trip Avro map data mismatch for nonnullable.impala.avro"
        );
        Ok(())
    }
}
