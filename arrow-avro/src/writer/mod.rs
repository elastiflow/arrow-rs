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

    /// Writes multiple [`RecordBatch`] in succession.
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
/// interface to write [`RecordBatch`] or close the writer.
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
    use std::collections::HashMap;
    use std::fs::File;
    use super::*;
    use arrow_array::{Array, Decimal128Array, Decimal256Array, DictionaryArray, FixedSizeBinaryArray, Int32Array, Int8Array, ListArray, MapArray, PrimitiveArray, StringArray, StructArray};
    use arrow_schema::{DataType, Field, Fields, IntervalUnit, Schema};
    use crate::reader::{Reader, ReaderBuilder};
    use std::io::{BufReader, Cursor};
    use arrow_array::builder::{Decimal128Builder, Decimal256Builder, FixedSizeBinaryBuilder, Int32Builder, MapBuilder, StringBuilder};
    use arrow_array::types::IntervalMonthDayNanoType;
    use arrow_buffer::{i256, Buffer, IntervalMonthDayNano};
    use arrow_data::ArrayData;
    use crate::test_util::arrow_test_data;

    fn read_file(path: &str, _schema: Option<Schema>) -> Reader<BufReader<File>> {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        let builder = ReaderBuilder::new().with_batch_size(64);
        builder.build(reader).unwrap()
    }

    #[test]
    fn test_round_trip_files() -> Result<(), ArrowError> {
        let files = [
            "avro/alltypes_plain.avro",
            "avro/alltypes_plain.snappy.avro",
            "avro/alltypes_plain.zstandard.avro",
            "avro/alltypes_plain.bzip2.avro",
            "avro/alltypes_plain.xz.avro",
            "avro/alltypes_dictionary.avro",
            "avro/alltypes_nulls_plain.avro",
            "avro/binary.avro",
        ];
        for file in files {
            let file = arrow_test_data(file);
            let mut original_reader = read_file(&file, None);
            let mut original_batches = Vec::new();
            while let Some(batch) = original_reader.next() {
                original_batches.push(batch?);
            }
            let mut buffer = Vec::new();
            if !original_batches.is_empty() {
                let schema = original_batches[0].schema();
                let mut writer = WriterBuilder::new(&mut buffer, schema.clone()).build()?;
                for batch in &original_batches {
                    writer.write(batch)?;
                }
                writer.finish()?;
            }
            let mut roundtrip_reader = ReaderBuilder::new().build(Cursor::new(&buffer))?;
            let mut roundtrip_batches = Vec::new();
            while let Some(batch) = roundtrip_reader.next() {
                roundtrip_batches.push(batch?);
            }
            assert_eq!(original_batches.len(), roundtrip_batches.len(),
                       "Mismatch in number of batches for file '{}'", file);
            for (i, (original, roundtrip)) in
                original_batches.iter().zip(roundtrip_batches.iter()).enumerate()
            {
                assert_eq!(
                    original.num_rows(), roundtrip.num_rows(),
                    "Row count mismatch in file '{}' batch {}",
                    file, i
                );
                assert_eq!(
                    original.num_columns(), roundtrip.num_columns(),
                    "Column count mismatch in file '{}' batch {}",
                    file, i
                );
                assert_eq!(
                    original, roundtrip,
                    "Mismatch in file '{}' batch {} after round-trip",
                    file, i
                );
            }
        }
        Ok(())
    }


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
    fn test_round_trip_duration() -> Result<(), ArrowError> {
        let row0 = [0u8; 12];
        let row1 = [0, 0, 0, 0,   1, 0, 0, 0,   0xF4, 0x01, 0, 0];
        let mut builder = FixedSizeBinaryBuilder::new(12);
        builder.append_value(&row0)?;
        builder.append_value(&row1)?;
        let duration_array: FixedSizeBinaryArray = builder.finish();
        let mut md = HashMap::new();
        md.insert("logicalType".to_string(), "duration".to_string());
        let field = Field::new("duration_col", DataType::FixedSizeBinary(12), true)
            .with_metadata(md);
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(duration_array) as ArrayRef])?;
        let out_batches = round_trip(&[batch], None)?;
        assert_eq!(out_batches.len(), 1);
        let out_batch = &out_batches[0];
        assert_eq!(out_batch.num_rows(), 2);
        assert_eq!(out_batch.num_columns(), 1);
        let out_arr = out_batch.column(0);
        assert_eq!(
        out_arr.data_type(),
        &DataType::Interval(IntervalUnit::MonthDayNano),
    );
        let out_mdn = out_arr
            .as_any()
            .downcast_ref::<PrimitiveArray<IntervalMonthDayNanoType>>()
            .unwrap();
        let row0_val: IntervalMonthDayNano = out_mdn.value(0);
        assert_eq!(row0_val.months, 0);
        assert_eq!(row0_val.days, 0);
        assert_eq!(row0_val.nanoseconds, 0);
        let row1_val: IntervalMonthDayNano = out_mdn.value(1);
        assert_eq!(row1_val.months, 0);
        assert_eq!(row1_val.days, 1);
        assert_eq!(row1_val.nanoseconds, 500_000_000);
        Ok(())
    }

    #[test]
    fn test_round_trip_nested_array_non_nullable() -> Result<(), ArrowError> {
        let int_values = Int32Array::from(vec![1, 2, 3]);
        let int_data = int_values.into_data();
        let offsets_child = [0i32, 2, 3];
        let child_list_data_type = DataType::List(Arc::new(Field::new(
            "item",
            DataType::Int32,
            false,
        )));
        let child_list_data = ArrayData::builder(child_list_data_type.clone())
            .len(2)
            .add_buffer(Buffer::from_slice_ref(&offsets_child))
            .add_child_data(int_data)
            .build()?;
        let child_list_arr = ListArray::from(child_list_data);
        let offsets_outer = [0i32, 1, 2];
        let outer_list_data_type = DataType::List(Arc::new(Field::new(
            "item",
            child_list_data_type,
            false
        )));
        let outer_list_data = ArrayData::builder(outer_list_data_type.clone())
            .len(2)
            .add_buffer(Buffer::from_slice_ref(&offsets_outer))
            .add_child_data(child_list_arr.into_data())
            .build()?;
        let outer_list_arr = ListArray::from(outer_list_data);
        let field = Field::new("outer_list_col", outer_list_data_type.clone(), false);
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(outer_list_arr) as ArrayRef])?;
        let result_batches = round_trip(&[batch.clone()], None)?;
        assert_eq!(result_batches.len(), 1, "Expected 1 roundtrip batch");
        let out_batch = &result_batches[0];
        assert_eq!(out_batch.num_rows(), 2);
        assert_eq!(out_batch.num_columns(), 1);
        let out_arr = out_batch
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("outer list array");
        assert_eq!(out_arr.len(), 2);
        let row0_list = out_arr.value(0);
        let row0_list = row0_list.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(row0_list.len(), 1, "row0 has 1 sub-list in this example");
        let sublist0 = row0_list.value(0);
        let sublist0_ints = sublist0.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(sublist0_ints.len(), 2);
        assert_eq!(sublist0_ints.value(0), 1);
        assert_eq!(sublist0_ints.value(1), 2);
        let row1_list = out_arr.value(1);
        let row1_list = row1_list.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(row1_list.len(), 1, "row1 has 1 sub-list");
        let sublist1 = row1_list.value(0);
        let sublist1_ints = sublist1.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(sublist1_ints.len(), 1);
        assert_eq!(sublist1_ints.value(0), 3);
        Ok(())
    }

    #[test]
    fn test_round_trip_nested_record() -> Result<(), ArrowError> {
        let address_fields = Fields::from(vec![
            Field::new("street", DataType::Utf8, false),
            Field::new("city", DataType::Utf8, false),
        ]);
        let street_array = StringArray::from(vec![Some("Sunset Blvd"), Some("2nd Street")]);
        let city_array = StringArray::from(vec![Some("LA"), Some("NYC")]);
        let address_struct_array = StructArray::try_new(
            address_fields.clone(),
            vec![
                Arc::new(street_array) as ArrayRef,
                Arc::new(city_array) as ArrayRef,
            ],
            None,
        )?;
        let person_fields = Fields::from(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int32, false),
            Field::new(
                "address",
                DataType::Struct(address_fields.clone()),
                false,
            ),
        ]);
        let name_array = StringArray::from(vec![Some("Alice"), Some("Bob")]);
        let age_array = Int32Array::from(vec![Some(30), Some(40)]);
        let person_struct_array = StructArray::try_new(
            person_fields.clone(),
            vec![
                Arc::new(name_array) as ArrayRef,
                Arc::new(age_array) as ArrayRef,
                Arc::new(address_struct_array) as ArrayRef,
            ],
            None,
        )?;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "person",
            DataType::Struct(person_fields),
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(person_struct_array)])?;
        let result_batches = round_trip(&[batch.clone()], None)?;
        assert_eq!(result_batches.len(), 1, "Expected 1 output batch");
        let out_batch = &result_batches[0];
        assert_eq!(out_batch.num_rows(), batch.num_rows());
        assert_eq!(out_batch.num_columns(), 1);
        let person_col = out_batch.column(0);
        let person_struct = person_col
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("person column should be StructArray");
        assert_eq!(person_struct.len(), 2);
        let name_arr = person_struct
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let age_arr = person_struct
            .column_by_name("age")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let address_arr = person_struct
            .column_by_name("address")
            .unwrap()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let street_arr = address_arr
            .column_by_name("street")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let city_arr = address_arr
            .column_by_name("city")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(name_arr.value(0), "Alice");
        assert_eq!(age_arr.value(0), 30);
        assert_eq!(street_arr.value(0), "Sunset Blvd");
        assert_eq!(city_arr.value(0), "LA");
        assert_eq!(name_arr.value(1), "Bob");
        assert_eq!(age_arr.value(1), 40);
        assert_eq!(street_arr.value(1), "2nd Street");
        assert_eq!(city_arr.value(1), "NYC");
        Ok(())
    }

    #[test]
    fn test_round_trip_enum_dictionary() -> Result<(), ArrowError> {
        let dictionary_values = StringArray::from(vec!["RED", "GREEN", "BLUE"]);
        let keys = Int8Array::from(vec![Some(0), Some(1), Some(2), Some(1), None, Some(2)]);
        let dict_array = DictionaryArray::try_new(keys, Arc::new(dictionary_values))?;
        let mut md = HashMap::new();
        md.insert(
            "avro.enum.symbols".to_string(),
            "[\"RED\",\"GREEN\",\"BLUE\"]".to_string(),
        );
        let field = arrow_schema::Field::new("enum_col", dict_array.data_type().clone(), true)
            .with_metadata(md);
        let schema = Arc::new(arrow_schema::Schema::new(vec![field]));
        let batch = arrow_array::RecordBatch::try_new(schema.clone(), vec![Arc::new(dict_array)])?;
        let result_batches = round_trip(&[batch.clone()], None)?;
        assert_eq!(result_batches.len(), 1, "Expected 1 batch after roundtrip");
        let out_batch = &result_batches[0];
        assert_eq!(out_batch.num_rows(), batch.num_rows());
        assert_eq!(out_batch.num_columns(), 1);
        let out_col = out_batch.column(0);
        let out_dict_arr = out_col
            .as_any()
            .downcast_ref::<DictionaryArray<arrow_array::types::Int32Type>>()
            .expect("Expected a Dictionary<Int32, Utf8> after reading Avro enum");
        let out_values = out_dict_arr
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Dictionary values should be a StringArray");
        assert_eq!(out_values.len(), 3, "Should have 3 enum symbols");
        assert_eq!(out_values.value(0), "RED");
        assert_eq!(out_values.value(1), "GREEN");
        assert_eq!(out_values.value(2), "BLUE");
        let out_keys = out_dict_arr.keys();
        assert_eq!(out_keys.len(), 6);
        assert_eq!(out_keys.is_null(4), true, "Row 4 was null in the original data");
        let out_keys_i64: Vec<Option<i64>> = out_keys
            .iter()
            .map(|k| k.map(|x| x as i64))
            .collect();
        assert_eq!(out_keys_i64[0], Some(0));
        assert_eq!(out_keys_i64[1], Some(1));
        assert_eq!(out_keys_i64[2], Some(2));
        assert_eq!(out_keys_i64[3], Some(1));
        assert_eq!(out_keys_i64[4], None);
        assert_eq!(out_keys_i64[5], Some(2));
        Ok(())
    }

    #[test]
    fn test_round_trip_map() -> Result<(), ArrowError> {
        let key_builder = StringBuilder::new();
        let value_builder = Int32Builder::new();
        let mut map_builder = MapBuilder::new(None, key_builder, value_builder);
        map_builder.keys().append_value("apple");
        map_builder.values().append_value(10);
        map_builder.keys().append_value("banana");
        map_builder.values().append_value(20);
        map_builder.append(true)?;
        map_builder.keys().append_value("hello");
        map_builder.values().append_value(42);
        map_builder.append(true)?;
        let map_array = map_builder.finish();
        let field = Field::new("my_map", map_array.data_type().clone(), true);
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(map_array) as ArrayRef])?;
        let result = round_trip(&[batch.clone()], None)?;
        assert_eq!(result.len(), 1, "Expected 1 batch after round trip");
        let out_batch = &result[0];
        assert_eq!(out_batch.num_rows(), batch.num_rows());
        assert_eq!(out_batch.num_columns(), 1);
        let out_map = out_batch
            .column(0)
            .as_any()
            .downcast_ref::<MapArray>()
            .expect("Failed to downcast to MapArray");
        assert_eq!(out_map.len(), 2, "Should still have 2 rows");
        assert_eq!(out_map.value_length(0), 2);
        assert_eq!(out_map.value_length(1), 1);
        let entries_struct = out_map.entries();
        let key_arr = entries_struct
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Map keys should be a StringArray");
        let val_arr = entries_struct
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Map values should be Int32Array");
        assert_eq!(key_arr.value(0), "apple");
        assert_eq!(val_arr.value(0), 10);
        assert_eq!(key_arr.value(1), "banana");
        assert_eq!(val_arr.value(1), 20);
        assert_eq!(key_arr.value(2), "hello");
        assert_eq!(val_arr.value(2), 42);
        Ok(())
    }

    #[test]
    fn test_round_trip_decimal128() -> Result<(), ArrowError> {
        let mut builder = Decimal128Builder::new()
            .with_precision_and_scale(4, 2)?;
        builder.append_value(12345);
        builder.append_value(-9999);
        builder.append_null();
        builder.append_value(5000);
        let decimal_arr = builder.finish();
        let field = Field::new("decimal_col", DataType::Decimal128(4, 2), true);
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(decimal_arr) as ArrayRef],
        )?;
        let result = round_trip(&[batch.clone()], None)?;
        assert_eq!(result.len(), 1, "Expected exactly 1 batch returned");
        let out_batch = &result[0];
        assert_eq!(out_batch.num_rows(), batch.num_rows());
        assert_eq!(out_batch.num_columns(), 1);
        let out_decimal = out_batch
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("Failed to downcast to Decimal128Array");
        assert_eq!(out_decimal.len(), 4);
        assert_eq!(out_decimal.null_count(), 1);
        assert_eq!(out_decimal.value(0), 12345);
        assert_eq!(out_decimal.value(1), -9999);
        assert!(out_decimal.is_null(2));
        assert_eq!(out_decimal.value(3), 5000);
        Ok(())
    }

    #[test]
    fn test_round_trip_decimal256() -> Result<(), ArrowError> {
        let mut builder = Decimal256Builder::new()
            .with_precision_and_scale(38, 6)?;
        builder.append_value(i256::from_i128(123_456_789));
        builder.append_value(i256::from_i128(-1_000_000));
        builder.append_null();
        builder.append_value(i256::from_i128(999_999_999_999));
        let decimal_arr = builder.finish();
        let field = Field::new("decimal256_col", DataType::Decimal256(38, 6), true);
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(decimal_arr) as ArrayRef],
        )?;
        let result = round_trip(&[batch.clone()], None)?;
        assert_eq!(result.len(), 1, "Expected exactly 1 batch returned");
        let out_batch = &result[0];
        assert_eq!(out_batch.num_rows(), batch.num_rows());
        assert_eq!(out_batch.num_columns(), 1);
        let out_decimal = out_batch
            .column(0)
            .as_any()
            .downcast_ref::<Decimal256Array>()
            .expect("Failed to downcast to Decimal256Array");
        assert_eq!(out_decimal.len(), 4);
        assert_eq!(out_decimal.null_count(), 1);
        assert_eq!(out_decimal.value(0), i256::from_i128(123_456_789));
        assert_eq!(out_decimal.value(1), i256::from_i128(-1_000_000));
        assert!(out_decimal.is_null(2));
        assert_eq!(out_decimal.value(3), i256::from_i128(999_999_999_999));
        Ok(())
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