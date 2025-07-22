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

//! Avro reader
//!
//! This module provides facilities to read Apache Avro-encoded files or streams
//! into Arrow's `RecordBatch` format. In particular, it introduces:
//!
//! * `ReaderBuilder`: Configures Avro reading, e.g., batch size
//! * `Reader`: Yields `RecordBatch` values, implementing `Iterator`
//! * `Decoder`: A low-level push-based decoder for Avro records
//!
//! # Basic Usage
//!
//! `Reader` can be used directly with synchronous data sources, such as [`std::fs::File`].
//!
//! ## Reading a Single Batch
//!
//! ```
//! # use std::fs::File;
//! # use std::io::BufReader;
//! # use arrow_avro::reader::ReaderBuilder;
//!
//! let file = File::open("../testing/data/avro/alltypes_plain.avro").unwrap();
//! let mut avro = ReaderBuilder::new().build(BufReader::new(file)).unwrap();
//! let batch = avro.next().unwrap();
//! ```
//!
//! # Async Usage
//!
//! The lower-level `Decoder` can be integrated with various forms of async data streams,
//! and is designed to be agnostic to different async IO primitives within
//! the Rust ecosystem. It works by incrementally decoding Avro data from byte slices.
//!
//! For example, see below for how it could be used with an arbitrary `Stream` of `Bytes`:
//!
//! ```
//! # use std::task::{Poll, ready};
//! # use bytes::{Buf, Bytes};
//! # use arrow_schema::ArrowError;
//! # use futures::stream::{Stream, StreamExt};
//! # use arrow_array::RecordBatch;
//! # use arrow_avro::reader::Decoder;
//!
//! fn decode_stream<S: Stream<Item = Bytes> + Unpin>(
//!     mut decoder: Decoder,
//!     mut input: S,
//! ) -> impl Stream<Item = Result<RecordBatch, ArrowError>> {
//!     let mut buffered = Bytes::new();
//!     futures::stream::poll_fn(move |cx| {
//!         loop {
//!             if buffered.is_empty() {
//!                 buffered = match ready!(input.poll_next_unpin(cx)) {
//!                     Some(b) => b,
//!                     None => break,
//!                 };
//!             }
//!             let decoded = match decoder.decode(buffered.as_ref()) {
//!                 Ok(decoded) => decoded,
//!                 Err(e) => return Poll::Ready(Some(Err(e))),
//!             };
//!             let read = buffered.len();
//!             buffered.advance(decoded);
//!             if decoded != read {
//!                 break
//!             }
//!         }
//!         // Convert any fully-decoded rows to a RecordBatch, if available
//!         Poll::Ready(decoder.flush().transpose())
//!     })
//! }
//! ```
//!

use crate::codec::AvroField;
use crate::schema::{compare_schemas, Fingerprint, HashType, Schema as AvroSchema, SchemaStore, SINGLE_OBJECT_MAGIC};
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use block::BlockDecoder;
use header::{Header, HeaderDecoder};
use record::RecordDecoder;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::io::BufRead;

mod block;
mod cursor;
mod header;
mod record;
mod vlq;

/// Maximum number of inactive [`RecordDecoder`]s kept across schema switches.
///
/// The list is maintained in *least‑recently‑used* (LRU) order, so eviction
/// is deterministic and keeps recently‑seen schemas hot in the cache.
/// LRU is a well‑known cache‑eviction policy that minimizes thrashing in the
/// presence of temporal locality
const MAX_DECODERS: usize = 20;

/// Fingerprint byte‑widths (excluding the 2‑byte magic prefix)
const RABIN_FP_LEN: usize = 8;
const MD5_FP_LEN: usize = 16;
const SHA256_FP_LEN: usize = 32;

/// Fast helper: how many bytes does a fingerprint prefix occupy, *including* the
/// 2‑byte magic?
#[inline]
const fn prefix_len(ht: HashType) -> usize {
    2 + match ht {
        HashType::Rabin => 8,
        HashType::MD5 => 16,
        HashType::SHA256 => 32,
    }
}

/// Read the Avro file header (magic, metadata, sync marker) from `reader`.
fn read_header<R: BufRead>(mut reader: R) -> Result<Header, ArrowError> {
    let mut decoder = HeaderDecoder::default();
    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            break;
        }
        let read = buf.len();
        let decoded = decoder.decode(buf)?;
        reader.consume(decoded);
        if decoded != read {
            break;
        }
    }
    decoder.flush().ok_or_else(|| {
        ArrowError::ParseError("Unexpected EOF while reading Avro header".to_string())
    })
}

/// A low-level interface for decoding Avro-encoded bytes into Arrow `RecordBatch`.
#[derive(Debug)]
pub struct Decoder {
    batch_size: usize,
    decoded_rows: usize,
    active_fp: Option<Fingerprint>,
    active: RecordDecoder,
    cache: HashMap<Fingerprint, RecordDecoder>,
    cache_order: VecDeque<Fingerprint>,
    reader_schema: Option<AvroSchema<'static>>,
    utf8_view: bool,
    store: Option<SchemaStore<'static>>,
    static_store: bool,
    pending_fp: Option<Fingerprint>,
    pending_decoder: Option<RecordDecoder>,
}

impl Decoder {
    fn new(
        batch_size: usize,
        active_fp: Option<Fingerprint>,
        active: RecordDecoder,
        cache: HashMap<Fingerprint, RecordDecoder>,
        reader_schema: Option<AvroSchema<'static>>,
        utf8_view: bool,
        store: Option<SchemaStore<'static>>,
        static_store: bool,
    ) -> Self {
        Self {
            batch_size,
            decoded_rows: 0,
            active_fp,
            active,
            cache,
            cache_order: VecDeque::new(),
            reader_schema,
            utf8_view,
            store,
            static_store,
            pending_fp: None,
            pending_decoder: None,
        }
    }

    /// Return the Arrow schema for the rows decoded by this decoder
    pub fn schema(&self) -> SchemaRef {
        self.active.schema().clone()
    }

    /// Return the configured maximum number of rows per batch
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Feed `data` into the decoder row by row until we either:
    /// - consume all bytes in `data`, or
    /// - reach `batch_size` decoded rows.
    ///
    /// Returns the number of bytes consumed.
    /// Feed `data` into the decoder until either all bytes are consumed **or**
    /// the current batch reaches `batch_size`.
    ///
    /// Returns the number of bytes consumed.
    pub fn decode(&mut self, data: &[u8]) -> Result<usize, ArrowError> {
        let mut total_consumed = 0;
        let hash_type = self
            .store
            .as_ref()
            .and_then(|s| s.lookup_keys_hash_type())
            .unwrap_or(HashType::Rabin);
        while total_consumed < data.len() && self.decoded_rows < self.batch_size {
            if let Some(consumed) = self.handle_prefix(&data[total_consumed..], hash_type)? {
                if consumed == 0 {
                    break;
                }
                total_consumed += consumed;
            }
            let consumed = self.active.decode(&data[total_consumed..], 1)?;
            if consumed == 0 {
                break;
            }
            total_consumed += consumed;
            self.decoded_rows += 1;
        }
        Ok(total_consumed)
    }

    /// Produce a `RecordBatch` if at least one row is fully decoded, returning
    /// `Ok(None)` if no new rows are available.
    /// Flush any fully‑decoded rows into a `RecordBatch`.
    ///
    /// If a pending schema switch was scheduled earlier, it is applied *after*
    /// the current batch is emitted.
    pub fn flush(&mut self) -> Result<Option<RecordBatch>, ArrowError> {
        if self.decoded_rows == 0 {
            return Ok(None);
        }
        let batch = self.active.flush()?;
        self.decoded_rows = 0;
        if let Some(new_dec) = self.pending_decoder.take() {
            let new_fp = self
                .pending_fp
                .take()
                .expect("pending_fp accompanies pending_decoder");
            if let Some(old_fp) = self.active_fp.replace(new_fp) {
                self.cache
                    .insert(old_fp, std::mem::replace(&mut self.active, new_dec));
                self.touch_cache_key(old_fp);
            } else {
                self.active = new_dec;
            }
        }
        // Enforce LRU bound
        self.evict_if_needed();
        Ok(Some(batch))
    }

    /// Touch a cache key (move it to the back = most recently used)
    #[inline]
    fn touch_cache_key(&mut self, fp: Fingerprint) {
        if let Some(pos) = self.cache_order.iter().position(|k| *k == fp) {
            self.cache_order.remove(pos);
        }
        self.cache_order.push_back(fp);
    }

    /// Evict LRU decoder if we exceed `MAX_DECODERS`.
    #[inline]
    fn evict_if_needed(&mut self) {
        while self.cache_order.len() > MAX_DECODERS {
            if let Some(lru) = self.cache_order.pop_front() {
                self.cache.remove(&lru);
            }
        }
    }

    /// Attempt to consume a single‑object fingerprint prefix at the start of
    /// `buf`.
    fn handle_prefix(
        &mut self,
        buf: &[u8],
        hash_type: HashType,
    ) -> Result<Option<usize>, ArrowError> {
        let store = match &self.store {
            Some(s) => s,
            None => return Ok(None),
        };
        if buf.len() < 2 {
            return Ok(Some(0));
        }
        if &buf[..2] != SINGLE_OBJECT_MAGIC {
            if self.active_fp.is_none() && self.static_store {
                return Err(ArrowError::ParseError(
                    "No schema fingerprint found in static_store_mode".into(),
                ));
            }
            return Ok(None);
        }
        let full_len = prefix_len(hash_type);
        if buf.len() < full_len {
            return Ok(Some(0));
        }
        let new_fp = match hash_type {
            HashType::Rabin => {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&buf[2..2 + 8]);
                Fingerprint::Rabin(u64::from_le_bytes(arr))
            }
            HashType::MD5 => {
                let mut arr = [0u8; 16];
                arr.copy_from_slice(&buf[2..2 + 16]);
                Fingerprint::MD5(arr)
            }
            HashType::SHA256 => {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&buf[2..2 + 32]);
                Fingerprint::SHA256(arr)
            }
        };
        if self.active_fp != Some(new_fp) {
            if self.static_store && self.active_fp.is_some() {
                return Err(ArrowError::ParseError(
                    "Schema fingerprint changed in static_store_mode".into(),
                ));
            }
            self.prepare_schema_switch(new_fp)?;
            if self.decoded_rows > 0 {
                self.decoded_rows = self.batch_size;
            }
        }
        Ok(Some(full_len))
    }

    /// Prepare `pending_decoder` to switch to `new_fp` after batch flush.
    fn prepare_schema_switch(&mut self, new_fp: Fingerprint) -> Result<(), ArrowError> {
        let new_dec = if let Some(dec) = self.cache.remove(&new_fp) {
            dec
        } else {
            let store = self.store.as_ref().ok_or_else(|| {
                ArrowError::ParseError("Schema store unavailable for fingerprint".into())
            })?;
            let writer_schema = store.lookup(&new_fp).ok_or_else(|| {
                ArrowError::ParseError(format!("Unknown schema fingerprint: {new_fp:?}"))
            })?;
            let reader_schema = self
                .reader_schema
                .clone()
                .ok_or_else(|| ArrowError::ParseError("Reader schema unavailable".into()))?;
            let resolved = AvroField::resolve_from_writer_and_reader(
                &writer_schema,
                &reader_schema,
                self.utf8_view,
            )?;
            RecordDecoder::try_new_with_options(resolved.data_type(), self.utf8_view)?
        };
        self.pending_fp = Some(new_fp);
        self.pending_decoder = Some(new_dec);
        Ok(())
    }

    /// Returns the number of rows that can be added to this decoder before it is full.
    pub fn capacity(&self) -> usize {
        self.batch_size.saturating_sub(self.decoded_rows)
    }

    /// Returns true if the decoder has reached its capacity for the current batch.
    pub fn batch_is_full(&self) -> bool {
        self.capacity() == 0
    }
}

/// A builder to create a `Reader` that reads Avro data
/// into Arrow `RecordBatch`.
#[derive(Debug)]
pub struct ReaderBuilder {
    batch_size: usize,
    strict_mode: bool,
    utf8_view: bool,
    reader_schema: Option<AvroSchema<'static>>,
    writer_schema_store: Option<SchemaStore<'static>>,
    active_fp: Option<Fingerprint>,
    static_store_mode: bool,
}

impl Default for ReaderBuilder {
    fn default() -> Self {
        Self {
            batch_size: 1024,
            strict_mode: false,
            utf8_view: false,
            reader_schema: None,
            writer_schema_store: None,
            active_fp: None,
            static_store_mode: false,
        }
    }
}

impl ReaderBuilder {
    /// Creates a new [`ReaderBuilder`] with default settings:
    /// - `batch_size` = 1024
    /// - `strict_mode` = false
    /// - `utf8_view` = false
    /// - `reader_schema` = None
    /// - `writer_schema_store` = None
    /// - `active_fp` = None
    /// - `static_store_mode` = false
    pub fn new() -> Self {
        Self::default()
    }

    fn validate_impl(&self) -> Result<(), ArrowError> {
        if self.writer_schema_store.is_some() && self.reader_schema.is_none() {
            return Err(ArrowError::ParseError(
                "Reader schema must be set when writer schema store is provided".into(),
            ));
        }
        if self.active_fp.is_some() && self.writer_schema_store.is_none() {
            return Err(ArrowError::ParseError(
                "Active fingerprint requires a writer schema store".into(),
            ));
        }
        if self.static_store_mode && self.writer_schema_store.is_none() {
            return Err(ArrowError::ParseError(
                "static_store_mode=true requires a writer schema store".into(),
            ));
        }
        Ok(())
    }

    fn make_record_decoder(&self, schema: &AvroSchema<'_>) -> Result<RecordDecoder, ArrowError> {
        let root_field = AvroField::try_from(schema)?;
        RecordDecoder::try_new_with_options(root_field.data_type(), self.utf8_view)
    }

    fn make_decoder(&self, header: Option<&Header>) -> Result<Decoder, ArrowError> {
        match header {
            Some(hdr) => {
                let writer_schema = hdr
                    .schema()
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?
                    .ok_or_else(|| {
                        ArrowError::ParseError("No Avro schema present in file header".into())
                    })?;
                let (root, reader_schema) = if let Some(reader_schema) = &self.reader_schema {
                    let root = if !compare_schemas(&writer_schema, reader_schema) {
                        AvroField::resolve_from_writer_and_reader(
                            &writer_schema,
                            reader_schema,
                            self.utf8_view,
                        )?
                    } else {
                        AvroField::try_from(reader_schema)?
                    };
                    (root, reader_schema.clone())
                } else {
                    let root = AvroField::try_from(&writer_schema)?;
                    (root, writer_schema.to_owned())
                };
                let record_decoder =
                    RecordDecoder::try_new_with_options(root.data_type(), self.utf8_view)?;
                Ok(Decoder::new(
                    self.batch_size,
                    None,
                    record_decoder,
                    HashMap::new(),
                    self.reader_schema.clone(),
                    self.utf8_view,
                    self.writer_schema_store.clone(),
                    true,
                ))
            }
            None => {
                let reader_schema = self.reader_schema.clone().ok_or_else(|| {
                    ArrowError::ParseError("Reader schema required for raw Avro".into())
                })?;
                let store = self.writer_schema_store.clone();
                let (init_fp, init_decoder) = if let Some(store) = &store {
                    if let Some(fp) = self.active_fp {
                        let writer_schema = store.lookup(&fp).ok_or_else(|| {
                            ArrowError::ParseError(
                                "Active fingerprint not found in schema store".into(),
                            )
                        })?;
                        let root = AvroField::resolve_from_writer_and_reader(
                            &writer_schema,
                            &reader_schema,
                            self.utf8_view,
                        )?;
                        let dec =
                            RecordDecoder::try_new_with_options(root.data_type(), self.utf8_view)?;
                        (Some(fp), dec)
                    } else {
                        let dec = self.make_record_decoder(&reader_schema)?;
                        (None, dec)
                    }
                } else {
                    let dec = self.make_record_decoder(&reader_schema)?;
                    (None, dec)
                };
                Ok(Decoder::new(
                    self.batch_size,
                    init_fp,
                    init_decoder,
                    HashMap::new(),
                    Some(reader_schema),
                    self.utf8_view,
                    store,
                    self.static_store_mode,
                ))
            }
        }
    }

    /// Sets the row-based batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set whether to use StringViewArray for string data
    ///
    /// When enabled, string data from Avro files will be loaded into
    /// Arrow's StringViewArray instead of the standard StringArray.
    pub fn with_utf8_view(mut self, utf8_view: bool) -> Self {
        self.utf8_view = utf8_view;
        self
    }

    /// Get whether StringViewArray is enabled for string data
    pub fn use_utf8view(&self) -> bool {
        self.utf8_view
    }

    /// Controls whether certain Avro unions of the form `[T, "null"]` should produce an error.
    pub fn with_strict_mode(mut self, strict_mode: bool) -> Self {
        self.strict_mode = strict_mode;
        self
    }

    /// Sets the Avro Reader Schema.
    ///
    /// If a reader schema is not provided, the reader schema will be read from the Avro file header.
    /// This schema must match the desired arrow `Schema` to be decoded into.
    pub fn with_reader_schema(mut self, reader_schema: AvroSchema<'static>) -> Self {
        self.reader_schema = Some(reader_schema);
        self
    }

    /// Sets the Avro Writer Schema Store.
    ///
    /// If a writer schema is not provided, the writer schema will be read from the Avro file header.
    /// `with_writer_schema_store` requires a reader schema having been set, i.e. `with_reader_schema()`.
    /// Note: All writer schemas must be resolvable with the set reader schema per the Avro specifications
    pub fn with_writer_schema_store(mut self, writer_schema_store: SchemaStore<'static>) -> Self {
        self.writer_schema_store = Some(writer_schema_store);
        self
    }

    /// Sets the active fingerprint
    ///
    /// If an active Fingerprint is not provided, then schema resolution will not start until the
    /// first Fingerprint is received from a single object encoding.
    /// Note: This setting assumes that a `SchemaStore` was provided via `with_writer_schema_store()`.
    pub fn with_active_fingerprint(mut self, active_fp: Fingerprint) -> Self {
        self.active_fp = Some(active_fp);
        self
    }

    /// Controls whether new Avro writer schemas will attempt to be resolved and decoded when a
    /// `SchemaStore` was provided. If set to `true`, then only schemas provided in the `SchemaStore` will be used.
    ///
    /// `with_static_store_mode` requires a writer schema store to have been set, i.e. `with_writer_schema_store()`.
    pub fn with_static_store_mode(mut self, static_store_mode: bool) -> Self {
        self.static_store_mode = static_store_mode;
        self
    }

    /// Create a [`Reader`] from this builder and a `BufRead`
    pub fn build<R: BufRead>(self, mut reader: R) -> Result<Reader<R>, ArrowError> {
        self.validate_impl()?;
        let header = read_header(&mut reader)?;
        let decoder = self.make_decoder(Some(&header))?;
        Ok(Reader {
            reader,
            header,
            decoder,
            block_decoder: BlockDecoder::default(),
            block_data: Vec::new(),
            block_cursor: 0,
            finished: false,
        })
    }

    /// Create a [`Decoder`] from this builder and a `BufRead` by
    /// reading and parsing the Avro file's header. This will
    /// not create a full [`Reader`].
    pub fn build_decoder<R: BufRead>(self, reader: R) -> Result<Decoder, ArrowError> {
        self.validate_impl()?;
        self.make_decoder(None)
    }
}

/// A high-level Avro `Reader` that reads container-file blocks
/// and feeds them into a row-level [`Decoder`].
#[derive(Debug)]
pub struct Reader<R: BufRead> {
    reader: R,
    header: Header,
    decoder: Decoder,
    block_decoder: BlockDecoder,
    block_data: Vec<u8>,
    block_cursor: usize,
    finished: bool,
}

impl<R: BufRead> Reader<R> {
    /// Return the Arrow schema discovered from the Avro file header
    pub fn schema(&self) -> SchemaRef {
        self.decoder.schema()
    }

    /// Return the Avro container-file header
    pub fn avro_header(&self) -> &Header {
        &self.header
    }

    /// Reads the next [`RecordBatch`] from the Avro file or `Ok(None)` on EOF
    fn read(&mut self) -> Result<Option<RecordBatch>, ArrowError> {
        'outer: while !self.finished && !self.decoder.batch_is_full() {
            while self.block_cursor == self.block_data.len() {
                let buf = self.reader.fill_buf()?;
                if buf.is_empty() {
                    self.finished = true;
                    break 'outer;
                }
                let consumed = self.block_decoder.decode(buf)?;
                self.reader.consume(consumed);
                if let Some(block) = self.block_decoder.flush() {
                    let block_data = if let Some(ref codec) = self.header.compression()? {
                        codec.decompress(&block.data)?
                    } else {
                        block.data
                    };
                    self.block_data = block_data;
                    self.block_cursor = 0;
                } else if consumed == 0 {
                    return Err(ArrowError::ParseError(
                        "Could not decode next Avro block from partial data".to_string(),
                    ));
                }
            }
            let consumed = self.decoder.decode(&self.block_data[self.block_cursor..])?;
            self.block_cursor += consumed;
        }
        self.decoder.flush()
    }
}

impl<R: BufRead> Iterator for Reader<R> {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.read().transpose()
    }
}

impl<R: BufRead> RecordBatchReader for Reader<R> {
    fn schema(&self) -> SchemaRef {
        self.schema()
    }
}

#[cfg(test)]
mod test {
    use crate::codec::{AvroDataType, AvroField, Codec};
    use crate::compression::CompressionCodec;
    use crate::reader::record::RecordDecoder;
    use crate::reader::vlq::VLQDecoder;
    use crate::reader::{read_header, Decoder, ReaderBuilder};
    use crate::test_util::arrow_test_data;
    use arrow_array::types::{Int32Type, IntervalMonthDayNanoType};
    use arrow_array::*;
    use arrow_schema::{ArrowError, DataType, Field, IntervalUnit, Schema};
    use bytes::{Buf, BufMut, Bytes};
    use futures::executor::block_on;
    use futures::{stream, Stream, StreamExt, TryStreamExt};
    use std::collections::HashMap;
    use std::fs;
    use std::fs::File;
    use std::io::{BufReader, Cursor, Read};
    use std::sync::Arc;
    use std::task::{ready, Poll};

    fn read_file(path: &str, batch_size: usize, utf8_view: bool) -> RecordBatch {
        let file = File::open(path).unwrap();
        let reader = ReaderBuilder::new()
            .with_batch_size(batch_size)
            .with_utf8_view(utf8_view)
            .build(BufReader::new(file))
            .unwrap();
        let schema = reader.schema();
        let batches = reader.collect::<Result<Vec<_>, _>>().unwrap();
        arrow::compute::concat_batches(&schema, &batches).unwrap()
    }

    fn decode_stream<S: Stream<Item = Bytes> + Unpin>(
        mut decoder: Decoder,
        mut input: S,
    ) -> impl Stream<Item = Result<RecordBatch, ArrowError>> {
        async_stream::try_stream! {
            if let Some(data) = input.next().await {
                let consumed = decoder.decode(&data)?;
                if consumed < data.len() {
                    Err(ArrowError::ParseError(
                        "did not consume all bytes".to_string(),
                    ))?;
                }
            }
            if let Some(batch) = decoder.flush()? {
                yield batch
            }
        }
    }

    #[test]
    fn test_utf8view_support() {
        let schema_json = r#"{
            "type": "record",
            "name": "test",
            "fields": [{
                "name": "str_field",
                "type": "string"
            }]
        }"#;

        let schema: crate::schema::Schema = serde_json::from_str(schema_json).unwrap();
        let avro_field = AvroField::try_from(&schema).unwrap();

        let data_type = avro_field.data_type();

        struct TestHelper;
        impl TestHelper {
            fn with_utf8view(field: &Field) -> Field {
                match field.data_type() {
                    DataType::Utf8 => {
                        Field::new(field.name(), DataType::Utf8View, field.is_nullable())
                            .with_metadata(field.metadata().clone())
                    }
                    _ => field.clone(),
                }
            }
        }

        let field = TestHelper::with_utf8view(&Field::new("str_field", DataType::Utf8, false));

        assert_eq!(field.data_type(), &DataType::Utf8View);

        let array = StringViewArray::from(vec!["test1", "test2"]);
        let batch =
            RecordBatch::try_from_iter(vec![("str_field", Arc::new(array) as ArrayRef)]).unwrap();

        assert!(batch.column(0).as_any().is::<StringViewArray>());
    }

    #[test]
    fn test_alltypes() {
        let files = [
            "avro/alltypes_plain.avro",
            "avro/alltypes_plain.snappy.avro",
            "avro/alltypes_plain.zstandard.avro",
            "avro/alltypes_plain.bzip2.avro",
            "avro/alltypes_plain.xz.avro",
        ];

        let expected = RecordBatch::try_from_iter_with_nullable([
            (
                "id",
                Arc::new(Int32Array::from(vec![4, 5, 6, 7, 2, 3, 0, 1])) as _,
                true,
            ),
            (
                "bool_col",
                Arc::new(BooleanArray::from_iter((0..8).map(|x| Some(x % 2 == 0)))) as _,
                true,
            ),
            (
                "tinyint_col",
                Arc::new(Int32Array::from_iter_values((0..8).map(|x| x % 2))) as _,
                true,
            ),
            (
                "smallint_col",
                Arc::new(Int32Array::from_iter_values((0..8).map(|x| x % 2))) as _,
                true,
            ),
            (
                "int_col",
                Arc::new(Int32Array::from_iter_values((0..8).map(|x| x % 2))) as _,
                true,
            ),
            (
                "bigint_col",
                Arc::new(Int64Array::from_iter_values((0..8).map(|x| (x % 2) * 10))) as _,
                true,
            ),
            (
                "float_col",
                Arc::new(Float32Array::from_iter_values(
                    (0..8).map(|x| (x % 2) as f32 * 1.1),
                )) as _,
                true,
            ),
            (
                "double_col",
                Arc::new(Float64Array::from_iter_values(
                    (0..8).map(|x| (x % 2) as f64 * 10.1),
                )) as _,
                true,
            ),
            (
                "date_string_col",
                Arc::new(BinaryArray::from_iter_values([
                    [48, 51, 47, 48, 49, 47, 48, 57],
                    [48, 51, 47, 48, 49, 47, 48, 57],
                    [48, 52, 47, 48, 49, 47, 48, 57],
                    [48, 52, 47, 48, 49, 47, 48, 57],
                    [48, 50, 47, 48, 49, 47, 48, 57],
                    [48, 50, 47, 48, 49, 47, 48, 57],
                    [48, 49, 47, 48, 49, 47, 48, 57],
                    [48, 49, 47, 48, 49, 47, 48, 57],
                ])) as _,
                true,
            ),
            (
                "string_col",
                Arc::new(BinaryArray::from_iter_values((0..8).map(|x| [48 + x % 2]))) as _,
                true,
            ),
            (
                "timestamp_col",
                Arc::new(
                    TimestampMicrosecondArray::from_iter_values([
                        1235865600000000, // 2009-03-01T00:00:00.000
                        1235865660000000, // 2009-03-01T00:01:00.000
                        1238544000000000, // 2009-04-01T00:00:00.000
                        1238544060000000, // 2009-04-01T00:01:00.000
                        1233446400000000, // 2009-02-01T00:00:00.000
                        1233446460000000, // 2009-02-01T00:01:00.000
                        1230768000000000, // 2009-01-01T00:00:00.000
                        1230768060000000, // 2009-01-01T00:01:00.000
                    ])
                    .with_timezone("+00:00"),
                ) as _,
                true,
            ),
        ])
        .unwrap();

        for file in files {
            let file = arrow_test_data(file);

            assert_eq!(read_file(&file, 8, false), expected);
            assert_eq!(read_file(&file, 3, false), expected);
        }
    }

    #[test]
    fn test_decode_stream_with_schema() {
        struct TestCase<'a> {
            name: &'a str,
            schema: &'a str,
            expected_error: Option<&'a str>,
        }
        let tests = vec![
            TestCase {
                name: "success",
                schema: r#"{"type":"record","name":"test","fields":[{"name":"f2","type":"string"}]}"#,
                expected_error: None,
            },
            TestCase {
                name: "valid schema invalid data",
                schema: r#"{"type":"record","name":"test","fields":[{"name":"f2","type":"long"}]}"#,
                expected_error: Some("did not consume all bytes"),
            },
        ];
        for test in tests {
            let schema_s2: crate::schema::Schema = serde_json::from_str(test.schema).unwrap();
            let record_val = "some_string";
            let mut body = vec![];
            body.push((record_val.len() as u8) << 1);
            body.extend_from_slice(record_val.as_bytes());
            let mut reader_placeholder = Cursor::new(&[] as &[u8]);
            let builder = ReaderBuilder::new()
                .with_batch_size(1)
                .with_reader_schema(schema_s2);
            let decoder_result = builder.build_decoder(&mut reader_placeholder);
            let decoder = match decoder_result {
                Ok(decoder) => decoder,
                Err(e) => {
                    if let Some(expected) = test.expected_error {
                        assert!(
                            e.to_string().contains(expected),
                            "Test '{}' failed: unexpected error message at build.\nExpected to contain: '{expected}'\nActual: '{e}'",
                            test.name,
                        );
                        continue;
                    } else {
                        panic!("Test '{}' failed at decoder build: {e}", test.name);
                    }
                }
            };
            let stream = Box::pin(stream::once(async { Bytes::from(body) }));
            let decoded_stream = decode_stream(decoder, stream);
            let batches_result: Result<Vec<RecordBatch>, ArrowError> =
                block_on(decoded_stream.try_collect());
            match (batches_result, test.expected_error) {
                (Ok(batches), None) => {
                    let batch =
                        arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
                    let expected_field = Field::new("f2", DataType::Utf8, false);
                    let expected_schema = Arc::new(Schema::new(vec![expected_field]));
                    let expected_array = Arc::new(StringArray::from(vec![record_val]));
                    let expected_batch =
                        RecordBatch::try_new(expected_schema, vec![expected_array]).unwrap();
                    assert_eq!(batch, expected_batch, "Test '{}' failed", test.name);
                    assert_eq!(
                        batch.schema().field(0).name(),
                        "f2",
                        "Test '{}' failed",
                        test.name
                    );
                }
                (Err(e), Some(expected)) => {
                    assert!(
                        e.to_string().contains(expected),
                        "Test '{}' failed: unexpected error message at decode.\nExpected to contain: '{expected}'\nActual: '{e}'",
                        test.name,
                    );
                }
                (Ok(batches), Some(expected)) => {
                    panic!(
                        "Test '{}' was expected to fail with '{expected}', but it succeeded with: {:?}",
                        test.name, batches
                    );
                }
                (Err(e), None) => {
                    panic!(
                        "Test '{}' was not expected to fail, but it did with '{e}'",
                        test.name
                    );
                }
            }
        }
    }

    #[test]
    fn test_decimal() {
        let files = [
            ("avro/fixed_length_decimal.avro", 25, 2),
            ("avro/fixed_length_decimal_legacy.avro", 13, 2),
            ("avro/int32_decimal.avro", 4, 2),
            ("avro/int64_decimal.avro", 10, 2),
        ];
        let decimal_values: Vec<i128> = (1..=24).map(|n| n as i128 * 100).collect();
        for (file, precision, scale) in files {
            let file_path = arrow_test_data(file);
            let actual_batch = read_file(&file_path, 8, false);
            let expected_array = Decimal128Array::from_iter_values(decimal_values.clone())
                .with_precision_and_scale(precision, scale)
                .unwrap();
            let mut meta = HashMap::new();
            meta.insert("precision".to_string(), precision.to_string());
            meta.insert("scale".to_string(), scale.to_string());
            let field_with_meta = Field::new("value", DataType::Decimal128(precision, scale), true)
                .with_metadata(meta);
            let expected_schema = Arc::new(Schema::new(vec![field_with_meta]));
            let expected_batch =
                RecordBatch::try_new(expected_schema.clone(), vec![Arc::new(expected_array)])
                    .expect("Failed to build expected RecordBatch");
            assert_eq!(
                actual_batch, expected_batch,
                "Decoded RecordBatch does not match the expected Decimal128 data for file {file}"
            );
            let actual_batch_small = read_file(&file_path, 3, false);
            assert_eq!(
                actual_batch_small,
                expected_batch,
                "Decoded RecordBatch does not match the expected Decimal128 data for file {file} with batch size 3"
            );
        }
    }

    #[test]
    fn test_simple() {
        let tests = [
            ("avro/simple_enum.avro", 4, build_expected_enum(), 2),
            ("avro/simple_fixed.avro", 2, build_expected_fixed(), 1),
        ];

        fn build_expected_enum() -> RecordBatch {
            // Build the DictionaryArrays for f1, f2, f3
            let keys_f1 = Int32Array::from(vec![0, 1, 2, 3]);
            let vals_f1 = StringArray::from(vec!["a", "b", "c", "d"]);
            let f1_dict =
                DictionaryArray::<Int32Type>::try_new(keys_f1, Arc::new(vals_f1)).unwrap();
            let keys_f2 = Int32Array::from(vec![2, 3, 0, 1]);
            let vals_f2 = StringArray::from(vec!["e", "f", "g", "h"]);
            let f2_dict =
                DictionaryArray::<Int32Type>::try_new(keys_f2, Arc::new(vals_f2)).unwrap();
            let keys_f3 = Int32Array::from(vec![Some(1), Some(2), None, Some(0)]);
            let vals_f3 = StringArray::from(vec!["i", "j", "k"]);
            let f3_dict =
                DictionaryArray::<Int32Type>::try_new(keys_f3, Arc::new(vals_f3)).unwrap();
            let dict_type =
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
            let mut md_f1 = HashMap::new();
            md_f1.insert(
                "avro.enum.symbols".to_string(),
                r#"["a","b","c","d"]"#.to_string(),
            );
            let f1_field = Field::new("f1", dict_type.clone(), false).with_metadata(md_f1);
            let mut md_f2 = HashMap::new();
            md_f2.insert(
                "avro.enum.symbols".to_string(),
                r#"["e","f","g","h"]"#.to_string(),
            );
            let f2_field = Field::new("f2", dict_type.clone(), false).with_metadata(md_f2);
            let mut md_f3 = HashMap::new();
            md_f3.insert(
                "avro.enum.symbols".to_string(),
                r#"["i","j","k"]"#.to_string(),
            );
            let f3_field = Field::new("f3", dict_type.clone(), true).with_metadata(md_f3);
            let expected_schema = Arc::new(Schema::new(vec![f1_field, f2_field, f3_field]));
            RecordBatch::try_new(
                expected_schema,
                vec![
                    Arc::new(f1_dict) as Arc<dyn Array>,
                    Arc::new(f2_dict) as Arc<dyn Array>,
                    Arc::new(f3_dict) as Arc<dyn Array>,
                ],
            )
            .unwrap()
        }

        fn build_expected_fixed() -> RecordBatch {
            let f1 =
                FixedSizeBinaryArray::try_from_iter(vec![b"abcde", b"12345"].into_iter()).unwrap();
            let f2 =
                FixedSizeBinaryArray::try_from_iter(vec![b"fghijklmno", b"1234567890"].into_iter())
                    .unwrap();
            let f3 = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                vec![Some(b"ABCDEF" as &[u8]), None].into_iter(),
                6,
            )
            .unwrap();
            let expected_schema = Arc::new(Schema::new(vec![
                Field::new("f1", DataType::FixedSizeBinary(5), false),
                Field::new("f2", DataType::FixedSizeBinary(10), false),
                Field::new("f3", DataType::FixedSizeBinary(6), true),
            ]));
            RecordBatch::try_new(
                expected_schema,
                vec![
                    Arc::new(f1) as Arc<dyn Array>,
                    Arc::new(f2) as Arc<dyn Array>,
                    Arc::new(f3) as Arc<dyn Array>,
                ],
            )
            .unwrap()
        }
        for (file_name, batch_size, expected, alt_batch_size) in tests {
            let file = arrow_test_data(file_name);
            let actual = read_file(&file, batch_size, false);
            assert_eq!(actual, expected);
            let actual2 = read_file(&file, alt_batch_size, false);
            assert_eq!(actual2, expected);
        }
    }

    #[test]
    fn test_duration_uuid() {
        let batch = read_file("test/data/duration_uuid.avro", 4, false);
        let schema = batch.schema();
        let fields = schema.fields();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name(), "duration_field");
        assert_eq!(
            fields[0].data_type(),
            &DataType::Interval(IntervalUnit::MonthDayNano)
        );
        assert_eq!(fields[1].name(), "uuid_field");
        assert_eq!(fields[1].data_type(), &DataType::FixedSizeBinary(16));
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 2);
        let duration_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<IntervalMonthDayNanoArray>()
            .unwrap();
        let expected_duration_array: IntervalMonthDayNanoArray = [
            Some(IntervalMonthDayNanoType::make_value(1, 15, 500_000_000)),
            Some(IntervalMonthDayNanoType::make_value(0, 5, 2_500_000_000)),
            Some(IntervalMonthDayNanoType::make_value(2, 0, 0)),
            Some(IntervalMonthDayNanoType::make_value(12, 31, 999_000_000)),
        ]
        .iter()
        .copied()
        .collect();
        assert_eq!(&expected_duration_array, duration_array);
        let uuid_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();
        let expected_uuid_array = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
            [
                Some([
                    0xfe, 0x7b, 0xc3, 0x0b, 0x4c, 0xe8, 0x4c, 0x5e, 0xb6, 0x7c, 0x22, 0x34, 0xa2,
                    0xd3, 0x8e, 0x66,
                ]),
                Some([
                    0xb3, 0x3f, 0x2a, 0xd7, 0x97, 0xb4, 0x4d, 0xe1, 0x8b, 0xfe, 0x94, 0x94, 0x1d,
                    0x60, 0x15, 0x6e,
                ]),
                Some([
                    0x5f, 0x74, 0x92, 0x64, 0x07, 0x4b, 0x40, 0x05, 0x84, 0xbf, 0x11, 0x5e, 0xa8,
                    0x4e, 0xd2, 0x0a,
                ]),
                Some([
                    0x08, 0x26, 0xcc, 0x06, 0xd2, 0xe3, 0x45, 0x99, 0xb4, 0xad, 0xaf, 0x5f, 0xa6,
                    0x90, 0x5c, 0xdb,
                ]),
            ]
            .into_iter(),
            16,
        )
        .unwrap();
        assert_eq!(&expected_uuid_array, uuid_array);
    }
}
