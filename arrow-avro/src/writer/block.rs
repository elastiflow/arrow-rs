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

//! Manages Avro data blocks (accumulating row data, compressing, flushing to disk).

use std::io::Write;

use arrow_schema::ArrowError;

use crate::compression::CompressionCodec;
use crate::writer::utils::{to_arrow_io_err};
use crate::writer::zigzag::write_zigzag_long;

/// Handles buffering Avro-encoded rows, compressing them if needed,
/// and writing them in Avro block format.
#[derive(Debug)]
pub struct BlockEncoder {
    block_buf: Vec<u8>,
    block_count: usize,
    max_block_size: usize,
    compression: Option<CompressionCodec>,
    sync_marker: [u8; 16],
}

impl BlockEncoder {
    /// Create a new `BlockEncoder`.
    ///
    /// * `compression`: Optional compression codec
    /// * `sync_marker`: The 16-byte sync marker used in the Avro container file
    /// * `max_block_size`: Threshold in bytes; once `block_buf` grows past this,
    ///    the block is flushed to the sink.
    pub fn new(
        compression: Option<CompressionCodec>,
        sync_marker: [u8; 16],
        max_block_size: usize,
    ) -> Self {
        Self {
            block_buf: Vec::new(),
            block_count: 0,
            max_block_size,
            compression,
            sync_marker,
        }
    }

    /// Appends encoded bytes for a row (or partial row chunk) into the internal buffer.
    pub fn append_encoded(&mut self, data: &[u8]) {
        self.block_buf.extend_from_slice(data);
    }

    /// Increments the row count for the next flush.
    pub fn inc_count(&mut self) {
        self.block_count += 1;
    }

    /// Flush the current buffer if it exceeds `max_block_size`.
    pub fn maybe_flush(&mut self, sink: &mut dyn Write) -> Result<(), ArrowError> {
        if self.block_buf.len() >= self.max_block_size {
            self.flush_block(sink)?;
        }
        Ok(())
    }

    /// Force a flush of the current block, if any rows are present.
    pub fn flush_block(&mut self, sink: &mut dyn Write) -> Result<(), ArrowError> {
        if self.block_count == 0 {
            return Ok(());
        }
        write_zigzag_long(self.block_count as i64, sink)?;
        let payload = if let Some(codec) = self.compression {
            codec.compress_block(&self.block_buf)?
        } else {
            self.block_buf.clone()
        };
        write_zigzag_long(payload.len() as i64, sink)?;
        sink.write_all(&payload)
            .map_err(|e| to_arrow_io_err(e, "Writing block data"))?;
        sink.write_all(&self.sync_marker)
            .map_err(|e| to_arrow_io_err(e, "Writing sync marker"))?;
        self.block_buf.clear();
        self.block_count = 0;
        Ok(())
    }

    /// Flush any leftover data. Called when finishing or closing the file.
    pub fn close(&mut self, sink: &mut dyn Write) -> Result<(), ArrowError> {
        self.flush_block(sink)?;
        Ok(())
    }
}
