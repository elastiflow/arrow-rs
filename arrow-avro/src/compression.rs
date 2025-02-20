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

use arrow_schema::ArrowError;
use std::io::Read;

/// The metadata key used for storing the JSON encoded [`CompressionCodec`]
pub const CODEC_METADATA_KEY: &str = "avro.codec";

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
/// CompressionCodec includes the enumerated types for each supported compression
/// type
pub enum CompressionCodec {
    /// Deflate - compression
    Deflate,
    /// Snappy - compression
    Snappy,
    /// ZStandard - compression
    ZStandard,
    /// Bzip2 - compression
    Bzip2,
    /// Xz - compression
    Xz,
}

impl CompressionCodec {
    /// Decompress an Avro block that was encoded with this codec.
    /// Used by the **reader** to decode block data from an Avro container file.
    pub(crate) fn decompress(&self, block: &[u8]) -> Result<Vec<u8>, ArrowError> {
        match self {
            #[cfg(feature = "deflate")]
            CompressionCodec::Deflate => {
                let mut decoder = flate2::read::DeflateDecoder::new(block);
                let mut out = Vec::new();
                decoder.read_to_end(&mut out)?;
                Ok(out)
            }
            #[cfg(not(feature = "deflate"))]
            CompressionCodec::Deflate => Err(ArrowError::ParseError(
                "Deflate codec requires deflate feature".to_string(),
            )),

            #[cfg(feature = "snappy")]
            CompressionCodec::Snappy => {
                if block.len() < 4 {
                    return Err(ArrowError::ParseError(
                        "Snappy block too short to contain trailing crc".to_string(),
                    ));
                }
                let crc = &block[block.len() - 4..];
                let block_data = &block[..block.len() - 4];
                let mut decoder = snap::raw::Decoder::new();
                let decoded = decoder
                    .decompress_vec(block_data)
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;

                let checksum = crc::Crc::<u32>::new(&crc::CRC_32_ISO_HDLC).checksum(&decoded);
                if checksum != u32::from_be_bytes(crc.try_into().unwrap()) {
                    return Err(ArrowError::ParseError("Snappy CRC mismatch".to_string()));
                }
                Ok(decoded)
            }
            #[cfg(not(feature = "snappy"))]
            CompressionCodec::Snappy => Err(ArrowError::ParseError(
                "Snappy codec requires snappy feature".to_string(),
            )),

            #[cfg(feature = "zstd")]
            CompressionCodec::ZStandard => {
                let mut decoder = zstd::Decoder::new(block)?;
                let mut out = Vec::new();
                decoder.read_to_end(&mut out)?;
                Ok(out)
            }
            #[cfg(not(feature = "zstd"))]
            CompressionCodec::ZStandard => Err(ArrowError::ParseError(
                "ZStandard codec requires zstd feature".to_string(),
            )),

            #[cfg(feature = "bzip2")]
            CompressionCodec::Bzip2 => {
                let mut decoder = bzip2::read::BzDecoder::new(block);
                let mut out = Vec::new();
                decoder.read_to_end(&mut out)?;
                Ok(out)
            }
            #[cfg(not(feature = "bzip2"))]
            CompressionCodec::Bzip2 => Err(ArrowError::ParseError(
                "Bzip2 codec requires bzip2 feature".to_string(),
            )),

            #[cfg(feature = "xz")]
            CompressionCodec::Xz => {
                let mut decoder = xz::read::XzDecoder::new(block);
                let mut out = Vec::new();
                decoder.read_to_end(&mut out)?;
                Ok(out)
            }
            #[cfg(not(feature = "xz"))]
            CompressionCodec::Xz => Err(ArrowError::ParseError(
                "XZ codec requires xz feature".to_string(),
            )),
        }
    }

    /// Compress a block using this Avro codec.
    /// Used by the **writer** to encode block data before writing it.
    ///
    /// Snappy: Avro requires a 4-byte big-endian CRC32 of the *uncompressed* data appended.
    pub(crate) fn compress_block(&self, data: &[u8]) -> Result<Vec<u8>, ArrowError> {
        match self {
            #[cfg(feature = "deflate")]
            CompressionCodec::Deflate => {
                use flate2::{write::DeflateEncoder, Compression};
                use std::io::Write;
                let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
                encoder
                    .write_all(data)
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                let compressed = encoder
                    .finish()
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                Ok(compressed)
            }
            #[cfg(not(feature = "deflate"))]
            CompressionCodec::Deflate => Err(ArrowError::ParseError(
                "Deflate codec requires deflate feature".to_string(),
            )),

            #[cfg(feature = "snappy")]
            CompressionCodec::Snappy => {
                let mut encoder = snap::raw::Encoder::new();
                let compressed = encoder
                    .compress_vec(data)
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                let crc = crc::Crc::<u32>::new(&crc::CRC_32_ISO_HDLC).checksum(data);
                let mut out = Vec::with_capacity(compressed.len() + 4);
                out.extend_from_slice(&compressed);
                out.extend_from_slice(&crc.to_be_bytes());
                Ok(out)
            }
            #[cfg(not(feature = "snappy"))]
            CompressionCodec::Snappy => Err(ArrowError::ParseError(
                "Snappy codec requires snappy feature".to_string(),
            )),

            #[cfg(feature = "zstd")]
            CompressionCodec::ZStandard => {
                use std::io::Write;
                let mut encoder = zstd::Encoder::new(Vec::new(), 0)
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                encoder
                    .write_all(data)
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                let compressed = encoder
                    .finish()
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                Ok(compressed)
            }
            #[cfg(not(feature = "zstd"))]
            CompressionCodec::ZStandard => Err(ArrowError::ParseError(
                "ZStandard codec requires zstd feature".to_string(),
            )),

            #[cfg(feature = "bzip2")]
            CompressionCodec::Bzip2 => {
                use std::io::Write;
                use bzip2::{write::BzEncoder, Compression};
                let mut encoder = BzEncoder::new(Vec::new(), Compression::default());
                encoder
                    .write_all(data)
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                let compressed = encoder
                    .finish()
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                Ok(compressed)
            }
            #[cfg(not(feature = "bzip2"))]
            CompressionCodec::Bzip2 => Err(ArrowError::ParseError(
                "Bzip2 codec requires bzip2 feature".to_string(),
            )),

            #[cfg(feature = "xz")]
            CompressionCodec::Xz => {
                use std::io::Write;
                let mut encoder = xz::write::XzEncoder::new(Vec::new(), 6);
                encoder
                    .write_all(data)
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                let compressed = encoder
                    .finish()
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                Ok(compressed)
            }
            #[cfg(not(feature = "xz"))]
            CompressionCodec::Xz => Err(ArrowError::ParseError(
                "XZ codec requires xz feature".to_string(),
            )),
        }
    }
}
