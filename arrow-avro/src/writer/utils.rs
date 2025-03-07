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

use std::io::{Error as IoError, Write};

use arrow_schema::ArrowError;

use crate::writer::zigzag::write_zigzag_long;

/// Wraps an `std::io::Error` with a helpful context, returning an `ArrowError`.
pub fn to_arrow_io_err(e: IoError, context: &str) -> ArrowError {
    let msg = format!("{context}: {e}");
    ArrowError::ExternalError(Box::new(std::io::Error::new(e.kind(), msg)))
}

/// Write a UTF-8 string in Avro format: zigzag-encoded length + raw bytes
pub fn write_string(s: &str, w: &mut dyn Write) -> Result<(), ArrowError> {
    write_bytes(s.as_bytes(), w)
}

/// Write raw bytes in Avro format: zigzag-encoded length + raw bytes
pub fn write_bytes(b: &[u8], w: &mut dyn Write) -> Result<(), ArrowError> {
    write_zigzag_long(b.len() as i64, w)?;
    w.write_all(b)
        .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
    Ok(())
}
