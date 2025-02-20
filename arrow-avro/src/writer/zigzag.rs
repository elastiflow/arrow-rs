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

//! Zigzag variable-length encoding for Avro "long" and "int" types

use std::io::Write;
use arrow_schema::ArrowError;

/// Write an Avro "long" (i64) in **zigzag** variable-length format into `writer`.
///
pub fn write_zigzag_long(n: i64, writer: &mut dyn Write) -> Result<(), ArrowError> {
    let zz = ((n << 1) ^ (n >> 63)) as u64;
    write_varint_u64(zz, writer)
}

/// Write an Avro "long" (i32) in **zigzag** variable-length format into `writer`.
///
pub fn write_zigzag_int(n: i32, writer: &mut dyn Write) -> Result<(), ArrowError> {
    let i64_val = n as i64;
    write_zigzag_long(i64_val, writer)
}

fn write_varint_u64(mut val: u64, writer: &mut dyn Write) -> Result<(), ArrowError> {
    let mut buf = [0u8; 10];
    let mut i = 0;
    loop {
        let b = (val & 0x7F) as u8;
        val >>= 7;
        if val != 0 {
            buf[i] = b | 0x80;
            i += 1;
        } else {
            buf[i] = b;
            i += 1;
            break;
        }
        if i >= buf.len() {
            return Err(ArrowError::ParseError(
                "Varint exceeded 10 bytes, not a valid Avro long".to_string(),
            ));
        }
    }
    writer
        .write_all(&buf[..i])
        .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_zigzag_long_positive() {
        let mut buffer = Cursor::new(Vec::new());
        write_zigzag_long(5, &mut buffer).unwrap();
        assert_eq!(buffer.into_inner(), vec![0x0A]);
    }

    #[test]
    fn test_zigzag_long_negative() {
        let mut buffer = Cursor::new(Vec::new());
        write_zigzag_long(-1, &mut buffer).unwrap();
        assert_eq!(buffer.into_inner(), vec![0x01]);
    }

    #[test]
    fn test_zigzag_long_zero() {
        let mut buffer = Cursor::new(Vec::new());
        write_zigzag_long(0, &mut buffer).unwrap();
        assert_eq!(buffer.into_inner(), vec![0x00]);
    }

    #[test]
    fn test_zigzag_int_positive() {
        let mut buffer = Cursor::new(Vec::new());
        write_zigzag_int(3, &mut buffer).unwrap();
        assert_eq!(buffer.into_inner(), vec![0x06]);
    }

    #[test]
    fn test_zigzag_int_negative() {
        let mut buffer = Cursor::new(Vec::new());
        write_zigzag_int(-3, &mut buffer).unwrap();
        assert_eq!(buffer.into_inner(), vec![0x05]);
    }
}
