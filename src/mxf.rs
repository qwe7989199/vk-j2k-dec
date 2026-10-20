//! Bounded, forward-only reading of frame-wrapped JPEG 2000 MXF picture essence.
//! Metadata/audio are skipped by their KLV lengths; no header dictionary is needed.
use std::io::{self, Read};

pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

pub struct MxfFrameReader<R> {
    reader: R,
    picture_key: Option<[u8; 16]>,
    pub frames: usize,
    pub essence_bytes: u64,
}

impl<R: Read> MxfFrameReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            picture_key: None,
            frames: 0,
            essence_bytes: 0,
        }
    }

    pub fn next_frame(&mut self) -> io::Result<Option<Vec<u8>>> {
        loop {
            let mut key = [0u8; 16];
            if self.reader.read(&mut key[..1])? == 0 {
                return Ok(None);
            }
            self.reader.read_exact(&mut key[1..])?;
            if key[..4] != [0x06, 0x0e, 0x2b, 0x34] {
                return Err(invalid("expected an MXF KLV universal label"));
            }
            let mut first = [0u8; 1];
            self.reader.read_exact(&mut first)?;
            let length = if first[0] < 0x80 {
                u64::from(first[0])
            } else {
                let count = usize::from(first[0] & 0x7f);
                if count == 0 || count > 8 {
                    return Err(invalid("invalid KLV BER length"));
                }
                let mut bytes = [0u8; 8];
                self.reader.read_exact(&mut bytes[8 - count..])?;
                u64::from_be_bytes(bytes)
            };
            // Generic Container picture essence (track number begins with 0x15).
            let picture = key[4..7] == [0x01, 0x02, 0x01]
                && key[8..12] == [0x0d, 0x01, 0x03, 0x01]
                && key[12] == 0x15;
            if !picture {
                let copied = io::copy(&mut (&mut self.reader).take(length), &mut io::sink())?;
                if copied != length {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "truncated MXF KLV value",
                    ));
                }
                continue;
            }
            if self.picture_key.is_some_and(|previous| previous != key) {
                return Err(invalid("multiple MXF picture tracks are not supported"));
            }
            self.picture_key = Some(key);
            if length < 4 || length > MAX_FRAME_BYTES as u64 {
                return Err(invalid("picture essence exceeds the 64 MiB frame limit"));
            }
            let mut frame = vec![0u8; length as usize];
            self.reader.read_exact(&mut frame)?;
            if !frame.starts_with(&[0xff, 0x4f, 0xff, 0x51]) || !frame.ends_with(&[0xff, 0xd9]) {
                return Err(invalid(
                    "expected one complete JPEG 2000 codestream per picture KLV",
                ));
            }
            self.frames += 1;
            self.essence_bytes += length;
            return Ok(Some(frame));
        }
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn klv(data: &mut Vec<u8>, picture: bool, payload: &[u8]) {
        data.extend_from_slice(&[
            6,
            14,
            43,
            52,
            1,
            2,
            1,
            1,
            13,
            1,
            3,
            1,
            if picture { 21 } else { 22 },
            1,
            8,
            1,
        ]);
        data.extend_from_slice(&[0x83, 0, 0, payload.len() as u8]);
        data.extend_from_slice(payload);
    }

    #[test]
    fn klv_boundaries_skip_audio_and_ignore_marker_lookalikes_inside_frames() {
        let frame = [0xff, 0x4f, 0xff, 0x51, 0xff, 0x4f, 0xff, 0x51, 0xff, 0xd9];
        let mut data = Vec::new();
        klv(&mut data, false, &frame);
        klv(&mut data, true, &frame);
        klv(&mut data, true, &frame[..4]); // A truncated picture must not pass as a frame.
        let mut reader = MxfFrameReader::new(Cursor::new(data));
        assert_eq!(reader.next_frame().unwrap().unwrap(), frame);
        assert_eq!(reader.frames, 1);
        assert!(reader.next_frame().is_err());
    }

    #[test]
    fn eof_is_distinct_from_a_truncated_klv() {
        assert!(
            MxfFrameReader::new(Cursor::new(Vec::new()))
                .next_frame()
                .unwrap()
                .is_none()
        );
        assert!(
            MxfFrameReader::new(Cursor::new(vec![6, 14]))
                .next_frame()
                .is_err()
        );
    }
}
