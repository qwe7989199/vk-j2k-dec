use std::io::{BufReader, Read};
use std::process::{Child, Command, Stdio};

pub struct FrameReader {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
    buffer: Vec<u8>,
    pending: Vec<u8>,
}

impl FrameReader {
    pub fn open(input_path: &str, ffmpeg_path: &str) -> Result<Self, String> {
        let mut child = Command::new(ffmpeg_path)
            .args([
                "-hide_banner",
                "-loglevel", "error",
                "-i", input_path,
                "-map", "0:v:0",
                "-c:v", "copy",
                "-f", "image2pipe",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to spawn ffmpeg: {e}"))?;

        let stdout = child.stdout.take().ok_or("no stdout from ffmpeg")?;

        Ok(Self {
            child,
            reader: BufReader::with_capacity(4 * 1024 * 1024, stdout),
            buffer: Vec::with_capacity(512 * 1024),
            pending: Vec::with_capacity(4 * 1024 * 1024),
        })
    }

    pub fn next_frame(&mut self) -> Option<&[u8]> {
        loop {
            if self.try_extract_frame() {
                return Some(&self.buffer);
            }
            let mut chunk = [0u8; 64 * 1024];
            let n = self.reader.read(&mut chunk).ok()?;
            if n == 0 {
                return None;
            }
            self.pending.extend_from_slice(&chunk[..n]);
        }
    }

    fn try_extract_frame(&mut self) -> bool {
        let data = &self.pending;
        if data.len() < 4 {
            return false;
        }

        let Some(soc_pos) = find_soc(data, 0) else {
            return false;
        };
        let search_start = soc_pos + 2;
        let Some(eoc_pos) = find_eoc(data, search_start) else {
            return false;
        };
        let frame_end = eoc_pos + 2;

        self.buffer.clear();
        self.buffer.extend_from_slice(&self.pending[soc_pos..frame_end]);
        self.pending.drain(..frame_end);
        true
    }
}

impl Drop for FrameReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn find_soc(data: &[u8], start: usize) -> Option<usize> {
    for i in start..data.len().saturating_sub(1) {
        if data[i] == 0xFF && data[i + 1] == 0x4F {
            return Some(i);
        }
    }
    None
}

fn find_eoc(data: &[u8], start: usize) -> Option<usize> {
    for i in start..data.len().saturating_sub(1) {
        if data[i] == 0xFF && data[i + 1] == 0xD9 {
            return Some(i);
        }
    }
    None
}
