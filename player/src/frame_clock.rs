use std::time::{Duration, Instant};

pub struct FrameClock {
    start: Instant,
    frame_interval: Duration,
    frame_index: u64,
}

impl FrameClock {
    pub fn new(fps: f64) -> Self {
        Self {
            start: Instant::now(),
            frame_interval: Duration::from_secs_f64(1.0 / fps),
            frame_index: 0,
        }
    }

    pub fn next_present_time(&self) -> Instant {
        self.start + self.frame_interval * (self.frame_index + 1) as u32
    }

    pub fn advance(&mut self) {
        self.frame_index += 1;
    }

    pub fn wait_for_next_frame(&mut self) {
        let target = self.next_present_time();
        let now = Instant::now();
        if target > now {
            std::thread::sleep(target - now);
        }
        self.advance();
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }
}
