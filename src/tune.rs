//! Runtime auto-tuning for the batched GPU Tier-1 pipeline.
//!
//! Neither knob has a value that is right for every machine: the best batch size depends on
//! how many SMs the GPU has to fill, and how much Tier-1 work the CPU can absorb depends on its
//! core count and on how fast the GPU is. Both are measured while decoding instead of guessed.

/// Frames run after switching batch size before measuring: the in-flight batch of the previous
/// size drains and the CPU offload re-settles.
const WARMUP_FRAMES: usize = 16;
/// Warm-up for the first candidate, while the CPU offload converges from its start value.
const INITIAL_WARMUP_FRAMES: usize = 48;
/// Frames averaged per candidate batch size.
const MEASURE_FRAMES: usize = 24;
/// Warm-up and measurement each span at least this many batches.
const MIN_BATCHES: usize = 2;
/// A batch counts as "as fast as the best" within this fraction; the smallest such batch wins.
const BATCH_TOLERANCE: f64 = 0.03;

/// Picks the frames-per-submit that gives the lowest steady-state time per frame.
///
/// Candidates are powers of two up to the pipeline's memory-derived maximum, probed largest
/// first: the largest batch is the most likely winner, and the CPU offload converges while it
/// runs. Probing stops once a smaller batch is clearly slower than the best one.
#[derive(Debug, Clone)]
pub struct BatchSizeTuner {
    candidates: Vec<usize>,
    probe_index: usize,
    warmup_frames: usize,
    warmup_batches: usize,
    probe_ms: f64,
    probe_frames: usize,
    probe_batches: usize,
    results: Vec<(usize, f64)>,
    chosen: Option<usize>,
}

impl BatchSizeTuner {
    pub fn new(max_batch_frames: usize) -> Self {
        let max_batch_frames = max_batch_frames.max(1);
        let mut candidates = Vec::new();
        let mut batch = 1;
        while batch <= max_batch_frames {
            candidates.push(batch);
            batch *= 2;
        }
        if candidates.last() != Some(&max_batch_frames) {
            candidates.push(max_batch_frames);
        }
        candidates.reverse();
        let chosen = (candidates.len() == 1).then_some(candidates[0]);
        Self::with_candidates(candidates, chosen)
    }

    /// Skips tuning and always uses `batch_frames`.
    pub fn fixed(batch_frames: usize) -> Self {
        let batch_frames = batch_frames.max(1);
        Self::with_candidates(vec![batch_frames], Some(batch_frames))
    }

    fn with_candidates(candidates: Vec<usize>, chosen: Option<usize>) -> Self {
        Self {
            candidates,
            probe_index: 0,
            warmup_frames: 0,
            warmup_batches: 0,
            probe_ms: 0.0,
            probe_frames: 0,
            probe_batches: 0,
            results: Vec::new(),
            chosen,
        }
    }

    /// Batch size to use for the next submit.
    pub fn current(&self) -> usize {
        self.chosen.unwrap_or(self.candidates[self.probe_index])
    }

    pub fn chosen(&self) -> Option<usize> {
        self.chosen
    }

    /// Measured `(batch_frames, ms_per_frame)` for every candidate probed so far.
    pub fn results(&self) -> &[(usize, f64)] {
        &self.results
    }

    /// Records one steady-state loop iteration that completed `frames` frames in `wall_ms`.
    pub fn record(&mut self, frames: usize, wall_ms: f64) {
        if self.chosen.is_some() || frames == 0 {
            return;
        }
        let warmup_target = if self.probe_index == 0 {
            INITIAL_WARMUP_FRAMES
        } else {
            WARMUP_FRAMES
        };
        if self.warmup_frames < warmup_target || self.warmup_batches < MIN_BATCHES {
            self.warmup_frames += frames;
            self.warmup_batches += 1;
            return;
        }
        self.probe_ms += wall_ms;
        self.probe_frames += frames;
        self.probe_batches += 1;
        if self.probe_frames < MEASURE_FRAMES || self.probe_batches < MIN_BATCHES {
            return;
        }

        let batch = self.candidates[self.probe_index];
        let ms_per_frame = self.probe_ms / self.probe_frames as f64;
        let best_so_far = self.best();
        self.results.push((batch, ms_per_frame));
        self.probe_index += 1;
        self.warmup_frames = 0;
        self.warmup_batches = 0;
        self.probe_ms = 0.0;
        self.probe_frames = 0;
        self.probe_batches = 0;

        // Smaller batches only get slower from here once one falls clearly behind.
        let fell_behind = best_so_far
            .map(|(_, best_ms)| ms_per_frame > best_ms * (1.0 + BATCH_TOLERANCE))
            .unwrap_or(false);
        if fell_behind || self.probe_index == self.candidates.len() {
            self.chosen = Some(self.pick());
        }
    }

    fn best(&self) -> Option<(usize, f64)> {
        self.results
            .iter()
            .copied()
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Smallest batch within `BATCH_TOLERANCE` of the fastest one: same throughput, less latency.
    fn pick(&self) -> usize {
        let Some((_, best_ms)) = self.best() else {
            return 1;
        };
        self.results
            .iter()
            .filter(|(_, ms)| *ms <= best_ms * (1.0 + BATCH_TOLERANCE))
            .map(|(batch, _)| *batch)
            .min()
            .unwrap_or(1)
    }
}

/// Below this share of an iteration spent waiting on the GPU, the host is the bottleneck.
const CPU_BOUND_WAIT_FRACTION: f64 = 0.02;
/// Above this share, the GPU is the bottleneck and the CPU can take more codeblocks.
const GPU_BOUND_WAIT_FRACTION: f64 = 0.15;
/// Wait share the controller steers towards when growing.
const TARGET_WAIT_FRACTION: f64 = 0.08;
/// Fraction of the estimated slack moved to the CPU per update; damps overshoot.
const GROW_GAIN: f64 = 0.5;
/// CPU share per frame when offloading starts.
const START_PER_FRAME: usize = 4;
/// Below this many hardware threads, offloading starts disabled and only begins when the
/// pipeline proves to be GPU-bound.
const MIN_OFFLOAD_THREADS: usize = 4;

/// Decides how many of each frame's heaviest codeblocks the host decodes on the CPU.
///
/// The CPU decode of batch `k` overlaps the GPU running batch `k-1`, so it is free as long as
/// the host still ends up waiting on the GPU. The controller keeps that wait small but nonzero:
/// it grows the CPU share in proportion to the measured slack and the measured CPU cost per
/// codeblock, and backs off as soon as the host stops waiting. A share that once made the
/// pipeline CPU-bound becomes a ceiling, so the controller does not oscillate around it.
#[derive(Debug, Clone)]
pub struct CpuOffloadController {
    per_frame: usize,
    ceiling: usize,
    adaptive: bool,
}

impl CpuOffloadController {
    pub fn new(max_per_frame: usize, cpu_threads: usize) -> Self {
        let start = if cpu_threads >= MIN_OFFLOAD_THREADS { START_PER_FRAME } else { 0 };
        Self {
            per_frame: start.min(max_per_frame),
            ceiling: max_per_frame,
            adaptive: true,
        }
    }

    /// Always offloads exactly `per_frame` codeblocks per frame.
    pub fn fixed(per_frame: usize) -> Self {
        Self {
            per_frame,
            ceiling: per_frame,
            adaptive: false,
        }
    }

    pub fn per_frame(&self) -> usize {
        self.per_frame
    }

    /// `cpu_ms`: CPU decode time for the batch just submitted (`frames` frames); `wait_ms`: time
    /// blocked on the GPU fence afterwards; `wall_ms`: the whole loop iteration.
    pub fn update(&mut self, cpu_ms: f64, wait_ms: f64, wall_ms: f64, frames: usize) {
        if !self.adaptive || wall_ms <= 0.0 || frames == 0 {
            return;
        }
        if wait_ms < wall_ms * CPU_BOUND_WAIT_FRACTION {
            if self.per_frame > 0 {
                self.ceiling = self.per_frame - 1;
                self.per_frame -= (self.per_frame / 4).max(1);
            }
        } else if wait_ms > wall_ms * GPU_BOUND_WAIT_FRACTION && self.per_frame < self.ceiling {
            if self.per_frame == 0 {
                self.per_frame = START_PER_FRAME.min(self.ceiling);
                return;
            }
            // Heaviest codeblocks go first, so the next ones are cheaper than this average and
            // the step undershoots rather than overshoots.
            let cost_ms = cpu_ms / (self.per_frame * frames) as f64;
            let slack_ms = (wait_ms - wall_ms * TARGET_WAIT_FRACTION) * GROW_GAIN;
            let extra = if cost_ms > 0.0 {
                (slack_ms / cost_ms / frames as f64) as usize
            } else {
                self.per_frame
            };
            self.per_frame = (self.per_frame + extra.clamp(1, self.per_frame)).min(self.ceiling);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds iterations at `ms_per_frame` until the current candidate's probe completes.
    fn run_probe(tuner: &mut BatchSizeTuner, ms_per_frame: f64) {
        let probed = tuner.results().len();
        while tuner.results().len() == probed && tuner.chosen().is_none() {
            let batch = tuner.current();
            tuner.record(batch, ms_per_frame * batch as f64);
        }
    }

    #[test]
    fn batch_candidates_run_largest_first() {
        assert_eq!(BatchSizeTuner::new(4).candidates, vec![4, 2, 1]);
        assert_eq!(BatchSizeTuner::new(6).candidates, vec![6, 4, 2, 1]);
        assert_eq!(BatchSizeTuner::new(1).chosen(), Some(1));
        assert_eq!(BatchSizeTuner::new(0).chosen(), Some(1));
    }

    #[test]
    fn batch_tuner_keeps_smallest_equally_fast_batch() {
        let mut tuner = BatchSizeTuner::new(8);
        run_probe(&mut tuner, 20.0);
        assert_eq!(tuner.current(), 4);
        run_probe(&mut tuner, 20.4);
        assert_eq!(tuner.current(), 2);
        // Batch 2 falls clearly behind: stop probing and keep batch 4.
        run_probe(&mut tuner, 22.0);
        assert_eq!(tuner.chosen(), Some(4));
        assert_eq!(tuner.results().len(), 3);
    }

    #[test]
    fn batch_tuner_ignores_warmup() {
        let mut tuner = BatchSizeTuner::new(2);
        for _ in 0..INITIAL_WARMUP_FRAMES / 2 {
            tuner.record(2, 1000.0);
        }
        run_probe(&mut tuner, 20.0);
        assert_eq!(tuner.results()[0], (2, 20.0));
    }

    #[test]
    fn fixed_batch_tuner_never_probes() {
        let mut tuner = BatchSizeTuner::fixed(3);
        tuner.record(3, 10.0);
        assert_eq!(tuner.current(), 3);
        assert!(tuner.results().is_empty());
    }

    #[test]
    fn offload_grows_in_proportion_to_gpu_slack() {
        let mut offload = CpuOffloadController::new(200, 8);
        assert_eq!(offload.per_frame(), START_PER_FRAME);
        // 4 codeblocks x 8 frames cost 3.2 ms (0.1 ms each); 10 ms of a 25 ms iteration is
        // spent waiting. Slack (10 - 2) * 0.5 = 4 ms -> 40 codeblocks -> 5 per frame, capped
        // at doubling.
        offload.update(3.2, 10.0, 25.0, 8);
        assert_eq!(offload.per_frame(), 8);
        // Same cost per codeblock, smaller slack: (5 - 2) * 0.5 / 0.1 / 8 = 1.875 -> +1.
        offload.update(6.4, 5.0, 25.0, 8);
        assert_eq!(offload.per_frame(), 9);
        // Inside the target band: hold.
        offload.update(7.2, 3.0, 25.0, 8);
        assert_eq!(offload.per_frame(), 9);
    }

    #[test]
    fn offload_backs_off_when_cpu_bound_and_remembers_the_ceiling() {
        let mut offload = CpuOffloadController::new(200, 8);
        for _ in 0..20 {
            offload.update(1.0, 20.0, 25.0, 1);
        }
        let grown = offload.per_frame();
        assert!(grown > 100, "grew to {grown}");
        // The host no longer waits on the GPU: back off a quarter.
        offload.update(24.0, 0.0, 25.0, 1);
        assert_eq!(offload.per_frame(), grown - grown / 4);
        // GPU-bound again, but never back up to the share that was CPU-bound.
        for _ in 0..20 {
            offload.update(1.0, 20.0, 25.0, 1);
        }
        assert_eq!(offload.per_frame(), grown - 1);
    }

    #[test]
    fn offload_starts_disabled_on_small_cpus_until_gpu_bound() {
        let mut offload = CpuOffloadController::new(200, 2);
        assert_eq!(offload.per_frame(), 0);
        offload.update(0.0, 1.0, 25.0, 1);
        assert_eq!(offload.per_frame(), 0);
        offload.update(0.0, 10.0, 25.0, 1);
        assert_eq!(offload.per_frame(), START_PER_FRAME);
    }

    #[test]
    fn fixed_offload_never_moves() {
        let mut fixed = CpuOffloadController::fixed(5);
        fixed.update(0.0, 0.0, 25.0, 1);
        fixed.update(1.0, 20.0, 25.0, 1);
        assert_eq!(fixed.per_frame(), 5);
    }
}
