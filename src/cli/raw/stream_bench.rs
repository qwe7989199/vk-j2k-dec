//! Identical sequential MXF source and RGB48LE sink for both decoder backends.
use super::*;
use std::collections::VecDeque;
use std::ffi::{CStr, c_char, c_void};
use std::io::{BufReader, Write};
use std::ptr;
use vk_j2k_dec::mxf::{MAX_FRAME_BYTES, MxfFrameReader};

pub fn benchmark_mxf_stream(
    input: &str,
    backend: &str,
    limit: usize,
    batch: usize,
    output: &str,
) -> ExitCode {
    match run(input, backend, limit, batch, output) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("bench-mxf-stream failed: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Single-threaded CPU cost of the per-frame stream prepare (parse, plan, align, sort) over
/// demuxed `.j2k` frames, without GPU or MXF I/O.
pub fn benchmark_prepare(dir: &str, limit: usize) -> ExitCode {
    let run = || -> Result<(), String> {
        let mut paths = fs::read_dir(dir)
            .map_err(|e| e.to_string())?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "j2k"))
            .collect::<Vec<_>>();
        paths.sort();
        paths.truncate(limit.max(1));
        let frames = paths
            .iter()
            .map(|path| fs::read(path).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let first = frames.first().ok_or("no .j2k frames")?;
        let mut layout = build_fast_tier1_rgb48_frame(first)?;
        let (template, scatter, _) = complete_template(first, &layout)?;
        align_tier1_records_to_template(&template, &scatter, &mut layout)?;
        let (mut parse_ms, mut align_ms, mut sort_ms) = (0.0, 0.0, 0.0);
        let mut checksum = 0u64;
        for frame in &frames {
            let start = Instant::now();
            let mut plan = build_fast_tier1_rgb48_frame(frame)?;
            parse_ms += elapsed_ms(start);
            let start = Instant::now();
            align_tier1_records_to_template(&template, &scatter, &mut plan)?;
            align_ms += elapsed_ms(start);
            let start = Instant::now();
            sort_codeblock_records_by_descending_work(&mut plan.records);
            sort_ms += elapsed_ms(start);
            checksum = plan.records.iter().fold(checksum, |sum, record| {
                let words: [u32; 16] = bytemuck_record(record);
                words.iter().fold(sum, |sum, word| {
                    (sum ^ u64::from(*word)).wrapping_mul(0x100000001b3)
                })
            });
            checksum = plan.compressed_words.iter().fold(checksum, |sum, word| {
                (sum ^ u64::from(*word)).wrapping_mul(0x100000001b3)
            });
        }
        let n = frames.len() as f64;
        println!(
            "bench-prepare: {} frames, ms/frame parse {:.3} align {:.3} sort {:.3} total {:.3}, checksum {checksum:016x}",
            frames.len(),
            parse_ms / n,
            align_ms / n,
            sort_ms / n,
            (parse_ms + align_ms + sort_ms) / n
        );
        Ok(())
    };
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("bench-prepare failed: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Every field of a record, in declaration order.
fn bytemuck_record(record: &GpuTier1CodeblockRecord) -> [u32; 16] {
    [
        record.component,
        record.resolution,
        record.subband,
        record.codeblock_x,
        record.codeblock_y,
        record.width,
        record.height,
        record.magnitude_bitplanes,
        record.zero_bit_planes,
        record.byte_offset,
        record.byte_len,
        record.coefficient_offset,
        record.coefficient_count,
        record.pass_offset,
        record.pass_count,
        record.reconstruction_mode,
    ]
}

struct Sink {
    file: fs::File,
    frames: usize,
    bytes: u64,
    hash: Option<u64>,
}

impl Sink {
    fn consume(&mut self, samples: &[u16]) -> Result<(), String> {
        if let Some(hash) = &mut self.hash {
            for &sample in samples {
                *hash = (*hash ^ u64::from(sample)).wrapping_mul(0x100000001b3);
            }
        }
        // Both backends supply MSB-aligned, packed RGB48 in host memory (Windows LE).
        let bytes =
            unsafe { std::slice::from_raw_parts(samples.as_ptr().cast::<u8>(), samples.len() * 2) };
        self.file.write_all(bytes).map_err(|e| e.to_string())?;
        self.frames += 1;
        self.bytes += bytes.len() as u64;
        Ok(())
    }
}

fn run(
    input: &str,
    backend: &str,
    limit: usize,
    requested_batch: usize,
    output: &str,
) -> Result<(), String> {
    if !matches!(backend, "vulkan" | "nvj2k") || !(1..=8).contains(&requested_batch) {
        return Err("backend must be vulkan or nvj2k; batch must be 1..8".into());
    }
    let start = Instant::now();
    let file = fs::File::open(input).map_err(|e| e.to_string())?;
    let mut source = MxfFrameReader::new(BufReader::with_capacity(1024 * 1024, file));
    let mut sink = Sink {
        file: fs::File::create(output).map_err(|e| e.to_string())?,
        frames: 0,
        bytes: 0,
        hash: std::env::var_os("VK_J2K_HASH_FRAMES").map(|_| 0xcbf29ce484222325),
    };
    let first = source
        .next_frame()
        .map_err(|e| e.to_string())?
        .ok_or("MXF has no picture frames")?;
    let stream = Codestream::parse(&first).map_err(|e| e.to_string())?;
    let siz = stream.siz.as_ref().ok_or("missing SIZ")?;
    let precision = siz.components.first().ok_or("missing component")?.precision;
    let samples = siz.width as usize * siz.height as usize * 3;
    let mut decoder = match backend {
        "vulkan" => Backend::Vulkan(VulkanDecoder::new(&first, requested_batch)?),
        _ => Backend::Nv(NvDecoder::new(
            siz.width,
            siz.height,
            precision,
            requested_batch,
            samples,
        )?),
    };
    let batch_size = decoder.batch_size();
    let setup_ms = elapsed_ms(start);
    println!("bench-mxf-stream: backend {backend}, input {input}");
    println!("  device {}", decoder.device_name());
    println!(
        "  format {}x{} {}-bit -> RGB48LE, sink {output}, raw_host_readback enabled",
        siz.width, siz.height, precision
    );
    println!("  demux MXF KLV sequential, batch {batch_size}, slots 2, cpu_codeblocks_per_frame 0");
    println!(
        "  frame_limit {limit} (0=EOF), hash {}",
        if sink.hash.is_some() {
            "enabled"
        } else {
            "disabled"
        }
    );
    let loop_start = Instant::now();
    let mut initial = Some(first);
    let mut pending = VecDeque::new();
    let mut batch_index = 0;
    let mut submitted = 0;
    let mut read_ms = 0.0;
    let mut prepare_submit_ms = 0.0;
    let mut wait_write_ms = 0.0;
    loop {
        let read_start = Instant::now();
        let mut frames = Vec::with_capacity(batch_size);
        while frames.len() < batch_size && (limit == 0 || submitted + frames.len() < limit) {
            let next = match initial.take() {
                Some(frame) => Some(frame),
                None => source.next_frame().map_err(|e| e.to_string())?,
            };
            match next {
                Some(frame) => frames.push(frame),
                None => break,
            }
        }
        read_ms += elapsed_ms(read_start);
        if frames.is_empty() {
            break;
        }
        let slot = batch_index % 2;
        if pending.len() == 2 {
            let (old_slot, count) = pending.pop_front().unwrap();
            let wait_start = Instant::now();
            decoder.consume(old_slot, count, &mut sink)?;
            wait_write_ms += elapsed_ms(wait_start);
        }
        let submit_start = Instant::now();
        decoder.submit(slot, &frames)?;
        prepare_submit_ms += elapsed_ms(submit_start);
        submitted += frames.len();
        pending.push_back((slot, frames.len()));
        batch_index += 1;
    }
    for (slot, count) in pending {
        let wait_start = Instant::now();
        decoder.consume(slot, count, &mut sink)?;
        wait_write_ms += elapsed_ms(wait_start);
    }
    sink.file.flush().map_err(|e| e.to_string())?;
    if submitted != sink.frames || (limit != 0 && submitted != limit) {
        return Err(format!(
            "frame count mismatch: requested {limit}, submitted {submitted}, output {}",
            sink.frames
        ));
    }
    let loop_ms = elapsed_ms(loop_start);
    if let Backend::Vulkan(v) = &decoder {
        let frames = v.gpu_times.iter().map(|(count, _)| count).sum::<usize>();
        if frames > 0 {
            let per_frame = |value: fn(&Tier1GpuStageTimes) -> f64| {
                v.gpu_times.iter().map(|(_, times)| value(times)).sum::<f64>() / frames as f64
            };
            println!(
                "  gpu_stage_ms_per_frame total {:.3} upload {:.3} tier1 {:.3} scatter {:.3} idwt {:.3} color {:.3} readback_copy {:.3}",
                per_frame(Tier1GpuStageTimes::total_ms),
                per_frame(|t| t.upload_ms),
                per_frame(|t| t.tier1_ms),
                per_frame(|t| t.scatter_ms),
                per_frame(|t| t.idwt_ms),
                per_frame(|t| t.color_ms),
                per_frame(|t| t.readback_copy_ms),
            );
        }
    }
    drop(decoder); // Include device teardown in the principal end-to-end timing.
    let total_ms = elapsed_ms(start);
    println!(
        "  frames {} output_bytes {} essence_bytes {}",
        sink.frames, sink.bytes, source.essence_bytes
    );
    println!(
        "  setup_ms {setup_ms:.3} read_ms {read_ms:.3} prepare_submit_ms {prepare_submit_ms:.3} wait_write_ms {wait_write_ms:.3}"
    );
    println!(
        "  stream_loop_ms {loop_ms:.3} fps {:.3}",
        sink.frames as f64 * 1000.0 / loop_ms
    );
    println!(
        "  end_to_end_ms {total_ms:.3} fps {:.3}",
        sink.frames as f64 * 1000.0 / total_ms
    );
    if let Some(hash) = sink.hash {
        println!("  frames_hash {hash:016x}");
    }
    Ok(())
}

enum Backend {
    Vulkan(VulkanDecoder),
    Nv(NvDecoder),
}
impl Backend {
    fn batch_size(&self) -> usize {
        match self {
            Self::Vulkan(v) => v.pipeline.max_batch_frames(),
            Self::Nv(n) => n.batch,
        }
    }
    fn device_name(&self) -> String {
        match self {
            Self::Vulkan(v) => v.pipeline.device_name().into(),
            Self::Nv(n) => n.device.clone(),
        }
    }
    fn submit(&mut self, slot: usize, frames: &[Vec<u8>]) -> Result<(), String> {
        match self {
            Self::Vulkan(v) => v.submit(slot, frames),
            Self::Nv(n) => {
                for (i, frame) in frames.iter().enumerate() {
                    if unsafe {
                        (n.submit)(
                            n.session,
                            (slot * n.batch + i) as u32,
                            frame.as_ptr(),
                            frame.len(),
                        )
                    } != 0
                    {
                        return Err(n.error());
                    }
                }
                Ok(())
            }
        }
    }
    fn consume(&mut self, slot: usize, count: usize, sink: &mut Sink) -> Result<(), String> {
        match self {
            Self::Vulkan(v) => {
                let mut failure = None;
                v.pipeline
                    .wait_and_read_batch(slot, |_, samples| {
                        if failure.is_none() {
                            failure = sink.consume(samples).err();
                        }
                    })
                    .map_err(|e| e.to_string())?;
                if let Some(times) = v.pipeline.last_gpu_stage_times(slot) {
                    v.gpu_times.push((count, times));
                }
                failure.map_or(Ok(()), Err)
            }
            Self::Nv(n) => {
                for i in 0..count {
                    let mut samples = ptr::null();
                    if unsafe { (n.wait)(n.session, (slot * n.batch + i) as u32, &mut samples) }
                        != 0
                    {
                        return Err(n.error());
                    }
                    sink.consume(unsafe { std::slice::from_raw_parts(samples, n.samples) })?;
                }
                Ok(())
            }
        }
    }
}

struct VulkanDecoder {
    pipeline: VulkanTier1Rgb48Pipeline,
    /// (frames, stage times) per read batch, with VK_J2K_GPU_TIMING=1.
    gpu_times: Vec<(usize, Tier1GpuStageTimes)>,
    template: Vec<GpuTier1CodeblockRecord>,
    scatter: Vec<GpuTier1ScatterRecord>,
    layout: FastTier1Rgb48Frame,
    first: Option<FastTier1Rgb48Frame>,
}

/// The stream template comes from geometry, not the first frame's nonzero blocks.
/// A block absent in frame 1 may carry data later; no movie-wide pre-scan is needed.
fn complete_template(
    data: &[u8],
    frame: &FastTier1Rgb48Frame,
) -> Result<
    (
        Vec<GpuTier1CodeblockRecord>,
        Vec<GpuTier1ScatterRecord>,
        usize,
    ),
    String,
> {
    let stream = Codestream::parse(data).map_err(|e| e.to_string())?;
    let layouts = stream.tile_layouts().map_err(|e| e.to_string())?;
    let tile = layouts.first().ok_or("missing tile layout")?;
    let mut records = Vec::new();
    let mut scatter = Vec::new();
    let mut offset = 0u32;
    for plane in &frame.scatter_plan.planes {
        let resolution = tile.components[plane.component as usize]
            .resolutions
            .iter()
            .find(|r| r.level == plane.resolution)
            .ok_or("missing resolution")?;
        let band = resolution
            .subbands
            .iter()
            .find(|b| b.kind == plane.subband)
            .ok_or("missing band")?;
        for y in 0..band.codeblocks_y {
            for x in 0..band.codeblocks_x {
                let rect = band
                    .codeblock_rect(x, y)
                    .ok_or("missing codeblock rectangle")?;
                let count = rect
                    .width()
                    .checked_mul(rect.height())
                    .ok_or("codeblock size overflow")?;
                records.push(GpuTier1CodeblockRecord {
                    component: plane.component as u32,
                    resolution: plane.resolution as u32,
                    subband: match plane.subband {
                        SubbandKind::Ll => 0,
                        SubbandKind::Hl => 1,
                        SubbandKind::Lh => 2,
                        SubbandKind::Hh => 3,
                    },
                    codeblock_x: x,
                    codeblock_y: y,
                    width: rect.width(),
                    height: rect.height(),
                    coefficient_offset: offset,
                    coefficient_count: count,
                    reconstruction_mode: if frame.transform == WaveletTransform::Irreversible97 {
                        2
                    } else {
                        0
                    },
                    ..Default::default()
                });
                scatter.push(GpuTier1ScatterRecord {
                    coefficient_offset: offset,
                    coefficient_count: count,
                    width: rect.width(),
                    height: rect.height(),
                    plane_offset: plane.plane_offset,
                    plane_width: plane.width,
                    plane_x0: rect.x0,
                    plane_y0: rect.y0,
                });
                offset = offset
                    .checked_add(count)
                    .ok_or("coefficient offset overflow")?;
            }
        }
    }
    if offset as usize != frame.scatter_plan.output_coefficients {
        return Err("incomplete coefficient grid".into());
    }
    Ok((records, scatter, offset as usize))
}

/// Parses one frame (or takes the already parsed first frame), checks it against the stream
/// layout and aligns its records to the template, ready for `submit_batch`.
fn prepare_stream_frame(
    frame: &[u8],
    parsed: Option<FastTier1Rgb48Frame>,
    layout: &FastTier1Rgb48Frame,
    template: &[GpuTier1CodeblockRecord],
    scatter: &[GpuTier1ScatterRecord],
) -> Result<FastTier1Rgb48Frame, String> {
    let mut plan = match parsed {
        Some(plan) => plan,
        None => build_fast_tier1_rgb48_frame(frame)?,
    };
    if plan.scatter_plan.planes != layout.scatter_plan.planes
        || plan.target_resolution != layout.target_resolution
        || plan.color_transform != layout.color_transform
        || plan.precision != layout.precision
        || plan.transform != layout.transform
        || plan.qcd != layout.qcd
        || plan.qccs != layout.qccs
    {
        return Err("stream frame layout/quantization changed".into());
    }
    align_tier1_records_to_template(template, scatter, &mut plan)?;
    sort_codeblock_records_by_descending_work(&mut plan.records);
    Ok(plan)
}

impl VulkanDecoder {
    fn new(data: &[u8], batch: usize) -> Result<Self, String> {
        let mut layout = build_fast_tier1_rgb48_frame(data)?;
        let (template, scatter, flat_count) = complete_template(data, &layout)?;
        align_tier1_records_to_template(&template, &scatter, &mut layout)?;
        let is_97 = layout.transform == WaveletTransform::Irreversible97;
        let scales = if is_97 {
            Some(
                vk_j2k_dec::gpu::build_scatter_dequant_scales(
                    &layout.scatter_plan,
                    layout.qcd.as_ref().ok_or("9/7 missing QCD")?,
                    &layout.qccs,
                    layout.precision,
                    layout.target_resolution,
                )
                .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let forced_shader = std::env::var("VK_J2K_TIER1_SHADER").ok();
        let pipeline = VulkanTier1Rgb48Pipeline::new(&VulkanTier1Rgb48PipelineConfig {
            // VK_J2K_TIER1_SHADER forces a Tier-1 shader (A/B testing).
            cleanup_shader_path: forced_shader
                .as_deref()
                .unwrap_or_else(|| packed_tier1_shader(&template)),
            thread_per_codeblock: false,
            scatter_shader_path: if is_97 {
                r"target\shaders\tier1_scatter_dequant.comp.spv"
            } else {
                r"target\shaders\tier1_scatter_smoke.comp.spv"
            },
            idwt53_shader_path: if is_97 {
                r"target\shaders\idwt97_offset_1d.comp.spv"
            } else {
                r"target\shaders\idwt53_offset_1d.comp.spv"
            },
            // VK_J2K_IDWT=separable keeps the three-pass-per-component IDWT (A/B testing).
            idwt53_fused_shader_path: if std::env::var("VK_J2K_IDWT").is_ok_and(|mode| mode == "separable") {
                None
            } else if is_97 {
                Some(r"target\shaders\idwt97_fused_2d.comp.spv")
            } else {
                Some(r"target\shaders\idwt53_fused_2d.comp.spv")
            },
            color_shader_path: if is_97 {
                r"target\shaders\color_convert_f32_packed.comp.spv"
            } else {
                r"target\shaders\color_convert_i32.comp.spv"
            },
            codeblock_count: template.len(),
            compressed_word_capacity: MAX_FRAME_BYTES / 4,
            flat_coefficient_count: flat_count,
            scatter_records: &scatter,
            scatter_planes: &layout.scatter_plan.planes,
            output_coefficients: layout.scatter_plan.output_coefficients,
            target_resolution: layout.target_resolution,
            color_transform: layout.color_transform,
            precision: layout.precision,
            max_batch_frames: batch,
            cpu_coefficient_capacity: 0,
            // VK_J2K_GPU_TIMING=1 records per-stage GPU timestamps.
            gpu_timestamps: std::env::var_os("VK_J2K_GPU_TIMING").is_some(),
            // VK_J2K_TRANSFER_QUEUE=0 keeps the readback copy on the compute queue (A/B testing).
            transfer_queue: !std::env::var("VK_J2K_TRANSFER_QUEUE").is_ok_and(|mode| mode == "0"),
            export_rgb48: false,
            scatter_dequant_scales: scales.as_deref(),
        })
        .map_err(|e| e.to_string())?;
        // Keep only the metadata/template after first-frame submission, not any movie-wide plan.
        let first = FastTier1Rgb48Frame {
            records: std::mem::take(&mut layout.records),
            compressed_words: std::mem::take(&mut layout.compressed_words),
            flat_coefficient_count: layout.flat_coefficient_count,
            scatter_plan: layout.scatter_plan.clone(),
            target_resolution: layout.target_resolution,
            color_transform: layout.color_transform,
            precision: layout.precision,
            codeblock_style: layout.codeblock_style,
            transform: layout.transform,
            qcd: layout.qcd.clone(),
            qccs: layout.qccs.clone(),
        };
        Ok(Self {
            pipeline,
            gpu_times: Vec::new(),
            template,
            scatter,
            layout,
            first: Some(first),
        })
    }

    fn submit(&mut self, slot: usize, frames: &[Vec<u8>]) -> Result<(), String> {
        // Frames are independent: parse and plan each one on its own thread.
        let (layout, template, scatter) = (&self.layout, &self.template, &self.scatter);
        let mut first = self.first.take();
        let plans = std::thread::scope(|scope| {
            let handles = frames
                .iter()
                .map(|frame| {
                    let parsed = first.take();
                    scope.spawn(move || prepare_stream_frame(frame, parsed, layout, template, scatter))
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| "frame prepare thread panicked".to_string())?)
                .collect::<Result<Vec<_>, String>>()
        })?;
        let batch = plans
            .iter()
            .map(|p| Tier1BatchFrame {
                gpu_records: &p.records,
                compressed_words: &p.compressed_words,
                cpu_codeblocks: &[],
            })
            .collect::<Vec<_>>();
        self.pipeline
            .submit_batch(slot, &batch)
            .map_err(|e| e.to_string())
    }
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
}
struct Module(*mut c_void);
impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            FreeLibrary(self.0);
        }
    }
}
type Create = unsafe extern "C" fn(u32, u32, u32, u32, i32) -> *mut c_void;
type Submit = unsafe extern "C" fn(*mut c_void, u32, *const u8, usize) -> i32;
type Wait = unsafe extern "C" fn(*mut c_void, u32, *mut *const u16) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Error = unsafe extern "C" fn() -> *const c_char;
type Device = unsafe extern "C" fn(*mut c_void) -> *const c_char;
struct NvDecoder {
    session: *mut c_void,
    _module: Module,
    submit: Submit,
    wait: Wait,
    destroy: Destroy,
    error_fn: Error,
    batch: usize,
    samples: usize,
    device: String,
}
impl Drop for NvDecoder {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.session);
        }
    }
}
impl NvDecoder {
    fn error(&self) -> String {
        unsafe { CStr::from_ptr((self.error_fn)()) }
            .to_string_lossy()
            .into_owned()
    }
    fn new(
        width: u32,
        height: u32,
        precision: u8,
        batch: usize,
        samples: usize,
    ) -> Result<Self, String> {
        let path = fs::canonicalize("target/nvj2k/nvjpeg2k_bridge.dll")
            .map_err(|e| format!("run scripts/build-nvj2k-bridge.ps1 first: {e}"))?;
        let wide = path
            .to_string_lossy()
            .encode_utf16()
            .chain(Some(0))
            .collect::<Vec<_>>();
        let module =
            Module(unsafe { LoadLibraryExW(wide.as_ptr(), ptr::null_mut(), 0x100 | 0x1000) });
        if module.0.is_null() {
            return Err(format!(
                "load nvJPEG2000 bridge: {}",
                std::io::Error::last_os_error()
            ));
        }
        let symbol = |name: &CStr| -> Result<*mut c_void, String> {
            let address = unsafe { GetProcAddress(module.0, name.as_ptr()) };
            if address.is_null() {
                Err(format!("missing symbol {name:?}"))
            } else {
                Ok(address)
            }
        };
        let create = unsafe { std::mem::transmute::<*mut c_void, Create>(symbol(c"nvj_create")?) };
        let submit = unsafe { std::mem::transmute::<*mut c_void, Submit>(symbol(c"nvj_submit")?) };
        let wait = unsafe { std::mem::transmute::<*mut c_void, Wait>(symbol(c"nvj_wait")?) };
        let destroy =
            unsafe { std::mem::transmute::<*mut c_void, Destroy>(symbol(c"nvj_destroy")?) };
        let error_fn = unsafe { std::mem::transmute::<*mut c_void, Error>(symbol(c"nvj_error")?) };
        let device_fn =
            unsafe { std::mem::transmute::<*mut c_void, Device>(symbol(c"nvj_device")?) };
        let device_index = std::env::var("VK_J2K_CUDA_DEVICE")
            .unwrap_or_else(|_| "0".into())
            .parse::<i32>()
            .map_err(|e| e.to_string())?;
        let session = unsafe {
            create(
                width,
                height,
                u32::from(precision),
                (batch * 2) as u32,
                device_index,
            )
        };
        if session.is_null() {
            return Err(unsafe { CStr::from_ptr(error_fn()) }
                .to_string_lossy()
                .into_owned());
        }
        let device = unsafe { CStr::from_ptr(device_fn(session)) }
            .to_string_lossy()
            .into_owned();
        Ok(Self {
            session,
            _module: module,
            submit,
            wait,
            destroy,
            error_fn,
            batch,
            samples,
            device,
        })
    }
}
