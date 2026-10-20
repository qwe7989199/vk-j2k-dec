use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::ExitCode;
use std::time::Instant;

use vk_j2k_dec::codestream::{CodeblockStyle, Codestream, SubbandKind, WaveletTransform};
use vk_j2k_dec::decode::{
    DecodeProfile, decode_coefficient_planes, decode_reversible_component_planes_with_discard,
    decode_rgb48_with_discard, decode_rgb48_with_discard_profiled, decode_rgb48_with_gpu_dwt,
    decode_rgb48_with_gpu_dwt_profiled, decode_rgb48_with_gpu_pipeline,
    decode_rgb48_with_gpu_pipeline_profiled, decode_rgb48_with_persistent_gpu_pipeline_profiled,
};
use vk_j2k_dec::gpu::{
    GpuTier1CodeblockRecord, GpuTier1Plan, GpuTier1ScatterPlan, GpuTier1ScatterRecord,
    cpu_decodable, decode_codeblock_record_cpu, sort_codeblock_records_by_descending_work,
    thread_per_codeblock_eligible,
};
use vk_j2k_dec::packet::parse_lrcp_tile_packets;
use vk_j2k_dec::reconstruct::{ComponentSubbandPlanes, ResolutionSubbandPlane};
use vk_j2k_dec::tier1::{
    CoefficientReconstruction, DecodedCodeblock, MqDecoder, Tier1CodingMode, Tier1WorkPlan,
};
use vk_j2k_dec::tune::{BatchSizeTuner, CpuOffloadController};
use vk_j2k_dec::vulkan::{
    TIER1_PIPELINE_MAX_BATCH_FRAMES, Tier1BatchFrame, Tier1CpuCodeblock, Tier1GpuStageTimes,
    VulkanRgb48Pipeline,
    VulkanTier1Rgb48Pipeline, VulkanTier1Rgb48PipelineConfig,
};

pub fn decode_rgb48_file(input: &str, output: &str) -> ExitCode {
    decode_rgb48_file_with_discard(input, 0, output)
}

pub fn decode_rgb48_file_with_discard(input: &str, discard_levels: u8, output: &str) -> ExitCode {
    let data = match fs::read(input) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {input}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let image = match decode_rgb48_with_discard(&data, discard_levels) {
        Ok(image) => image,
        Err(err) => {
            eprintln!("decode-rgb48 failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    let bytes = image.to_le_bytes();
    if let Err(err) = fs::write(output, bytes) {
        eprintln!("failed to write {output}: {err}");
        return ExitCode::FAILURE;
    }
    println!(
        "decode-rgb48: {}x{} {}-bit, discard {}, {} samples -> {output}",
        image.width,
        image.height,
        image.precision,
        discard_levels,
        image.samples.len()
    );
    ExitCode::SUCCESS
}

pub fn decode_rgb48_gpu_dwt_file(
    input: &str,
    discard_levels: u8,
    output: &str,
    idwt53_shader_path: &str,
) -> ExitCode {
    let data = match fs::read(input) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {input}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let image = match decode_rgb48_with_gpu_dwt(&data, discard_levels, idwt53_shader_path) {
        Ok(image) => image,
        Err(err) => {
            eprintln!("decode-rgb48-gpu-dwt failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    let bytes = image.to_le_bytes();
    if let Err(err) = fs::write(output, bytes) {
        eprintln!("failed to write {output}: {err}");
        return ExitCode::FAILURE;
    }
    println!(
        "decode-rgb48-gpu-dwt: {}x{} {}-bit, discard {}, {} samples -> {output}",
        image.width,
        image.height,
        image.precision,
        discard_levels,
        image.samples.len()
    );
    ExitCode::SUCCESS
}

pub fn decode_rgb48_gpu_file(
    input: &str,
    discard_levels: u8,
    output: &str,
    idwt53_shader_path: &str,
    color_shader_path: &str,
) -> ExitCode {
    let data = match fs::read(input) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {input}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let image = match decode_rgb48_with_gpu_pipeline(
        &data,
        discard_levels,
        idwt53_shader_path,
        color_shader_path,
    ) {
        Ok(image) => image,
        Err(err) => {
            eprintln!("decode-rgb48-gpu failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    let bytes = image.to_le_bytes();
    if let Err(err) = fs::write(output, bytes) {
        eprintln!("failed to write {output}: {err}");
        return ExitCode::FAILURE;
    }
    println!(
        "decode-rgb48-gpu: {}x{} {}-bit, discard {}, {} samples -> {output}",
        image.width,
        image.height,
        image.precision,
        discard_levels,
        image.samples.len()
    );
    ExitCode::SUCCESS
}

pub fn profile_rgb48_file(
    input: &str,
    discard_levels: u8,
    mode: &str,
    output: &str,
    idwt53_shader_path: &str,
    color_shader_path: &str,
) -> ExitCode {
    let total_start = Instant::now();
    let read_start = Instant::now();
    let data = match fs::read(input) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {input}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let read_ms = elapsed_ms(read_start);

    let decoded = match mode {
        "cpu" => decode_rgb48_with_discard_profiled(&data, discard_levels),
        "gpu-dwt" => decode_rgb48_with_gpu_dwt_profiled(&data, discard_levels, idwt53_shader_path),
        "gpu" => decode_rgb48_with_gpu_pipeline_profiled(
            &data,
            discard_levels,
            idwt53_shader_path,
            color_shader_path,
        ),
        other => {
            eprintln!("profile-rgb48 failed: mode must be cpu, gpu-dwt, or gpu, got {other}");
            return ExitCode::FAILURE;
        }
    };
    let decoded = match decoded {
        Ok(decoded) => decoded,
        Err(err) => {
            eprintln!("profile-rgb48 failed: {err}");
            return ExitCode::FAILURE;
        }
    };

    let write_start = Instant::now();
    let bytes = decoded.image.to_le_bytes();
    if let Err(err) = fs::write(output, bytes) {
        eprintln!("failed to write {output}: {err}");
        return ExitCode::FAILURE;
    }
    let write_ms = elapsed_ms(write_start);
    let total_ms = elapsed_ms(total_start);

    println!(
        "profile-rgb48: mode {mode}, {}x{} {}-bit, discard {}, {} samples -> {output}",
        decoded.image.width,
        decoded.image.height,
        decoded.image.precision,
        discard_levels,
        decoded.image.samples.len()
    );
    println!("  read_ms {:.3}", read_ms);
    print_decode_profile(&decoded.profile);
    println!("  write_ms {:.3}", write_ms);
    println!("  cli_total_ms {:.3}", total_ms);
    ExitCode::SUCCESS
}

pub fn profile_rgb48_bench_file(
    input: &str,
    discard_levels: u8,
    mode: &str,
    iterations: usize,
    output: &str,
    idwt53_shader_path: &str,
    color_shader_path: &str,
) -> ExitCode {
    if iterations == 0 {
        eprintln!("profile-rgb48-bench failed: iterations must be greater than 0");
        return ExitCode::FAILURE;
    }

    let read_start = Instant::now();
    let data = match fs::read(input) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {input}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let read_ms = elapsed_ms(read_start);

    let pipeline = if mode == "gpu" {
        match VulkanRgb48Pipeline::new(idwt53_shader_path, color_shader_path) {
            Ok(pipeline) => Some(pipeline),
            Err(err) => {
                eprintln!("profile-rgb48-bench failed: GPU pipeline setup failed: {err}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };
    let pipeline_setup_ms = pipeline
        .as_ref()
        .map(VulkanRgb48Pipeline::setup_ms)
        .unwrap_or(0.0);

    let mut profiles = Vec::with_capacity(iterations);
    let mut last_image = None;
    let bench_start = Instant::now();
    for iteration in 0..iterations {
        let decoded = match (mode, pipeline.as_ref()) {
            ("cpu", _) => decode_rgb48_with_discard_profiled(&data, discard_levels),
            ("gpu-dwt", _) => {
                decode_rgb48_with_gpu_dwt_profiled(&data, discard_levels, idwt53_shader_path)
            }
            ("gpu", Some(pipeline)) => {
                decode_rgb48_with_persistent_gpu_pipeline_profiled(&data, discard_levels, pipeline)
            }
            ("gpu", None) => unreachable!("GPU pipeline setup checked above"),
            (other, _) => {
                eprintln!(
                    "profile-rgb48-bench failed: mode must be cpu, gpu-dwt, or gpu, got {other}"
                );
                return ExitCode::FAILURE;
            }
        };
        let decoded = match decoded {
            Ok(decoded) => decoded,
            Err(err) => {
                eprintln!(
                    "profile-rgb48-bench failed on iteration {}: {err}",
                    iteration + 1
                );
                return ExitCode::FAILURE;
            }
        };
        println!(
            "  iter {:>2}: decode_total_ms {:.3} tier1_ms {:.3} gpu_setup_ms {:.3} reconstruct_ms {:.3} reconstruct_upload_ms {:.3} reconstruct_alloc_ms {:.3} reconstruct_dispatch_ms {:.3} pack_ms {:.3} pack_shader_ms {:.3} pack_copy_ms {:.3} pack_readback_ms {:.3}",
            iteration + 1,
            decoded.profile.timings.total_ms,
            decoded.profile.timings.tier1_decode_ms,
            decoded.profile.timings.gpu_setup_ms,
            decoded.profile.timings.reconstruct_ms,
            decoded.profile.timings.reconstruct_upload_ms,
            decoded.profile.timings.reconstruct_alloc_ms,
            decoded.profile.timings.reconstruct_dispatch_ms,
            decoded.profile.timings.pack_ms,
            decoded.profile.timings.pack_shader_ms,
            decoded.profile.timings.pack_copy_ms,
            decoded.profile.timings.pack_readback_ms
        );
        profiles.push(decoded.profile);
        last_image = Some(decoded.image);
    }
    let bench_ms = elapsed_ms(bench_start);

    let Some(image) = last_image else {
        eprintln!("profile-rgb48-bench failed: no decode iterations completed");
        return ExitCode::FAILURE;
    };
    let write_start = Instant::now();
    let bytes = image.to_le_bytes();
    if let Err(err) = fs::write(output, bytes) {
        eprintln!("failed to write {output}: {err}");
        return ExitCode::FAILURE;
    }
    let write_ms = elapsed_ms(write_start);

    println!(
        "profile-rgb48-bench: mode {mode}, {}x{} {}-bit, discard {}, iterations {}, {} samples -> {output}",
        image.width,
        image.height,
        image.precision,
        discard_levels,
        iterations,
        image.samples.len()
    );
    println!("  read_once_ms {:.3}", read_ms);
    println!("  pipeline_setup_ms {:.3}", pipeline_setup_ms);
    println!("  bench_loop_ms {:.3}", bench_ms);
    println!("  write_last_ms {:.3}", write_ms);
    print_profile_average("avg_all", &profiles);
    if profiles.len() > 1 {
        print_profile_average("avg_warm_skip1", &profiles[1..]);
    }
    ExitCode::SUCCESS
}

pub fn profile_mxf_ffmpeg_file(
    input: &str,
    frame_count: usize,
    discard_levels: u8,
    mode: &str,
    output: &str,
    ffmpeg_path: &str,
    work_dir: &str,
    idwt53_shader_path: &str,
    color_shader_path: &str,
) -> ExitCode {
    if frame_count == 0 {
        eprintln!("profile-mxf-ffmpeg failed: frames must be greater than 0");
        return ExitCode::FAILURE;
    }
    if !matches!(mode, "cpu" | "gpu-dwt" | "gpu" | "gpu-tier1") {
        eprintln!(
            "profile-mxf-ffmpeg failed: mode must be cpu, gpu-dwt, gpu, or gpu-tier1, got {mode}"
        );
        return ExitCode::FAILURE;
    }

    let work_dir = Path::new(work_dir);
    if let Err(err) = fs::create_dir_all(work_dir) {
        eprintln!(
            "profile-mxf-ffmpeg failed: cannot create {}: {err}",
            work_dir.display()
        );
        return ExitCode::FAILURE;
    }
    let pattern = work_dir.join("frame_%06d.j2k");
    let demux_start = Instant::now();
    let status = Command::new(ffmpeg_path)
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-i")
        .arg(input)
        .arg("-map")
        .arg("0:v:0")
        .arg("-frames:v")
        .arg(frame_count.to_string())
        .arg("-c:v")
        .arg("copy")
        .arg("-f")
        .arg("image2")
        .arg(&pattern)
        .status();
    let demux_ms = elapsed_ms(demux_start);
    match status {
        Ok(status) if status.success() => {}
        Ok(status) => {
            eprintln!("profile-mxf-ffmpeg failed: ffmpeg exited with {status}");
            return ExitCode::FAILURE;
        }
        Err(err) => {
            eprintln!("profile-mxf-ffmpeg failed: failed to run {ffmpeg_path}: {err}");
            return ExitCode::FAILURE;
        }
    }

    if mode == "gpu-tier1" {
        return profile_mxf_ffmpeg_gpu_tier1_file(
            input,
            frame_count,
            discard_levels,
            output,
            ffmpeg_path,
            work_dir,
            demux_ms,
            idwt53_shader_path,
            color_shader_path,
        );
    }

    let pipeline_setup_start = Instant::now();
    let pipeline = if mode == "gpu" {
        match VulkanRgb48Pipeline::new(idwt53_shader_path, color_shader_path) {
            Ok(pipeline) => Some(pipeline),
            Err(err) => {
                eprintln!("profile-mxf-ffmpeg failed: GPU pipeline setup failed: {err}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };
    let pipeline_setup_ms = pipeline
        .as_ref()
        .map(VulkanRgb48Pipeline::setup_ms)
        .unwrap_or_else(|| elapsed_ms(pipeline_setup_start));

    let mut profiles = Vec::with_capacity(frame_count);
    let mut read_total_ms = 0.0;
    let mut decode_loop_ms = 0.0;
    let mut last_image = None;
    let mut total_bytes = 0usize;
    for frame_index in 0..frame_count {
        let frame_path = frame_path(work_dir, frame_index + 1);
        let read_start = Instant::now();
        let data = match fs::read(&frame_path) {
            Ok(data) => data,
            Err(err) => {
                eprintln!(
                    "profile-mxf-ffmpeg failed: cannot read {}: {err}",
                    frame_path.display()
                );
                return ExitCode::FAILURE;
            }
        };
        read_total_ms += elapsed_ms(read_start);
        total_bytes += data.len();

        let decode_start = Instant::now();
        let decoded = match (mode, pipeline.as_ref()) {
            ("cpu", _) => decode_rgb48_with_discard_profiled(&data, discard_levels),
            ("gpu-dwt", _) => {
                decode_rgb48_with_gpu_dwt_profiled(&data, discard_levels, idwt53_shader_path)
            }
            ("gpu", Some(pipeline)) => {
                decode_rgb48_with_persistent_gpu_pipeline_profiled(&data, discard_levels, pipeline)
            }
            ("gpu", None) => unreachable!("GPU pipeline setup checked above"),
            _ => unreachable!("mode checked above"),
        };
        decode_loop_ms += elapsed_ms(decode_start);
        let decoded = match decoded {
            Ok(decoded) => decoded,
            Err(err) => {
                eprintln!(
                    "profile-mxf-ffmpeg failed on frame {} ({}): {err}",
                    frame_index + 1,
                    frame_path.display()
                );
                return ExitCode::FAILURE;
            }
        };
        println!(
            "  frame {:>4}: bytes {} decode_total_ms {:.3} tier1_ms {:.3} reconstruct_ms {:.3} pack_ms {:.3}",
            frame_index + 1,
            data.len(),
            decoded.profile.timings.total_ms,
            decoded.profile.timings.tier1_decode_ms,
            decoded.profile.timings.reconstruct_ms,
            decoded.profile.timings.pack_ms
        );
        profiles.push(decoded.profile);
        last_image = Some(decoded.image);
    }

    let Some(image) = last_image else {
        eprintln!("profile-mxf-ffmpeg failed: no frames decoded");
        return ExitCode::FAILURE;
    };
    let write_start = Instant::now();
    if let Err(err) = fs::write(output, image.to_le_bytes()) {
        eprintln!("failed to write {output}: {err}");
        return ExitCode::FAILURE;
    }
    let write_ms = elapsed_ms(write_start);

    println!(
        "profile-mxf-ffmpeg: mode {mode}, {}x{} {}-bit, discard {}, frames {}, demuxed {} bytes -> {output}",
        image.width, image.height, image.precision, discard_levels, frame_count, total_bytes
    );
    println!("  ffmpeg_path {ffmpeg_path}");
    println!("  work_dir {}", work_dir.display());
    println!("  ffmpeg_demux_ms {:.3}", demux_ms);
    println!("  read_frames_ms {:.3}", read_total_ms);
    println!("  pipeline_setup_ms {:.3}", pipeline_setup_ms);
    println!("  decode_loop_ms {:.3}", decode_loop_ms);
    println!("  write_last_ms {:.3}", write_ms);
    print_profile_average("avg_all_frames", &profiles);
    if profiles.len() > 1 {
        print_profile_average("avg_skip_first_frame", &profiles[1..]);
    }
    ExitCode::SUCCESS
}

fn frame_path(work_dir: &Path, one_based_index: usize) -> PathBuf {
    work_dir.join(format!("frame_{one_based_index:06}.j2k"))
}

struct FastTier1Rgb48Frame {
    records: Vec<GpuTier1CodeblockRecord>,
    compressed_words: Vec<u32>,
    flat_coefficient_count: usize,
    scatter_plan: GpuTier1ScatterPlan,
    target_resolution: u8,
    color_transform: u32,
    precision: u8,
    codeblock_style: CodeblockStyle,
}

fn profile_mxf_ffmpeg_gpu_tier1_file(
    input: &str,
    frame_count: usize,
    discard_levels: u8,
    output: &str,
    ffmpeg_path: &str,
    work_dir: &Path,
    demux_ms: f64,
    idwt53_shader_path: &str,
    color_shader_path: &str,
) -> ExitCode {
    if discard_levels != 0 {
        eprintln!("profile-mxf-ffmpeg gpu-tier1 currently supports discard 0 only");
        return ExitCode::FAILURE;
    }
    let scatter_shader_path = r"target\shaders\tier1_scatter_smoke.comp.spv";
    let mut read_total_ms = 0.0;
    let mut frame_plans = Vec::with_capacity(frame_count);
    let mut total_bytes = 0usize;
    for frame_index in 0..frame_count {
        let frame_path = frame_path(work_dir, frame_index + 1);
        let read_start = Instant::now();
        let data = match fs::read(&frame_path) {
            Ok(data) => data,
            Err(err) => {
                eprintln!(
                    "profile-mxf-ffmpeg gpu-tier1 failed: cannot read {}: {err}",
                    frame_path.display()
                );
                return ExitCode::FAILURE;
            }
        };
        read_total_ms += elapsed_ms(read_start);
        total_bytes += data.len();
        let frame_plan = match build_fast_tier1_rgb48_frame(&data) {
            Ok(frame_plan) => frame_plan,
            Err(err) => {
                eprintln!(
                    "profile-mxf-ffmpeg gpu-tier1 failed on frame {}: {err}",
                    frame_index + 1
                );
                return ExitCode::FAILURE;
            }
        };
        frame_plans.push(frame_plan);
    }
    if frame_plans.is_empty() {
        eprintln!("profile-mxf-ffmpeg gpu-tier1 failed: no frames decoded");
        return ExitCode::FAILURE;
    };
    let (template_records, template_scatter_records, flat_coefficient_count) =
        match build_tier1_union_template(&frame_plans) {
            Ok(template) => template,
            Err(err) => {
                eprintln!("profile-mxf-ffmpeg gpu-tier1 failed: {err}");
                return ExitCode::FAILURE;
            }
        };
    for (index, frame) in frame_plans.iter_mut().enumerate() {
        if let Err(err) =
            align_tier1_records_to_template(&template_records, &template_scatter_records, frame)
        {
            eprintln!(
                "profile-mxf-ffmpeg gpu-tier1 failed: frame {} cannot use frame 1 layout: {err}",
                index + 1
            );
            return ExitCode::FAILURE;
        }
        // Heaviest first: the CPU offload takes each frame's leading records.
        sort_codeblock_records_by_descending_work(&mut frame.records);
    }
    let first_frame = frame_plans
        .first()
        .expect("frame_plans was checked to be non-empty");
    let codeblock_count = first_frame.records.len();
    let scatter_plan = first_frame.scatter_plan.clone();
    let target_resolution = first_frame.target_resolution;
    let color_transform = first_frame.color_transform;
    let precision = first_frame.precision;
    let compressed_word_capacity = frame_plans
        .iter()
        .map(|frame| frame.compressed_words.len())
        .max()
        .unwrap_or(0);
    for (index, frame) in frame_plans.iter().enumerate() {
        if frame.records.len() != codeblock_count {
            eprintln!(
                "profile-mxf-ffmpeg gpu-tier1 failed: frame {} has {} records, expected {}",
                index + 1,
                frame.records.len(),
                codeblock_count
            );
            return ExitCode::FAILURE;
        }
        if frame.flat_coefficient_count != flat_coefficient_count
            || frame.scatter_plan.records != scatter_plan.records
            || frame.scatter_plan.planes != scatter_plan.planes
            || frame.scatter_plan.output_coefficients != scatter_plan.output_coefficients
            || frame.target_resolution != target_resolution
            || frame.color_transform != color_transform
            || frame.precision != precision
        {
            eprintln!(
                "profile-mxf-ffmpeg gpu-tier1 failed: frame {} layout differs from frame 1",
                index + 1
            );
            return ExitCode::FAILURE;
        }
    }

    // VK_J2K_TIER1_SHADER forces a warp-per-codeblock Tier-1 shader (A/B testing).
    let forced_shader = std::env::var("VK_J2K_TIER1_SHADER").ok();
    // The packed-state warp shader beats thread-per-codeblock even on 16x16 codeblocks; the
    // latter is only a fallback when the packed build is missing.
    let thread_per_codeblock = forced_shader.is_none()
        && !Path::new(TIER1_WARP_SHADERS[0]).exists()
        && thread_per_codeblock_eligible(&template_records)
        && first_existing(TIER1_THREAD_SHADERS).is_some();
    let cleanup_shader_path: &str = if let Some(path) = forced_shader.as_deref() {
        path
    } else if thread_per_codeblock {
        first_existing(TIER1_THREAD_SHADERS).unwrap_or(TIER1_WARP_SHADERS[TIER1_WARP_SHADERS.len() - 1])
    } else {
        first_existing(TIER1_WARP_SHADERS).unwrap_or(TIER1_WARP_SHADERS[TIER1_WARP_SHADERS.len() - 1])
    };

    // The CPU may take up to a quarter of a frame's codeblocks; size its upload region for the
    // largest codeblocks so any choice the controller makes fits.
    let max_cpu_per_frame = (codeblock_count / 4).min(TIER1_MAX_CPU_CODEBLOCKS_PER_FRAME);
    let mut coefficient_counts = template_records
        .iter()
        .map(|record| record.coefficient_count as usize)
        .collect::<Vec<_>>();
    coefficient_counts.sort_unstable_by(|a, b| b.cmp(a));
    let cpu_coefficient_capacity = coefficient_counts.iter().take(max_cpu_per_frame).sum();
    let fixed_batch = env_usize("VK_J2K_BATCH").filter(|batch| *batch > 0);
    let fixed_cpu_codeblocks = env_usize("VK_J2K_CPU_CBLKS").map(|k| k.min(max_cpu_per_frame));

    let mut pipeline = match VulkanTier1Rgb48Pipeline::new(&VulkanTier1Rgb48PipelineConfig {
        cleanup_shader_path,
        thread_per_codeblock,
        scatter_shader_path,
        idwt53_shader_path,
        // VK_J2K_IDWT=separable keeps the three-pass-per-component IDWT (A/B testing).
        idwt53_fused_shader_path: if std::env::var("VK_J2K_IDWT").is_ok_and(|mode| mode == "separable") {
            None
        } else {
            first_existing(&[r"target\shaders\idwt53_fused_2d.comp.spv"])
        },
        color_shader_path,
        codeblock_count,
        compressed_word_capacity,
        flat_coefficient_count,
        scatter_records: &scatter_plan.records,
        scatter_planes: &scatter_plan.planes,
        output_coefficients: scatter_plan.output_coefficients,
        target_resolution,
        color_transform,
        precision,
        max_batch_frames: fixed_batch
            .unwrap_or(TIER1_PIPELINE_MAX_BATCH_FRAMES)
            .min(frame_plans.len()),
        cpu_coefficient_capacity,
        // VK_J2K_GPU_TIMING=1 records per-stage GPU timestamps.
        gpu_timestamps: std::env::var_os("VK_J2K_GPU_TIMING").is_some(),
        // VK_J2K_TRANSFER_QUEUE=0 keeps the readback copy on the compute queue (A/B testing).
        transfer_queue: !std::env::var("VK_J2K_TRANSFER_QUEUE").is_ok_and(|mode| mode == "0"),
    }) {
        Ok(pipeline) => pipeline,
        Err(err) => {
            eprintln!("profile-mxf-ffmpeg gpu-tier1 failed: pipeline setup failed: {err}");
            return ExitCode::FAILURE;
        }
    };

    let cpu_threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let mut tuner = match fixed_batch {
        Some(batch) => BatchSizeTuner::fixed(batch.min(pipeline.max_batch_frames())),
        None => BatchSizeTuner::new(pipeline.max_batch_frames()),
    };
    let mut offload = match fixed_cpu_codeblocks {
        Some(per_frame) => CpuOffloadController::fixed(per_frame),
        None => CpuOffloadController::new(max_cpu_per_frame, cpu_threads),
    };
    let hash_frames = std::env::var_os("VK_J2K_HASH_FRAMES").is_some();

    let decode_loop_start = Instant::now();
    let outcome = match run_tier1_mxf_decode_loop(
        &mut pipeline,
        &frame_plans,
        &mut tuner,
        &mut offload,
        cpu_threads,
        hash_frames,
    ) {
        Ok(outcome) => outcome,
        Err(err) => {
            eprintln!("profile-mxf-ffmpeg gpu-tier1 failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    let decode_loop_ms = elapsed_ms(decode_loop_start);
    let samples = outcome.last_samples;
    if samples.is_empty() {
        eprintln!("profile-mxf-ffmpeg gpu-tier1 failed: no frames decoded");
        return ExitCode::FAILURE;
    }
    let write_start = Instant::now();
    if let Err(err) = fs::write(output, samples_to_le_bytes(&samples)) {
        eprintln!("failed to write {output}: {err}");
        return ExitCode::FAILURE;
    }
    let write_ms = elapsed_ms(write_start);
    let sample_count = samples.len() / 3;
    println!(
        "profile-mxf-ffmpeg: mode gpu-tier1, {} samples/frame, discard {}, frames {}, demuxed {} bytes -> {output}",
        samples.len(),
        discard_levels,
        frame_count,
        total_bytes
    );
    println!("  ffmpeg_path {ffmpeg_path}");
    println!("  input {input}");
    println!("  work_dir {}", work_dir.display());
    println!("  vulkan_device {}", pipeline.device_name());
    println!(
        "  device_local_heap_mib {} cpu_threads {}",
        pipeline.device_local_heap_bytes() / (1024 * 1024),
        cpu_threads
    );
    println!(
        "  tier1_shader {} ({})",
        cleanup_shader_path,
        if thread_per_codeblock { "thread-per-codeblock" } else { "warp-per-codeblock" }
    );
    println!("  idwt {}", if pipeline.fused_idwt() { "fused 2d" } else { "separable 1d" });
    println!(
        "  readback {}",
        if pipeline.transfer_readback() { "transfer queue" } else { "compute queue" }
    );
    println!("  ffmpeg_demux_ms {:.3}", demux_ms);
    println!("  read_frames_ms {:.3}", read_total_ms);
    println!("  pipeline_setup_ms {:.3}", pipeline.setup_ms());
    println!("  decode_loop_ms {:.3}", decode_loop_ms);
    println!("  write_last_ms {:.3}", write_ms);
    let probes = tuner
        .results()
        .iter()
        .map(|(batch, ms)| format!("{batch}:{ms:.2}ms"))
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "  autotune batch {} (max {}{}) probes [{}] cpu_codeblocks_per_frame {} (max {}{})",
        tuner.current(),
        pipeline.max_batch_frames(),
        if fixed_batch.is_some() { ", fixed" } else { "" },
        probes,
        offload.per_frame(),
        max_cpu_per_frame,
        if fixed_cpu_codeblocks.is_some() { ", fixed" } else { "" },
    );
    println!(
        "  loop_avg_ms_per_frame total {:.3} fps {:.2}",
        decode_loop_ms / frame_plans.len() as f64,
        frame_plans.len() as f64 * 1000.0 / decode_loop_ms
    );
    print_tier1_iteration_average("tuning", &outcome.iterations, |it| !it.steady);
    print_tier1_iteration_average("steady", &outcome.iterations, |it| it.steady);
    print_tier1_gpu_stage_average(&outcome.iterations);
    if let Some(hash) = outcome.frames_hash {
        println!("  frames_hash {hash:016x}");
    }
    println!("  output_pixels {}", sample_count);
    ExitCode::SUCCESS
}

/// Production Tier-1 shaders without diagnostic counters first, then the validation builds.
const TIER1_WARP_SHADERS: &[&str] = &[
    r"target\shaders\tier1_packed.comp.spv",
    r"target\shaders\tier1_cleanup_nostats.comp.spv",
    r"target\shaders\tier1_cleanup_smoke.comp.spv",
];
const TIER1_THREAD_SHADERS: &[&str] = &[
    r"target\shaders\tier1_thread_per_codeblock_nostats.comp.spv",
    r"target\shaders\tier1_cleanup_smoke_16.comp.spv",
];
const TIER1_MAX_CPU_CODEBLOCKS_PER_FRAME: usize = 256;

fn first_existing(paths: &[&'static str]) -> Option<&'static str> {
    paths.iter().copied().find(|path| Path::new(path).exists())
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

struct Tier1DecodeIteration {
    frames_read: usize,
    wall_ms: f64,
    cpu_ms: f64,
    wait_ms: f64,
    read_ms: f64,
    /// Batch size was settled (and offload adapting) when this iteration ran.
    steady: bool,
    /// GPU stage times of the batch read in this iteration, with VK_J2K_GPU_TIMING.
    gpu: Option<Tier1GpuStageTimes>,
}

struct Tier1DecodeLoopOutcome {
    iterations: Vec<Tier1DecodeIteration>,
    last_samples: Vec<u16>,
    frames_hash: Option<u64>,
}

/// Double-buffered batch loop. Iteration `k` decodes batch `k`'s heaviest codeblocks on the CPU
/// while the GPU runs batch `k-1`, submits batch `k`, then reads batch `k-1` back.
fn run_tier1_mxf_decode_loop(
    pipeline: &mut VulkanTier1Rgb48Pipeline,
    frame_plans: &[FastTier1Rgb48Frame],
    tuner: &mut BatchSizeTuner,
    offload: &mut CpuOffloadController,
    cpu_threads: usize,
    hash_frames: bool,
) -> Result<Tier1DecodeLoopOutcome, String> {
    let frame_total = frame_plans.len();
    let mut in_flight: [Option<(usize, usize)>; 2] = [None, None];
    let mut iterations = Vec::new();
    let mut last_samples = Vec::new();
    let mut frames_hash = 0xcbf2_9ce4_8422_2325u64;
    let mut next_frame = 0;
    let mut slot = 0;
    while next_frame < frame_total || in_flight.iter().any(Option::is_some) {
        let iteration_start = Instant::now();
        let mut cpu_ms = 0.0;
        let mut frames_submitted = 0;
        let submitted = next_frame < frame_total;
        if submitted {
            let batch = tuner
                .current()
                .min(pipeline.max_batch_frames())
                .min(frame_total - next_frame);
            let batch_plans = &frame_plans[next_frame..next_frame + batch];
            let cpu_start = Instant::now();
            let cpu_codeblocks =
                decode_heaviest_codeblocks_cpu(batch_plans, offload.per_frame(), cpu_threads)?;
            cpu_ms = elapsed_ms(cpu_start);
            let batch_frames = batch_plans
                .iter()
                .zip(&cpu_codeblocks)
                .map(|(plan, cpu)| Tier1BatchFrame {
                    gpu_records: &plan.records[cpu.len()..],
                    compressed_words: &plan.compressed_words,
                    cpu_codeblocks: cpu,
                })
                .collect::<Vec<_>>();
            pipeline
                .submit_batch(slot, &batch_frames)
                .map_err(|err| format!("frame {}: {err}", next_frame + 1))?;
            in_flight[slot] = Some((next_frame, batch));
            next_frame += batch;
            frames_submitted = batch;
        }

        // Read the older batch; once nothing is left to submit, drain this slot too.
        let read_slot = if in_flight[1 - slot].is_some() {
            Some(1 - slot)
        } else if !submitted {
            Some(slot)
        } else {
            None
        };
        let mut wait_ms = 0.0;
        let mut read_ms = 0.0;
        let mut frames_read = 0;
        let mut gpu = None;
        if let Some((read_slot, (first_frame, count))) =
            read_slot.and_then(|read_slot| in_flight[read_slot].take().map(|batch| (read_slot, batch)))
        {
            let read_start = Instant::now();
            wait_ms = pipeline
                .wait_and_read_batch(read_slot, |index, samples| {
                    if hash_frames {
                        frames_hash = samples.iter().fold(frames_hash, |hash, sample| {
                            (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
                        });
                    }
                    if first_frame + index + 1 == frame_total {
                        last_samples = samples.to_vec();
                    }
                })
                .map_err(|err| format!("reading frame {}: {err}", first_frame + 1))?;
            read_ms = elapsed_ms(read_start) - wait_ms;
            frames_read = count;
            gpu = pipeline.last_gpu_stage_times(read_slot);
        }
        slot = 1 - slot;

        // Only iterations that both decoded on the CPU and waited on the GPU are steady state.
        if submitted && frames_read > 0 {
            let wall_ms = elapsed_ms(iteration_start);
            let steady = tuner.chosen().is_some();
            iterations.push(Tier1DecodeIteration {
                frames_read,
                wall_ms,
                cpu_ms,
                wait_ms,
                read_ms,
                steady,
                gpu,
            });
            tuner.record(frames_read, wall_ms);
            offload.update(cpu_ms, wait_ms, wall_ms, frames_submitted);
        }
    }
    Ok(Tier1DecodeLoopOutcome {
        iterations,
        last_samples,
        frames_hash: hash_frames.then_some(frames_hash),
    })
}

/// Decodes the `per_frame` heaviest codeblocks of every frame on a scoped thread pool. Records
/// are sorted heaviest first, so these are each frame's leading records.
fn decode_heaviest_codeblocks_cpu(
    frames: &[FastTier1Rgb48Frame],
    per_frame: usize,
    threads: usize,
) -> Result<Vec<Vec<Tier1CpuCodeblock>>, String> {
    // The GPU decodes `records[cpu_count..]`, so the CPU share must be a prefix; it stops at the
    // first record the CPU decoder cannot reproduce bit-exactly.
    let cpu_counts = frames
        .iter()
        .map(|frame| {
            frame
                .records
                .iter()
                .take(per_frame)
                .take_while(|record| cpu_decodable(record))
                .count()
        })
        .collect::<Vec<_>>();
    // Rank-major: every frame's heaviest codeblock is handed out before anyone's second.
    let jobs = (0..per_frame)
        .flat_map(|rank| {
            let cpu_counts = &cpu_counts;
            (0..frames.len())
                .filter(move |frame| rank < cpu_counts[*frame])
                .map(move |frame| (frame, rank))
        })
        .collect::<Vec<_>>();
    let mut decoded = cpu_counts
        .iter()
        .map(|count| {
            (0..*count)
                .map(|_| None)
                .collect::<Vec<Option<Tier1CpuCodeblock>>>()
        })
        .collect::<Vec<_>>();
    if jobs.is_empty() {
        return Ok(decoded.into_iter().map(|_| Vec::new()).collect());
    }
    let next_job = std::sync::atomic::AtomicUsize::new(0);
    let worker_results = std::thread::scope(|scope| {
        let handles = (0..threads.clamp(1, jobs.len()))
            .map(|_| {
                scope.spawn(|| {
                    let mut results = Vec::new();
                    loop {
                        let job = next_job.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(frame_index, rank)) = jobs.get(job) else {
                            break;
                        };
                        let frame = &frames[frame_index];
                        let record = &frame.records[rank];
                        let coefficients = decode_codeblock_record_cpu(
                            record,
                            &frame.compressed_words,
                            frame.codeblock_style,
                        )
                        .map_err(|err| {
                            format!(
                                "C{} R{} S{} {},{} {}x{} mbp {} zbp {} passes {} bytes {}: {err}",
                                record.component,
                                record.resolution,
                                record.subband,
                                record.codeblock_x,
                                record.codeblock_y,
                                record.width,
                                record.height,
                                record.magnitude_bitplanes,
                                record.zero_bit_planes,
                                record.pass_count,
                                record.byte_len
                            )
                        });
                        results.push((frame_index, rank, record.coefficient_offset, coefficients));
                    }
                    results
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join())
            .collect::<Vec<_>>()
    });
    for results in worker_results {
        let results = results.map_err(|_| "CPU Tier-1 worker thread panicked".to_string())?;
        for (frame_index, rank, coefficient_offset, coefficients) in results {
            let coefficients = coefficients.map_err(|err| {
                format!("CPU Tier-1 decode of frame-local codeblock {rank} failed: {err}")
            })?;
            decoded[frame_index][rank] = Some(Tier1CpuCodeblock {
                coefficient_offset,
                coefficients,
            });
        }
    }
    decoded
        .into_iter()
        .map(|frame| {
            frame
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| "CPU Tier-1 decode skipped a codeblock".to_string())
        })
        .collect()
}

/// Per-frame GPU stage times over steady iterations (all timed iterations if none are steady).
fn print_tier1_gpu_stage_average(iterations: &[Tier1DecodeIteration]) {
    let timed = |steady_only: bool| {
        iterations
            .iter()
            .filter(|it| it.gpu.is_some() && (!steady_only || it.steady))
            .collect::<Vec<_>>()
    };
    let mut selected = timed(true);
    if selected.is_empty() {
        selected = timed(false);
    }
    let frames = selected.iter().map(|it| it.frames_read).sum::<usize>();
    if frames == 0 {
        return;
    }
    let per_frame = |value: fn(&Tier1GpuStageTimes) -> f64| {
        selected
            .iter()
            .filter_map(|it| it.gpu.as_ref())
            .map(value)
            .sum::<f64>()
            / frames as f64
    };
    println!(
        "  gpu_stage_ms_per_frame total {:.3} upload {:.3} tier1 {:.3} scatter {:.3} idwt {:.3} color {:.3} readback_copy {:.3} (batches {}, frames {})",
        per_frame(Tier1GpuStageTimes::total_ms),
        per_frame(|t| t.upload_ms),
        per_frame(|t| t.tier1_ms),
        per_frame(|t| t.scatter_ms),
        per_frame(|t| t.idwt_ms),
        per_frame(|t| t.color_ms),
        per_frame(|t| t.readback_copy_ms),
        selected.len(),
        frames
    );
}

fn print_tier1_iteration_average(
    label: &str,
    iterations: &[Tier1DecodeIteration],
    include: impl Fn(&Tier1DecodeIteration) -> bool,
) {
    let selected = iterations.iter().filter(|it| include(it)).collect::<Vec<_>>();
    let frames = selected.iter().map(|it| it.frames_read).sum::<usize>();
    if frames == 0 {
        return;
    }
    let per_frame = |value: fn(&Tier1DecodeIteration) -> f64| {
        selected.iter().map(|it| value(it)).sum::<f64>() / frames as f64
    };
    let total = per_frame(|it| it.wall_ms);
    println!(
        "  {label}_ms_per_frame total {:.3} fps {:.2} cpu_tier1 {:.3} gpu_wait {:.3} readback {:.3} (iterations {}, frames {})",
        total,
        1000.0 / total,
        per_frame(|it| it.cpu_ms),
        per_frame(|it| it.wait_ms),
        per_frame(|it| it.read_ms),
        selected.len(),
        frames
    );
}

fn build_fast_tier1_rgb48_frame(data: &[u8]) -> Result<FastTier1Rgb48Frame, String> {
    let codestream = Codestream::parse(data).map_err(|err| err.to_string())?;
    let siz = codestream
        .siz
        .as_ref()
        .ok_or_else(|| "missing SIZ marker".to_string())?;
    let cod = codestream
        .cod
        .as_ref()
        .ok_or_else(|| "missing COD marker".to_string())?;
    if cod.transform != WaveletTransform::Reversible53 {
        return Err(format!(
            "gpu-tier1 currently requires reversible 5/3, got {}",
            cod.transform
        ));
    }
    if codestream.tile_parts.len() != 1 {
        return Err(format!(
            "gpu-tier1 currently supports one tile-part, found {}",
            codestream.tile_parts.len()
        ));
    }
    let layouts = codestream.tile_layouts().map_err(|err| err.to_string())?;
    let tile_part = &codestream.tile_parts[0];
    let layout = layouts
        .iter()
        .find(|layout| layout.tile_index == tile_part.tile_index)
        .ok_or_else(|| format!("missing layout for tile {}", tile_part.tile_index))?;
    let tile_data = data
        .get(tile_part.data_start..tile_part.data_end)
        .ok_or_else(|| "tile payload outside codestream".to_string())?;
    let packet_summary =
        parse_lrcp_tile_packets(tile_data, layout, cod.layers).map_err(|err| err.to_string())?;
    let work_plan = Tier1WorkPlan::from_packet_summary_with_quantization(
        &packet_summary,
        codestream.qcd.as_ref(),
        &codestream.qccs,
    )
    .map_err(|err| err.to_string())?;
    let bound = work_plan
        .bind_segment_data(tile_data)
        .map_err(|err| err.to_string())?;
    let plan = GpuTier1Plan::from_bound_work_plan(&bound, cod.codeblock_style_flags())
        .map_err(|err| err.to_string())?;
    let scatter_plan = plan
        .scatter_plan(&layout.components)
        .map_err(|err| err.to_string())?;
    let precision = siz
        .components
        .first()
        .map(|component| component.precision)
        .unwrap_or(0);
    Ok(FastTier1Rgb48Frame {
        records: plan.codeblock_records(),
        compressed_words: plan.compressed_u32_words(),
        flat_coefficient_count: plan.output_coefficients,
        scatter_plan,
        target_resolution: cod.decomposition_levels,
        color_transform: u32::from(cod.multiple_component_transform),
        precision,
        codeblock_style: cod.codeblock_style_flags(),
    })
}

fn align_tier1_records_to_template(
    template: &[GpuTier1CodeblockRecord],
    template_scatter_records: &[vk_j2k_dec::gpu::GpuTier1ScatterRecord],
    frame: &mut FastTier1Rgb48Frame,
) -> Result<(), String> {
    let mut by_key = BTreeMap::new();
    for record in frame.records.iter().copied() {
        by_key.insert(tier1_record_key(&record), record);
    }
    let mut aligned = Vec::with_capacity(template.len());
    for template_record in template {
        let mut record = match by_key.remove(&tier1_record_key(template_record)) {
            Some(record) => record,
            None => {
                let mut zero = *template_record;
                zero.magnitude_bitplanes = 0;
                zero.zero_bit_planes = 0;
                zero.byte_offset = 0;
                zero.byte_len = 0;
                zero.pass_offset = 0;
                zero.pass_count = 0;
                zero
            }
        };
        if record.width != template_record.width || record.height != template_record.height {
            return Err(format!(
                "codeblock C{} R{} S{} {},{} changed from {}x{} to {}x{}",
                template_record.component,
                template_record.resolution,
                template_record.subband,
                template_record.codeblock_x,
                template_record.codeblock_y,
                template_record.width,
                template_record.height,
                record.width,
                record.height
            ));
        }
        record.coefficient_offset = template_record.coefficient_offset;
        record.coefficient_count = template_record.coefficient_count;
        aligned.push(record);
    }
    if let Some(record) = by_key.values().next() {
        return Err(format!(
            "frame contains codeblock not present in frame 1 template: C{} R{} S{} {},{}",
            record.component,
            record.resolution,
            record.subband,
            record.codeblock_x,
            record.codeblock_y
        ));
    }
    frame.records = aligned;
    frame.scatter_plan.records = template_scatter_records.to_vec();
    frame.flat_coefficient_count = template
        .iter()
        .map(|record| record.coefficient_offset + record.coefficient_count)
        .max()
        .unwrap_or(0) as usize;
    Ok(())
}

fn build_tier1_union_template(
    frames: &[FastTier1Rgb48Frame],
) -> Result<
    (
        Vec<GpuTier1CodeblockRecord>,
        Vec<GpuTier1ScatterRecord>,
        usize,
    ),
    String,
> {
    let mut by_key = BTreeMap::<
        (u32, u32, u32, u32, u32),
        (GpuTier1CodeblockRecord, GpuTier1ScatterRecord),
    >::new();
    for (frame_index, frame) in frames.iter().enumerate() {
        if frame.records.len() != frame.scatter_plan.records.len() {
            return Err(format!(
                "frame {} has {} codeblock records but {} scatter records",
                frame_index + 1,
                frame.records.len(),
                frame.scatter_plan.records.len()
            ));
        }
        for (record, scatter) in frame.records.iter().zip(&frame.scatter_plan.records) {
            let key = tier1_record_key(record);
            if let Some((existing, _)) = by_key.get(&key) {
                if existing.width != record.width
                    || existing.height != record.height
                    || existing.coefficient_count != record.coefficient_count
                {
                    return Err(format!(
                        "codeblock C{} R{} S{} {},{} geometry changed across frames",
                        record.component,
                        record.resolution,
                        record.subband,
                        record.codeblock_x,
                        record.codeblock_y
                    ));
                }
            } else {
                by_key.insert(key, (*record, *scatter));
            }
        }
    }

    let mut coefficient_offset = 0u32;
    let mut template_records = Vec::with_capacity(by_key.len());
    let mut template_scatter = Vec::with_capacity(by_key.len());
    for (_, (mut record, mut scatter)) in by_key {
        record.byte_offset = 0;
        record.byte_len = 0;
        record.pass_offset = 0;
        record.pass_count = 0;
        record.coefficient_offset = coefficient_offset;
        scatter.coefficient_offset = coefficient_offset;
        template_records.push(record);
        template_scatter.push(scatter);
        coefficient_offset = coefficient_offset
            .checked_add(record.coefficient_count)
            .ok_or_else(|| "GPU Tier-1 union coefficient offset overflow".to_string())?;
    }
    Ok((
        template_records,
        template_scatter,
        coefficient_offset as usize,
    ))
}

fn tier1_record_key(record: &GpuTier1CodeblockRecord) -> (u32, u32, u32, u32, u32) {
    (
        record.component,
        record.resolution,
        record.subband,
        record.codeblock_x,
        record.codeblock_y,
    )
}

fn samples_to_le_bytes(samples: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

pub fn diff_rgb48_files(actual: &str, reference: &str) -> ExitCode {
    let actual_bytes = match fs::read(actual) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {actual}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let reference_bytes = match fs::read(reference) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {reference}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let stats = match diff_rgb48(&actual_bytes, &reference_bytes) {
        Ok(stats) => stats,
        Err(message) => {
            eprintln!("diff-rgb48 failed: {message}");
            return ExitCode::FAILURE;
        }
    };

    println!(
        "diff-rgb48: samples {}, differing {}, max_abs {} at sample {} (x {}, y {}, channel {}), mean_abs {:.12}",
        stats.samples,
        stats.differing,
        stats.max_abs,
        stats.max_index,
        stats.max_x,
        stats.max_y,
        channel_name(stats.max_channel),
        stats.mean_abs
    );
    println!(
        "  signed_delta min {} max {}",
        stats.min_delta, stats.max_delta
    );
    println!(
        "  channel_diff R={} G={} B={}",
        stats.channel_differing[0], stats.channel_differing[1], stats.channel_differing[2]
    );
    println!(
        "  channel_mean R={:.12} G={:.12} B={:.12}",
        stats.channel_mean_abs[0], stats.channel_mean_abs[1], stats.channel_mean_abs[2]
    );
    ExitCode::SUCCESS
}

pub fn diff_component_plane_files(actual: &str, reference: &str) -> ExitCode {
    let actual_bytes = match fs::read(actual) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {actual}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let reference_bytes = match fs::read(reference) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {reference}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let actual_samples = match read_i32le_samples(&actual_bytes) {
        Ok(samples) => samples,
        Err(message) => {
            eprintln!("diff-component-plane failed: {message}");
            return ExitCode::FAILURE;
        }
    };
    let reference_samples = match read_i32le_samples(&reference_bytes) {
        Ok(samples) => samples,
        Err(message) => {
            eprintln!("diff-component-plane failed: {message}");
            return ExitCode::FAILURE;
        }
    };
    let stats = match diff_i32_samples(&actual_samples, &reference_samples) {
        Ok(stats) => stats,
        Err(message) => {
            eprintln!("diff-component-plane failed: {message}");
            return ExitCode::FAILURE;
        }
    };

    print_i32_diff_stats("diff-component-plane", &stats);
    ExitCode::SUCCESS
}

pub fn diff_component_plane_rgb48_file(
    actual: &str,
    reference: &str,
    component: usize,
    precision: u8,
) -> ExitCode {
    let actual_bytes = match fs::read(actual) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {actual}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let reference_bytes = match fs::read(reference) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {reference}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let actual_samples = match read_i32le_samples(&actual_bytes) {
        Ok(samples) => samples,
        Err(message) => {
            eprintln!("diff-component-rgb48 failed: {message}");
            return ExitCode::FAILURE;
        }
    };
    let stats = match diff_i32_samples_against_rgb48(
        &actual_samples,
        &reference_bytes,
        component,
        precision,
    ) {
        Ok(stats) => stats,
        Err(message) => {
            eprintln!("diff-component-rgb48 failed: {message}");
            return ExitCode::FAILURE;
        }
    };

    print_i32_diff_stats("diff-component-rgb48", &stats.diff);
    println!(
        "  clipped_or_saturated {}, comparable {}, lowbit_misaligned {}",
        stats.clipped_or_saturated, stats.comparable, stats.lowbit_misaligned
    );
    ExitCode::SUCCESS
}

pub fn dump_coeff_plane_file(
    input: &str,
    component: u16,
    resolution: u8,
    subband: &str,
    output: &str,
) -> ExitCode {
    let subband = match parse_subband(subband) {
        Some(subband) => subband,
        None => {
            eprintln!("dump-coeff-plane failed: subband must be LL, HL, LH, or HH");
            return ExitCode::FAILURE;
        }
    };
    let data = match fs::read(input) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {input}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let planes = match decode_coefficient_planes(&data) {
        Ok(planes) => planes,
        Err(err) => {
            eprintln!("dump-coeff-plane failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    let Some(component_planes) = planes
        .components
        .iter()
        .find(|planes| planes.component == component)
    else {
        eprintln!("dump-coeff-plane failed: missing component {component}");
        return ExitCode::FAILURE;
    };
    let Some(plane) = component_planes.plane(resolution, subband) else {
        eprintln!("dump-coeff-plane failed: missing component {component} R{resolution} {subband}");
        return ExitCode::FAILURE;
    };

    let mut bytes = Vec::with_capacity(plane.coefficients.len() * 4);
    for coefficient in &plane.coefficients {
        bytes.extend_from_slice(&coefficient.to_le_bytes());
    }
    if let Err(err) = fs::write(output, bytes) {
        eprintln!("failed to write {output}: {err}");
        return ExitCode::FAILURE;
    }
    let nonzero = plane
        .coefficients
        .iter()
        .filter(|coefficient| **coefficient != 0)
        .count();
    let min = plane.coefficients.iter().min().copied().unwrap_or(0);
    let max = plane.coefficients.iter().max().copied().unwrap_or(0);
    println!(
        "dump-coeff-plane: component {component} R{resolution} {subband} {}x{}, {} samples, {} nonzero, min {}, max {} -> {output}",
        plane.width,
        plane.height,
        plane.coefficients.len(),
        nonzero,
        min,
        max
    );
    ExitCode::SUCCESS
}

pub fn dump_component_plane_file(
    input: &str,
    component: usize,
    discard_levels: u8,
    output: &str,
) -> ExitCode {
    let data = match fs::read(input) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {input}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let planes = match decode_reversible_component_planes_with_discard(&data, discard_levels) {
        Ok(planes) => planes,
        Err(err) => {
            eprintln!("dump-component-plane failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    let Some(plane) = planes.components.get(component) else {
        eprintln!("dump-component-plane failed: missing component {component}");
        return ExitCode::FAILURE;
    };

    let mut bytes = Vec::with_capacity(plane.samples.len() * 4);
    for sample in &plane.samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    if let Err(err) = fs::write(output, bytes) {
        eprintln!("failed to write {output}: {err}");
        return ExitCode::FAILURE;
    }
    let nonzero = plane.samples.iter().filter(|sample| **sample != 0).count();
    let min = plane.samples.iter().min().copied().unwrap_or(0);
    let max = plane.samples.iter().max().copied().unwrap_or(0);
    println!(
        "dump-component-plane: component {component} {}x{} {}-bit discard {}, {} samples, {} nonzero, min {}, max {} -> {output}",
        plane.width,
        plane.height,
        planes.precision,
        discard_levels,
        plane.samples.len(),
        nonzero,
        min,
        max
    );
    ExitCode::SUCCESS
}

pub fn trace_component53_sample_file(
    input: &str,
    component: usize,
    discard_levels: u8,
    x: u32,
    y: u32,
    max_dependencies: usize,
) -> ExitCode {
    let data = match fs::read(input) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {input}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let decoded = match decode_coefficient_planes(&data) {
        Ok(decoded) => decoded,
        Err(err) => {
            eprintln!("trace-component53-sample failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    let Some(component_planes) = decoded
        .components
        .iter()
        .find(|planes| planes.component as usize == component)
    else {
        eprintln!("trace-component53-sample failed: missing component {component}");
        return ExitCode::FAILURE;
    };
    let max_resolution = component_planes
        .planes
        .iter()
        .map(|plane| plane.resolution)
        .max()
        .unwrap_or(0);
    if discard_levels > max_resolution {
        eprintln!(
            "trace-component53-sample failed: cannot discard {discard_levels} levels from {max_resolution} decomposition levels"
        );
        return ExitCode::FAILURE;
    }
    let target_resolution = max_resolution - discard_levels;
    let Some((width, height)) = resolution_extent(component_planes, target_resolution) else {
        eprintln!(
            "trace-component53-sample failed: missing component {component} resolution {target_resolution}"
        );
        return ExitCode::FAILURE;
    };
    if x >= width || y >= height {
        eprintln!(
            "trace-component53-sample failed: sample {x},{y} outside component {component} R{target_resolution} extent {width}x{height}"
        );
        return ExitCode::FAILURE;
    }

    let reconstructed = match component_planes.inverse_5_3_until(target_resolution) {
        Ok(plane) => plane,
        Err(err) => {
            eprintln!("trace-component53-sample failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    let sample = reconstructed
        .sample(x, y)
        .expect("coordinate checked against reconstructed plane");
    let mut dependencies = BTreeSet::new();
    collect_component53_dependencies(component_planes, target_resolution, x, y, &mut dependencies);

    println!(
        "trace-component53-sample: component {component} discard {discard_levels} -> R{target_resolution} {width}x{height}, sample ({x},{y}) = {sample}"
    );
    println!("  coefficient dependencies: {}", dependencies.len());

    let mut by_codeblock = BTreeMap::<(u8, SubbandKind, u32, u32), usize>::new();
    for dependency in &dependencies {
        if let Some(plane) =
            find_resolution_subband(component_planes, dependency.resolution, dependency.subband)
        {
            let codeblock_x = dependency.x / plane.layout.codeblock_width;
            let codeblock_y = dependency.y / plane.layout.codeblock_height;
            *by_codeblock
                .entry((
                    dependency.resolution,
                    dependency.subband,
                    codeblock_x,
                    codeblock_y,
                ))
                .or_default() += 1;
        }
    }
    println!("  touched codeblocks: {}", by_codeblock.len());
    for ((resolution, subband, codeblock_x, codeblock_y), count) in &by_codeblock {
        println!("    R{resolution} {subband} cblk {codeblock_x},{codeblock_y}: {count} coeff");
    }

    for dependency in dependencies.iter().take(max_dependencies) {
        match trace_dependency_value(component_planes, dependency) {
            Some(value) => println!(
                "    R{} {} ({},{}): {}",
                dependency.resolution, dependency.subband, dependency.x, dependency.y, value
            ),
            None => println!(
                "    R{} {} ({},{}): unavailable",
                dependency.resolution, dependency.subband, dependency.x, dependency.y
            ),
        }
    }
    if dependencies.len() > max_dependencies {
        println!(
            "    ... {} more dependencies",
            dependencies.len() - max_dependencies
        );
    }

    ExitCode::SUCCESS
}

pub fn trace_tier1_codeblock_file(
    input: &str,
    component: u16,
    resolution: u8,
    subband: &str,
    codeblock_x: u32,
    codeblock_y: u32,
) -> ExitCode {
    let subband = match parse_subband(subband) {
        Some(subband) => subband,
        None => {
            eprintln!("tier1-codeblock-trace failed: subband must be LL, HL, LH, or HH");
            return ExitCode::FAILURE;
        }
    };
    let data = match fs::read(input) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {input}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let codestream = match Codestream::parse(&data) {
        Ok(codestream) => codestream,
        Err(err) => {
            eprintln!("tier1-codeblock-trace failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    let Some(cod) = &codestream.cod else {
        eprintln!("tier1-codeblock-trace failed: missing COD marker");
        return ExitCode::FAILURE;
    };
    if codestream.tile_parts.len() != 1 {
        eprintln!(
            "tier1-codeblock-trace failed: only one tile-part is supported for now, found {}",
            codestream.tile_parts.len()
        );
        return ExitCode::FAILURE;
    }
    let layouts = match codestream.tile_layouts() {
        Ok(layouts) => layouts,
        Err(err) => {
            eprintln!("tier1-codeblock-trace failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    let tile_part = &codestream.tile_parts[0];
    let Some(layout) = layouts
        .iter()
        .find(|layout| layout.tile_index == tile_part.tile_index)
    else {
        eprintln!(
            "tier1-codeblock-trace failed: missing layout for tile {}",
            tile_part.tile_index
        );
        return ExitCode::FAILURE;
    };
    let Some(tile_data) = data.get(tile_part.data_start..tile_part.data_end) else {
        eprintln!("tier1-codeblock-trace failed: tile payload is outside codestream data");
        return ExitCode::FAILURE;
    };
    let packet_summary = match parse_lrcp_tile_packets(tile_data, layout, cod.layers) {
        Ok(summary) => summary,
        Err(err) => {
            eprintln!("tier1-codeblock-trace failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    let work_plan = match Tier1WorkPlan::from_packet_summary_with_quantization(
        &packet_summary,
        codestream.qcd.as_ref(),
        &codestream.qccs,
    ) {
        Ok(plan) => plan,
        Err(err) => {
            eprintln!("tier1-codeblock-trace failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    let bound = match work_plan.bind_segment_data(tile_data) {
        Ok(bound) => bound,
        Err(err) => {
            eprintln!("tier1-codeblock-trace failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    let Some(codeblock) = bound.codeblocks.iter().find(|codeblock| {
        codeblock.id.component == component
            && codeblock.id.resolution == resolution
            && codeblock.id.subband == subband
            && codeblock.id.codeblock_x == codeblock_x
            && codeblock.id.codeblock_y == codeblock_y
    }) else {
        eprintln!(
            "tier1-codeblock-trace failed: missing C{component} R{resolution} {subband} cblk {codeblock_x},{codeblock_y}"
        );
        return ExitCode::FAILURE;
    };
    let execution = match codeblock.execution_plan(cod.codeblock_style_flags()) {
        Ok(execution) => execution,
        Err(err) => {
            eprintln!("tier1-codeblock-trace failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    if execution
        .passes
        .iter()
        .any(|pass| pass.coding_mode == Tier1CodingMode::RawBypass)
    {
        eprintln!("tier1-codeblock-trace failed: raw-bypass passes are not traced yet");
        return ExitCode::FAILURE;
    }

    let mut segment_data = Vec::with_capacity(codeblock.total_bytes());
    for segment in &codeblock.segments {
        segment_data.extend_from_slice(segment.data);
    }
    let mut mq = MqDecoder::with_data(&segment_data);
    let mut decoded = match DecodedCodeblock::new_bound_with_reconstruction(
        codeblock,
        coefficient_reconstruction_for_wavelet(cod.transform),
    ) {
        Ok(decoded) => decoded,
        Err(err) => {
            eprintln!("tier1-codeblock-trace failed: {err}");
            return ExitCode::FAILURE;
        }
    };

    println!(
        "tier1-codeblock-trace: C{component} R{resolution} {subband} cblk {codeblock_x},{codeblock_y} {}x{}, zbp {}, mbp {}, {} segment(s), {} byte(s), {} pass(es)",
        codeblock.width,
        codeblock.height,
        codeblock.zero_bit_planes,
        codeblock.magnitude_bitplanes,
        codeblock.segments.len(),
        codeblock.total_bytes(),
        execution.passes.len()
    );
    for (segment_index, segment) in codeblock.segments.iter().enumerate() {
        println!(
            "  segment {segment_index}: packet {}, layer {}, {} byte(s), {} pass(es)",
            segment.packet_index,
            segment.layer,
            segment.data.len(),
            segment.coding_passes
        );
    }

    for pass in execution.passes {
        let summary = match decoded.decode_pass(pass.pass, &mut mq) {
            Ok(summary) => summary,
            Err(err) => {
                eprintln!("tier1-codeblock-trace failed: {err}");
                return ExitCode::FAILURE;
            }
        };
        println!(
            "  pass {:>2}: bp {:>2} {:>12}, packet {}, layer {}, seg {}:{}, visited {}, sig {}, sign {}, refine {}, run {}, uniform {}, became {}, consumed {}",
            pass.pass.index,
            pass.pass.bitplane,
            pass.pass.kind,
            pass.packet_index,
            pass.layer,
            pass.segment_index,
            pass.segment_pass_index,
            summary.visited,
            summary.significance_bits,
            summary.sign_bits,
            summary.refinement_bits,
            summary.run_length_bits,
            summary.uniform_bits,
            summary.became_significant,
            mq.consumed_bytes()
        );
    }

    let nonzero = decoded
        .coefficients
        .iter()
        .filter(|coefficient| **coefficient != 0)
        .count();
    println!(
        "  decoded: {} significant, {} nonzero, consumed {}/{} bytes",
        decoded.state.significant_count(),
        nonzero,
        mq.consumed_bytes(),
        segment_data.len()
    );

    ExitCode::SUCCESS
}

fn parse_subband(value: &str) -> Option<SubbandKind> {
    match value.to_ascii_uppercase().as_str() {
        "LL" => Some(SubbandKind::Ll),
        "HL" => Some(SubbandKind::Hl),
        "LH" => Some(SubbandKind::Lh),
        "HH" => Some(SubbandKind::Hh),
        _ => None,
    }
}

fn print_decode_profile(profile: &DecodeProfile) {
    println!(
        "  input_bytes {}, tile_bytes {}, codeblocks decoded {}/{} (skipped {}), passes {}, nonzero {}, output_samples {}",
        profile.input_bytes,
        profile.tile_bytes,
        profile.decoded_codeblocks,
        profile.total_codeblocks,
        profile.skipped_codeblocks,
        profile.decoded_passes,
        profile.nonzero_coefficients,
        profile.output_samples
    );
    println!("  decode_total_ms {:.3}", profile.timings.total_ms);
    println!("    parse_ms {:.3}", profile.timings.parse_ms);
    println!("    layout_ms {:.3}", profile.timings.layout_ms);
    println!("    packet_ms {:.3}", profile.timings.packet_ms);
    println!("    work_plan_ms {:.3}", profile.timings.work_plan_ms);
    println!("    bind_ms {:.3}", profile.timings.bind_ms);
    println!("    tier1_decode_ms {:.3}", profile.timings.tier1_decode_ms);
    println!("    gpu_setup_ms {:.3}", profile.timings.gpu_setup_ms);
    println!("    reconstruct_ms {:.3}", profile.timings.reconstruct_ms);
    println!(
        "    reconstruct_upload_ms {:.3}",
        profile.timings.reconstruct_upload_ms
    );
    println!(
        "    reconstruct_alloc_ms {:.3}",
        profile.timings.reconstruct_alloc_ms
    );
    println!(
        "    reconstruct_dispatch_ms {:.3}",
        profile.timings.reconstruct_dispatch_ms
    );
    println!("    pack_ms {:.3}", profile.timings.pack_ms);
    println!("    pack_shader_ms {:.3}", profile.timings.pack_shader_ms);
    println!("    pack_copy_ms {:.3}", profile.timings.pack_copy_ms);
    println!(
        "    pack_readback_ms {:.3}",
        profile.timings.pack_readback_ms
    );
}

fn print_profile_average(label: &str, profiles: &[DecodeProfile]) {
    let count = profiles.len() as f64;
    let sum = profiles
        .iter()
        .fold(DecodeProfile::default(), |mut acc, profile| {
            acc.timings.total_ms += profile.timings.total_ms;
            acc.timings.parse_ms += profile.timings.parse_ms;
            acc.timings.layout_ms += profile.timings.layout_ms;
            acc.timings.packet_ms += profile.timings.packet_ms;
            acc.timings.work_plan_ms += profile.timings.work_plan_ms;
            acc.timings.bind_ms += profile.timings.bind_ms;
            acc.timings.tier1_decode_ms += profile.timings.tier1_decode_ms;
            acc.timings.gpu_setup_ms += profile.timings.gpu_setup_ms;
            acc.timings.reconstruct_ms += profile.timings.reconstruct_ms;
            acc.timings.reconstruct_upload_ms += profile.timings.reconstruct_upload_ms;
            acc.timings.reconstruct_alloc_ms += profile.timings.reconstruct_alloc_ms;
            acc.timings.reconstruct_dispatch_ms += profile.timings.reconstruct_dispatch_ms;
            acc.timings.pack_ms += profile.timings.pack_ms;
            acc.timings.pack_shader_ms += profile.timings.pack_shader_ms;
            acc.timings.pack_copy_ms += profile.timings.pack_copy_ms;
            acc.timings.pack_readback_ms += profile.timings.pack_readback_ms;
            acc
        });
    let decode_total_ms = sum.timings.total_ms / count;
    let fps = if decode_total_ms > 0.0 {
        1000.0 / decode_total_ms
    } else {
        0.0
    };
    println!(
        "  {label}_ms decode_total {:.3} fps {:.2} parse {:.3} layout {:.3} packet {:.3} work_plan {:.3} bind {:.3} tier1 {:.3} gpu_setup {:.3} reconstruct {:.3} reconstruct_upload {:.3} reconstruct_alloc {:.3} reconstruct_dispatch {:.3} pack {:.3} pack_shader {:.3} pack_copy {:.3} pack_readback {:.3}",
        decode_total_ms,
        fps,
        sum.timings.parse_ms / count,
        sum.timings.layout_ms / count,
        sum.timings.packet_ms / count,
        sum.timings.work_plan_ms / count,
        sum.timings.bind_ms / count,
        sum.timings.tier1_decode_ms / count,
        sum.timings.gpu_setup_ms / count,
        sum.timings.reconstruct_ms / count,
        sum.timings.reconstruct_upload_ms / count,
        sum.timings.reconstruct_alloc_ms / count,
        sum.timings.reconstruct_dispatch_ms / count,
        sum.timings.pack_ms / count,
        sum.timings.pack_shader_ms / count,
        sum.timings.pack_copy_ms / count,
        sum.timings.pack_readback_ms / count
    );
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TraceDependency {
    resolution: u8,
    subband: SubbandKind,
    x: u32,
    y: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Inverse53SourceIndices {
    low: Vec<u32>,
    high: Vec<u32>,
}

fn collect_component53_dependencies(
    component: &ComponentSubbandPlanes,
    resolution: u8,
    x: u32,
    y: u32,
    dependencies: &mut BTreeSet<TraceDependency>,
) {
    if resolution == 0 {
        dependencies.insert(TraceDependency {
            resolution,
            subband: SubbandKind::Ll,
            x,
            y,
        });
        return;
    }

    let Some((low_width, low_height)) = resolution_extent(component, resolution - 1) else {
        return;
    };
    let Some(hl) = find_resolution_subband(component, resolution, SubbandKind::Hl) else {
        return;
    };
    let Some(lh) = find_resolution_subband(component, resolution, SubbandKind::Lh) else {
        return;
    };

    let x_sources = inverse_53_source_indices(x, low_width, hl.layout.width);
    let y_sources = inverse_53_source_indices(y, low_height, lh.layout.height);

    for low_y in &y_sources.low {
        for low_x in &x_sources.low {
            collect_component53_dependencies(
                component,
                resolution - 1,
                *low_x,
                *low_y,
                dependencies,
            );
        }
        for high_x in &x_sources.high {
            dependencies.insert(TraceDependency {
                resolution,
                subband: SubbandKind::Hl,
                x: *high_x,
                y: *low_y,
            });
        }
    }
    for high_y in &y_sources.high {
        for low_x in &x_sources.low {
            dependencies.insert(TraceDependency {
                resolution,
                subband: SubbandKind::Lh,
                x: *low_x,
                y: *high_y,
            });
        }
        for high_x in &x_sources.high {
            dependencies.insert(TraceDependency {
                resolution,
                subband: SubbandKind::Hh,
                x: *high_x,
                y: *high_y,
            });
        }
    }
}

fn inverse_53_source_indices(
    output_index: u32,
    low_len: u32,
    high_len: u32,
) -> Inverse53SourceIndices {
    if low_len == 0 {
        return Inverse53SourceIndices {
            low: Vec::new(),
            high: Vec::new(),
        };
    }
    if high_len == 0 {
        return Inverse53SourceIndices {
            low: vec![output_index.min(low_len - 1)],
            high: Vec::new(),
        };
    }

    let mut low = BTreeSet::new();
    let mut high = BTreeSet::new();
    if output_index.is_multiple_of(2) {
        let index = (output_index / 2).min(low_len - 1);
        add_inverse_53_updated_low_sources(index, low_len, high_len, &mut low, &mut high);
    } else {
        let index = (output_index / 2).min(high_len - 1);
        high.insert(index);
        add_inverse_53_updated_low_sources(index, low_len, high_len, &mut low, &mut high);
        add_inverse_53_updated_low_sources(
            (index + 1).min(low_len - 1),
            low_len,
            high_len,
            &mut low,
            &mut high,
        );
    }

    Inverse53SourceIndices {
        low: low.into_iter().collect(),
        high: high.into_iter().collect(),
    }
}

fn add_inverse_53_updated_low_sources(
    index: u32,
    low_len: u32,
    high_len: u32,
    low: &mut BTreeSet<u32>,
    high: &mut BTreeSet<u32>,
) {
    if index >= low_len {
        return;
    }
    low.insert(index);
    if high_len == 0 {
        return;
    }
    high.insert(index.saturating_sub(1).min(high_len - 1));
    high.insert(index.min(high_len - 1));
}

fn resolution_extent(component: &ComponentSubbandPlanes, resolution: u8) -> Option<(u32, u32)> {
    if resolution == 0 {
        let plane = find_resolution_subband(component, 0, SubbandKind::Ll)?;
        return Some((plane.layout.width, plane.layout.height));
    }
    let hl = find_resolution_subband(component, resolution, SubbandKind::Hl)?;
    let lh = find_resolution_subband(component, resolution, SubbandKind::Lh)?;
    Some((
        lh.layout.width + hl.layout.width,
        hl.layout.height + lh.layout.height,
    ))
}

fn find_resolution_subband(
    component: &ComponentSubbandPlanes,
    resolution: u8,
    subband: SubbandKind,
) -> Option<&ResolutionSubbandPlane> {
    component
        .planes
        .iter()
        .find(|plane| plane.resolution == resolution && plane.layout.kind == subband)
}

fn trace_dependency_value(
    component: &ComponentSubbandPlanes,
    dependency: &TraceDependency,
) -> Option<i32> {
    let plane = find_resolution_subband(component, dependency.resolution, dependency.subband)?;
    plane.plane.coefficient(dependency.x, dependency.y)
}

fn coefficient_reconstruction_for_wavelet(
    transform: WaveletTransform,
) -> CoefficientReconstruction {
    match transform {
        WaveletTransform::Reversible53 => CoefficientReconstruction::Exact,
        WaveletTransform::Irreversible97 | WaveletTransform::Unknown(_) => {
            CoefficientReconstruction::Midpoint
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Rgb48DiffStats {
    samples: usize,
    differing: u64,
    max_abs: u16,
    max_index: usize,
    max_x: usize,
    max_y: usize,
    max_channel: usize,
    min_delta: i32,
    max_delta: i32,
    mean_abs: f64,
    channel_differing: [u64; 3],
    channel_mean_abs: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
struct I32DiffStats {
    samples: usize,
    differing: u64,
    max_abs: u32,
    max_index: usize,
    max_x: usize,
    max_y: usize,
    min_delta: i64,
    max_delta: i64,
    mean_abs: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct ComponentRgb48DiffStats {
    diff: I32DiffStats,
    comparable: usize,
    clipped_or_saturated: usize,
    lowbit_misaligned: usize,
}

fn diff_rgb48(actual: &[u8], reference: &[u8]) -> Result<Rgb48DiffStats, String> {
    if actual.len() != reference.len() {
        return Err(format!(
            "length mismatch: actual {} bytes, reference {} bytes",
            actual.len(),
            reference.len()
        ));
    }
    if !actual.len().is_multiple_of(6) {
        return Err(format!(
            "rgb48 buffers must contain whole pixels, got {} bytes",
            actual.len()
        ));
    }

    let samples = actual.len() / 2;
    let pixels = samples / 3;
    let width = infer_rgb48_width(pixels);
    let mut differing = 0u64;
    let mut sum_abs = 0u64;
    let mut max_abs = 0u16;
    let mut max_index = 0usize;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut max_channel = 0usize;
    let mut min_delta = 0i32;
    let mut max_delta = 0i32;
    let mut channel_differing = [0u64; 3];
    let mut channel_sum_abs = [0u64; 3];

    for index in 0..samples {
        let actual_sample = u16::from_le_bytes([actual[index * 2], actual[index * 2 + 1]]);
        let reference_sample = u16::from_le_bytes([reference[index * 2], reference[index * 2 + 1]]);
        let delta_signed = i32::from(actual_sample) - i32::from(reference_sample);
        let delta = actual_sample.abs_diff(reference_sample);
        if delta != 0 {
            differing += 1;
            channel_differing[index % 3] += 1;
        }
        sum_abs += u64::from(delta);
        channel_sum_abs[index % 3] += u64::from(delta);
        min_delta = min_delta.min(delta_signed);
        max_delta = max_delta.max(delta_signed);
        if delta > max_abs {
            max_abs = delta;
            max_index = index;
            max_channel = index % 3;
            let pixel = index / 3;
            max_x = width.map(|value| pixel % value).unwrap_or(pixel);
            max_y = width.map(|value| pixel / value).unwrap_or(0);
        }
    }

    let channel_samples = (samples / 3) as f64;
    Ok(Rgb48DiffStats {
        samples,
        differing,
        max_abs,
        max_index,
        max_x,
        max_y,
        max_channel,
        min_delta,
        max_delta,
        mean_abs: sum_abs as f64 / samples as f64,
        channel_differing,
        channel_mean_abs: [
            channel_sum_abs[0] as f64 / channel_samples,
            channel_sum_abs[1] as f64 / channel_samples,
            channel_sum_abs[2] as f64 / channel_samples,
        ],
    })
}

fn read_i32le_samples(bytes: &[u8]) -> Result<Vec<i32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "i32le plane buffers must contain whole samples, got {} bytes",
            bytes.len()
        ));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

fn diff_i32_samples(actual: &[i32], reference: &[i32]) -> Result<I32DiffStats, String> {
    if actual.len() != reference.len() {
        return Err(format!(
            "length mismatch: actual {} samples, reference {} samples",
            actual.len(),
            reference.len()
        ));
    }

    Ok(diff_i32_indexed_sample_pairs(
        actual
            .iter()
            .copied()
            .zip(reference.iter().copied())
            .enumerate()
            .map(|(index, (actual, reference))| (index, actual, reference)),
        actual.len(),
        actual.len(),
    ))
}

fn diff_i32_samples_against_rgb48(
    actual: &[i32],
    reference_rgb48: &[u8],
    component: usize,
    precision: u8,
) -> Result<ComponentRgb48DiffStats, String> {
    if component >= 3 {
        return Err(format!("component must be 0, 1, or 2, got {component}"));
    }
    if precision == 0 || precision > 16 {
        return Err(format!("precision must be in 1..=16, got {precision}"));
    }
    if !reference_rgb48.len().is_multiple_of(6) {
        return Err(format!(
            "rgb48 buffer must contain whole pixels, got {} bytes",
            reference_rgb48.len()
        ));
    }
    let pixels = reference_rgb48.len() / 6;
    if actual.len() != pixels {
        return Err(format!(
            "length mismatch: actual {} component samples, reference {} pixels",
            actual.len(),
            pixels
        ));
    }

    let shift = 16u8 - precision;
    let low_mask = if shift == 0 { 0 } else { (1u16 << shift) - 1 };
    let level_shift = 1i32 << (precision - 1);
    let native_min = -level_shift;
    let native_max = level_shift - 1;
    let unsigned_max = (1i32 << precision) - 1;
    let mut lowbit_misaligned = 0usize;
    let mut clipped_or_saturated = 0usize;
    let mut comparable = Vec::new();

    for (index, actual_sample) in actual.iter().copied().enumerate() {
        let byte_index = (index * 3 + component) * 2;
        let packed =
            u16::from_le_bytes([reference_rgb48[byte_index], reference_rgb48[byte_index + 1]]);
        if packed & low_mask != 0 {
            lowbit_misaligned += 1;
        }
        let reference_native = i32::from(packed >> shift) - level_shift;
        let reference_unsigned = reference_native + level_shift;
        if actual_sample < native_min
            || actual_sample > native_max
            || reference_unsigned == 0
            || reference_unsigned == unsigned_max
        {
            clipped_or_saturated += 1;
            continue;
        }
        comparable.push((index, actual_sample, reference_native));
    }

    let diff =
        diff_i32_indexed_sample_pairs(comparable.iter().copied(), comparable.len(), actual.len());
    Ok(ComponentRgb48DiffStats {
        diff,
        comparable: comparable.len(),
        clipped_or_saturated,
        lowbit_misaligned,
    })
}

fn diff_i32_indexed_sample_pairs(
    pairs: impl Iterator<Item = (usize, i32, i32)>,
    samples: usize,
    coordinate_samples: usize,
) -> I32DiffStats {
    let width = infer_plane_width(coordinate_samples);
    let mut differing = 0u64;
    let mut max_abs = 0u32;
    let mut max_index = 0usize;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut min_delta = 0i64;
    let mut max_delta = 0i64;
    let mut sum_abs = 0u64;

    for (index, actual, reference) in pairs {
        let delta = i64::from(actual) - i64::from(reference);
        let abs_delta = delta.unsigned_abs();
        if delta != 0 {
            differing += 1;
        }
        sum_abs += abs_delta;
        min_delta = min_delta.min(delta);
        max_delta = max_delta.max(delta);
        if abs_delta > u64::from(max_abs) {
            max_abs = abs_delta.min(u64::from(u32::MAX)) as u32;
            max_index = index;
            max_x = width.map(|value| index % value).unwrap_or(index);
            max_y = width.map(|value| index / value).unwrap_or(0);
        }
    }

    I32DiffStats {
        samples,
        differing,
        max_abs,
        max_index,
        max_x,
        max_y,
        min_delta,
        max_delta,
        mean_abs: if samples == 0 {
            0.0
        } else {
            sum_abs as f64 / samples as f64
        },
    }
}

fn print_i32_diff_stats(label: &str, stats: &I32DiffStats) {
    println!(
        "{label}: samples {}, differing {}, max_abs {} at sample {} (x {}, y {}), mean_abs {:.12}",
        stats.samples,
        stats.differing,
        stats.max_abs,
        stats.max_index,
        stats.max_x,
        stats.max_y,
        stats.mean_abs
    );
    println!(
        "  signed_delta min {} max {}",
        stats.min_delta, stats.max_delta
    );
}

fn infer_rgb48_width(pixels: usize) -> Option<usize> {
    infer_plane_width(pixels)
}

fn infer_plane_width(samples: usize) -> Option<usize> {
    match samples {
        value if value == 60 * 34 => Some(60),
        value if value == 120 * 68 => Some(120),
        value if value == 240 * 135 => Some(240),
        value if value == 480 * 270 => Some(480),
        value if value == 960 * 540 => Some(960),
        value if value == 1920 * 1080 => Some(1920),
        _ => None,
    }
}

fn channel_name(channel: usize) -> &'static str {
    match channel {
        0 => "R",
        1 => "G",
        2 => "B",
        _ => "?",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_rgb48_diff_preserves_source_coordinates() {
        let mut actual = vec![10i32; 120 * 68];
        actual[2 * 120 + 29] = 12;
        let mut reference = vec![0u8; 120 * 68 * 6];
        for index in 0..120 * 68 {
            let packed = ((2048u16 + 10) << 4).to_le_bytes();
            let byte_index = (index * 3 + 2) * 2;
            reference[byte_index..byte_index + 2].copy_from_slice(&packed);
        }

        let stats = diff_i32_samples_against_rgb48(&actual, &reference, 2, 12).expect("diff stats");

        assert_eq!(stats.diff.max_abs, 2);
        assert_eq!(stats.diff.max_index, 269);
        assert_eq!(stats.diff.max_x, 29);
        assert_eq!(stats.diff.max_y, 2);
        assert_eq!(stats.lowbit_misaligned, 0);
    }

    #[test]
    fn component_rgb48_diff_skips_clipped_samples() {
        let actual = vec![-3000, 0, 3000];
        let mut reference = Vec::new();
        for sample in [0u16, 2048u16 << 4, 4095u16 << 4] {
            reference.extend_from_slice(&sample.to_le_bytes());
            reference.extend_from_slice(&sample.to_le_bytes());
            reference.extend_from_slice(&sample.to_le_bytes());
        }

        let stats = diff_i32_samples_against_rgb48(&actual, &reference, 0, 12).expect("diff stats");

        assert_eq!(stats.clipped_or_saturated, 2);
        assert_eq!(stats.comparable, 1);
        assert_eq!(stats.diff.differing, 0);
    }

    #[test]
    fn inverse_53_source_indices_tracks_interleaved_high_sample() {
        let sources = inverse_53_source_indices(29, 60, 60);

        assert_eq!(sources.low, vec![14, 15]);
        assert_eq!(sources.high, vec![13, 14, 15]);
    }

    #[test]
    fn inverse_53_source_indices_clamps_edge_neighbors() {
        let first_low = inverse_53_source_indices(0, 3, 2);
        let first_high = inverse_53_source_indices(1, 3, 2);
        let last_low = inverse_53_source_indices(4, 3, 2);

        assert_eq!(first_low.low, vec![0]);
        assert_eq!(first_low.high, vec![0]);
        assert_eq!(first_high.low, vec![0, 1]);
        assert_eq!(first_high.high, vec![0, 1]);
        assert_eq!(last_low.low, vec![2]);
        assert_eq!(last_low.high, vec![1]);
    }
}
