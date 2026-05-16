use std::fs;
use std::process::ExitCode;

use vk_j2k_dec::codestream::SubbandKind;
use vk_j2k_dec::decode::{
    decode_coefficient_planes, decode_reversible_component_planes_with_discard,
    decode_rgb48_with_discard, decode_rgb48_with_gpu_dwt, decode_rgb48_with_gpu_pipeline,
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

fn parse_subband(value: &str) -> Option<SubbandKind> {
    match value.to_ascii_uppercase().as_str() {
        "LL" => Some(SubbandKind::Ll),
        "HL" => Some(SubbandKind::Hl),
        "LH" => Some(SubbandKind::Lh),
        "HH" => Some(SubbandKind::Hh),
        _ => None,
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

fn infer_rgb48_width(pixels: usize) -> Option<usize> {
    match pixels {
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
