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
}
