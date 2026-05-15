use std::fs;
use std::process::ExitCode;

use vk_j2k_dec::decode::{
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
        "diff-rgb48: samples {}, differing {}, max_abs {} at sample {}, mean_abs {:.12}",
        stats.samples, stats.differing, stats.max_abs, stats.max_index, stats.mean_abs
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

#[derive(Debug, Clone, PartialEq)]
struct Rgb48DiffStats {
    samples: usize,
    differing: u64,
    max_abs: u16,
    max_index: usize,
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
    let mut differing = 0u64;
    let mut sum_abs = 0u64;
    let mut max_abs = 0u16;
    let mut max_index = 0usize;
    let mut channel_differing = [0u64; 3];
    let mut channel_sum_abs = [0u64; 3];

    for index in 0..samples {
        let actual_sample = u16::from_le_bytes([actual[index * 2], actual[index * 2 + 1]]);
        let reference_sample = u16::from_le_bytes([reference[index * 2], reference[index * 2 + 1]]);
        let delta = actual_sample.abs_diff(reference_sample);
        if delta != 0 {
            differing += 1;
            channel_differing[index % 3] += 1;
        }
        sum_abs += u64::from(delta);
        channel_sum_abs[index % 3] += u64::from(delta);
        if delta > max_abs {
            max_abs = delta;
            max_index = index;
        }
    }

    let channel_samples = (samples / 3) as f64;
    Ok(Rgb48DiffStats {
        samples,
        differing,
        max_abs,
        max_index,
        mean_abs: sum_abs as f64 / samples as f64,
        channel_differing,
        channel_mean_abs: [
            channel_sum_abs[0] as f64 / channel_samples,
            channel_sum_abs[1] as f64 / channel_samples,
            channel_sum_abs[2] as f64 / channel_samples,
        ],
    })
}
