use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use vk_j2k_dec::codestream::{Codestream, SubbandKind, WaveletTransform};
use vk_j2k_dec::decode::{
    DecodeProfile, decode_coefficient_planes, decode_reversible_component_planes_with_discard,
    decode_rgb48_with_discard, decode_rgb48_with_discard_profiled, decode_rgb48_with_gpu_dwt,
    decode_rgb48_with_gpu_dwt_profiled, decode_rgb48_with_gpu_pipeline,
    decode_rgb48_with_gpu_pipeline_profiled,
};
use vk_j2k_dec::packet::parse_lrcp_tile_packets;
use vk_j2k_dec::reconstruct::{ComponentSubbandPlanes, ResolutionSubbandPlane};
use vk_j2k_dec::tier1::{
    CoefficientReconstruction, DecodedCodeblock, MqDecoder, Tier1CodingMode, Tier1WorkPlan,
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
    println!("    pack_ms {:.3}", profile.timings.pack_ms);
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
