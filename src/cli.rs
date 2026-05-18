use std::env;
use std::fs;
use std::process::ExitCode;

use vk_j2k_dec::codestream::{Codestream, WaveletTransform, scan_codestreams};
use vk_j2k_dec::gpu::{GpuDwtDispatchPlan, GpuPipelinePlan, GpuTier1Plan};
use vk_j2k_dec::packet::{TilePacketHeaderState, parse_lrcp_tile_packets};
use vk_j2k_dec::reconstruct::TileCoefficientPlanes;
use vk_j2k_dec::tier1::{
    BoundCompressedCodeblock, CoefficientReconstruction, DecodedCodeblock, MqDecoder,
    Tier1WorkPlan, tier1_pass_for_index,
};

mod raw;
mod scan;
mod vulkan;

pub fn run() -> ExitCode {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "vk-j2k-dec".to_string());

    match (args.next().as_deref(), args.next()) {
        (Some("dump"), Some(path)) => dump_codestream(&path),
        (Some("scan"), Some(path)) => {
            let max_hits = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(8);
            scan::scan_file(&path, max_hits)
        }
        (Some("scan-summary"), Some(path)) => scan::scan_summary(&path),
        (Some("extract"), Some(path)) => {
            let Some(index) = args.next().and_then(|value| value.parse::<usize>().ok()) else {
                eprintln!(
                    "usage: {program} extract <container-or-bytestream> <index> <output.j2k>"
                );
                return ExitCode::FAILURE;
            };
            let Some(output) = args.next() else {
                eprintln!(
                    "usage: {program} extract <container-or-bytestream> <index> <output.j2k>"
                );
                return ExitCode::FAILURE;
            };
            extract_codestream(&path, index, &output)
        }
        (Some("tier1-smoke"), Some(path)) => {
            let max_codeblocks = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(8);
            tier1_smoke(&path, max_codeblocks)
        }
        (Some("tier1-subband-stats"), Some(path)) => tier1_subband_stats(&path),
        (Some("tier1-gpu-plan"), Some(path)) => tier1_gpu_plan(&path),
        (Some("tier1-codeblock-trace"), Some(path)) => {
            let Some(component) = args.next().and_then(|value| value.parse::<u16>().ok()) else {
                eprintln!(
                    "usage: {program} tier1-codeblock-trace <input.j2k> <component> <resolution> <LL|HL|LH|HH> <codeblock-x> <codeblock-y>"
                );
                return ExitCode::FAILURE;
            };
            let Some(resolution) = args.next().and_then(|value| value.parse::<u8>().ok()) else {
                eprintln!(
                    "usage: {program} tier1-codeblock-trace <input.j2k> <component> <resolution> <LL|HL|LH|HH> <codeblock-x> <codeblock-y>"
                );
                return ExitCode::FAILURE;
            };
            let Some(subband) = args.next() else {
                eprintln!(
                    "usage: {program} tier1-codeblock-trace <input.j2k> <component> <resolution> <LL|HL|LH|HH> <codeblock-x> <codeblock-y>"
                );
                return ExitCode::FAILURE;
            };
            let Some(codeblock_x) = args.next().and_then(|value| value.parse::<u32>().ok()) else {
                eprintln!(
                    "usage: {program} tier1-codeblock-trace <input.j2k> <component> <resolution> <LL|HL|LH|HH> <codeblock-x> <codeblock-y>"
                );
                return ExitCode::FAILURE;
            };
            let Some(codeblock_y) = args.next().and_then(|value| value.parse::<u32>().ok()) else {
                eprintln!(
                    "usage: {program} tier1-codeblock-trace <input.j2k> <component> <resolution> <LL|HL|LH|HH> <codeblock-x> <codeblock-y>"
                );
                return ExitCode::FAILURE;
            };
            raw::trace_tier1_codeblock_file(
                &path,
                component,
                resolution,
                &subband,
                codeblock_x,
                codeblock_y,
            )
        }
        (Some("reconstruct-smoke"), Some(path)) => reconstruct_smoke(&path),
        (Some("decode-rgb48"), Some(path)) => {
            let Some(output) = args.next() else {
                eprintln!("usage: {program} decode-rgb48 <input.j2k> <output.rgb48le>");
                return ExitCode::FAILURE;
            };
            raw::decode_rgb48_file(&path, &output)
        }
        (Some("decode-rgb48-lowres"), Some(path)) => {
            let Some(discard_levels) = args.next().and_then(|value| value.parse::<u8>().ok())
            else {
                eprintln!(
                    "usage: {program} decode-rgb48-lowres <input.j2k> <discard-levels> <output.rgb48le>"
                );
                return ExitCode::FAILURE;
            };
            let Some(output) = args.next() else {
                eprintln!(
                    "usage: {program} decode-rgb48-lowres <input.j2k> <discard-levels> <output.rgb48le>"
                );
                return ExitCode::FAILURE;
            };
            raw::decode_rgb48_file_with_discard(&path, discard_levels, &output)
        }
        (Some("decode-rgb48-gpu-dwt"), Some(path)) => {
            let Some(discard_levels) = args.next().and_then(|value| value.parse::<u8>().ok())
            else {
                eprintln!(
                    "usage: {program} decode-rgb48-gpu-dwt <input.j2k> <discard-levels> <output.rgb48le> [idwt53.comp.spv]"
                );
                return ExitCode::FAILURE;
            };
            let Some(output) = args.next() else {
                eprintln!(
                    "usage: {program} decode-rgb48-gpu-dwt <input.j2k> <discard-levels> <output.rgb48le> [idwt53.comp.spv]"
                );
                return ExitCode::FAILURE;
            };
            let shader_path = args
                .next()
                .unwrap_or_else(|| "target\\shaders\\idwt53_1d.comp.spv".to_string());
            raw::decode_rgb48_gpu_dwt_file(&path, discard_levels, &output, &shader_path)
        }
        (Some("decode-rgb48-gpu"), Some(path)) => {
            let Some(discard_levels) = args.next().and_then(|value| value.parse::<u8>().ok())
            else {
                eprintln!(
                    "usage: {program} decode-rgb48-gpu <input.j2k> <discard-levels> <output.rgb48le> [idwt53.comp.spv] [color.comp.spv]"
                );
                return ExitCode::FAILURE;
            };
            let Some(output) = args.next() else {
                eprintln!(
                    "usage: {program} decode-rgb48-gpu <input.j2k> <discard-levels> <output.rgb48le> [idwt53.comp.spv] [color.comp.spv]"
                );
                return ExitCode::FAILURE;
            };
            let idwt53_shader_path = args
                .next()
                .unwrap_or_else(|| "target\\shaders\\idwt53_1d.comp.spv".to_string());
            let color_shader_path = args
                .next()
                .unwrap_or_else(|| "target\\shaders\\color_convert.comp.spv".to_string());
            raw::decode_rgb48_gpu_file(
                &path,
                discard_levels,
                &output,
                &idwt53_shader_path,
                &color_shader_path,
            )
        }
        (Some("profile-rgb48"), Some(path)) => {
            let Some(discard_levels) = args.next().and_then(|value| value.parse::<u8>().ok())
            else {
                eprintln!(
                    "usage: {program} profile-rgb48 <input.j2k> <discard-levels> <cpu|gpu-dwt|gpu> <output.rgb48le> [idwt53.comp.spv] [color.comp.spv]"
                );
                return ExitCode::FAILURE;
            };
            let Some(mode) = args.next() else {
                eprintln!(
                    "usage: {program} profile-rgb48 <input.j2k> <discard-levels> <cpu|gpu-dwt|gpu> <output.rgb48le> [idwt53.comp.spv] [color.comp.spv]"
                );
                return ExitCode::FAILURE;
            };
            let Some(output) = args.next() else {
                eprintln!(
                    "usage: {program} profile-rgb48 <input.j2k> <discard-levels> <cpu|gpu-dwt|gpu> <output.rgb48le> [idwt53.comp.spv] [color.comp.spv]"
                );
                return ExitCode::FAILURE;
            };
            let idwt53_shader_path = args
                .next()
                .unwrap_or_else(|| "target\\shaders\\idwt53_1d.comp.spv".to_string());
            let color_shader_path = args
                .next()
                .unwrap_or_else(|| "target\\shaders\\color_convert.comp.spv".to_string());
            raw::profile_rgb48_file(
                &path,
                discard_levels,
                &mode,
                &output,
                &idwt53_shader_path,
                &color_shader_path,
            )
        }
        (Some("profile-rgb48-bench"), Some(path)) => {
            let Some(discard_levels) = args.next().and_then(|value| value.parse::<u8>().ok())
            else {
                eprintln!(
                    "usage: {program} profile-rgb48-bench <input.j2k> <discard-levels> <cpu|gpu-dwt|gpu> <iterations> <output.rgb48le> [idwt53.comp.spv] [color.comp.spv]"
                );
                return ExitCode::FAILURE;
            };
            let Some(mode) = args.next() else {
                eprintln!(
                    "usage: {program} profile-rgb48-bench <input.j2k> <discard-levels> <cpu|gpu-dwt|gpu> <iterations> <output.rgb48le> [idwt53.comp.spv] [color.comp.spv]"
                );
                return ExitCode::FAILURE;
            };
            let Some(iterations) = args.next().and_then(|value| value.parse::<usize>().ok()) else {
                eprintln!(
                    "usage: {program} profile-rgb48-bench <input.j2k> <discard-levels> <cpu|gpu-dwt|gpu> <iterations> <output.rgb48le> [idwt53.comp.spv] [color.comp.spv]"
                );
                return ExitCode::FAILURE;
            };
            let Some(output) = args.next() else {
                eprintln!(
                    "usage: {program} profile-rgb48-bench <input.j2k> <discard-levels> <cpu|gpu-dwt|gpu> <iterations> <output.rgb48le> [idwt53.comp.spv] [color.comp.spv]"
                );
                return ExitCode::FAILURE;
            };
            let idwt53_shader_path = args
                .next()
                .unwrap_or_else(|| "target\\shaders\\idwt53_1d.comp.spv".to_string());
            let color_shader_path = args
                .next()
                .unwrap_or_else(|| "target\\shaders\\color_convert.comp.spv".to_string());
            raw::profile_rgb48_bench_file(
                &path,
                discard_levels,
                &mode,
                iterations,
                &output,
                &idwt53_shader_path,
                &color_shader_path,
            )
        }
        (Some("diff-rgb48"), Some(actual)) => {
            let Some(reference) = args.next() else {
                eprintln!("usage: {program} diff-rgb48 <actual.rgb48le> <reference.rgb48le>");
                return ExitCode::FAILURE;
            };
            raw::diff_rgb48_files(&actual, &reference)
        }
        (Some("diff-component-plane"), Some(actual)) => {
            let Some(reference) = args.next() else {
                eprintln!("usage: {program} diff-component-plane <actual.i32le> <reference.i32le>");
                return ExitCode::FAILURE;
            };
            raw::diff_component_plane_files(&actual, &reference)
        }
        (Some("diff-component-rgb48"), Some(actual)) => {
            let Some(reference) = args.next() else {
                eprintln!(
                    "usage: {program} diff-component-rgb48 <actual.i32le> <reference.rgb48le> <component> <precision>"
                );
                return ExitCode::FAILURE;
            };
            let Some(component) = args.next().and_then(|value| value.parse::<usize>().ok()) else {
                eprintln!(
                    "usage: {program} diff-component-rgb48 <actual.i32le> <reference.rgb48le> <component> <precision>"
                );
                return ExitCode::FAILURE;
            };
            let Some(precision) = args.next().and_then(|value| value.parse::<u8>().ok()) else {
                eprintln!(
                    "usage: {program} diff-component-rgb48 <actual.i32le> <reference.rgb48le> <component> <precision>"
                );
                return ExitCode::FAILURE;
            };
            raw::diff_component_plane_rgb48_file(&actual, &reference, component, precision)
        }
        (Some("trace-component53-sample"), Some(path)) => {
            let Some(component) = args.next().and_then(|value| value.parse::<usize>().ok()) else {
                eprintln!(
                    "usage: {program} trace-component53-sample <input.j2k> <component> <discard-levels> <x> <y> [max-deps]"
                );
                return ExitCode::FAILURE;
            };
            let Some(discard_levels) = args.next().and_then(|value| value.parse::<u8>().ok())
            else {
                eprintln!(
                    "usage: {program} trace-component53-sample <input.j2k> <component> <discard-levels> <x> <y> [max-deps]"
                );
                return ExitCode::FAILURE;
            };
            let Some(x) = args.next().and_then(|value| value.parse::<u32>().ok()) else {
                eprintln!(
                    "usage: {program} trace-component53-sample <input.j2k> <component> <discard-levels> <x> <y> [max-deps]"
                );
                return ExitCode::FAILURE;
            };
            let Some(y) = args.next().and_then(|value| value.parse::<u32>().ok()) else {
                eprintln!(
                    "usage: {program} trace-component53-sample <input.j2k> <component> <discard-levels> <x> <y> [max-deps]"
                );
                return ExitCode::FAILURE;
            };
            let max_dependencies = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(96);
            raw::trace_component53_sample_file(
                &path,
                component,
                discard_levels,
                x,
                y,
                max_dependencies,
            )
        }
        (Some("dump-coeff-plane"), Some(path)) => {
            let Some(component) = args.next().and_then(|value| value.parse::<u16>().ok()) else {
                eprintln!(
                    "usage: {program} dump-coeff-plane <input.j2k> <component> <resolution> <LL|HL|LH|HH> <output.i32le>"
                );
                return ExitCode::FAILURE;
            };
            let Some(resolution) = args.next().and_then(|value| value.parse::<u8>().ok()) else {
                eprintln!(
                    "usage: {program} dump-coeff-plane <input.j2k> <component> <resolution> <LL|HL|LH|HH> <output.i32le>"
                );
                return ExitCode::FAILURE;
            };
            let Some(subband) = args.next() else {
                eprintln!(
                    "usage: {program} dump-coeff-plane <input.j2k> <component> <resolution> <LL|HL|LH|HH> <output.i32le>"
                );
                return ExitCode::FAILURE;
            };
            let Some(output) = args.next() else {
                eprintln!(
                    "usage: {program} dump-coeff-plane <input.j2k> <component> <resolution> <LL|HL|LH|HH> <output.i32le>"
                );
                return ExitCode::FAILURE;
            };
            raw::dump_coeff_plane_file(&path, component, resolution, &subband, &output)
        }
        (Some("dump-component-plane"), Some(path)) => {
            let Some(component) = args.next().and_then(|value| value.parse::<usize>().ok()) else {
                eprintln!(
                    "usage: {program} dump-component-plane <input.j2k> <component> <discard-levels> <output.i32le>"
                );
                return ExitCode::FAILURE;
            };
            let Some(discard_levels) = args.next().and_then(|value| value.parse::<u8>().ok())
            else {
                eprintln!(
                    "usage: {program} dump-component-plane <input.j2k> <component> <discard-levels> <output.i32le>"
                );
                return ExitCode::FAILURE;
            };
            let Some(output) = args.next() else {
                eprintln!(
                    "usage: {program} dump-component-plane <input.j2k> <component> <discard-levels> <output.i32le>"
                );
                return ExitCode::FAILURE;
            };
            raw::dump_component_plane_file(&path, component, discard_levels, &output)
        }
        (Some("vk-info"), _) => vulkan::print_vulkan_info(),
        (Some("vk-smoke"), shader_path) => vulkan::run_vulkan_smoke(
            shader_path
                .as_deref()
                .unwrap_or("target\\shaders\\dequant.comp.spv"),
        ),
        (Some("vk-color-smoke"), shader_path) => vulkan::run_vulkan_color_smoke(
            shader_path
                .as_deref()
                .unwrap_or("target\\shaders\\color_convert.comp.spv"),
        ),
        (Some("vk-idwt53-smoke"), shader_path) => vulkan::run_vulkan_idwt53_smoke(
            shader_path
                .as_deref()
                .unwrap_or("target\\shaders\\idwt53_1d.comp.spv"),
        ),
        (Some("vk-idwt97-smoke"), shader_path) => vulkan::run_vulkan_idwt97_smoke(
            shader_path
                .as_deref()
                .unwrap_or("target\\shaders\\idwt97_1d.comp.spv"),
        ),
        (Some("vk-irreversible-smoke"), idwt97_shader_path) => {
            let idwt97_shader_path = idwt97_shader_path
                .unwrap_or_else(|| "target\\shaders\\idwt97_1d.comp.spv".to_string());
            let color_shader_path = args
                .next()
                .unwrap_or_else(|| "target\\shaders\\color_convert.comp.spv".to_string());
            vulkan::run_vulkan_irreversible_smoke(&idwt97_shader_path, &color_shader_path)
        }
        _ => {
            eprintln!("usage:");
            eprintln!("  {program} dump <input.j2k>");
            eprintln!("  {program} scan <container-or-bytestream> [max_hits]");
            eprintln!("  {program} scan-summary <container-or-bytestream>");
            eprintln!("  {program} extract <container-or-bytestream> <index> <output.j2k>");
            eprintln!("  {program} tier1-smoke <input.j2k> [max_codeblocks]");
            eprintln!("  {program} tier1-subband-stats <input.j2k>");
            eprintln!("  {program} tier1-gpu-plan <input.j2k>");
            eprintln!(
                "  {program} tier1-codeblock-trace <input.j2k> <component> <resolution> <LL|HL|LH|HH> <codeblock-x> <codeblock-y>"
            );
            eprintln!("  {program} reconstruct-smoke <input.j2k>");
            eprintln!("  {program} decode-rgb48 <input.j2k> <output.rgb48le>");
            eprintln!(
                "  {program} decode-rgb48-lowres <input.j2k> <discard-levels> <output.rgb48le>"
            );
            eprintln!(
                "  {program} decode-rgb48-gpu-dwt <input.j2k> <discard-levels> <output.rgb48le> [idwt53.comp.spv]"
            );
            eprintln!(
                "  {program} decode-rgb48-gpu <input.j2k> <discard-levels> <output.rgb48le> [idwt53.comp.spv] [color.comp.spv]"
            );
            eprintln!(
                "  {program} profile-rgb48 <input.j2k> <discard-levels> <cpu|gpu-dwt|gpu> <output.rgb48le> [idwt53.comp.spv] [color.comp.spv]"
            );
            eprintln!(
                "  {program} profile-rgb48-bench <input.j2k> <discard-levels> <cpu|gpu-dwt|gpu> <iterations> <output.rgb48le> [idwt53.comp.spv] [color.comp.spv]"
            );
            eprintln!("  {program} diff-rgb48 <actual.rgb48le> <reference.rgb48le>");
            eprintln!("  {program} diff-component-plane <actual.i32le> <reference.i32le>");
            eprintln!(
                "  {program} diff-component-rgb48 <actual.i32le> <reference.rgb48le> <component> <precision>"
            );
            eprintln!(
                "  {program} trace-component53-sample <input.j2k> <component> <discard-levels> <x> <y> [max-deps]"
            );
            eprintln!(
                "  {program} dump-coeff-plane <input.j2k> <component> <resolution> <LL|HL|LH|HH> <output.i32le>"
            );
            eprintln!(
                "  {program} dump-component-plane <input.j2k> <component> <discard-levels> <output.i32le>"
            );
            eprintln!("  {program} vk-info");
            eprintln!("  {program} vk-smoke [dequant.comp.spv]");
            eprintln!("  {program} vk-color-smoke [color_convert.comp.spv]");
            eprintln!("  {program} vk-idwt53-smoke [idwt53_1d.comp.spv]");
            eprintln!("  {program} vk-idwt97-smoke [idwt97_1d.comp.spv]");
            eprintln!(
                "  {program} vk-irreversible-smoke [idwt97_1d.comp.spv] [color_convert.comp.spv]"
            );
            ExitCode::FAILURE
        }
    }
}

fn tier1_subband_stats(path: &str) -> ExitCode {
    let data = match fs::read(path) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {path}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let codestream = match Codestream::parse(&data) {
        Ok(codestream) => codestream,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    };
    let Some(cod) = &codestream.cod else {
        eprintln!("tier1 subband stats: missing COD marker");
        return ExitCode::FAILURE;
    };
    let layouts = match codestream.tile_layouts() {
        Ok(layouts) => layouts,
        Err(err) => {
            eprintln!("tier1 subband stats: tile layout unavailable ({err})");
            return ExitCode::FAILURE;
        }
    };

    println!("tier1 subband stats: {path}");
    for tile_part in &codestream.tile_parts {
        let Some(layout) = layouts
            .iter()
            .find(|layout| layout.tile_index == tile_part.tile_index)
        else {
            eprintln!(
                "tier1 subband stats: missing layout for tile {}",
                tile_part.tile_index
            );
            return ExitCode::FAILURE;
        };
        let Some(tile_data) = data.get(tile_part.data_start..tile_part.data_end) else {
            eprintln!(
                "tier1 subband stats: tile {} payload outside input",
                tile_part.tile_index
            );
            return ExitCode::FAILURE;
        };
        let packet_summary = match parse_lrcp_tile_packets(tile_data, layout, cod.layers) {
            Ok(summary) => summary,
            Err(err) => {
                eprintln!(
                    "tier1 subband stats: packet parse failed for tile {}: {err}",
                    tile_part.tile_index
                );
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
                eprintln!(
                    "tier1 subband stats: work plan failed for tile {}: {err}",
                    tile_part.tile_index
                );
                return ExitCode::FAILURE;
            }
        };
        let bound = match work_plan.bind_segment_data(tile_data) {
            Ok(bound) => bound,
            Err(err) => {
                eprintln!(
                    "tier1 subband stats: segment bind failed for tile {}: {err}",
                    tile_part.tile_index
                );
                return ExitCode::FAILURE;
            }
        };
        let mut stats = Vec::<SubbandWorkStats>::new();
        for codeblock in &bound.codeblocks {
            let stat = match stats.iter_mut().find(|stat| {
                stat.component == codeblock.id.component
                    && stat.resolution == codeblock.id.resolution
                    && stat.subband == codeblock.id.subband
            }) {
                Some(stat) => stat,
                None => {
                    stats.push(SubbandWorkStats::new(
                        codeblock.id.component,
                        codeblock.id.resolution,
                        codeblock.id.subband,
                    ));
                    stats.last_mut().expect("just pushed")
                }
            };
            stat.add(codeblock);
            let decoded = match codeblock.decode_arithmetic_passes_with_reconstruction(
                cod.codeblock_style_flags(),
                coefficient_reconstruction_for_transform(cod.transform),
            ) {
                Ok(decoded) => decoded,
                Err(err) => {
                    eprintln!(
                        "tier1 subband stats: codeblock {:?} failed: {err}",
                        codeblock.id
                    );
                    return ExitCode::FAILURE;
                }
            };
            stat.add_decoded(&decoded);
        }
        stats.sort_by_key(|stat| (stat.component, stat.resolution, stat.subband));
        println!("  tile {}:", tile_part.tile_index);
        for stat in stats {
            println!(
                "    C{} R{} {}: {} cblk, {} bytes, consumed {}, {} passes, zbp {}..{}, mbp {}..{}, size {}..{} x {}..{}, {} nonzero",
                stat.component,
                stat.resolution,
                stat.subband,
                stat.codeblocks,
                stat.bytes,
                stat.consumed_bytes,
                stat.passes,
                stat.min_zero_bit_planes,
                stat.max_zero_bit_planes,
                stat.min_magnitude_bitplanes,
                stat.max_magnitude_bitplanes,
                stat.min_width,
                stat.max_width,
                stat.min_height,
                stat.max_height,
                stat.nonzero_coefficients
            );
        }
    }

    ExitCode::SUCCESS
}

fn tier1_gpu_plan(path: &str) -> ExitCode {
    let data = match fs::read(path) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("tier1 gpu plan: failed to read {path}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let codestream = match Codestream::parse(&data) {
        Ok(codestream) => codestream,
        Err(err) => {
            eprintln!("tier1 gpu plan: {err}");
            return ExitCode::FAILURE;
        }
    };
    let Some(cod) = &codestream.cod else {
        eprintln!("tier1 gpu plan: missing COD marker");
        return ExitCode::FAILURE;
    };
    let layouts = match codestream.tile_layouts() {
        Ok(layouts) => layouts,
        Err(err) => {
            eprintln!("tier1 gpu plan: tile layout unavailable ({err})");
            return ExitCode::FAILURE;
        }
    };

    println!("tier1 gpu plan: {path}");
    for tile_part in &codestream.tile_parts {
        let Some(layout) = layouts
            .iter()
            .find(|layout| layout.tile_index == tile_part.tile_index)
        else {
            eprintln!(
                "tier1 gpu plan: missing layout for tile {}",
                tile_part.tile_index
            );
            return ExitCode::FAILURE;
        };
        let Some(tile_data) = data.get(tile_part.data_start..tile_part.data_end) else {
            eprintln!(
                "tier1 gpu plan: tile {} payload outside input",
                tile_part.tile_index
            );
            return ExitCode::FAILURE;
        };
        let packet_summary = match parse_lrcp_tile_packets(tile_data, layout, cod.layers) {
            Ok(summary) => summary,
            Err(err) => {
                eprintln!(
                    "tier1 gpu plan: packet parse failed for tile {}: {err}",
                    tile_part.tile_index
                );
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
                eprintln!(
                    "tier1 gpu plan: work plan failed for tile {}: {err}",
                    tile_part.tile_index
                );
                return ExitCode::FAILURE;
            }
        };
        let bound = match work_plan.bind_segment_data(tile_data) {
            Ok(bound) => bound,
            Err(err) => {
                eprintln!(
                    "tier1 gpu plan: segment bind failed for tile {}: {err}",
                    tile_part.tile_index
                );
                return ExitCode::FAILURE;
            }
        };
        let plan = match GpuTier1Plan::from_bound_work_plan(&bound, cod.codeblock_style_flags()) {
            Ok(plan) => plan,
            Err(err) => {
                eprintln!(
                    "tier1 gpu plan: flatten failed for tile {}: {err}",
                    tile_part.tile_index
                );
                return ExitCode::FAILURE;
            }
        };
        println!(
            "  tile {}: {} cblk, {} compressed bytes, {} coeff, {} passes ({} mq, {} raw)",
            tile_part.tile_index,
            plan.codeblocks.len(),
            plan.compressed_bytes.len(),
            plan.output_coefficients,
            plan.total_passes,
            plan.arithmetic_passes,
            plan.raw_bypass_passes
        );
        if let Some(first) = plan.codeblocks.first() {
            println!(
                "    first: C{} R{} {} cblk {},{} {}x{}, bytes {}..{}, coeff {}..{}, passes {}..{}",
                first.component,
                first.resolution,
                first.subband,
                first.codeblock_x,
                first.codeblock_y,
                first.width,
                first.height,
                first.byte_offset,
                first.byte_offset + first.byte_len,
                first.coefficient_offset,
                first.coefficient_offset + first.coefficient_count,
                first.pass_offset,
                first.pass_offset + first.pass_count
            );
        }
        let shader_path = "target\\shaders\\tier1_plan_smoke.comp.spv";
        if std::path::Path::new(shader_path).exists() {
            let records = plan.codeblock_records();
            let compressed_words = plan.compressed_u32_words();
            let smoke = match vk_j2k_dec::vulkan::run_tier1_plan_smoke(
                shader_path,
                &records,
                &compressed_words,
            ) {
                Ok(smoke) => smoke,
                Err(err) => {
                    eprintln!("tier1 gpu plan: Vulkan smoke failed: {err}");
                    return ExitCode::FAILURE;
                }
            };
            if let Some(first) = plan.codeblocks.first() {
                let expected = [
                    first.byte_offset + first.byte_len,
                    first.coefficient_offset + first.coefficient_count,
                    first.pass_offset + first.pass_count,
                    plan.codeblock_byte_checksum(0).unwrap_or(0),
                ];
                if smoke.output.get(0..4) != Some(expected.as_slice()) {
                    eprintln!(
                        "tier1 gpu plan: smoke mismatch: got {:?}, expected {:?}",
                        smoke.output.get(0..4).unwrap_or(&[]),
                        expected
                    );
                    return ExitCode::FAILURE;
                }
            }
            println!(
                "    vulkan smoke: device {}, queue {}, {} records, {} byte words",
                smoke.device_name,
                smoke.queue_family_index,
                records.len(),
                compressed_words.len()
            );
            let mq_shader_path = "target\\shaders\\tier1_mq_smoke.comp.spv";
            if std::path::Path::new(mq_shader_path).exists() {
                let mq_smoke = match vk_j2k_dec::vulkan::run_tier1_mq_smoke(
                    mq_shader_path,
                    &records,
                    &compressed_words,
                    0,
                    32,
                    0,
                ) {
                    Ok(smoke) => smoke,
                    Err(err) => {
                        eprintln!("tier1 gpu plan: Vulkan MQ smoke failed: {err}");
                        return ExitCode::FAILURE;
                    }
                };
                let expected = match tier1_mq_smoke_expected(&plan, 0, 32, 0) {
                    Ok(expected) => expected,
                    Err(err) => {
                        eprintln!("tier1 gpu plan: CPU MQ smoke failed: {err}");
                        return ExitCode::FAILURE;
                    }
                };
                if mq_smoke.output != expected {
                    eprintln!(
                        "tier1 gpu plan: MQ smoke mismatch: got {:?}, expected {:?}",
                        mq_smoke.output, expected
                    );
                    return ExitCode::FAILURE;
                }
                println!(
                    "    vulkan mq smoke: device {}, queue {}, cblk 0, context 0, 32 bits 0x{:08x}",
                    mq_smoke.device_name, mq_smoke.queue_family_index, mq_smoke.output[0]
                );
                let cleanup_shader_path = "target\\shaders\\tier1_cleanup_smoke.comp.spv";
                if std::path::Path::new(cleanup_shader_path).exists() {
                    let cleanup_smoke = match vk_j2k_dec::vulkan::run_tier1_cleanup_smoke(
                        cleanup_shader_path,
                        &records,
                        &compressed_words,
                        0,
                    ) {
                        Ok(smoke) => smoke,
                        Err(err) => {
                            eprintln!("tier1 gpu plan: Vulkan cleanup smoke failed: {err}");
                            return ExitCode::FAILURE;
                        }
                    };
                    let Some(first_bound) = bound.codeblocks.first() else {
                        eprintln!("tier1 gpu plan: CPU cleanup smoke missing first codeblock");
                        return ExitCode::FAILURE;
                    };
                    let expected = match tier1_cleanup_smoke_expected(
                        first_bound,
                        coefficient_reconstruction_for_transform(cod.transform),
                    ) {
                        Ok(expected) => expected,
                        Err(err) => {
                            eprintln!("tier1 gpu plan: CPU cleanup smoke failed: {err}");
                            return ExitCode::FAILURE;
                        }
                    };
                    if cleanup_smoke.output != expected {
                        eprintln!(
                            "tier1 gpu plan: cleanup smoke mismatch: got {:?}, expected {:?}",
                            cleanup_smoke.output, expected
                        );
                        return ExitCode::FAILURE;
                    }
                    println!(
                        "    vulkan cleanup smoke: device {}, queue {}, cblk 0, nonzero {}, checksum 0x{:08x}",
                        cleanup_smoke.device_name,
                        cleanup_smoke.queue_family_index,
                        cleanup_smoke.output[0],
                        cleanup_smoke.output[1]
                    );
                } else {
                    println!("    vulkan cleanup smoke: skipped, missing {cleanup_shader_path}");
                }
            } else {
                println!("    vulkan mq smoke: skipped, missing {mq_shader_path}");
            }
        } else {
            println!("    vulkan smoke: skipped, missing {shader_path}");
        }
    }
    ExitCode::SUCCESS
}

fn tier1_mq_smoke_expected(
    plan: &GpuTier1Plan,
    codeblock_index: usize,
    bit_count: u32,
    context_index: usize,
) -> Result<Vec<u32>, String> {
    let codeblock = plan
        .codeblocks
        .get(codeblock_index)
        .ok_or_else(|| format!("missing codeblock {codeblock_index}"))?;
    let start = codeblock.byte_offset as usize;
    let end = start
        .checked_add(codeblock.byte_len as usize)
        .ok_or_else(|| "codeblock byte range overflow".to_string())?;
    let bytes = plan
        .compressed_bytes
        .get(start..end)
        .ok_or_else(|| format!("codeblock {codeblock_index} byte range outside stream"))?;
    let mut mq = MqDecoder::with_data(bytes);
    let mut bits = 0u32;
    for _ in 0..bit_count.min(32) {
        let bit = mq
            .decode_bit(context_index)
            .map_err(|err| err.to_string())?;
        bits = (bits << 1) | if bit { 1 } else { 0 };
    }
    let registers = mq.input_registers();
    let context = mq
        .context(context_index)
        .ok_or_else(|| format!("missing MQ context {context_index}"))?;
    Ok(vec![
        bits,
        u32::try_from(registers.byte_pos).map_err(|_| "MQ byte_pos exceeds u32".to_string())?,
        registers.a,
        registers.c,
        registers.ct,
        u32::from(context.state) | ((if context.mps { 1 } else { 0 }) << 8),
    ])
}

fn tier1_cleanup_smoke_expected(
    codeblock: &BoundCompressedCodeblock<'_>,
    reconstruction: CoefficientReconstruction,
) -> Result<Vec<u32>, String> {
    let pass = tier1_pass_for_index(codeblock.magnitude_bitplanes, codeblock.zero_bit_planes, 0);
    let mut decoded = DecodedCodeblock::new_bound_with_reconstruction(codeblock, reconstruction)
        .map_err(|err| err.to_string())?;
    let mut mq = if codeblock.segments.len() == 1 {
        MqDecoder::with_data(codeblock.segments[0].data)
    } else {
        let mut data = Vec::with_capacity(codeblock.total_bytes());
        for segment in &codeblock.segments {
            data.extend_from_slice(segment.data);
        }
        MqDecoder::with_owned_data(data)
    };
    let summary = decoded
        .decode_pass(pass, &mut mq)
        .map_err(|err| err.to_string())?;
    let mut nonzero = 0u32;
    let mut checksum = 0u32;
    for coefficient in &decoded.coefficients {
        if *coefficient != 0 {
            nonzero += 1;
            checksum = checksum.wrapping_add(*coefficient as u32);
        }
    }
    Ok(vec![
        nonzero,
        checksum,
        u32::try_from(mq.consumed_bytes()).map_err(|_| "MQ byte_pos exceeds u32".to_string())?,
        summary.significance_bits,
        summary.sign_bits,
        summary.run_length_bits,
        summary.uniform_bits,
        summary.became_significant,
        summary.visited,
    ])
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SubbandWorkStats {
    component: u16,
    resolution: u8,
    subband: vk_j2k_dec::codestream::SubbandKind,
    codeblocks: usize,
    bytes: usize,
    consumed_bytes: usize,
    passes: u32,
    nonzero_coefficients: usize,
    min_zero_bit_planes: u32,
    max_zero_bit_planes: u32,
    min_magnitude_bitplanes: u32,
    max_magnitude_bitplanes: u32,
    min_width: u32,
    max_width: u32,
    min_height: u32,
    max_height: u32,
}

impl SubbandWorkStats {
    fn new(component: u16, resolution: u8, subband: vk_j2k_dec::codestream::SubbandKind) -> Self {
        Self {
            component,
            resolution,
            subband,
            codeblocks: 0,
            bytes: 0,
            consumed_bytes: 0,
            passes: 0,
            nonzero_coefficients: 0,
            min_zero_bit_planes: u32::MAX,
            max_zero_bit_planes: 0,
            min_magnitude_bitplanes: u32::MAX,
            max_magnitude_bitplanes: 0,
            min_width: u32::MAX,
            max_width: 0,
            min_height: u32::MAX,
            max_height: 0,
        }
    }

    fn add(&mut self, codeblock: &vk_j2k_dec::tier1::BoundCompressedCodeblock<'_>) {
        self.codeblocks += 1;
        self.bytes += codeblock.total_bytes();
        self.passes += codeblock.total_coding_passes();
        self.min_zero_bit_planes = self.min_zero_bit_planes.min(codeblock.zero_bit_planes);
        self.max_zero_bit_planes = self.max_zero_bit_planes.max(codeblock.zero_bit_planes);
        self.min_magnitude_bitplanes = self
            .min_magnitude_bitplanes
            .min(codeblock.magnitude_bitplanes);
        self.max_magnitude_bitplanes = self
            .max_magnitude_bitplanes
            .max(codeblock.magnitude_bitplanes);
        self.min_width = self.min_width.min(codeblock.width);
        self.max_width = self.max_width.max(codeblock.width);
        self.min_height = self.min_height.min(codeblock.height);
        self.max_height = self.max_height.max(codeblock.height);
    }

    fn add_decoded(&mut self, decoded: &vk_j2k_dec::tier1::DecodedCodeblock) {
        self.consumed_bytes += decoded.decoder_bytes_consumed;
        self.nonzero_coefficients += decoded
            .coefficients
            .iter()
            .filter(|coefficient| **coefficient != 0)
            .count();
    }
}

fn reconstruct_smoke(path: &str) -> ExitCode {
    let data = match fs::read(path) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {path}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let codestream = match Codestream::parse(&data) {
        Ok(codestream) => codestream,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    };
    let Some(cod) = &codestream.cod else {
        eprintln!("reconstruct smoke: missing COD marker");
        return ExitCode::FAILURE;
    };
    let layouts = match codestream.tile_layouts() {
        Ok(layouts) => layouts,
        Err(err) => {
            eprintln!("reconstruct smoke: tile layout unavailable ({err})");
            return ExitCode::FAILURE;
        }
    };

    println!("reconstruct smoke: {path}");
    let mut total_codeblocks = 0usize;
    let mut total_passes = 0u32;
    let mut total_nonzero = 0usize;
    for tile_part in &codestream.tile_parts {
        let Some(layout) = layouts
            .iter()
            .find(|layout| layout.tile_index == tile_part.tile_index)
        else {
            eprintln!(
                "reconstruct smoke: missing layout for tile {}",
                tile_part.tile_index
            );
            return ExitCode::FAILURE;
        };
        let Some(tile_data) = data.get(tile_part.data_start..tile_part.data_end) else {
            eprintln!(
                "reconstruct smoke: tile {} payload outside input",
                tile_part.tile_index
            );
            return ExitCode::FAILURE;
        };
        let packet_summary = match parse_lrcp_tile_packets(tile_data, layout, cod.layers) {
            Ok(summary) => summary,
            Err(err) => {
                eprintln!(
                    "reconstruct smoke: packet parse failed for tile {}: {err}",
                    tile_part.tile_index
                );
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
                eprintln!(
                    "reconstruct smoke: work plan failed for tile {}: {err}",
                    tile_part.tile_index
                );
                return ExitCode::FAILURE;
            }
        };
        let bound = match work_plan.bind_segment_data(tile_data) {
            Ok(bound) => bound,
            Err(err) => {
                eprintln!(
                    "reconstruct smoke: segment bind failed for tile {}: {err}",
                    tile_part.tile_index
                );
                return ExitCode::FAILURE;
            }
        };
        let planes = match TileCoefficientPlanes::decode_arithmetic_with_reconstruction(
            &layout.components,
            &bound,
            cod.codeblock_style_flags(),
            coefficient_reconstruction_for_transform(cod.transform),
        ) {
            Ok(planes) => planes,
            Err(err) => {
                eprintln!(
                    "reconstruct smoke: tile {} coefficient plane build failed: {err}",
                    tile_part.tile_index
                );
                return ExitCode::FAILURE;
            }
        };
        println!(
            "  tile {}: {} components, {} codeblocks, {} passes, {} nonzero coefficients",
            tile_part.tile_index,
            planes.components.len(),
            planes.decoded_codeblocks,
            planes.decoded_passes,
            planes.nonzero_coefficients
        );
        for component in &planes.components {
            let component_nonzero: usize = component
                .planes
                .iter()
                .map(|plane| {
                    plane
                        .plane
                        .coefficients
                        .iter()
                        .filter(|coefficient| **coefficient != 0)
                        .count()
                })
                .sum();
            println!(
                "    component {}: {} subband planes, {} nonzero coefficients",
                component.component,
                component.planes.len(),
                component_nonzero
            );
        }
        for component in &planes.components {
            let plane = match component.inverse_5_3() {
                Ok(plane) => plane,
                Err(err) => {
                    eprintln!(
                        "reconstruct smoke: component {} inverse 5/3 failed: {err}",
                        component.component
                    );
                    return ExitCode::FAILURE;
                }
            };
            let nonzero = plane.samples.iter().filter(|sample| **sample != 0).count();
            println!(
                "    component {} reconstructed: {}x{}, {} nonzero samples",
                component.component, plane.width, plane.height, nonzero
            );
        }
        total_codeblocks += planes.decoded_codeblocks;
        total_passes += planes.decoded_passes;
        total_nonzero += planes.nonzero_coefficients;
    }

    println!(
        "reconstruct smoke summary: {} codeblocks, {} passes, {} nonzero coefficients",
        total_codeblocks, total_passes, total_nonzero
    );
    ExitCode::SUCCESS
}

fn tier1_smoke(path: &str, max_codeblocks: usize) -> ExitCode {
    let data = match fs::read(path) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {path}: {err}");
            return ExitCode::FAILURE;
        }
    };

    let codestream = match Codestream::parse(&data) {
        Ok(codestream) => codestream,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    };
    let Some(cod) = &codestream.cod else {
        eprintln!("tier1 smoke: missing COD marker");
        return ExitCode::FAILURE;
    };
    let Ok(layouts) = codestream.tile_layouts() else {
        eprintln!("tier1 smoke: tile layout unavailable");
        return ExitCode::FAILURE;
    };

    let mut decoded_codeblocks = 0usize;
    let mut decoded_passes = 0u32;
    let mut nonzero_coefficients = 0usize;
    let mut source_bytes = 0usize;
    let mut consumed_bytes = 0usize;
    println!("tier1 smoke: {path}");
    for tile_part in &codestream.tile_parts {
        let Some(layout) = layouts
            .iter()
            .find(|layout| layout.tile_index == tile_part.tile_index)
        else {
            eprintln!(
                "tier1 smoke: missing layout for tile {}",
                tile_part.tile_index
            );
            return ExitCode::FAILURE;
        };
        let Some(tile_data) = data.get(tile_part.data_start..tile_part.data_end) else {
            eprintln!(
                "tier1 smoke: tile {} payload outside input",
                tile_part.tile_index
            );
            return ExitCode::FAILURE;
        };
        let packet_summary = match parse_lrcp_tile_packets(tile_data, layout, cod.layers) {
            Ok(summary) => summary,
            Err(err) => {
                eprintln!(
                    "tier1 smoke: packet parse failed for tile {}: {err}",
                    tile_part.tile_index
                );
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
                eprintln!(
                    "tier1 smoke: work plan failed for tile {}: {err}",
                    tile_part.tile_index
                );
                return ExitCode::FAILURE;
            }
        };
        let bound = match work_plan.bind_segment_data(tile_data) {
            Ok(bound) => bound,
            Err(err) => {
                eprintln!(
                    "tier1 smoke: segment bind failed for tile {}: {err}",
                    tile_part.tile_index
                );
                return ExitCode::FAILURE;
            }
        };

        println!(
            "  tile {}: {} codeblocks, decoding up to {}",
            tile_part.tile_index,
            bound.codeblock_count(),
            max_codeblocks.saturating_sub(decoded_codeblocks)
        );
        for codeblock in &bound.codeblocks {
            if decoded_codeblocks >= max_codeblocks {
                break;
            }
            let decoded = match codeblock.decode_arithmetic_passes_with_reconstruction(
                cod.codeblock_style_flags(),
                coefficient_reconstruction_for_transform(cod.transform),
            ) {
                Ok(decoded) => decoded,
                Err(err) => {
                    eprintln!("tier1 smoke: codeblock {:?} failed: {err}", codeblock.id);
                    return ExitCode::FAILURE;
                }
            };
            let nonzero = decoded
                .coefficients
                .iter()
                .filter(|coefficient| **coefficient != 0)
                .count();
            println!(
                "    {:?}: {} passes, {} significant, {} nonzero, consumed {}/{} bytes",
                decoded.id,
                decoded.decoded_passes,
                decoded.state.significant_count(),
                nonzero,
                decoded.decoder_bytes_consumed,
                decoded.source_bytes
            );
            decoded_codeblocks += 1;
            decoded_passes += decoded.decoded_passes;
            nonzero_coefficients += nonzero;
            source_bytes += decoded.source_bytes;
            consumed_bytes += decoded.decoder_bytes_consumed;
        }
    }

    println!(
        "tier1 smoke summary: {} codeblocks, {} passes, {} nonzero coefficients, consumed {}/{} bytes",
        decoded_codeblocks, decoded_passes, nonzero_coefficients, consumed_bytes, source_bytes
    );
    ExitCode::SUCCESS
}

fn coefficient_reconstruction_for_transform(
    transform: WaveletTransform,
) -> CoefficientReconstruction {
    match transform {
        WaveletTransform::Reversible53 => CoefficientReconstruction::Exact,
        WaveletTransform::Irreversible97 | WaveletTransform::Unknown(_) => {
            CoefficientReconstruction::Midpoint
        }
    }
}

fn dump_codestream(path: &str) -> ExitCode {
    let data = match fs::read(path) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {path}: {err}");
            return ExitCode::FAILURE;
        }
    };

    let codestream = match Codestream::parse(&data) {
        Ok(codestream) => codestream,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    };

    println!("file: {path}");

    let profile = codestream.first_profile_report();
    if profile.is_supported() {
        println!("first profile: supported");
    } else {
        println!("first profile: unsupported");
        for issue in &profile.issues {
            println!("  - {}", issue.message);
        }
    }

    if let Some(siz) = &codestream.siz {
        println!("image: {}x{}", siz.width, siz.height);
        println!(
            "origin: {}, {}  tile: {}x{} at {}, {}",
            siz.x_origin,
            siz.y_origin,
            siz.tile_width,
            siz.tile_height,
            siz.tile_x_origin,
            siz.tile_y_origin
        );
        println!("components: {}", siz.components.len());
        for (index, component) in siz.components.iter().enumerate() {
            let signed = if component.is_signed {
                "signed"
            } else {
                "unsigned"
            };
            println!(
                "  component {index}: {}-bit {signed}, subsampling {}x{}",
                component.precision, component.x_subsampling, component.y_subsampling
            );
        }
    }

    if let Some(cod) = &codestream.cod {
        println!("coding style: 0x{:02x}", cod.coding_style);
        println!("progression: {}", cod.progression_order);
        println!("layers: {}", cod.layers);
        println!("mct: {}", cod.multiple_component_transform);
        println!("decomposition levels: {}", cod.decomposition_levels);
        println!(
            "codeblock: {}x{}",
            cod.codeblock_width, cod.codeblock_height
        );
        println!("codeblock style: {}", cod.codeblock_style_flags());
        println!("transform: {}", cod.transform);
        if !cod.precinct_sizes.is_empty() {
            println!("custom precinct bytes: {}", cod.precinct_sizes.len());
        }
    }

    if let Some(qcd) = &codestream.qcd {
        println!("quantization: {}, guard bits {}", qcd.style, qcd.guard_bits);
        println!("quantization bytes: {}", qcd.raw_steps.len());
        match qcd.steps() {
            Ok(steps) => println!("quantization steps: {}", steps.len()),
            Err(err) => println!("quantization steps: unavailable ({err})"),
        }
    }
    if !codestream.qccs.is_empty() {
        println!(
            "component quantization overrides: {}",
            codestream.qccs.len()
        );
        for qcc in &codestream.qccs {
            match qcc.quantization.steps() {
                Ok(steps) => println!(
                    "  component {}: {}, guard bits {}, {} step(s)",
                    qcc.component,
                    qcc.quantization.style,
                    qcc.quantization.guard_bits,
                    steps.len()
                ),
                Err(err) => println!(
                    "  component {}: {}, guard bits {}, steps unavailable ({err})",
                    qcc.component, qcc.quantization.style, qcc.quantization.guard_bits
                ),
            }
        }
    }

    match GpuPipelinePlan::from_codestream(&codestream) {
        Ok(plan) => println!(
            "gpu plan: {} samples, color {}, shader mode {}, output max {}, level shift {}, qcd steps {}, qcc overrides {}",
            plan.sample_count,
            plan.color_transform,
            plan.color_transform.shader_value(),
            plan.output_max,
            plan.level_shift,
            plan.qcd_steps.len(),
            plan.qcc_steps.len()
        ),
        Err(err) => println!("gpu plan: unavailable ({err})"),
    }
    match GpuDwtDispatchPlan::from_codestream(&codestream) {
        Ok(plan) => {
            println!(
                "gpu dwt plan: {}, {} tiles, {} components, {} stages, {} workgroups, {} output samples touched",
                plan.transform,
                plan.tile_count,
                plan.component_count,
                plan.stages.len(),
                plan.total_workgroups(),
                plan.total_output_samples()
            );
            for stage in plan.stages.iter().take(6) {
                println!(
                    "  gpu dwt stage: tile {} component {} r{} {}, low {}, high {}, lines {}, output {}, groups {}x{}, output stride {}/{}",
                    stage.tile_index,
                    stage.component,
                    stage.resolution,
                    stage.pass,
                    stage.low_count,
                    stage.high_count,
                    stage.line_count,
                    stage.output_count,
                    stage.groups_x,
                    stage.groups_y,
                    stage.output_line_stride,
                    stage.output_sample_stride
                );
            }
            if plan.stages.len() > 6 {
                println!("  ... {} more dwt stages", plan.stages.len() - 6);
            }
        }
        Err(err) => println!("gpu dwt plan: unavailable ({err})"),
    }

    if !codestream.comments.is_empty() {
        println!("comments: {}", codestream.comments.len());
        for comment in &codestream.comments {
            match comment.text() {
                Some(text) => println!("  registration {}: {text}", comment.registration),
                None => println!(
                    "  registration {}: {} bytes",
                    comment.registration,
                    comment.bytes.len()
                ),
            }
        }
    }

    println!("tile parts: {}", codestream.tile_parts.len());
    for tile_part in &codestream.tile_parts {
        println!(
            "  tile {} part {}/{} length {}, data 0x{:08x}..0x{:08x} ({} bytes)",
            tile_part.tile_index,
            tile_part.tile_part_index,
            tile_part.tile_part_count,
            tile_part.tile_part_length,
            tile_part.data_start,
            tile_part.data_end,
            tile_part.data_end.saturating_sub(tile_part.data_start)
        );
    }

    print_tile_layouts(&codestream);
    print_packet_parse_summary(&codestream, &data);

    println!("markers:");
    for record in &codestream.markers {
        println!("  0x{:08x}: {}", record.offset, record.marker);
    }

    ExitCode::SUCCESS
}

fn print_packet_parse_summary(codestream: &Codestream, data: &[u8]) {
    let Some(cod) = &codestream.cod else {
        return;
    };
    let Ok(layouts) = codestream.tile_layouts() else {
        return;
    };

    for tile_part in &codestream.tile_parts {
        let Some(layout) = layouts
            .iter()
            .find(|layout| layout.tile_index == tile_part.tile_index)
        else {
            println!(
                "packet parse tile {}: unavailable (missing tile layout)",
                tile_part.tile_index
            );
            continue;
        };
        let Some(tile_data) = data.get(tile_part.data_start..tile_part.data_end) else {
            println!(
                "packet parse tile {}: unavailable (tile-part payload outside input)",
                tile_part.tile_index
            );
            continue;
        };

        match parse_lrcp_tile_packets(tile_data, layout, cod.layers) {
            Ok(summary) => {
                let status = if summary.consumed_bytes == tile_data.len() {
                    "complete"
                } else {
                    "partial"
                };
                println!(
                    "packet parse tile {}: {status}, {} packets, {} included codeblocks, header {} bytes, body {} bytes, consumed {} / {} bytes",
                    tile_part.tile_index,
                    summary.packet_count(),
                    summary.included_codeblocks(),
                    summary.header_bytes(),
                    summary.body_bytes(),
                    summary.consumed_bytes,
                    tile_data.len()
                );
                for (index, packet) in summary.packets.iter().take(6).enumerate() {
                    println!(
                        "  packet {index}: offset 0x{:06x}, L{} R{} C{} present {}, header {} bytes, body {} bytes, included codeblocks {}",
                        packet.offset,
                        packet.layer,
                        packet.resolution,
                        packet.component,
                        packet.present,
                        packet.header_bytes,
                        packet.body_bytes,
                        packet.segments.len()
                    );
                }
                match Tier1WorkPlan::from_packet_summary_with_quantization(
                    &summary,
                    codestream.qcd.as_ref(),
                    &codestream.qccs,
                ) {
                    Ok(plan) => match plan.dimension_bounds() {
                        Some(bounds) => {
                            println!(
                                "tier1 work tile {}: {} codeblocks, {} segments, {} bytes, {} coding passes, cblk size {}..{} x {}..{}",
                                tile_part.tile_index,
                                plan.codeblock_count(),
                                plan.segment_count(),
                                plan.total_bytes(),
                                plan.total_coding_passes(),
                                bounds.min_width,
                                bounds.max_width,
                                bounds.min_height,
                                bounds.max_height
                            );
                            let pass_counts = plan.pass_kind_counts();
                            println!(
                                "tier1 passes tile {}: cleanup {}, significance {}, refinement {}",
                                tile_part.tile_index,
                                pass_counts.cleanup,
                                pass_counts.significance,
                                pass_counts.refinement
                            );
                            match plan.allocate_decode_stubs() {
                                Ok(stubs) => println!(
                                    "tier1 decode staging tile {}: {} coefficient samples allocated, {} pending coding passes",
                                    tile_part.tile_index,
                                    stubs.coefficient_samples(),
                                    stubs.pending_passes
                                ),
                                Err(err) => println!(
                                    "tier1 decode staging tile {}: unavailable ({err})",
                                    tile_part.tile_index
                                ),
                            }
                            match plan.bind_segment_data(tile_data) {
                                Ok(bound) => {
                                    println!(
                                        "tier1 input tile {}: {} codeblocks, {} segments, {} bound bytes",
                                        tile_part.tile_index,
                                        bound.codeblock_count(),
                                        bound.segment_count(),
                                        bound.total_bytes()
                                    );
                                    match bound.execution_summary(cod.codeblock_style_flags()) {
                                        Ok(execution) => println!(
                                            "tier1 execution tile {}: {} passes, {} mq, {} raw bypass",
                                            tile_part.tile_index,
                                            execution.passes,
                                            execution.arithmetic_passes,
                                            execution.raw_bypass_passes
                                        ),
                                        Err(err) => println!(
                                            "tier1 execution tile {}: unavailable ({err})",
                                            tile_part.tile_index
                                        ),
                                    }
                                }
                                Err(err) => println!(
                                    "tier1 input tile {}: unavailable ({err})",
                                    tile_part.tile_index
                                ),
                            }
                        }
                        None => println!(
                            "tier1 work tile {}: no included codeblocks",
                            tile_part.tile_index
                        ),
                    },
                    Err(err) => {
                        println!(
                            "tier1 work tile {}: unavailable ({err})",
                            tile_part.tile_index
                        );
                    }
                }
            }
            Err(err) => {
                println!(
                    "packet parse tile {}: unavailable ({err})",
                    tile_part.tile_index
                );
            }
        }
    }
}

fn extract_codestream(path: &str, index: usize, output: &str) -> ExitCode {
    let data = match fs::read(path) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {path}: {err}");
            return ExitCode::FAILURE;
        }
    };

    let hits = scan_codestreams(&data, index.saturating_add(1));
    let Some(hit) = hits.get(index) else {
        eprintln!(
            "codestream index {index} not found in {path}; scanned {} hit(s)",
            hits.len()
        );
        return ExitCode::FAILURE;
    };
    let Some(bytes) = data.get(hit.offset..hit.offset.saturating_add(hit.codestream.byte_len))
    else {
        eprintln!("codestream index {index} points outside input");
        return ExitCode::FAILURE;
    };

    match fs::write(output, bytes) {
        Ok(()) => {
            println!(
                "extracted codestream {index}: offset 0x{:08x}, length {} bytes -> {output}",
                hit.offset, hit.codestream.byte_len
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("failed to write {output}: {err}");
            ExitCode::FAILURE
        }
    }
}

fn print_tile_layouts(codestream: &Codestream) {
    match codestream.tile_layouts() {
        Ok(layouts) => {
            println!("tile layouts: {}", layouts.len());
            for layout in layouts.iter().take(4) {
                let packet_state = TilePacketHeaderState::from_tile_layout(layout);
                println!(
                    "  tile {}: grid {}x{} at {},{}..{},{}",
                    layout.tile_index,
                    layout.x1.saturating_sub(layout.x0),
                    layout.y1.saturating_sub(layout.y0),
                    layout.x0,
                    layout.y0,
                    layout.x1,
                    layout.y1
                );
                println!(
                    "    packet header state: {} subbands, {} codeblocks",
                    packet_state.subbands.len(),
                    packet_state.codeblock_count()
                );
                for (component_index, component) in layout.components.iter().enumerate() {
                    let packet_count: u32 = component
                        .resolutions
                        .iter()
                        .map(|resolution| resolution.precincts_x * resolution.precincts_y)
                        .sum();
                    let codeblock_count: u32 = component
                        .resolutions
                        .iter()
                        .flat_map(|resolution| &resolution.subbands)
                        .map(|subband| subband.codeblock_count())
                        .sum();
                    println!(
                        "    component {component_index}: {}x{}, resolutions {}, precinct packets/layer {}, codeblocks {}",
                        component.width,
                        component.height,
                        component.resolutions.len(),
                        packet_count,
                        codeblock_count
                    );
                    for resolution in &component.resolutions {
                        let subbands = resolution
                            .subbands
                            .iter()
                            .map(|subband| {
                                format!(
                                    "{} {}x{} cblk {}x{}",
                                    subband.kind,
                                    subband.width,
                                    subband.height,
                                    subband.codeblocks_x,
                                    subband.codeblocks_y
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        println!(
                            "      r{}: {}x{}, precincts {}x{}, {subbands}",
                            resolution.level,
                            resolution.width,
                            resolution.height,
                            resolution.precincts_x,
                            resolution.precincts_y
                        );
                    }
                }
            }
            if layouts.len() > 4 {
                println!("  ... {} more tiles", layouts.len() - 4);
            }
        }
        Err(err) => println!("tile layouts: unavailable ({err})"),
    }
}
