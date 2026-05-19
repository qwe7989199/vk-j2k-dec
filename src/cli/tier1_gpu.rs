use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use vk_j2k_dec::codestream::{Codestream, WaveletTransform};
use vk_j2k_dec::decode::decode_rgb48_with_discard;
use vk_j2k_dec::gpu::GpuTier1Plan;
use vk_j2k_dec::packet::parse_lrcp_tile_packets;
use vk_j2k_dec::reconstruct::TileCoefficientPlanes;
use vk_j2k_dec::tier1::{
    BoundCompressedCodeblock, CoefficientReconstruction, DecodedCodeblock, MqDecoder,
    Tier1WorkPlan, tier1_pass_for_index,
};

use super::{coefficient_reconstruction_for_transform, elapsed_ms};

pub fn tier1_gpu_plan(path: &str) -> ExitCode {
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
                    let gpu_tier1_start = Instant::now();
                    let cleanup_smoke = match vk_j2k_dec::vulkan::run_tier1_cleanup_smoke(
                        cleanup_shader_path,
                        &records,
                        &compressed_words,
                    ) {
                        Ok(smoke) => smoke,
                        Err(err) => {
                            eprintln!("tier1 gpu plan: Vulkan cleanup smoke failed: {err}");
                            return ExitCode::FAILURE;
                        }
                    };
                    let gpu_tier1_ms = elapsed_ms(gpu_tier1_start);
                    let cpu_tier1_start = Instant::now();
                    let reconstruction = coefficient_reconstruction_for_transform(cod.transform);
                    let (expected, expected_coefficients) =
                        match tier1_cleanup_smoke_expected_all(&bound.codeblocks, reconstruction) {
                            Ok(expected) => expected,
                            Err(err) => {
                                eprintln!("tier1 gpu plan: CPU cleanup smoke failed: {err}");
                                return ExitCode::FAILURE;
                            }
                        };
                    let cpu_tier1_ms = elapsed_ms(cpu_tier1_start);
                    if cleanup_smoke.output != expected {
                        eprintln!(
                            "tier1 gpu plan: cleanup smoke mismatch: got {:?}, expected {:?}",
                            cleanup_smoke.output, expected
                        );
                        return ExitCode::FAILURE;
                    }
                    if cleanup_smoke.coefficients != expected_coefficients {
                        let mismatch = cleanup_smoke
                            .coefficients
                            .iter()
                            .zip(&expected_coefficients)
                            .position(|(actual, expected)| actual != expected)
                            .unwrap_or(cleanup_smoke.coefficients.len());
                        eprintln!(
                            "tier1 gpu plan: coefficient smoke mismatch at {mismatch}: got {:?}, expected {:?}",
                            cleanup_smoke.coefficients.get(mismatch),
                            expected_coefficients.get(mismatch)
                        );
                        return ExitCode::FAILURE;
                    }
                    let scatter_shader_path = "target\\shaders\\tier1_scatter_smoke.comp.spv";
                    if std::path::Path::new(scatter_shader_path).exists() {
                        let scatter_plan = match plan.scatter_plan(&layout.components) {
                            Ok(scatter_plan) => scatter_plan,
                            Err(err) => {
                                eprintln!("tier1 gpu plan: scatter plan failed: {err}");
                                return ExitCode::FAILURE;
                            }
                        };
                        let scatter_smoke = match vk_j2k_dec::vulkan::run_tier1_scatter_smoke(
                            scatter_shader_path,
                            &scatter_plan.records,
                            &cleanup_smoke.coefficients,
                            scatter_plan.output_coefficients,
                        ) {
                            Ok(smoke) => smoke,
                            Err(err) => {
                                eprintln!("tier1 gpu plan: Vulkan scatter smoke failed: {err}");
                                return ExitCode::FAILURE;
                            }
                        };
                        let expected_scatter = match tier1_scatter_expected(
                            &scatter_plan.records,
                            &expected_coefficients,
                            scatter_plan.output_coefficients,
                        ) {
                            Ok(expected) => expected,
                            Err(err) => {
                                eprintln!("tier1 gpu plan: CPU scatter failed: {err}");
                                return ExitCode::FAILURE;
                            }
                        };
                        if scatter_smoke.output != expected_scatter {
                            let mismatch = scatter_smoke
                                .output
                                .iter()
                                .zip(&expected_scatter)
                                .position(|(actual, expected)| actual != expected)
                                .unwrap_or(scatter_smoke.output.len());
                            eprintln!(
                                "tier1 gpu plan: scatter mismatch at {mismatch}: got {:?}, expected {:?}",
                                scatter_smoke.output.get(mismatch),
                                expected_scatter.get(mismatch)
                            );
                            return ExitCode::FAILURE;
                        }
                        println!(
                            "    vulkan tier1 scatter smoke: device {}, queue {}, {} records, {} plane coeff, upload {:.3} ms, dispatch {:.3} ms, readback {:.3} ms",
                            scatter_smoke.device_name,
                            scatter_smoke.queue_family_index,
                            scatter_plan.records.len(),
                            scatter_smoke.output.len(),
                            scatter_smoke.upload_ms,
                            scatter_smoke.dispatch_ms,
                            scatter_smoke.readback_ms
                        );
                        let cleanup_scatter_smoke =
                            match vk_j2k_dec::vulkan::run_tier1_cleanup_scatter_smoke(
                                cleanup_shader_path,
                                scatter_shader_path,
                                &records,
                                &compressed_words,
                                &scatter_plan.records,
                                scatter_plan.output_coefficients,
                            ) {
                                Ok(smoke) => smoke,
                                Err(err) => {
                                    eprintln!(
                                        "tier1 gpu plan: Vulkan cleanup+scatter smoke failed: {err}"
                                    );
                                    return ExitCode::FAILURE;
                                }
                            };
                        if cleanup_scatter_smoke.output != expected {
                            eprintln!(
                                "tier1 gpu plan: cleanup+scatter summary mismatch: got {:?}, expected {:?}",
                                cleanup_scatter_smoke.output, expected
                            );
                            return ExitCode::FAILURE;
                        }
                        if cleanup_scatter_smoke.plane_coefficients != expected_scatter {
                            let mismatch = cleanup_scatter_smoke
                                .plane_coefficients
                                .iter()
                                .zip(&expected_scatter)
                                .position(|(actual, expected)| actual != expected)
                                .unwrap_or(cleanup_scatter_smoke.plane_coefficients.len());
                            eprintln!(
                                "tier1 gpu plan: cleanup+scatter plane mismatch at {mismatch}: got {:?}, expected {:?}",
                                cleanup_scatter_smoke.plane_coefficients.get(mismatch),
                                expected_scatter.get(mismatch)
                            );
                            return ExitCode::FAILURE;
                        }
                        println!(
                            "    vulkan tier1_cleanup_pass+scatter: device {}, queue {}, upload {:.3} ms, tier1_cleanup_pass {:.3} ms, scatter {:.3} ms, readback {:.3} ms",
                            cleanup_scatter_smoke.device_name,
                            cleanup_scatter_smoke.queue_family_index,
                            cleanup_scatter_smoke.upload_ms,
                            cleanup_scatter_smoke.cleanup_dispatch_ms,
                            cleanup_scatter_smoke.scatter_dispatch_ms,
                            cleanup_scatter_smoke.readback_ms
                        );
                        let idwt53_offset_shader_path =
                            "target\\shaders\\idwt53_offset_1d.comp.spv";
                        if cod.transform == WaveletTransform::Reversible53
                            && std::path::Path::new(idwt53_offset_shader_path).exists()
                        {
                            let scattered_idwt =
                                match vk_j2k_dec::vulkan::run_idwt53_scattered_components(
                                    idwt53_offset_shader_path,
                                    &scatter_plan.planes,
                                    &cleanup_scatter_smoke.plane_coefficients,
                                    cod.decomposition_levels,
                                ) {
                                    Ok(result) => result,
                                    Err(err) => {
                                        eprintln!(
                                            "tier1 gpu plan: Vulkan scattered IDWT failed: {err}"
                                        );
                                        return ExitCode::FAILURE;
                                    }
                                };
                            let cpu_tile_planes =
                                match TileCoefficientPlanes::decode_arithmetic_with_reconstruction(
                                    &layout.components,
                                    &bound,
                                    cod.codeblock_style_flags(),
                                    reconstruction,
                                ) {
                                    Ok(planes) => planes,
                                    Err(err) => {
                                        eprintln!(
                                            "tier1 gpu plan: CPU scattered IDWT reference failed: {err}"
                                        );
                                        return ExitCode::FAILURE;
                                    }
                                };
                            if scattered_idwt.planes.len() != cpu_tile_planes.components.len() {
                                eprintln!(
                                    "tier1 gpu plan: scattered IDWT component count mismatch: got {}, expected {}",
                                    scattered_idwt.planes.len(),
                                    cpu_tile_planes.components.len()
                                );
                                return ExitCode::FAILURE;
                            }
                            for (component_index, (gpu_plane, cpu_component)) in scattered_idwt
                                .planes
                                .iter()
                                .zip(&cpu_tile_planes.components)
                                .enumerate()
                            {
                                let cpu_plane = match cpu_component
                                    .inverse_5_3_until(cod.decomposition_levels)
                                {
                                    Ok(plane) => plane,
                                    Err(err) => {
                                        eprintln!(
                                            "tier1 gpu plan: CPU IDWT component {component_index} failed: {err}"
                                        );
                                        return ExitCode::FAILURE;
                                    }
                                };
                                if gpu_plane != &cpu_plane {
                                    let mismatch = gpu_plane
                                        .samples
                                        .iter()
                                        .zip(&cpu_plane.samples)
                                        .position(|(actual, expected)| actual != expected)
                                        .unwrap_or(gpu_plane.samples.len());
                                    eprintln!(
                                        "tier1 gpu plan: scattered IDWT component {component_index} mismatch at {mismatch}: got {:?}, expected {:?}",
                                        gpu_plane.samples.get(mismatch),
                                        cpu_plane.samples.get(mismatch)
                                    );
                                    return ExitCode::FAILURE;
                                }
                            }
                            println!(
                                "    vulkan scattered idwt53: device {}, queue {}, {} components, upload {:.3} ms, dispatch {:.3} ms, readback {:.3} ms",
                                scattered_idwt.device_name,
                                scattered_idwt.queue_family_index,
                                scattered_idwt.planes.len(),
                                scattered_idwt.upload_ms,
                                scattered_idwt.dispatch_ms,
                                scattered_idwt.readback_ms
                            );
                            let combined_idwt =
                                match vk_j2k_dec::vulkan::run_tier1_cleanup_scatter_idwt53_smoke(
                                    cleanup_shader_path,
                                    scatter_shader_path,
                                    idwt53_offset_shader_path,
                                    &records,
                                    &compressed_words,
                                    &scatter_plan.records,
                                    &scatter_plan.planes,
                                    scatter_plan.output_coefficients,
                                    cod.decomposition_levels,
                                ) {
                                    Ok(result) => result,
                                    Err(err) => {
                                        eprintln!(
                                            "tier1 gpu plan: Vulkan cleanup+scatter+IDWT failed: {err}"
                                        );
                                        return ExitCode::FAILURE;
                                    }
                                };
                            if combined_idwt.output != expected {
                                eprintln!(
                                    "tier1 gpu plan: cleanup+scatter+IDWT summary mismatch: got {:?}, expected {:?}",
                                    combined_idwt.output, expected
                                );
                                return ExitCode::FAILURE;
                            }
                            if combined_idwt.planes != scattered_idwt.planes {
                                let mismatch_component = combined_idwt
                                    .planes
                                    .iter()
                                    .zip(&scattered_idwt.planes)
                                    .position(|(actual, expected)| actual != expected)
                                    .unwrap_or(combined_idwt.planes.len());
                                eprintln!(
                                    "tier1 gpu plan: cleanup+scatter+IDWT component mismatch at {mismatch_component}"
                                );
                                return ExitCode::FAILURE;
                            }
                            println!(
                                "    vulkan tier1_cleanup_pass+scatter+idwt53: device {}, queue {}, upload {:.3} ms, tier1_cleanup_pass {:.3} ms, scatter {:.3} ms, idwt_dispatch {:.3} ms, idwt_alloc {:.3} ms, readback {:.3} ms",
                                combined_idwt.device_name,
                                combined_idwt.queue_family_index,
                                combined_idwt.upload_ms,
                                combined_idwt.cleanup_dispatch_ms,
                                combined_idwt.scatter_dispatch_ms,
                                combined_idwt.idwt_dispatch_ms,
                                combined_idwt.idwt_alloc_ms,
                                combined_idwt.readback_ms
                            );
                            let color_i32_shader_path =
                                "target\\shaders\\color_convert_i32.comp.spv";
                            if std::path::Path::new(color_i32_shader_path).exists() {
                                let precision = codestream
                                    .siz
                                    .as_ref()
                                    .and_then(|siz| siz.components.first())
                                    .map(|component| component.precision)
                                    .unwrap_or(0);
                                let combined_rgb =
                                    match vk_j2k_dec::vulkan::run_tier1_cleanup_scatter_rgb48_smoke(
                                        cleanup_shader_path,
                                        scatter_shader_path,
                                        idwt53_offset_shader_path,
                                        color_i32_shader_path,
                                        &records,
                                        &compressed_words,
                                        &scatter_plan.records,
                                        &scatter_plan.planes,
                                        scatter_plan.output_coefficients,
                                        cod.decomposition_levels,
                                        u32::from(cod.multiple_component_transform),
                                        precision,
                                    ) {
                                        Ok(result) => result,
                                        Err(err) => {
                                            eprintln!(
                                                "tier1 gpu plan: Vulkan cleanup+scatter+RGB48 failed: {err}"
                                            );
                                            return ExitCode::FAILURE;
                                        }
                                    };
                                if combined_rgb.output != expected {
                                    eprintln!(
                                        "tier1 gpu plan: cleanup+scatter+RGB48 summary mismatch: got {:?}, expected {:?}",
                                        combined_rgb.output, expected
                                    );
                                    return ExitCode::FAILURE;
                                }
                                let expected_rgb = match decode_rgb48_with_discard(&data, 0) {
                                    Ok(image) => image,
                                    Err(err) => {
                                        eprintln!(
                                            "tier1 gpu plan: CPU RGB48 reference failed: {err}"
                                        );
                                        return ExitCode::FAILURE;
                                    }
                                };
                                if combined_rgb.samples != expected_rgb.samples {
                                    let mismatch = combined_rgb
                                        .samples
                                        .iter()
                                        .zip(&expected_rgb.samples)
                                        .position(|(actual, expected)| actual != expected)
                                        .unwrap_or(combined_rgb.samples.len());
                                    eprintln!(
                                        "tier1 gpu plan: cleanup+scatter+RGB48 mismatch at {mismatch}: got {:?}, expected {:?}",
                                        combined_rgb.samples.get(mismatch),
                                        expected_rgb.samples.get(mismatch)
                                    );
                                    return ExitCode::FAILURE;
                                }
                                println!(
                                    "    vulkan tier1_cleanup_pass+scatter+rgb48: device {}, queue {}, {} samples, upload {:.3} ms, tier1_cleanup_pass {:.3} ms, scatter {:.3} ms, idwt_dispatch {:.3} ms, idwt_alloc {:.3} ms, color_shader {:.3} ms, color_copy {:.3} ms, color_readback {:.3} ms, summary_readback {:.3} ms",
                                    combined_rgb.device_name,
                                    combined_rgb.queue_family_index,
                                    combined_rgb.samples.len(),
                                    combined_rgb.upload_ms,
                                    combined_rgb.cleanup_dispatch_ms,
                                    combined_rgb.scatter_dispatch_ms,
                                    combined_rgb.idwt_dispatch_ms,
                                    combined_rgb.idwt_alloc_ms,
                                    combined_rgb.color_shader_ms,
                                    combined_rgb.color_copy_ms,
                                    combined_rgb.color_readback_ms,
                                    combined_rgb.summary_readback_ms
                                );
                            }
                        }
                    } else {
                        println!(
                            "    vulkan tier1 scatter smoke: skipped, missing {scatter_shader_path}"
                        );
                    }
                    println!(
                        "    vulkan tier1 full smoke: device {}, queue {}, {} cblk summaries, {} coeff, first nonzero {}, checksum 0x{:08x}",
                        cleanup_smoke.device_name,
                        cleanup_smoke.queue_family_index,
                        records.len(),
                        cleanup_smoke.coefficients.len(),
                        cleanup_smoke.output[0],
                        cleanup_smoke.output[1]
                    );
                    println!(
                        "      tier1 full smoke timings: gpu_roundtrip {:.3} ms (upload {:.3}, dispatch {:.3}, readback {:.3}), cpu_reference {:.3} ms",
                        gpu_tier1_ms,
                        cleanup_smoke.upload_ms,
                        cleanup_smoke.dispatch_ms,
                        cleanup_smoke.readback_ms,
                        cpu_tier1_ms
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

pub fn tier1_gpu_bench(path: &str, iterations: usize) -> ExitCode {
    if iterations == 0 {
        eprintln!("tier1 gpu bench: iterations must be at least 1");
        return ExitCode::FAILURE;
    }
    let data = match fs::read(path) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("tier1 gpu bench: failed to read {path}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let codestream = match Codestream::parse(&data) {
        Ok(codestream) => codestream,
        Err(err) => {
            eprintln!("tier1 gpu bench: {err}");
            return ExitCode::FAILURE;
        }
    };
    let Some(cod) = &codestream.cod else {
        eprintln!("tier1 gpu bench: missing COD marker");
        return ExitCode::FAILURE;
    };
    let layouts = match codestream.tile_layouts() {
        Ok(layouts) => layouts,
        Err(err) => {
            eprintln!("tier1 gpu bench: tile layout unavailable ({err})");
            return ExitCode::FAILURE;
        }
    };
    let cleanup_shader_path = "target\\shaders\\tier1_cleanup_smoke.comp.spv";
    if !std::path::Path::new(cleanup_shader_path).exists() {
        eprintln!("tier1 gpu bench: missing {cleanup_shader_path}");
        return ExitCode::FAILURE;
    }

    println!("tier1 gpu bench: {path}, {iterations} iteration(s)");
    for tile_part in &codestream.tile_parts {
        let Some(layout) = layouts
            .iter()
            .find(|layout| layout.tile_index == tile_part.tile_index)
        else {
            eprintln!(
                "tier1 gpu bench: missing layout for tile {}",
                tile_part.tile_index
            );
            return ExitCode::FAILURE;
        };
        let Some(tile_data) = data.get(tile_part.data_start..tile_part.data_end) else {
            eprintln!(
                "tier1 gpu bench: tile {} payload outside input",
                tile_part.tile_index
            );
            return ExitCode::FAILURE;
        };
        let packet_summary = match parse_lrcp_tile_packets(tile_data, layout, cod.layers) {
            Ok(summary) => summary,
            Err(err) => {
                eprintln!(
                    "tier1 gpu bench: packet parse failed for tile {}: {err}",
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
                    "tier1 gpu bench: work plan failed for tile {}: {err}",
                    tile_part.tile_index
                );
                return ExitCode::FAILURE;
            }
        };
        let bound = match work_plan.bind_segment_data(tile_data) {
            Ok(bound) => bound,
            Err(err) => {
                eprintln!(
                    "tier1 gpu bench: segment bind failed for tile {}: {err}",
                    tile_part.tile_index
                );
                return ExitCode::FAILURE;
            }
        };
        let plan = match GpuTier1Plan::from_bound_work_plan(&bound, cod.codeblock_style_flags()) {
            Ok(plan) => plan,
            Err(err) => {
                eprintln!(
                    "tier1 gpu bench: flatten failed for tile {}: {err}",
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

        let records = plan.codeblock_records();
        let compressed_words = plan.compressed_u32_words();
        let bench = match vk_j2k_dec::vulkan::run_tier1_cleanup_bench(
            cleanup_shader_path,
            &records,
            &compressed_words,
            iterations,
        ) {
            Ok(bench) => bench,
            Err(err) => {
                eprintln!("tier1 gpu bench: Vulkan cleanup bench failed: {err}");
                return ExitCode::FAILURE;
            }
        };

        let reconstruction = coefficient_reconstruction_for_transform(cod.transform);
        let cpu_start = Instant::now();
        let mut expected_output = Vec::new();
        let mut expected_coefficients = Vec::new();
        for iteration in 0..iterations {
            let (output, coefficients) =
                match tier1_cleanup_smoke_expected_all(&bound.codeblocks, reconstruction) {
                    Ok(expected) => expected,
                    Err(err) => {
                        eprintln!("tier1 gpu bench: CPU cleanup reference failed: {err}");
                        return ExitCode::FAILURE;
                    }
                };
            if iteration == 0 {
                expected_output = output;
                expected_coefficients = coefficients;
            }
        }
        let cpu_total_ms = elapsed_ms(cpu_start);
        if bench.output != expected_output {
            eprintln!(
                "tier1 gpu bench: cleanup output mismatch: got {:?}, expected {:?}",
                bench.output, expected_output
            );
            return ExitCode::FAILURE;
        }
        if bench.coefficients != expected_coefficients {
            let mismatch = bench
                .coefficients
                .iter()
                .zip(&expected_coefficients)
                .position(|(actual, expected)| actual != expected)
                .unwrap_or(bench.coefficients.len());
            eprintln!(
                "tier1 gpu bench: coefficient mismatch at {mismatch}: got {:?}, expected {:?}",
                bench.coefficients.get(mismatch),
                expected_coefficients.get(mismatch)
            );
            return ExitCode::FAILURE;
        }
        let scatter_shader_path = "target\\shaders\\tier1_scatter_smoke.comp.spv";
        let mut cleanup_scatter_average_ms = None;
        if std::path::Path::new(scatter_shader_path).exists() {
            let scatter_plan = match plan.scatter_plan(&layout.components) {
                Ok(scatter_plan) => scatter_plan,
                Err(err) => {
                    eprintln!("tier1 gpu bench: scatter plan failed: {err}");
                    return ExitCode::FAILURE;
                }
            };
            let expected_scatter = match tier1_scatter_expected(
                &scatter_plan.records,
                &expected_coefficients,
                scatter_plan.output_coefficients,
            ) {
                Ok(expected) => expected,
                Err(err) => {
                    eprintln!("tier1 gpu bench: CPU scatter failed: {err}");
                    return ExitCode::FAILURE;
                }
            };
            let cleanup_scatter_bench = match vk_j2k_dec::vulkan::run_tier1_cleanup_scatter_bench(
                cleanup_shader_path,
                scatter_shader_path,
                &records,
                &compressed_words,
                &scatter_plan.records,
                scatter_plan.output_coefficients,
                iterations,
            ) {
                Ok(bench) => bench,
                Err(err) => {
                    eprintln!("tier1 gpu bench: Vulkan cleanup+scatter bench failed: {err}");
                    return ExitCode::FAILURE;
                }
            };
            if cleanup_scatter_bench.output != expected_output {
                eprintln!(
                    "tier1 gpu bench: cleanup+scatter summary mismatch: got {:?}, expected {:?}",
                    cleanup_scatter_bench.output, expected_output
                );
                return ExitCode::FAILURE;
            }
            if cleanup_scatter_bench.plane_coefficients != expected_scatter {
                let mismatch = cleanup_scatter_bench
                    .plane_coefficients
                    .iter()
                    .zip(&expected_scatter)
                    .position(|(actual, expected)| actual != expected)
                    .unwrap_or(cleanup_scatter_bench.plane_coefficients.len());
                eprintln!(
                    "tier1 gpu bench: cleanup+scatter plane mismatch at {mismatch}: got {:?}, expected {:?}",
                    cleanup_scatter_bench.plane_coefficients.get(mismatch),
                    expected_scatter.get(mismatch)
                );
                return ExitCode::FAILURE;
            }
            cleanup_scatter_average_ms = Some(cleanup_scatter_bench.dispatch_average_ms);
            println!(
                "    vulkan tier1_cleanup_pass+scatter bench: device {}, queue {}, setup {:.3} ms, hot_dispatch {:.3} ms total / {:.3} ms avg, readback {:.3} ms",
                cleanup_scatter_bench.device_name,
                cleanup_scatter_bench.queue_family_index,
                cleanup_scatter_bench.setup_ms,
                cleanup_scatter_bench.dispatch_total_ms,
                cleanup_scatter_bench.dispatch_average_ms,
                cleanup_scatter_bench.readback_ms
            );
            let idwt53_offset_shader_path = "target\\shaders\\idwt53_offset_1d.comp.spv";
            let color_i32_shader_path = "target\\shaders\\color_convert_i32.comp.spv";
            if cod.transform == WaveletTransform::Reversible53
                && std::path::Path::new(idwt53_offset_shader_path).exists()
                && std::path::Path::new(color_i32_shader_path).exists()
            {
                let precision = codestream
                    .siz
                    .as_ref()
                    .and_then(|siz| siz.components.first())
                    .map(|component| component.precision)
                    .unwrap_or(0);
                let rgb_bench = match vk_j2k_dec::vulkan::run_tier1_cleanup_scatter_rgb48_bench(
                    cleanup_shader_path,
                    scatter_shader_path,
                    idwt53_offset_shader_path,
                    color_i32_shader_path,
                    &records,
                    &compressed_words,
                    &scatter_plan.records,
                    &scatter_plan.planes,
                    scatter_plan.output_coefficients,
                    cod.decomposition_levels,
                    u32::from(cod.multiple_component_transform),
                    precision,
                    iterations,
                ) {
                    Ok(bench) => bench,
                    Err(err) => {
                        eprintln!("tier1 gpu bench: Vulkan full RGB48 bench failed: {err}");
                        return ExitCode::FAILURE;
                    }
                };
                if rgb_bench.output != expected_output {
                    eprintln!(
                        "tier1 gpu bench: full RGB48 summary mismatch: got {:?}, expected {:?}",
                        rgb_bench.output, expected_output
                    );
                    return ExitCode::FAILURE;
                }

                let cpu_rgb_start = Instant::now();
                let mut expected_rgb = None;
                for iteration in 0..iterations {
                    let image = match decode_rgb48_with_discard(&data, 0) {
                        Ok(image) => image,
                        Err(err) => {
                            eprintln!("tier1 gpu bench: CPU RGB48 reference failed: {err}");
                            return ExitCode::FAILURE;
                        }
                    };
                    if iteration == 0 {
                        expected_rgb = Some(image);
                    }
                }
                let cpu_rgb_total_ms = elapsed_ms(cpu_rgb_start);
                let Some(expected_rgb) = expected_rgb else {
                    eprintln!("tier1 gpu bench: CPU RGB48 reference produced no image");
                    return ExitCode::FAILURE;
                };
                if rgb_bench.samples != expected_rgb.samples {
                    let mismatch = rgb_bench
                        .samples
                        .iter()
                        .zip(&expected_rgb.samples)
                        .position(|(actual, expected)| actual != expected)
                        .unwrap_or(rgb_bench.samples.len());
                    eprintln!(
                        "tier1 gpu bench: full RGB48 mismatch at {mismatch}: got {:?}, expected {:?}",
                        rgb_bench.samples.get(mismatch),
                        expected_rgb.samples.get(mismatch)
                    );
                    return ExitCode::FAILURE;
                }
                let cpu_rgb_average_ms = cpu_rgb_total_ms / iterations as f64;
                let gpu_fps = 1000.0 / rgb_bench.decode_average_ms;
                let cpu_fps = 1000.0 / cpu_rgb_average_ms;
                println!(
                    "    vulkan full RGB48 hot bench: device {}, queue {}, setup_once {:.3} ms, {:.3} ms total / {:.3} ms avg, {:.2} fps",
                    rgb_bench.device_name,
                    rgb_bench.queue_family_index,
                    rgb_bench.setup_ms,
                    rgb_bench.decode_total_ms,
                    rgb_bench.decode_average_ms,
                    gpu_fps
                );
                println!(
                    "      avg breakdown: fused_gpu_dispatch {:.3} ms, tier1_cleanup_pass {:.3} ms, scatter {:.3} ms, idwt_dispatch {:.3} ms, idwt_alloc {:.3} ms, color_shader {:.3} ms, color_copy {:.3} ms, color_readback {:.3} ms, summary_readback_once {:.3} ms",
                    rgb_bench.fused_dispatch_average_ms,
                    rgb_bench.cleanup_dispatch_average_ms,
                    rgb_bench.scatter_dispatch_average_ms,
                    rgb_bench.idwt_dispatch_average_ms,
                    rgb_bench.idwt_alloc_average_ms,
                    rgb_bench.color_shader_average_ms,
                    rgb_bench.color_copy_average_ms,
                    rgb_bench.color_readback_average_ms,
                    rgb_bench.summary_readback_ms
                );
                println!(
                    "      cpu full RGB48 reference: {:.3} ms total / {:.3} ms avg, {:.2} fps; GPU speedup {:.2}x",
                    cpu_rgb_total_ms,
                    cpu_rgb_average_ms,
                    cpu_fps,
                    cpu_rgb_average_ms / rgb_bench.decode_average_ms
                );
            }
        }
        let cpu_average_ms = cpu_total_ms / iterations as f64;
        let hot_speedup = cpu_average_ms / bench.dispatch_average_ms;
        println!(
            "    vulkan tier1 bench: device {}, queue {}, setup {:.3} ms, hot_dispatch {:.3} ms total / {:.3} ms avg, readback {:.3} ms",
            bench.device_name,
            bench.queue_family_index,
            bench.setup_ms,
            bench.dispatch_total_ms,
            bench.dispatch_average_ms,
            bench.readback_ms
        );
        if let Some(cleanup_scatter_average_ms) = cleanup_scatter_average_ms {
            println!(
                "    gpu cleanup+scatter vs cpu reference: {:.2}x hot speedup",
                cpu_average_ms / cleanup_scatter_average_ms
            );
        }
        println!(
            "    cpu tier1 reference: {:.3} ms total / {:.3} ms avg, hot GPU speedup {:.2}x, first nonzero {}, checksum 0x{:08x}",
            cpu_total_ms, cpu_average_ms, hot_speedup, bench.output[0], bench.output[1]
        );
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
) -> Result<(Vec<u32>, Vec<i32>), String> {
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
    let mut significance_bits = 0u32;
    let mut sign_bits = 0u32;
    let mut refinement_bits = 0u32;
    let mut became_significant = 0u32;
    let mut visited = 0u32;
    for pass_index in 0..codeblock.total_coding_passes() {
        let pass = tier1_pass_for_index(
            codeblock.magnitude_bitplanes,
            codeblock.zero_bit_planes,
            pass_index,
        );
        let summary = decoded
            .decode_pass(pass, &mut mq)
            .map_err(|err| err.to_string())?;
        significance_bits += summary.significance_bits;
        sign_bits += summary.sign_bits;
        refinement_bits += summary.refinement_bits;
        became_significant += summary.became_significant;
        visited += summary.visited;
    }
    let mut nonzero = 0u32;
    let mut checksum = 0u32;
    for coefficient in &decoded.coefficients {
        if *coefficient != 0 {
            nonzero += 1;
            checksum = checksum.wrapping_add(*coefficient as u32);
        }
    }
    Ok((
        vec![
            nonzero,
            checksum,
            u32::try_from(mq.consumed_bytes())
                .map_err(|_| "MQ byte_pos exceeds u32".to_string())?,
            decoded.decoded_passes,
            significance_bits,
            sign_bits,
            refinement_bits,
            became_significant,
            visited,
        ],
        decoded.coefficients,
    ))
}

fn tier1_cleanup_smoke_expected_all(
    codeblocks: &[BoundCompressedCodeblock<'_>],
    reconstruction: CoefficientReconstruction,
) -> Result<(Vec<u32>, Vec<i32>), String> {
    let mut expected = Vec::with_capacity(codeblocks.len() * 9);
    let mut coefficients = Vec::new();
    for codeblock in codeblocks {
        let (summary, mut codeblock_coefficients) =
            tier1_cleanup_smoke_expected(codeblock, reconstruction)?;
        expected.extend(summary);
        coefficients.append(&mut codeblock_coefficients);
    }
    Ok((expected, coefficients))
}

fn tier1_scatter_expected(
    records: &[vk_j2k_dec::gpu::GpuTier1ScatterRecord],
    flat_coefficients: &[i32],
    output_coefficients: usize,
) -> Result<Vec<i32>, String> {
    let mut output = vec![0i32; output_coefficients];
    for (record_index, record) in records.iter().enumerate() {
        if record.width == 0 || record.height == 0 {
            return Err(format!(
                "scatter record {record_index} has empty dimensions"
            ));
        }
        if record.width.checked_mul(record.height) != Some(record.coefficient_count) {
            return Err(format!(
                "scatter record {record_index} dimensions {}x{} do not match {} coefficients",
                record.width, record.height, record.coefficient_count
            ));
        }
        let source_start = record.coefficient_offset as usize;
        let source_end = source_start
            .checked_add(record.coefficient_count as usize)
            .ok_or_else(|| format!("scatter record {record_index} source overflows"))?;
        if source_end > flat_coefficients.len() {
            return Err(format!(
                "scatter record {record_index} source {source_start}..{source_end} outside {} coefficients",
                flat_coefficients.len()
            ));
        }
        for y in 0..record.height {
            for x in 0..record.width {
                let source = source_start + (y * record.width + x) as usize;
                let destination = record.plane_offset as usize
                    + ((record.plane_y0 + y) * record.plane_width + record.plane_x0 + x) as usize;
                let Some(slot) = output.get_mut(destination) else {
                    return Err(format!(
                        "scatter record {record_index} destination {destination} outside {output_coefficients} coefficients"
                    ));
                };
                *slot = flat_coefficients[source];
            }
        }
    }
    Ok(output)
}
