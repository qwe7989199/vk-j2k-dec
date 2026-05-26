use std::time::Instant;

use crate::codestream::{Codestream, WaveletTransform, component_layout};
use crate::color::{
    ColorError, inverse_ict_sample, inverse_rct_sample, level_shift_and_clamp,
    level_shift_and_clamp_float,
};
use crate::dwt::{PlaneF32, PlaneI32};
use crate::packet::parse_lrcp_tile_packets;
use crate::reconstruct::{
    ComponentFloatSubbandPlanes, ComponentSubbandPlanes, TileCoefficientPlanes,
};
use crate::tier1::{CoefficientReconstruction, Tier1WorkPlan};
use crate::vulkan::{
    VulkanRgb48Pipeline, run_color_components, run_color_float_components, run_idwt53_components,
    run_idwt53_rgb48_components, run_idwt97_float_components,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedRgb48 {
    pub width: u32,
    pub height: u32,
    pub precision: u8,
    pub samples: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProfiledDecodedRgb48 {
    pub image: DecodedRgb48,
    pub profile: DecodeProfile,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecodeProfile {
    pub input_bytes: usize,
    pub tile_bytes: usize,
    pub total_codeblocks: usize,
    pub decoded_codeblocks: usize,
    pub skipped_codeblocks: usize,
    pub decoded_passes: u32,
    pub nonzero_coefficients: usize,
    pub output_samples: usize,
    pub timings: DecodeTimings,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DecodeTimings {
    pub parse_ms: f64,
    pub layout_ms: f64,
    pub packet_ms: f64,
    pub work_plan_ms: f64,
    pub bind_ms: f64,
    pub tier1_decode_ms: f64,
    pub gpu_setup_ms: f64,
    pub reconstruct_ms: f64,
    pub reconstruct_upload_ms: f64,
    pub reconstruct_alloc_ms: f64,
    pub reconstruct_dispatch_ms: f64,
    pub pack_ms: f64,
    pub pack_shader_ms: f64,
    pub pack_copy_ms: f64,
    pub pack_readback_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedCoefficientPlanes {
    pub components: Vec<ComponentSubbandPlanes>,
    pub decoded_codeblocks: usize,
    pub decoded_passes: u32,
    pub nonzero_coefficients: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedI32ComponentPlanes {
    pub width: u32,
    pub height: u32,
    pub precision: u8,
    pub components: Vec<PlaneI32>,
}

impl DecodedRgb48 {
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.samples.len() * 2);
        let shift = u32::from(16 - self.precision);
        for sample in &self.samples {
            let value = sample.checked_shl(shift).unwrap_or(0);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError {
    pub message: String,
}

impl DecodeError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DecodeError {}

impl From<ColorError> for DecodeError {
    fn from(value: ColorError) -> Self {
        Self::new(value.to_string())
    }
}

pub fn decode_rgb48(data: &[u8]) -> Result<DecodedRgb48, DecodeError> {
    decode_rgb48_with_discard(data, 0)
}

pub fn decode_rgb48_with_discard(
    data: &[u8],
    discard_levels: u8,
) -> Result<DecodedRgb48, DecodeError> {
    decode_rgb48_with_backends_profiled(data, discard_levels, DwtBackend::Cpu, ColorBackend::Cpu)
        .map(|decoded| decoded.image)
}

pub fn decode_rgb48_with_discard_profiled(
    data: &[u8],
    discard_levels: u8,
) -> Result<ProfiledDecodedRgb48, DecodeError> {
    decode_rgb48_with_backends_profiled(data, discard_levels, DwtBackend::Cpu, ColorBackend::Cpu)
}

pub fn decode_rgb48_with_gpu_dwt(
    data: &[u8],
    discard_levels: u8,
    idwt53_shader_path: &str,
) -> Result<DecodedRgb48, DecodeError> {
    decode_rgb48_with_backends_profiled(
        data,
        discard_levels,
        DwtBackend::Gpu { idwt53_shader_path },
        ColorBackend::Cpu,
    )
    .map(|decoded| decoded.image)
}

pub fn decode_rgb48_with_gpu_dwt_profiled(
    data: &[u8],
    discard_levels: u8,
    idwt53_shader_path: &str,
) -> Result<ProfiledDecodedRgb48, DecodeError> {
    decode_rgb48_with_backends_profiled(
        data,
        discard_levels,
        DwtBackend::Gpu { idwt53_shader_path },
        ColorBackend::Cpu,
    )
}

pub fn decode_rgb48_with_gpu_pipeline(
    data: &[u8],
    discard_levels: u8,
    idwt53_shader_path: &str,
    color_shader_path: &str,
) -> Result<DecodedRgb48, DecodeError> {
    decode_rgb48_with_backends_profiled(
        data,
        discard_levels,
        DwtBackend::Gpu { idwt53_shader_path },
        ColorBackend::Gpu { color_shader_path },
    )
    .map(|decoded| decoded.image)
}

pub fn decode_rgb48_with_gpu_pipeline_profiled(
    data: &[u8],
    discard_levels: u8,
    idwt53_shader_path: &str,
    color_shader_path: &str,
) -> Result<ProfiledDecodedRgb48, DecodeError> {
    decode_rgb48_with_backends_profiled(
        data,
        discard_levels,
        DwtBackend::Gpu { idwt53_shader_path },
        ColorBackend::Gpu { color_shader_path },
    )
}

pub fn decode_rgb48_with_persistent_gpu_pipeline_profiled(
    data: &[u8],
    discard_levels: u8,
    pipeline: &VulkanRgb48Pipeline,
) -> Result<ProfiledDecodedRgb48, DecodeError> {
    decode_rgb48_with_backends_profiled(
        data,
        discard_levels,
        DwtBackend::PersistentGpu { pipeline },
        ColorBackend::Cpu,
    )
}

pub fn decode_rgb48_full_gpu(
    data: &[u8],
    cleanup_shader_path: &str,
    scatter_shader_path: &str,
    idwt_shader_path: &str,
    color_shader_path: &str,
) -> Result<DecodedRgb48, DecodeError> {
    use crate::gpu::{GpuTier1Plan, sort_codeblock_records_by_descending_work};
    use crate::packet::parse_lrcp_tile_packets;
    use crate::tier1::Tier1WorkPlan;

    let codestream = Codestream::parse(data).map_err(|e| DecodeError::new(e.to_string()))?;
    let cod = codestream.cod.as_ref().ok_or_else(|| DecodeError::new("missing COD"))?;
    let siz = codestream.siz.as_ref().ok_or_else(|| DecodeError::new("missing SIZ"))?;
    let layouts = codestream.tile_layouts().map_err(|e| DecodeError::new(e.to_string()))?;

    let precision = siz.components.first().map(|c| c.precision).unwrap_or(12);
    let width = siz.width;
    let height = siz.height;

    let mut all_plans = Vec::new();
    for tile_part in &codestream.tile_parts {
        let layout = layouts.iter().find(|l| l.tile_index == tile_part.tile_index)
            .ok_or_else(|| DecodeError::new(format!("missing layout for tile {}", tile_part.tile_index)))?;
        let tile_data = data.get(tile_part.data_start..tile_part.data_end)
            .ok_or_else(|| DecodeError::new("tile data out of bounds"))?;
        let packet_summary = parse_lrcp_tile_packets(tile_data, layout, cod.layers)
            .map_err(|e| DecodeError::new(e.to_string()))?;
        let work_plan = Tier1WorkPlan::from_packet_summary_with_quantization(
            &packet_summary, codestream.qcd.as_ref(), &codestream.qccs,
        ).map_err(|e| DecodeError::new(e.to_string()))?;
        let bound = work_plan.bind_segment_data(tile_data)
            .map_err(|e| DecodeError::new(e.to_string()))?;
        let plan = GpuTier1Plan::from_bound_work_plan(&bound, cod.codeblock_style_flags())
            .map_err(|e| DecodeError::new(e.to_string()))?;
        all_plans.push(plan);
    }

    let merged = GpuTier1Plan::merge(&all_plans);
    let reconstruction_mode = match cod.transform {
        WaveletTransform::Reversible53 => 0u32,
        _ => 1u32,
    };
    let mut records = merged.codeblock_records_with_reconstruction(reconstruction_mode);
    sort_codeblock_records_by_descending_work(&mut records);
    let compressed_words = merged.compressed_u32_words();

    let scatter_plan = if layouts.len() == 1 {
        merged.scatter_plan(&layouts[0].components)
    } else {
        let full_layouts: Vec<_> = siz.components.iter().map(|comp| {
            component_layout(comp, cod, siz.x_origin, siz.y_origin, siz.width, siz.height)
        }).collect();
        GpuTier1Plan::scatter_plan_multi_tile(&merged, &all_plans, &layouts, &full_layouts)
    }.map_err(|e| DecodeError::new(e.to_string()))?;

    let color_transform = if cod.multiple_component_transform != 0 { 1u32 } else { 0u32 };

    let result = crate::vulkan::run_tier1_cleanup_scatter_rgb48_smoke(
        cleanup_shader_path,
        scatter_shader_path,
        idwt_shader_path,
        color_shader_path,
        &records,
        &compressed_words,
        &scatter_plan.records,
        &scatter_plan.planes,
        scatter_plan.output_coefficients,
        cod.decomposition_levels,
        color_transform,
        precision,
    ).map_err(|e| DecodeError::new(e.to_string()))?;

    Ok(DecodedRgb48 {
        width,
        height,
        precision,
        samples: result.samples,
    })
}

pub fn decode_coefficient_planes(data: &[u8]) -> Result<DecodedCoefficientPlanes, DecodeError> {
    let codestream = Codestream::parse(data).map_err(|err| DecodeError::new(err.to_string()))?;
    let cod = codestream
        .cod
        .as_ref()
        .ok_or_else(|| DecodeError::new("missing COD marker"))?;
    if codestream.tile_parts.len() != 1 {
        return Err(DecodeError::new(format!(
            "coefficient-plane decode only supports a single tile-part for now, found {}",
            codestream.tile_parts.len()
        )));
    }
    let layouts = codestream
        .tile_layouts()
        .map_err(|err| DecodeError::new(err.to_string()))?;
    let tile_part = &codestream.tile_parts[0];
    let layout = layouts
        .iter()
        .find(|layout| layout.tile_index == tile_part.tile_index)
        .ok_or_else(|| {
            DecodeError::new(format!("missing layout for tile {}", tile_part.tile_index))
        })?;
    let tile_data = data
        .get(tile_part.data_start..tile_part.data_end)
        .ok_or_else(|| DecodeError::new("tile payload is outside codestream data"))?;
    let packet_summary = parse_lrcp_tile_packets(tile_data, layout, cod.layers)
        .map_err(|err| DecodeError::new(err.to_string()))?;
    let work_plan = Tier1WorkPlan::from_packet_summary_with_quantization(
        &packet_summary,
        codestream.qcd.as_ref(),
        &codestream.qccs,
    )
    .map_err(|err| DecodeError::new(err.to_string()))?;
    let bound = work_plan
        .bind_segment_data(tile_data)
        .map_err(|err| DecodeError::new(err.to_string()))?;
    let coefficient_planes = TileCoefficientPlanes::decode_arithmetic_with_reconstruction(
        &layout.components,
        &bound,
        cod.codeblock_style_flags(),
        coefficient_reconstruction_for_transform(cod.transform),
    )
    .map_err(|err| DecodeError::new(err.to_string()))?;

    Ok(DecodedCoefficientPlanes {
        components: coefficient_planes.components,
        decoded_codeblocks: coefficient_planes.decoded_codeblocks,
        decoded_passes: coefficient_planes.decoded_passes,
        nonzero_coefficients: coefficient_planes.nonzero_coefficients,
    })
}

pub fn decode_reversible_component_planes_with_discard(
    data: &[u8],
    discard_levels: u8,
) -> Result<DecodedI32ComponentPlanes, DecodeError> {
    let codestream = Codestream::parse(data).map_err(|err| DecodeError::new(err.to_string()))?;
    let siz = codestream
        .siz
        .as_ref()
        .ok_or_else(|| DecodeError::new("missing SIZ marker"))?;
    let cod = codestream
        .cod
        .as_ref()
        .ok_or_else(|| DecodeError::new("missing COD marker"))?;
    if cod.transform != WaveletTransform::Reversible53 {
        return Err(DecodeError::new(format!(
            "component-plane dump requires reversible 5/3, got {}",
            cod.transform
        )));
    }
    if discard_levels > cod.decomposition_levels {
        return Err(DecodeError::new(format!(
            "cannot discard {discard_levels} levels from {} decomposition levels",
            cod.decomposition_levels
        )));
    }
    if codestream.tile_parts.len() != 1 {
        return Err(DecodeError::new(format!(
            "component-plane decode only supports a single tile-part for now, found {}",
            codestream.tile_parts.len()
        )));
    }

    let layouts = codestream
        .tile_layouts()
        .map_err(|err| DecodeError::new(err.to_string()))?;
    let tile_part = &codestream.tile_parts[0];
    let layout = layouts
        .iter()
        .find(|layout| layout.tile_index == tile_part.tile_index)
        .ok_or_else(|| {
            DecodeError::new(format!("missing layout for tile {}", tile_part.tile_index))
        })?;
    let tile_data = data
        .get(tile_part.data_start..tile_part.data_end)
        .ok_or_else(|| DecodeError::new("tile payload is outside codestream data"))?;
    let packet_summary = parse_lrcp_tile_packets(tile_data, layout, cod.layers)
        .map_err(|err| DecodeError::new(err.to_string()))?;
    let target_resolution = cod.decomposition_levels - discard_levels;
    let mut work_plan = Tier1WorkPlan::from_packet_summary_with_quantization(
        &packet_summary,
        codestream.qcd.as_ref(),
        &codestream.qccs,
    )
    .map_err(|err| DecodeError::new(err.to_string()))?;
    work_plan.retain_max_resolution(target_resolution);
    let bound = work_plan
        .bind_segment_data(tile_data)
        .map_err(|err| DecodeError::new(err.to_string()))?;
    let coefficient_planes = TileCoefficientPlanes::decode_arithmetic_with_reconstruction(
        &layout.components,
        &bound,
        cod.codeblock_style_flags(),
        CoefficientReconstruction::Exact,
    )
    .map_err(|err| DecodeError::new(err.to_string()))?;

    let component_refs = (0..siz.components.len())
        .map(|component| {
            coefficient_planes
                .component(component as u16)
                .ok_or_else(|| DecodeError::new(format!("missing component {component}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected_width = discard_extent(siz.width, discard_levels);
    let expected_height = discard_extent(siz.height, discard_levels);
    let components = reconstruct_components(&component_refs, target_resolution, DwtBackend::Cpu)?;
    validate_component_planes_i32(&components, expected_width, expected_height)?;
    let precision = siz
        .components
        .first()
        .map(|component| component.precision)
        .unwrap_or(0);

    Ok(DecodedI32ComponentPlanes {
        width: expected_width,
        height: expected_height,
        precision,
        components,
    })
}

#[derive(Clone, Copy)]
enum DwtBackend<'a> {
    Cpu,
    Gpu { idwt53_shader_path: &'a str },
    PersistentGpu { pipeline: &'a VulkanRgb48Pipeline },
}

#[derive(Debug, Clone, Copy)]
enum ColorBackend<'a> {
    Cpu,
    Gpu { color_shader_path: &'a str },
}

fn decode_rgb48_with_backends_profiled(
    data: &[u8],
    discard_levels: u8,
    dwt_backend: DwtBackend<'_>,
    color_backend: ColorBackend<'_>,
) -> Result<ProfiledDecodedRgb48, DecodeError> {
    let total_start = Instant::now();
    let mut profile = DecodeProfile {
        input_bytes: data.len(),
        ..DecodeProfile::default()
    };

    let stage_start = Instant::now();
    let codestream = Codestream::parse(data).map_err(|err| DecodeError::new(err.to_string()))?;
    profile.timings.parse_ms = elapsed_ms(stage_start);
    let siz = codestream
        .siz
        .as_ref()
        .ok_or_else(|| DecodeError::new("missing SIZ marker"))?;
    let cod = codestream
        .cod
        .as_ref()
        .ok_or_else(|| DecodeError::new("missing COD marker"))?;
    if siz.components.len() != 3 {
        return Err(DecodeError::new(format!(
            "rgb48 output requires 3 components, found {}",
            siz.components.len()
        )));
    }
    if siz.components.iter().any(|component| component.is_signed) {
        return Err(DecodeError::new(
            "rgb48 output does not support signed components yet",
        ));
    }
    if !matches!(
        cod.transform,
        WaveletTransform::Reversible53 | WaveletTransform::Irreversible97
    ) {
        return Err(DecodeError::new(format!(
            "rgb48 output only supports reversible 5/3 and irreversible 9/7 for now, got {}",
            cod.transform
        )));
    }
    if discard_levels > cod.decomposition_levels {
        return Err(DecodeError::new(format!(
            "cannot discard {discard_levels} levels from {} decomposition levels",
            cod.decomposition_levels
        )));
    }
    if codestream.tile_parts.len() == 1 {
        // Fast path: single tile (no stitching needed)
    } else if !matches!(dwt_backend, DwtBackend::Cpu) {
        return Err(DecodeError::new(format!(
            "multi-tile GPU decode is not supported yet ({} tile-parts), use CPU backend",
            codestream.tile_parts.len()
        )));
    }

    let stage_start = Instant::now();
    let layouts = codestream
        .tile_layouts()
        .map_err(|err| DecodeError::new(err.to_string()))?;
    profile.timings.layout_ms = elapsed_ms(stage_start);
    let target_resolution = cod.decomposition_levels - discard_levels;
    let expected_width = discard_extent(siz.width, discard_levels);
    let expected_height = discard_extent(siz.height, discard_levels);

    let stage_start = Instant::now();
    let coefficient_planes_storage;
    let component_planes_storage: Vec<PlaneI32>;

    if codestream.tile_parts.len() == 1 {
        // Single-tile fast path (supports all backends)
        let tile_part = &codestream.tile_parts[0];
        let layout = layouts
            .iter()
            .find(|l| l.tile_index == tile_part.tile_index)
            .ok_or_else(|| {
                DecodeError::new(format!("missing layout for tile {}", tile_part.tile_index))
            })?;
        let tile_data = data
            .get(tile_part.data_start..tile_part.data_end)
            .ok_or_else(|| DecodeError::new("tile payload is outside codestream data"))?;
        profile.tile_bytes = tile_data.len();

        let packet_summary = parse_lrcp_tile_packets(tile_data, layout, cod.layers)
            .map_err(|err| DecodeError::new(err.to_string()))?;
        let mut work_plan = Tier1WorkPlan::from_packet_summary_with_quantization(
            &packet_summary,
            codestream.qcd.as_ref(),
            &codestream.qccs,
        )
        .map_err(|err| DecodeError::new(err.to_string()))?;
        profile.total_codeblocks = work_plan.codeblock_count();
        work_plan.retain_max_resolution(target_resolution);
        profile.decoded_codeblocks = work_plan.codeblock_count();
        let bound = work_plan
            .bind_segment_data(tile_data)
            .map_err(|err| DecodeError::new(err.to_string()))?;
        coefficient_planes_storage =
            TileCoefficientPlanes::decode_arithmetic_with_reconstruction(
                &layout.components,
                &bound,
                cod.codeblock_style_flags(),
                coefficient_reconstruction_for_transform(cod.transform),
            )
            .map_err(|err| DecodeError::new(err.to_string()))?;
        profile.decoded_passes = coefficient_planes_storage.decoded_passes;
        profile.nonzero_coefficients = coefficient_planes_storage.nonzero_coefficients;
        component_planes_storage = Vec::new();
    } else {
        // Multi-tile path (CPU only): decode each tile, reconstruct, stitch
        let mut full_planes: Vec<Vec<i32>> = (0..3)
            .map(|_| vec![0i32; (expected_width as usize) * (expected_height as usize)])
            .collect();
        let mut full_float_planes: Vec<Vec<f32>> = if matches!(cod.transform, WaveletTransform::Irreversible97) {
            (0..3).map(|_| vec![0f32; (expected_width as usize) * (expected_height as usize)]).collect()
        } else {
            Vec::new()
        };

        for tile_part in &codestream.tile_parts {
            let layout = layouts
                .iter()
                .find(|l| l.tile_index == tile_part.tile_index)
                .ok_or_else(|| {
                    DecodeError::new(format!("missing layout for tile {}", tile_part.tile_index))
                })?;
            let tile_data = data
                .get(tile_part.data_start..tile_part.data_end)
                .ok_or_else(|| DecodeError::new("tile payload is outside codestream data"))?;
            profile.tile_bytes += tile_data.len();

            let packet_summary = parse_lrcp_tile_packets(tile_data, layout, cod.layers)
                .map_err(|err| DecodeError::new(err.to_string()))?;
            let mut work_plan = Tier1WorkPlan::from_packet_summary_with_quantization(
                &packet_summary,
                codestream.qcd.as_ref(),
                &codestream.qccs,
            )
            .map_err(|err| DecodeError::new(err.to_string()))?;
            profile.total_codeblocks += work_plan.codeblock_count();
            work_plan.retain_max_resolution(target_resolution);
            profile.decoded_codeblocks += work_plan.codeblock_count();
            let bound = work_plan
                .bind_segment_data(tile_data)
                .map_err(|err| DecodeError::new(err.to_string()))?;
            let tile_coefficients =
                TileCoefficientPlanes::decode_arithmetic_with_reconstruction(
                    &layout.components,
                    &bound,
                    cod.codeblock_style_flags(),
                    coefficient_reconstruction_for_transform(cod.transform),
                )
                .map_err(|err| DecodeError::new(err.to_string()))?;
            profile.decoded_passes += tile_coefficients.decoded_passes;
            profile.nonzero_coefficients += tile_coefficients.nonzero_coefficients;

            let tile_x0 = discard_extent(layout.x0, discard_levels);
            let tile_y0 = discard_extent(layout.y0, discard_levels);
            if tile_part.tile_index < 3 {
                eprintln!("  tile {}: layout ({},{})..({},{}) -> pixel ({},{})",
                    tile_part.tile_index, layout.x0, layout.y0, layout.x1, layout.y1, tile_x0, tile_y0);
            }

            for comp_idx in 0..3u16 {
                let comp = tile_coefficients
                    .component(comp_idx)
                    .ok_or_else(|| DecodeError::new(format!("missing component {comp_idx}")))?;

                match cod.transform {
                    WaveletTransform::Reversible53 => {
                        let tile_plane = comp
                            .inverse_5_3_until(target_resolution)
                            .map_err(|err| DecodeError::new(err.to_string()))?;
                        let dst = &mut full_planes[comp_idx as usize];
                        for y in 0..tile_plane.height {
                            let dst_y = tile_y0 + y;
                            if dst_y >= expected_height { break; }
                            let src_start = (y * tile_plane.width) as usize;
                            let dst_start = (dst_y * expected_width + tile_x0) as usize;
                            let w = tile_plane.width.min(expected_width - tile_x0) as usize;
                            dst[dst_start..dst_start + w]
                                .copy_from_slice(&tile_plane.samples[src_start..src_start + w]);
                        }
                    }
                    WaveletTransform::Irreversible97 => {
                        let qcd = codestream
                            .quantization_for_component(comp_idx)
                            .ok_or_else(|| DecodeError::new(format!(
                                "missing QCD/QCC for component {comp_idx}"
                            )))?;
                        let float_comp = ComponentFloatSubbandPlanes::from_component_with_qcd(
                            comp, qcd, siz.components[0].precision, cod.decomposition_levels,
                        ).map_err(|err| DecodeError::new(err.to_string()))?;
                        let tile_plane = float_comp
                            .inverse_9_7_until(target_resolution)
                            .map_err(|err| DecodeError::new(err.to_string()))?;
                        if tile_part.tile_index == 0 && comp_idx == 0 {
                            let min = tile_plane.samples.iter().cloned().fold(f32::INFINITY, f32::min);
                            let max = tile_plane.samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            eprintln!("  tile 0 comp 0 float range: [{min}, {max}], size {}x{}", tile_plane.width, tile_plane.height);
                        }
                        let dst = &mut full_float_planes[comp_idx as usize];
                        for y in 0..tile_plane.height {
                            let dst_y = tile_y0 + y;
                            if dst_y >= expected_height { break; }
                            let src_start = (y * tile_plane.width) as usize;
                            let dst_start = (dst_y * expected_width + tile_x0) as usize;
                            let w = tile_plane.width.min(expected_width - tile_x0) as usize;
                            dst[dst_start..dst_start + w]
                                .copy_from_slice(&tile_plane.samples[src_start..src_start + w]);
                        }
                    }
                    _ => {}
                }
            }
        }

        if matches!(cod.transform, WaveletTransform::Irreversible97) {
            component_planes_storage = Vec::new();
            // Use float planes path
            let float_planes: Vec<PlaneF32> = full_float_planes
                .into_iter()
                .map(|samples| PlaneF32 { width: expected_width, height: expected_height, samples })
                .collect();
            profile.timings.tier1_decode_ms = elapsed_ms(stage_start);
            let precision = siz.components[0].precision;
            let stage_start = Instant::now();
            let samples = pack_rgb48_float_samples(
                &float_planes, precision, cod.multiple_component_transform, cod.transform, color_backend,
            )?;
            profile.timings.pack_ms = elapsed_ms(stage_start);
            return finish_profiled_rgb48(expected_width, expected_height, precision, samples, profile, total_start);
        }

        component_planes_storage = full_planes
            .into_iter()
            .map(|samples| PlaneI32 {
                width: expected_width,
                height: expected_height,
                samples,
            })
            .collect();
        coefficient_planes_storage = TileCoefficientPlanes {
            components: Vec::new(),
            decoded_codeblocks: 0,
            decoded_passes: 0,
            nonzero_coefficients: 0,
        };
    }
    profile.timings.tier1_decode_ms = elapsed_ms(stage_start);

    let precision = siz.components[0].precision;
    if siz
        .components
        .iter()
        .any(|component| component.precision != precision)
    {
        return Err(DecodeError::new(
            "rgb48 output does not support mixed component precision yet",
        ));
    }

    // For multi-tile, we already have reconstructed planes; skip DWT and go straight to packing
    if !component_planes_storage.is_empty() {
        let stage_start = Instant::now();
        let samples = pack_rgb48_samples(
            &component_planes_storage,
            precision,
            cod.multiple_component_transform,
            cod.transform,
            color_backend,
        )?;
        profile.timings.pack_ms = elapsed_ms(stage_start);
        return finish_profiled_rgb48(
            expected_width,
            expected_height,
            precision,
            samples,
            profile,
            total_start,
        );
    }

    // Single-tile path continues: extract component references for DWT
    let components = (0..3)
        .map(|c| {
            coefficient_planes_storage
                .component(c)
                .ok_or_else(|| DecodeError::new(format!("missing component {c}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (width, height, samples) = match cod.transform {
        WaveletTransform::Reversible53 => {
            if let DwtBackend::PersistentGpu { pipeline } = dwt_backend {
                let color_transform =
                    gpu_color_transform_value(cod.multiple_component_transform, cod.transform)?;
                let result = pipeline
                    .decode_reversible53_rgb48(
                        &components,
                        target_resolution,
                        color_transform,
                        precision,
                    )
                    .map_err(|err| DecodeError::new(format!("GPU RGB48 pipeline failed: {err}")))?;
                profile.timings.gpu_setup_ms = result.setup_ms;
                profile.timings.reconstruct_ms = result.reconstruct_ms;
                profile.timings.reconstruct_upload_ms = result.reconstruct_upload_ms;
                profile.timings.reconstruct_alloc_ms = result.reconstruct_alloc_ms;
                profile.timings.reconstruct_dispatch_ms = result.reconstruct_dispatch_ms;
                profile.timings.pack_ms = result.pack_ms;
                profile.timings.pack_shader_ms = result.pack_shader_ms;
                profile.timings.pack_copy_ms = result.pack_copy_ms;
                profile.timings.pack_readback_ms = result.pack_readback_ms;
                if result.width != expected_width || result.height != expected_height {
                    return Err(DecodeError::new(format!(
                        "GPU RGB48 pipeline produced {}x{}, expected {}x{}",
                        result.width, result.height, expected_width, expected_height
                    )));
                }
                return finish_profiled_rgb48(
                    result.width,
                    result.height,
                    precision,
                    result.samples,
                    profile,
                    total_start,
                );
            }
            if let (
                DwtBackend::Gpu { idwt53_shader_path },
                ColorBackend::Gpu { color_shader_path },
            ) = (dwt_backend, color_backend)
            {
                let color_transform =
                    gpu_color_transform_value(cod.multiple_component_transform, cod.transform)?;
                let result = run_idwt53_rgb48_components(
                    idwt53_shader_path,
                    color_shader_path,
                    &components,
                    target_resolution,
                    color_transform,
                    precision,
                )
                .map_err(|err| DecodeError::new(format!("GPU RGB48 pipeline failed: {err}")))?;
                profile.timings.gpu_setup_ms = result.setup_ms;
                profile.timings.reconstruct_ms = result.reconstruct_ms;
                profile.timings.reconstruct_upload_ms = result.reconstruct_upload_ms;
                profile.timings.reconstruct_alloc_ms = result.reconstruct_alloc_ms;
                profile.timings.reconstruct_dispatch_ms = result.reconstruct_dispatch_ms;
                profile.timings.pack_ms = result.pack_ms;
                profile.timings.pack_shader_ms = result.pack_shader_ms;
                profile.timings.pack_copy_ms = result.pack_copy_ms;
                profile.timings.pack_readback_ms = result.pack_readback_ms;
                if result.width != expected_width || result.height != expected_height {
                    return Err(DecodeError::new(format!(
                        "GPU RGB48 pipeline produced {}x{}, expected {}x{}",
                        result.width, result.height, expected_width, expected_height
                    )));
                }
                (result.width, result.height, result.samples)
            } else {
                let stage_start = Instant::now();
                let component_planes =
                    reconstruct_components(&components, target_resolution, dwt_backend)?;
                profile.timings.reconstruct_ms = elapsed_ms(stage_start);
                validate_component_planes_i32(&component_planes, expected_width, expected_height)?;
                let stage_start = Instant::now();
                let samples = pack_rgb48_samples(
                    &component_planes,
                    precision,
                    cod.multiple_component_transform,
                    cod.transform,
                    color_backend,
                )?;
                profile.timings.pack_ms = elapsed_ms(stage_start);
                (
                    component_planes[0].width,
                    component_planes[0].height,
                    samples,
                )
            }
        }
        WaveletTransform::Irreversible97 => {
            let stage_start = Instant::now();
            let component_planes = reconstruct_float_components(
                &components,
                target_resolution,
                dwt_backend,
                &codestream,
                precision,
                cod.decomposition_levels,
            )?;
            profile.timings.reconstruct_ms = elapsed_ms(stage_start);
            validate_component_planes_f32(&component_planes, expected_width, expected_height)?;
            let stage_start = Instant::now();
            let samples = pack_rgb48_float_samples(
                &component_planes,
                precision,
                cod.multiple_component_transform,
                cod.transform,
                color_backend,
            )?;
            profile.timings.pack_ms = elapsed_ms(stage_start);
            (
                component_planes[0].width,
                component_planes[0].height,
                samples,
            )
        }
        WaveletTransform::Unknown(_) => unreachable!("unsupported transform checked above"),
    };

    finish_profiled_rgb48(width, height, precision, samples, profile, total_start)
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn finish_profiled_rgb48(
    width: u32,
    height: u32,
    precision: u8,
    samples: Vec<u16>,
    mut profile: DecodeProfile,
    total_start: Instant,
) -> Result<ProfiledDecodedRgb48, DecodeError> {
    profile.output_samples = samples.len();
    profile.timings.total_ms = elapsed_ms(total_start);

    Ok(ProfiledDecodedRgb48 {
        image: DecodedRgb48 {
            width,
            height,
            precision,
            samples,
        },
        profile,
    })
}

fn reconstruct_components(
    components: &[&ComponentSubbandPlanes],
    target_resolution: u8,
    dwt_backend: DwtBackend<'_>,
) -> Result<Vec<PlaneI32>, DecodeError> {
    match dwt_backend {
        DwtBackend::Cpu => components
            .iter()
            .map(|component| {
                component
                    .inverse_5_3_until(target_resolution)
                    .map_err(|err| DecodeError::new(err.to_string()))
            })
            .collect(),
        DwtBackend::Gpu { idwt53_shader_path } => {
            run_idwt53_components(idwt53_shader_path, components, target_resolution)
                .map(|result| result.planes)
                .map_err(|err| DecodeError::new(format!("GPU inverse 5/3 failed: {err}")))
        }
        DwtBackend::PersistentGpu { .. } => Err(DecodeError::new(
            "persistent GPU backend must use the fused RGB48 pipeline",
        )),
    }
}

fn reconstruct_float_components(
    components: &[&ComponentSubbandPlanes],
    target_resolution: u8,
    dwt_backend: DwtBackend<'_>,
    codestream: &Codestream,
    precision: u8,
    decomposition_levels: u8,
) -> Result<Vec<PlaneF32>, DecodeError> {
    let float_components = components
        .iter()
        .map(|component| {
            let qcd = codestream
                .quantization_for_component(component.component)
                .ok_or_else(|| {
                    DecodeError::new(format!(
                        "irreversible 9/7 output requires QCD/QCC marker for component {}",
                        component.component
                    ))
                })?;
            ComponentFloatSubbandPlanes::from_component_with_qcd(
                component,
                qcd,
                precision,
                decomposition_levels,
            )
            .map_err(|err| DecodeError::new(err.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    match dwt_backend {
        DwtBackend::Cpu => float_components
            .iter()
            .map(|component| {
                component
                    .inverse_9_7_until(target_resolution)
                    .map_err(|err| DecodeError::new(err.to_string()))
            })
            .collect(),
        DwtBackend::Gpu { idwt53_shader_path } => {
            let component_refs = float_components.iter().collect::<Vec<_>>();
            run_idwt97_float_components(idwt53_shader_path, &component_refs, target_resolution)
                .map(|result| result.planes)
                .map_err(|err| DecodeError::new(format!("GPU inverse 9/7 failed: {err}")))
        }
        DwtBackend::PersistentGpu { .. } => Err(DecodeError::new(
            "persistent GPU backend currently supports only reversible RGB48",
        )),
    }
}

fn pack_rgb48_samples(
    component_planes: &[PlaneI32],
    precision: u8,
    multiple_component_transform: u8,
    transform: WaveletTransform,
    color_backend: ColorBackend<'_>,
) -> Result<Vec<u16>, DecodeError> {
    match color_backend {
        ColorBackend::Cpu => {
            pack_rgb48_samples_cpu(component_planes, precision, multiple_component_transform)
        }
        ColorBackend::Gpu { color_shader_path } => {
            let color_transform =
                gpu_color_transform_value(multiple_component_transform, transform)?;
            run_color_components(
                color_shader_path,
                component_planes,
                color_transform,
                precision,
            )
            .map(|result| result.samples)
            .map_err(|err| DecodeError::new(format!("GPU color conversion failed: {err}")))
        }
    }
}

fn pack_rgb48_float_samples(
    component_planes: &[PlaneF32],
    precision: u8,
    multiple_component_transform: u8,
    transform: WaveletTransform,
    color_backend: ColorBackend<'_>,
) -> Result<Vec<u16>, DecodeError> {
    match color_backend {
        ColorBackend::Cpu => pack_rgb48_float_samples_cpu(
            component_planes,
            precision,
            multiple_component_transform,
            transform,
        ),
        ColorBackend::Gpu { color_shader_path } => {
            let color_transform =
                gpu_color_transform_value(multiple_component_transform, transform)?;
            run_color_float_components(
                color_shader_path,
                component_planes,
                color_transform,
                precision,
            )
            .map(|result| result.samples)
            .map_err(|err| DecodeError::new(format!("GPU float color conversion failed: {err}")))
        }
    }
}

fn pack_rgb48_samples_cpu(
    component_planes: &[PlaneI32],
    precision: u8,
    multiple_component_transform: u8,
) -> Result<Vec<u16>, DecodeError> {
    let pixel_count = component_planes[0].samples.len();
    let mut samples = Vec::with_capacity(pixel_count * 3);
    match multiple_component_transform {
        0 => {
            for index in 0..pixel_count {
                samples.push(level_shift_and_clamp(
                    component_planes[0].samples[index],
                    precision,
                )?);
                samples.push(level_shift_and_clamp(
                    component_planes[1].samples[index],
                    precision,
                )?);
                samples.push(level_shift_and_clamp(
                    component_planes[2].samples[index],
                    precision,
                )?);
            }
        }
        1 => {
            for index in 0..pixel_count {
                let (r, g, b) = inverse_rct_sample(
                    component_planes[0].samples[index],
                    component_planes[1].samples[index],
                    component_planes[2].samples[index],
                );
                samples.push(level_shift_and_clamp(r, precision)?);
                samples.push(level_shift_and_clamp(g, precision)?);
                samples.push(level_shift_and_clamp(b, precision)?);
            }
        }
        other => {
            return Err(DecodeError::new(format!(
                "unsupported multiple component transform {other}"
            )));
        }
    }
    Ok(samples)
}

pub fn pack_rgb48_float_samples_cpu(
    component_planes: &[PlaneF32],
    precision: u8,
    multiple_component_transform: u8,
    transform: WaveletTransform,
) -> Result<Vec<u16>, DecodeError> {
    validate_float_component_planes(component_planes)?;
    let pixel_count = component_planes[0].samples.len();
    let mut samples = Vec::with_capacity(pixel_count * 3);
    match (multiple_component_transform, transform) {
        (0, _) => {
            for index in 0..pixel_count {
                samples.push(level_shift_and_clamp_float(
                    component_planes[0].samples[index],
                    precision,
                )?);
                samples.push(level_shift_and_clamp_float(
                    component_planes[1].samples[index],
                    precision,
                )?);
                samples.push(level_shift_and_clamp_float(
                    component_planes[2].samples[index],
                    precision,
                )?);
            }
        }
        (1, WaveletTransform::Irreversible97) => {
            for index in 0..pixel_count {
                let (r, g, b) = inverse_ict_sample(
                    component_planes[0].samples[index],
                    component_planes[1].samples[index],
                    component_planes[2].samples[index],
                );
                samples.push(level_shift_and_clamp_float(r, precision)?);
                samples.push(level_shift_and_clamp_float(g, precision)?);
                samples.push(level_shift_and_clamp_float(b, precision)?);
            }
        }
        (other, _) => {
            return Err(DecodeError::new(format!(
                "unsupported float multiple component transform {other} for {transform}"
            )));
        }
    }
    Ok(samples)
}

fn validate_float_component_planes(component_planes: &[PlaneF32]) -> Result<(), DecodeError> {
    if component_planes.len() != 3 {
        return Err(DecodeError::new(format!(
            "rgb48 float packing requires 3 component planes, got {}",
            component_planes.len()
        )));
    }
    let width = component_planes[0].width;
    let height = component_planes[0].height;
    let sample_count = component_planes[0].samples.len();
    for (index, plane) in component_planes.iter().enumerate() {
        if plane.width != width || plane.height != height || plane.samples.len() != sample_count {
            return Err(DecodeError::new(format!(
                "float component {index} has {}x{} / {} samples, expected {}x{} / {sample_count}",
                plane.width,
                plane.height,
                plane.samples.len(),
                width,
                height
            )));
        }
    }
    Ok(())
}

fn validate_component_planes_i32(
    component_planes: &[PlaneI32],
    expected_width: u32,
    expected_height: u32,
) -> Result<(), DecodeError> {
    for (component, plane) in component_planes.iter().enumerate() {
        if plane.width != expected_width || plane.height != expected_height {
            return Err(DecodeError::new(format!(
                "component {component} reconstructed as {}x{}, expected {}x{}",
                plane.width, plane.height, expected_width, expected_height
            )));
        }
    }
    Ok(())
}

fn validate_component_planes_f32(
    component_planes: &[PlaneF32],
    expected_width: u32,
    expected_height: u32,
) -> Result<(), DecodeError> {
    for (component, plane) in component_planes.iter().enumerate() {
        if plane.width != expected_width || plane.height != expected_height {
            return Err(DecodeError::new(format!(
                "float component {component} reconstructed as {}x{}, expected {}x{}",
                plane.width, plane.height, expected_width, expected_height
            )));
        }
    }
    Ok(())
}

fn gpu_color_transform_value(
    multiple_component_transform: u8,
    transform: WaveletTransform,
) -> Result<u32, DecodeError> {
    match (multiple_component_transform, transform) {
        (0, _) => Ok(0),
        (1, WaveletTransform::Reversible53) => Ok(1),
        (1, WaveletTransform::Irreversible97) => Ok(2),
        (other, _) => Err(DecodeError::new(format!(
            "unsupported multiple component transform {other}"
        ))),
    }
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

fn discard_extent(value: u32, discard_levels: u8) -> u32 {
    value.div_ceil(1u32 << discard_levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_float_rgb48_without_transform() {
        let planes = vec![
            plane_f32(&[-1.0, 0.4]),
            plane_f32(&[2.0, -2048.0]),
            plane_f32(&[5.0, 2048.0]),
        ];

        let samples =
            pack_rgb48_float_samples_cpu(&planes, 12, 0, WaveletTransform::Irreversible97)
                .expect("pack");

        assert_eq!(samples, vec![2047, 2050, 2053, 2048, 0, 4095]);
    }

    #[test]
    fn packs_float_rgb48_with_ict() {
        let planes = vec![
            plane_f32(&[100.0, -20.0]),
            plane_f32(&[10.0, -12.0]),
            plane_f32(&[-5.0, 7.0]),
        ];

        let samples =
            pack_rgb48_float_samples_cpu(&planes, 12, 1, WaveletTransform::Irreversible97)
                .expect("pack");

        assert_eq!(samples, vec![2141, 2148, 2166, 2038, 2027, 2007]);
    }

    #[test]
    fn rejects_float_rgb48_component_shape_mismatch() {
        let planes = vec![
            plane_f32(&[0.0, 1.0]),
            plane_f32(&[2.0]),
            plane_f32(&[3.0, 4.0]),
        ];

        let err = pack_rgb48_float_samples_cpu(&planes, 12, 0, WaveletTransform::Irreversible97)
            .expect_err("shape mismatch must fail");

        assert!(err.message.contains("float component 1"));
    }

    fn plane_f32(samples: &[f32]) -> PlaneF32 {
        PlaneF32 {
            width: samples.len() as u32,
            height: 1,
            samples: samples.to_vec(),
        }
    }
}
