use crate::codestream::{Codestream, WaveletTransform};
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
    run_color_components, run_color_float_components, run_idwt53_components,
    run_idwt97_float_components,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedRgb48 {
    pub width: u32,
    pub height: u32,
    pub precision: u8,
    pub samples: Vec<u16>,
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
    decode_rgb48_with_backends(data, discard_levels, DwtBackend::Cpu, ColorBackend::Cpu)
}

pub fn decode_rgb48_with_gpu_dwt(
    data: &[u8],
    discard_levels: u8,
    idwt53_shader_path: &str,
) -> Result<DecodedRgb48, DecodeError> {
    decode_rgb48_with_backends(
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
    decode_rgb48_with_backends(
        data,
        discard_levels,
        DwtBackend::Gpu { idwt53_shader_path },
        ColorBackend::Gpu { color_shader_path },
    )
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
    let target_resolution = cod.decomposition_levels - discard_levels;
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

#[derive(Debug, Clone, Copy)]
enum DwtBackend<'a> {
    Cpu,
    Gpu { idwt53_shader_path: &'a str },
}

#[derive(Debug, Clone, Copy)]
enum ColorBackend<'a> {
    Cpu,
    Gpu { color_shader_path: &'a str },
}

fn decode_rgb48_with_backends(
    data: &[u8],
    discard_levels: u8,
    dwt_backend: DwtBackend<'_>,
    color_backend: ColorBackend<'_>,
) -> Result<DecodedRgb48, DecodeError> {
    let codestream = Codestream::parse(data).map_err(|err| DecodeError::new(err.to_string()))?;
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
    if codestream.tile_parts.len() != 1 {
        return Err(DecodeError::new(format!(
            "rgb48 output only supports a single tile-part for now, found {}",
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

    let components = (0..3)
        .map(|component| {
            coefficient_planes
                .component(component)
                .ok_or_else(|| DecodeError::new(format!("missing component {component}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
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

    let target_resolution = cod.decomposition_levels - discard_levels;
    let expected_width = discard_extent(siz.width, discard_levels);
    let expected_height = discard_extent(siz.height, discard_levels);
    let (width, height, samples) = match cod.transform {
        WaveletTransform::Reversible53 => {
            let component_planes =
                reconstruct_components(&components, target_resolution, dwt_backend)?;
            validate_component_planes_i32(&component_planes, expected_width, expected_height)?;
            let samples = pack_rgb48_samples(
                &component_planes,
                precision,
                cod.multiple_component_transform,
                cod.transform,
                color_backend,
            )?;
            (
                component_planes[0].width,
                component_planes[0].height,
                samples,
            )
        }
        WaveletTransform::Irreversible97 => {
            let component_planes = reconstruct_float_components(
                &components,
                target_resolution,
                dwt_backend,
                &codestream,
                precision,
                cod.decomposition_levels,
            )?;
            validate_component_planes_f32(&component_planes, expected_width, expected_height)?;
            let samples = pack_rgb48_float_samples(
                &component_planes,
                precision,
                cod.multiple_component_transform,
                cod.transform,
                color_backend,
            )?;
            (
                component_planes[0].width,
                component_planes[0].height,
                samples,
            )
        }
        WaveletTransform::Unknown(_) => unreachable!("unsupported transform checked above"),
    };

    Ok(DecodedRgb48 {
        width,
        height,
        precision,
        samples,
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
