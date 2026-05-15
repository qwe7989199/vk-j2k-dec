use crate::codestream::{Codestream, WaveletTransform};
use crate::color::{ColorError, inverse_rct_sample, level_shift_and_clamp};
use crate::dwt::PlaneI32;
use crate::packet::parse_lrcp_tile_packets;
use crate::reconstruct::{ComponentSubbandPlanes, TileCoefficientPlanes};
use crate::tier1::Tier1WorkPlan;
use crate::vulkan::{run_color_components, run_idwt53_components};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedRgb48 {
    pub width: u32,
    pub height: u32,
    pub precision: u8,
    pub samples: Vec<u16>,
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
    if cod.transform != WaveletTransform::Reversible53 {
        return Err(DecodeError::new(format!(
            "rgb48 output only supports reversible 5/3 for now, got {}",
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
    let work_plan =
        Tier1WorkPlan::from_packet_summary_with_qcd(&packet_summary, codestream.qcd.as_ref())
            .map_err(|err| DecodeError::new(err.to_string()))?;
    let bound = work_plan
        .bind_segment_data(tile_data)
        .map_err(|err| DecodeError::new(err.to_string()))?;
    let coefficient_planes = TileCoefficientPlanes::decode_arithmetic(
        &layout.components,
        &bound,
        cod.codeblock_style_flags(),
    )
    .map_err(|err| DecodeError::new(err.to_string()))?;

    let target_resolution = cod.decomposition_levels - discard_levels;
    let components = (0..3)
        .map(|component| {
            coefficient_planes
                .component(component)
                .ok_or_else(|| DecodeError::new(format!("missing component {component}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let component_planes = reconstruct_components(&components, target_resolution, dwt_backend)?;
    for (component, plane) in component_planes.iter().enumerate() {
        let expected_width = discard_extent(siz.width, discard_levels);
        let expected_height = discard_extent(siz.height, discard_levels);
        if plane.width != expected_width || plane.height != expected_height {
            return Err(DecodeError::new(format!(
                "component {component} reconstructed as {}x{}, expected {}x{}",
                plane.width, plane.height, expected_width, expected_height
            )));
        }
    }

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

    let width = component_planes[0].width;
    let height = component_planes[0].height;
    let samples = pack_rgb48_samples(
        &component_planes,
        precision,
        cod.multiple_component_transform,
        cod.transform,
        color_backend,
    )?;

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

fn discard_extent(value: u32, discard_levels: u8) -> u32 {
    value.div_ceil(1u32 << discard_levels)
}
