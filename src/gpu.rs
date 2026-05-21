use crate::codestream::{Codestream, QuantizationStep, SubbandKind, WaveletTransform};
use crate::tier1::{Tier1BoundWorkPlan, Tier1CodingMode, Tier1DecodeError};

const DWT_LOCAL_SIZE_X: u32 = 16;
const DWT_LOCAL_SIZE_Y: u32 = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuTier1Plan {
    pub codeblocks: Vec<GpuTier1Codeblock>,
    pub compressed_bytes: Vec<u8>,
    pub total_passes: u32,
    pub arithmetic_passes: u32,
    pub raw_bypass_passes: u32,
    pub output_coefficients: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuTier1Codeblock {
    pub component: u16,
    pub resolution: u8,
    pub subband: SubbandKind,
    pub codeblock_x: u32,
    pub codeblock_y: u32,
    pub width: u32,
    pub height: u32,
    pub magnitude_bitplanes: u32,
    pub zero_bit_planes: u32,
    pub byte_offset: u32,
    pub byte_len: u32,
    pub coefficient_offset: u32,
    pub coefficient_count: u32,
    pub pass_offset: u32,
    pub pass_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuTier1CodeblockRecord {
    pub component: u32,
    pub resolution: u32,
    pub subband: u32,
    pub codeblock_x: u32,
    pub codeblock_y: u32,
    pub width: u32,
    pub height: u32,
    pub magnitude_bitplanes: u32,
    pub zero_bit_planes: u32,
    pub byte_offset: u32,
    pub byte_len: u32,
    pub coefficient_offset: u32,
    pub coefficient_count: u32,
    pub pass_offset: u32,
    pub pass_count: u32,
    pub _padding: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuTier1ScatterPlan {
    pub records: Vec<GpuTier1ScatterRecord>,
    pub planes: Vec<GpuTier1ScatterPlane>,
    pub output_coefficients: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuTier1ScatterPlane {
    pub component: u16,
    pub resolution: u8,
    pub subband: SubbandKind,
    pub plane_offset: u32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuTier1ScatterRecord {
    pub coefficient_offset: u32,
    pub coefficient_count: u32,
    pub width: u32,
    pub height: u32,
    pub plane_offset: u32,
    pub plane_width: u32,
    pub plane_x0: u32,
    pub plane_y0: u32,
}

pub fn sort_codeblock_records_by_descending_work(records: &mut [GpuTier1CodeblockRecord]) {
    records.sort_by(|a, b| {
        let a_work = u64::from(a.byte_len) * u64::from(a.pass_count);
        let b_work = u64::from(b.byte_len) * u64::from(b.pass_count);
        b_work
            .cmp(&a_work)
            .then_with(|| b.byte_len.cmp(&a.byte_len))
            .then_with(|| b.pass_count.cmp(&a.pass_count))
            .then_with(|| tier1_record_key(a).cmp(&tier1_record_key(b)))
    });
}

impl GpuTier1Plan {
    pub fn from_bound_work_plan(
        work_plan: &Tier1BoundWorkPlan<'_>,
        codeblock_style: crate::codestream::CodeblockStyle,
    ) -> Result<Self, Tier1DecodeError> {
        let mut codeblocks = Vec::with_capacity(work_plan.codeblocks.len());
        let mut compressed_bytes = Vec::with_capacity(work_plan.total_bytes());
        let mut total_passes = 0u32;
        let mut arithmetic_passes = 0u32;
        let mut raw_bypass_passes = 0u32;
        let mut output_coefficients = 0usize;

        for codeblock in &work_plan.codeblocks {
            let execution = codeblock.execution_plan(codeblock_style)?;
            let byte_offset = checked_u32(compressed_bytes.len(), "Tier-1 byte offset")?;
            for segment in &codeblock.segments {
                compressed_bytes.extend_from_slice(segment.data);
            }
            let byte_len = checked_u32(execution.bytes, "Tier-1 byte length")?;
            let coefficient_offset = checked_u32(output_coefficients, "Tier-1 coefficient offset")?;
            let coefficient_count = codeblock
                .width
                .checked_mul(codeblock.height)
                .ok_or_else(|| Tier1DecodeError::new("Tier-1 coefficient count overflow"))?;
            let pass_offset = total_passes;
            let pass_count = checked_u32(execution.passes.len(), "Tier-1 pass count")?;
            total_passes = total_passes
                .checked_add(pass_count)
                .ok_or_else(|| Tier1DecodeError::new("Tier-1 pass count overflow"))?;
            output_coefficients = output_coefficients
                .checked_add(coefficient_count as usize)
                .ok_or_else(|| Tier1DecodeError::new("Tier-1 coefficient count overflow"))?;

            for pass in execution.passes {
                match pass.coding_mode {
                    Tier1CodingMode::Arithmetic => arithmetic_passes += 1,
                    Tier1CodingMode::RawBypass => raw_bypass_passes += 1,
                }
            }

            codeblocks.push(GpuTier1Codeblock {
                component: codeblock.id.component,
                resolution: codeblock.id.resolution,
                subband: codeblock.id.subband,
                codeblock_x: codeblock.id.codeblock_x,
                codeblock_y: codeblock.id.codeblock_y,
                width: codeblock.width,
                height: codeblock.height,
                magnitude_bitplanes: codeblock.magnitude_bitplanes,
                zero_bit_planes: codeblock.zero_bit_planes,
                byte_offset,
                byte_len,
                coefficient_offset,
                coefficient_count,
                pass_offset,
                pass_count,
            });
        }

        Ok(Self {
            codeblocks,
            compressed_bytes,
            total_passes,
            arithmetic_passes,
            raw_bypass_passes,
            output_coefficients,
        })
    }

    pub fn codeblock_records(&self) -> Vec<GpuTier1CodeblockRecord> {
        self.codeblocks
            .iter()
            .map(GpuTier1CodeblockRecord::from)
            .collect()
    }

    pub fn compressed_u32_words(&self) -> Vec<u32> {
        self.compressed_bytes
            .chunks(4)
            .map(|chunk| {
                let mut word = 0u32;
                for (shift, byte) in chunk.iter().enumerate() {
                    word |= u32::from(*byte) << (shift * 8);
                }
                word
            })
            .collect()
    }

    pub fn codeblock_byte_checksum(&self, index: usize) -> Option<u32> {
        let codeblock = self.codeblocks.get(index)?;
        let start = codeblock.byte_offset as usize;
        let end = start.checked_add(codeblock.byte_len as usize)?;
        let bytes = self.compressed_bytes.get(start..end)?;
        Some(
            bytes
                .iter()
                .fold(0u32, |sum, byte| sum.wrapping_add(u32::from(*byte))),
        )
    }

    pub fn scatter_plan(
        &self,
        component_layouts: &[crate::codestream::TileComponentLayout],
    ) -> Result<GpuTier1ScatterPlan, Tier1DecodeError> {
        let mut plane_offsets = Vec::new();
        let mut output_coefficients = 0usize;
        for (component, component_layout) in component_layouts.iter().enumerate() {
            let component = u16::try_from(component)
                .map_err(|_| Tier1DecodeError::new("Tier-1 scatter component index exceeds u16"))?;
            for resolution in &component_layout.resolutions {
                for subband in &resolution.subbands {
                    let plane_offset =
                        checked_u32(output_coefficients, "Tier-1 scatter plane offset")?;
                    let plane_samples = usize::try_from(subband.width)
                        .ok()
                        .and_then(|width| {
                            usize::try_from(subband.height)
                                .ok()
                                .and_then(|height| width.checked_mul(height))
                        })
                        .ok_or_else(|| {
                            Tier1DecodeError::new("Tier-1 scatter plane size overflow")
                        })?;
                    output_coefficients = output_coefficients
                        .checked_add(plane_samples)
                        .ok_or_else(|| {
                            Tier1DecodeError::new("Tier-1 scatter output size overflow")
                        })?;
                    plane_offsets.push(GpuTier1PlaneOffset {
                        component,
                        resolution: resolution.level,
                        subband: subband.kind,
                        plane_offset,
                        layout: subband,
                    });
                }
            }
        }

        let mut records = Vec::with_capacity(self.codeblocks.len());
        for codeblock in &self.codeblocks {
            let plane = plane_offsets
                .iter()
                .find(|plane| {
                    plane.component == codeblock.component
                        && plane.resolution == codeblock.resolution
                        && plane.subband == codeblock.subband
                })
                .ok_or_else(|| {
                    Tier1DecodeError::new(format!(
                        "Tier-1 scatter missing plane C{} R{} {}",
                        codeblock.component, codeblock.resolution, codeblock.subband
                    ))
                })?;
            let rect = plane
                .layout
                .codeblock_rect(codeblock.codeblock_x, codeblock.codeblock_y)
                .ok_or_else(|| {
                    Tier1DecodeError::new(format!(
                        "Tier-1 scatter codeblock {},{} outside C{} R{} {}",
                        codeblock.codeblock_x,
                        codeblock.codeblock_y,
                        codeblock.component,
                        codeblock.resolution,
                        codeblock.subband
                    ))
                })?;
            if rect.width() != codeblock.width || rect.height() != codeblock.height {
                return Err(Tier1DecodeError::new(format!(
                    "Tier-1 scatter codeblock C{} R{} {} {},{} has {}x{}, expected {}x{} from layout",
                    codeblock.component,
                    codeblock.resolution,
                    codeblock.subband,
                    codeblock.codeblock_x,
                    codeblock.codeblock_y,
                    codeblock.width,
                    codeblock.height,
                    rect.width(),
                    rect.height()
                )));
            }
            if codeblock.width.checked_mul(codeblock.height) != Some(codeblock.coefficient_count) {
                return Err(Tier1DecodeError::new(format!(
                    "Tier-1 scatter codeblock C{} R{} {} {},{} dimensions {}x{} do not match {} coefficients",
                    codeblock.component,
                    codeblock.resolution,
                    codeblock.subband,
                    codeblock.codeblock_x,
                    codeblock.codeblock_y,
                    codeblock.width,
                    codeblock.height,
                    codeblock.coefficient_count
                )));
            }
            records.push(GpuTier1ScatterRecord {
                coefficient_offset: codeblock.coefficient_offset,
                coefficient_count: codeblock.coefficient_count,
                width: codeblock.width,
                height: codeblock.height,
                plane_offset: plane.plane_offset,
                plane_width: plane.layout.width,
                plane_x0: rect.x0,
                plane_y0: rect.y0,
            });
        }

        Ok(GpuTier1ScatterPlan {
            records,
            planes: plane_offsets
                .iter()
                .map(|plane| GpuTier1ScatterPlane {
                    component: plane.component,
                    resolution: plane.resolution,
                    subband: plane.subband,
                    plane_offset: plane.plane_offset,
                    width: plane.layout.width,
                    height: plane.layout.height,
                })
                .collect(),
            output_coefficients,
        })
    }
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

struct GpuTier1PlaneOffset<'a> {
    component: u16,
    resolution: u8,
    subband: SubbandKind,
    plane_offset: u32,
    layout: &'a crate::codestream::SubbandLayout,
}

impl From<&GpuTier1Codeblock> for GpuTier1CodeblockRecord {
    fn from(codeblock: &GpuTier1Codeblock) -> Self {
        Self {
            component: u32::from(codeblock.component),
            resolution: u32::from(codeblock.resolution),
            subband: gpu_subband_id(codeblock.subband),
            codeblock_x: codeblock.codeblock_x,
            codeblock_y: codeblock.codeblock_y,
            width: codeblock.width,
            height: codeblock.height,
            magnitude_bitplanes: codeblock.magnitude_bitplanes,
            zero_bit_planes: codeblock.zero_bit_planes,
            byte_offset: codeblock.byte_offset,
            byte_len: codeblock.byte_len,
            coefficient_offset: codeblock.coefficient_offset,
            coefficient_count: codeblock.coefficient_count,
            pass_offset: codeblock.pass_offset,
            pass_count: codeblock.pass_count,
            _padding: 0,
        }
    }
}

fn gpu_subband_id(subband: SubbandKind) -> u32 {
    match subband {
        SubbandKind::Ll => 0,
        SubbandKind::Hl => 1,
        SubbandKind::Lh => 2,
        SubbandKind::Hh => 3,
    }
}

fn checked_u32(value: usize, label: &str) -> Result<u32, Tier1DecodeError> {
    u32::try_from(value).map_err(|_| Tier1DecodeError::new(format!("{label} exceeds u32")))
}

#[derive(Debug, Clone, PartialEq)]
pub struct GpuPipelinePlan {
    pub width: u32,
    pub height: u32,
    pub component_count: u16,
    pub sample_count: u32,
    pub color_transform: GpuColorTransform,
    pub output_max: u32,
    pub level_shift: f32,
    pub qcd_steps: Vec<GpuQuantizationStep>,
    pub qcc_steps: Vec<GpuComponentQuantizationSteps>,
}

impl GpuPipelinePlan {
    pub fn from_codestream(codestream: &Codestream) -> Result<Self, GpuPlanError> {
        let siz = codestream
            .siz
            .as_ref()
            .ok_or_else(|| GpuPlanError::new("missing SIZ marker"))?;
        let cod = codestream
            .cod
            .as_ref()
            .ok_or_else(|| GpuPlanError::new("missing COD marker"))?;
        let qcd = codestream.qcd.as_ref();
        let precision = siz
            .components
            .iter()
            .map(|component| component.precision)
            .max()
            .ok_or_else(|| GpuPlanError::new("SIZ has no components"))?;
        if precision > 16 {
            return Err(GpuPlanError::new(format!(
                "component precision {precision} exceeds RGB16 staging"
            )));
        }
        let sample_count = siz
            .width
            .checked_mul(siz.height)
            .ok_or_else(|| GpuPlanError::new("image dimensions overflow sample count"))?;

        Ok(Self {
            width: siz.width,
            height: siz.height,
            component_count: siz.components.len() as u16,
            sample_count,
            color_transform: GpuColorTransform::from_codestream(
                cod.multiple_component_transform,
                cod.transform,
            )?,
            output_max: (1u32 << precision) - 1,
            level_shift: (1u32 << (precision - 1)) as f32,
            qcd_steps: qcd
                .map(|qcd| qcd.steps())
                .transpose()
                .map_err(|err| GpuPlanError::new(err.message))?
                .unwrap_or_default()
                .into_iter()
                .map(GpuQuantizationStep::from)
                .collect(),
            qcc_steps: codestream
                .qccs
                .iter()
                .map(|qcc| {
                    qcc.quantization
                        .steps()
                        .map_err(|err| GpuPlanError::new(err.message))
                        .map(|steps| GpuComponentQuantizationSteps {
                            component: qcc.component,
                            steps: steps.into_iter().map(GpuQuantizationStep::from).collect(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    pub fn qcd_step_for(
        &self,
        resolution: u8,
        subband: crate::codestream::SubbandKind,
    ) -> Result<&GpuQuantizationStep, GpuPlanError> {
        self.quantization_step_for(None, resolution, subband)
    }

    pub fn qcd_step_for_component(
        &self,
        component: u16,
        resolution: u8,
        subband: crate::codestream::SubbandKind,
    ) -> Result<&GpuQuantizationStep, GpuPlanError> {
        self.quantization_step_for(Some(component), resolution, subband)
    }

    fn quantization_step_for(
        &self,
        component: Option<u16>,
        resolution: u8,
        subband: crate::codestream::SubbandKind,
    ) -> Result<&GpuQuantizationStep, GpuPlanError> {
        let index = qcd_step_index(resolution, subband).ok_or_else(|| {
            GpuPlanError::new(format!(
                "resolution {resolution} subband {subband} has no QCD step index"
            ))
        })?;
        let steps = component
            .and_then(|component| {
                self.qcc_steps
                    .iter()
                    .rev()
                    .find(|qcc| qcc.component == component)
            })
            .map(|qcc| qcc.steps.as_slice())
            .unwrap_or(&self.qcd_steps);
        steps.get(index).ok_or_else(|| {
            GpuPlanError::new(format!(
                "quantization table has {} step(s), missing index {index} for R{resolution} {subband}",
                steps.len()
            ))
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GpuComponentQuantizationSteps {
    pub component: u16,
    pub steps: Vec<GpuQuantizationStep>,
}

fn qcd_step_index(resolution: u8, subband: crate::codestream::SubbandKind) -> Option<usize> {
    match (resolution, subband) {
        (0, crate::codestream::SubbandKind::Ll) => Some(0),
        (1.., crate::codestream::SubbandKind::Hl) => Some(1 + (usize::from(resolution) - 1) * 3),
        (1.., crate::codestream::SubbandKind::Lh) => Some(2 + (usize::from(resolution) - 1) * 3),
        (1.., crate::codestream::SubbandKind::Hh) => Some(3 + (usize::from(resolution) - 1) * 3),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDwtDispatchPlan {
    pub transform: WaveletTransform,
    pub tile_count: u32,
    pub component_count: u16,
    pub stages: Vec<GpuDwtStagePlan>,
}

impl GpuDwtDispatchPlan {
    pub fn from_codestream(codestream: &Codestream) -> Result<Self, GpuPlanError> {
        let cod = codestream
            .cod
            .as_ref()
            .ok_or_else(|| GpuPlanError::new("missing COD marker"))?;
        if !matches!(
            cod.transform,
            WaveletTransform::Reversible53 | WaveletTransform::Irreversible97
        ) {
            return Err(GpuPlanError::new(format!(
                "DWT dispatch planning does not support {}",
                cod.transform
            )));
        }

        let layouts = codestream
            .tile_layouts()
            .map_err(|err| GpuPlanError::new(err.message))?;
        let component_count = layouts
            .first()
            .map(|tile| tile.components.len() as u16)
            .unwrap_or(0);
        let mut stages = Vec::new();

        for tile in &layouts {
            for (component_index, component) in tile.components.iter().enumerate() {
                for resolution in component.resolutions.iter().skip(1) {
                    let previous = component
                        .resolutions
                        .get(resolution.level.saturating_sub(1) as usize)
                        .ok_or_else(|| {
                            GpuPlanError::new(format!(
                                "missing previous resolution for tile {} component {} r{}",
                                tile.tile_index, component_index, resolution.level
                            ))
                        })?;
                    let high_width = resolution.width.saturating_sub(previous.width);
                    let high_height = resolution.height.saturating_sub(previous.height);

                    let location = DwtStageLocation {
                        tile_index: tile.tile_index,
                        component: component_index as u16,
                        resolution: resolution.level,
                        pass: GpuDwtPassKind::VerticalLowColumns,
                    };
                    stages.push(GpuDwtStagePlan::new(
                        location.with_pass(GpuDwtPassKind::VerticalLowColumns),
                        DwtLineShape {
                            low_count: previous.height,
                            high_count: high_height,
                            line_count: previous.width,
                            output_count: resolution.height,
                            low_line_stride: 1,
                            low_sample_stride: previous.width,
                            high_line_stride: 1,
                            high_sample_stride: previous.width,
                            output_line_stride: 1,
                            output_sample_stride: previous.width,
                        },
                    ));
                    stages.push(GpuDwtStagePlan::new(
                        location.with_pass(GpuDwtPassKind::VerticalHighColumns),
                        DwtLineShape {
                            low_count: previous.height,
                            high_count: high_height,
                            line_count: high_width,
                            output_count: resolution.height,
                            low_line_stride: 1,
                            low_sample_stride: high_width,
                            high_line_stride: 1,
                            high_sample_stride: high_width,
                            output_line_stride: 1,
                            output_sample_stride: high_width,
                        },
                    ));
                    stages.push(GpuDwtStagePlan::new(
                        location.with_pass(GpuDwtPassKind::HorizontalRows),
                        DwtLineShape {
                            low_count: previous.width,
                            high_count: high_width,
                            line_count: resolution.height,
                            output_count: resolution.width,
                            low_line_stride: previous.width,
                            low_sample_stride: 1,
                            high_line_stride: high_width,
                            high_sample_stride: 1,
                            output_line_stride: resolution.width,
                            output_sample_stride: 1,
                        },
                    ));
                }
            }
        }

        Ok(Self {
            transform: cod.transform,
            tile_count: layouts.len() as u32,
            component_count,
            stages,
        })
    }

    pub fn total_workgroups(&self) -> u32 {
        self.stages.iter().map(GpuDwtStagePlan::workgroups).sum()
    }

    pub fn total_output_samples(&self) -> u64 {
        self.stages
            .iter()
            .map(GpuDwtStagePlan::output_samples)
            .sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuDwtStagePlan {
    pub tile_index: u16,
    pub component: u16,
    pub resolution: u8,
    pub pass: GpuDwtPassKind,
    pub low_count: u32,
    pub high_count: u32,
    pub line_count: u32,
    pub output_count: u32,
    pub low_line_stride: u32,
    pub low_sample_stride: u32,
    pub high_line_stride: u32,
    pub high_sample_stride: u32,
    pub output_line_stride: u32,
    pub output_sample_stride: u32,
    pub groups_x: u32,
    pub groups_y: u32,
}

impl GpuDwtStagePlan {
    fn new(location: DwtStageLocation, shape: DwtLineShape) -> Self {
        Self {
            tile_index: location.tile_index,
            component: location.component,
            resolution: location.resolution,
            pass: location.pass,
            low_count: shape.low_count,
            high_count: shape.high_count,
            line_count: shape.line_count,
            output_count: shape.output_count,
            low_line_stride: shape.low_line_stride,
            low_sample_stride: shape.low_sample_stride,
            high_line_stride: shape.high_line_stride,
            high_sample_stride: shape.high_sample_stride,
            output_line_stride: shape.output_line_stride,
            output_sample_stride: shape.output_sample_stride,
            groups_x: shape.output_count.div_ceil(DWT_LOCAL_SIZE_X),
            groups_y: shape.line_count.div_ceil(DWT_LOCAL_SIZE_Y),
        }
    }

    pub fn workgroups(&self) -> u32 {
        self.groups_x.saturating_mul(self.groups_y)
    }

    pub fn output_samples(&self) -> u64 {
        u64::from(self.output_count) * u64::from(self.line_count)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DwtStageLocation {
    tile_index: u16,
    component: u16,
    resolution: u8,
    pass: GpuDwtPassKind,
}

impl DwtStageLocation {
    fn with_pass(self, pass: GpuDwtPassKind) -> Self {
        Self { pass, ..self }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DwtLineShape {
    low_count: u32,
    high_count: u32,
    line_count: u32,
    output_count: u32,
    low_line_stride: u32,
    low_sample_stride: u32,
    high_line_stride: u32,
    high_sample_stride: u32,
    output_line_stride: u32,
    output_sample_stride: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDwtPassKind {
    VerticalLowColumns,
    VerticalHighColumns,
    HorizontalRows,
}

impl std::fmt::Display for GpuDwtPassKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VerticalLowColumns => f.write_str("vertical-low-columns"),
            Self::VerticalHighColumns => f.write_str("vertical-high-columns"),
            Self::HorizontalRows => f.write_str("horizontal-rows"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuColorTransform {
    None = 0,
    Rct = 1,
    Ict = 2,
}

impl GpuColorTransform {
    fn from_codestream(value: u8, transform: WaveletTransform) -> Result<Self, GpuPlanError> {
        match (value, transform) {
            (0, _) => Ok(Self::None),
            (1, WaveletTransform::Reversible53) => Ok(Self::Rct),
            (1, WaveletTransform::Irreversible97) => Ok(Self::Ict),
            (value, _) => Err(GpuPlanError::new(format!(
                "unsupported multiple component transform value {value}"
            ))),
        }
    }

    pub fn shader_value(self) -> u32 {
        self as u32
    }
}

impl std::fmt::Display for GpuColorTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Rct => f.write_str("RCT"),
            Self::Ict => f.write_str("ICT"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuQuantizationStep {
    pub exponent: u8,
    pub mantissa: u16,
    pub raw: u16,
}

impl From<QuantizationStep> for GpuQuantizationStep {
    fn from(value: QuantizationStep) -> Self {
        Self {
            exponent: value.exponent,
            mantissa: value.mantissa,
            raw: value.raw,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuPlanError {
    pub message: String,
}

impl GpuPlanError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for GpuPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for GpuPlanError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codestream::{
        CodSegment, ComponentSpec, ProgressionOrder, QccSegment, QcdSegment, QuantizationStyle,
        ResolutionLayout, SizSegment, SubbandKind, SubbandLayout, TileComponentLayout,
    };

    #[test]
    fn tier1_scatter_plan_maps_codeblocks_into_planes() {
        let component_layout = TileComponentLayout {
            width: 8,
            height: 4,
            resolutions: vec![ResolutionLayout {
                level: 0,
                width: 8,
                height: 4,
                precincts_x: 1,
                precincts_y: 1,
                subbands: vec![SubbandLayout {
                    kind: SubbandKind::Ll,
                    width: 8,
                    height: 4,
                    codeblock_width: 4,
                    codeblock_height: 4,
                    codeblocks_x: 2,
                    codeblocks_y: 1,
                }],
            }],
        };
        let plan = GpuTier1Plan {
            codeblocks: vec![GpuTier1Codeblock {
                component: 0,
                resolution: 0,
                subband: SubbandKind::Ll,
                codeblock_x: 1,
                codeblock_y: 0,
                width: 4,
                height: 4,
                magnitude_bitplanes: 8,
                zero_bit_planes: 0,
                byte_offset: 0,
                byte_len: 3,
                coefficient_offset: 16,
                coefficient_count: 16,
                pass_offset: 0,
                pass_count: 1,
            }],
            compressed_bytes: vec![0, 0, 0],
            total_passes: 1,
            arithmetic_passes: 1,
            raw_bypass_passes: 0,
            output_coefficients: 32,
        };

        let scatter = plan
            .scatter_plan(&[component_layout])
            .expect("scatter plan");

        assert_eq!(scatter.output_coefficients, 32);
        assert_eq!(
            scatter.planes,
            vec![GpuTier1ScatterPlane {
                component: 0,
                resolution: 0,
                subband: SubbandKind::Ll,
                plane_offset: 0,
                width: 8,
                height: 4,
            }]
        );
        assert_eq!(
            scatter.records,
            vec![GpuTier1ScatterRecord {
                coefficient_offset: 16,
                coefficient_count: 16,
                width: 4,
                height: 4,
                plane_offset: 0,
                plane_width: 8,
                plane_x0: 4,
                plane_y0: 0,
            }]
        );
    }

    #[test]
    fn builds_gpu_plan_for_reversible_rgb12_codestream() {
        let codestream = Codestream {
            byte_len: 0,
            siz: Some(SizSegment {
                capabilities: 0,
                width: 1920,
                height: 1080,
                x_origin: 0,
                y_origin: 0,
                tile_width: 1920,
                tile_height: 1080,
                tile_x_origin: 0,
                tile_y_origin: 0,
                components: vec![
                    ComponentSpec {
                        precision: 12,
                        is_signed: false,
                        x_subsampling: 1,
                        y_subsampling: 1,
                    };
                    3
                ],
            }),
            cod: Some(CodSegment {
                coding_style: 0,
                progression_order: ProgressionOrder::Lrcp,
                layers: 1,
                multiple_component_transform: 0,
                decomposition_levels: 5,
                codeblock_width: 64,
                codeblock_height: 64,
                codeblock_style: 0,
                transform: WaveletTransform::Reversible53,
                precinct_sizes: Vec::new(),
            }),
            qcd: Some(QcdSegment {
                guard_bits: 2,
                style: QuantizationStyle::NoQuantization,
                raw_steps: vec![0x28; 16],
            }),
            qccs: Vec::new(),
            comments: Vec::new(),
            tile_parts: Vec::new(),
            markers: Vec::new(),
        };

        let plan = GpuPipelinePlan::from_codestream(&codestream).expect("gpu plan");
        let dwt_plan = GpuDwtDispatchPlan::from_codestream(&codestream).expect("dwt plan");

        assert_eq!(plan.sample_count, 1920 * 1080);
        assert_eq!(plan.component_count, 3);
        assert_eq!(plan.color_transform, GpuColorTransform::None);
        assert_eq!(plan.color_transform.shader_value(), 0);
        assert_eq!(plan.output_max, 4095);
        assert_eq!(plan.level_shift, 2048.0);
        assert_eq!(plan.qcd_steps.len(), 16);
        assert_eq!(plan.qcd_steps[0].exponent, 5);
        assert_eq!(plan.qcd_step_for(0, SubbandKind::Ll).unwrap().raw, 0x28);
        assert_eq!(plan.qcd_step_for(1, SubbandKind::Hl).unwrap().raw, 0x28);
        assert_eq!(plan.qcd_step_for(1, SubbandKind::Lh).unwrap().raw, 0x28);
        assert_eq!(plan.qcd_step_for(1, SubbandKind::Hh).unwrap().raw, 0x28);
        assert!(plan.qcd_step_for(0, SubbandKind::Hl).is_err());
        assert_eq!(dwt_plan.transform, WaveletTransform::Reversible53);
        assert_eq!(dwt_plan.tile_count, 1);
        assert_eq!(dwt_plan.component_count, 3);
        assert_eq!(dwt_plan.stages.len(), 45);
        assert_eq!(
            dwt_plan.stages[0],
            GpuDwtStagePlan {
                tile_index: 0,
                component: 0,
                resolution: 1,
                pass: GpuDwtPassKind::VerticalLowColumns,
                low_count: 34,
                high_count: 34,
                line_count: 60,
                output_count: 68,
                low_line_stride: 1,
                low_sample_stride: 60,
                high_line_stride: 1,
                high_sample_stride: 60,
                output_line_stride: 1,
                output_sample_stride: 60,
                groups_x: 5,
                groups_y: 4,
            }
        );
        assert_eq!(dwt_plan.stages[2].pass, GpuDwtPassKind::HorizontalRows);
        assert_eq!(dwt_plan.stages[2].low_sample_stride, 1);
        assert_eq!(dwt_plan.stages[2].output_line_stride, 120);
    }

    #[test]
    fn builds_dwt_dispatch_plan_for_irreversible_97_codestream() {
        let mut codestream = reversible_rgb12_codestream();
        codestream.cod.as_mut().expect("COD").transform = WaveletTransform::Irreversible97;

        let dwt_plan = GpuDwtDispatchPlan::from_codestream(&codestream).expect("dwt plan");

        assert_eq!(dwt_plan.transform, WaveletTransform::Irreversible97);
        assert_eq!(dwt_plan.tile_count, 1);
        assert_eq!(dwt_plan.component_count, 3);
        assert_eq!(dwt_plan.stages.len(), 45);
        assert_eq!(dwt_plan.total_workgroups(), 65337);
    }

    #[test]
    fn builds_gpu_plan_for_irreversible_97_ict_codestream() {
        let mut codestream = reversible_rgb12_codestream();
        let cod = codestream.cod.as_mut().expect("COD");
        cod.transform = WaveletTransform::Irreversible97;
        cod.multiple_component_transform = 1;
        codestream.qcd = Some(QcdSegment {
            guard_bits: 2,
            style: QuantizationStyle::ScalarExpounded,
            raw_steps: vec![0x12, 0x34],
        });

        let plan = GpuPipelinePlan::from_codestream(&codestream).expect("gpu plan");

        assert_eq!(plan.color_transform, GpuColorTransform::Ict);
        assert_eq!(plan.color_transform.shader_value(), 2);
        assert_eq!(plan.qcd_steps.len(), 1);
        assert_eq!(plan.qcd_steps[0].raw, 0x1234);
        assert!(plan.qcd_step_for(1, SubbandKind::Hl).is_err());
    }

    #[test]
    fn gpu_plan_tracks_component_quantization_overrides() {
        let mut codestream = reversible_rgb12_codestream();
        codestream.qccs.push(QccSegment {
            component: 2,
            quantization: QcdSegment {
                guard_bits: 2,
                style: QuantizationStyle::NoQuantization,
                raw_steps: vec![0x28, 0x30, 0x38, 0x40],
            },
        });

        let plan = GpuPipelinePlan::from_codestream(&codestream).expect("gpu plan");

        assert_eq!(plan.qcc_steps.len(), 1);
        assert_eq!(
            plan.qcd_step_for_component(2, 1, SubbandKind::Hl)
                .expect("QCC HL")
                .raw,
            0x30
        );
        assert_eq!(
            plan.qcd_step_for_component(1, 1, SubbandKind::Hl)
                .expect("QCD fallback")
                .raw,
            0x28
        );
    }

    fn reversible_rgb12_codestream() -> Codestream {
        Codestream {
            byte_len: 0,
            siz: Some(SizSegment {
                capabilities: 0,
                width: 1920,
                height: 1080,
                x_origin: 0,
                y_origin: 0,
                tile_width: 1920,
                tile_height: 1080,
                tile_x_origin: 0,
                tile_y_origin: 0,
                components: vec![
                    ComponentSpec {
                        precision: 12,
                        is_signed: false,
                        x_subsampling: 1,
                        y_subsampling: 1,
                    };
                    3
                ],
            }),
            cod: Some(CodSegment {
                coding_style: 0,
                progression_order: ProgressionOrder::Lrcp,
                layers: 1,
                multiple_component_transform: 0,
                decomposition_levels: 5,
                codeblock_width: 64,
                codeblock_height: 64,
                codeblock_style: 0,
                transform: WaveletTransform::Reversible53,
                precinct_sizes: Vec::new(),
            }),
            qcd: Some(QcdSegment {
                guard_bits: 2,
                style: QuantizationStyle::NoQuantization,
                raw_steps: vec![0x28; 16],
            }),
            qccs: Vec::new(),
            comments: Vec::new(),
            tile_parts: Vec::new(),
            markers: Vec::new(),
        }
    }
}
