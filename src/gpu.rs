use crate::codestream::{Codestream, QuantizationStep, WaveletTransform};

const DWT_LOCAL_SIZE_X: u32 = 16;
const DWT_LOCAL_SIZE_Y: u32 = 16;

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
        let qcd = codestream
            .qcd
            .as_ref()
            .ok_or_else(|| GpuPlanError::new("missing QCD marker"))?;
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
                .steps()
                .map_err(|err| GpuPlanError::new(err.message))?
                .into_iter()
                .map(GpuQuantizationStep::from)
                .collect(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDwtDispatchPlan {
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
        if cod.transform != WaveletTransform::Reversible53 {
            return Err(GpuPlanError::new(format!(
                "DWT dispatch planning currently targets reversible 5/3, got {}",
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
        CodSegment, ComponentSpec, ProgressionOrder, QcdSegment, QuantizationStyle, SizSegment,
    };

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
}
