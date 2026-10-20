use crate::codestream::{
    QcdSegment, QuantizationStep, QuantizationStyle, SubbandKind, SubbandLayout,
    TileComponentLayout,
};
use crate::dwt::{Band2D, Band2DF32, PlaneF32, PlaneI32, inverse_5_3_level, inverse_9_7_level};
use crate::tier1::{
    CoefficientReconstruction, DecodedCodeblock, SubbandCoefficientPlane, Tier1BoundWorkPlan,
    Tier1DecodeError,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileCoefficientPlanes {
    pub components: Vec<ComponentSubbandPlanes>,
    pub decoded_codeblocks: usize,
    pub decoded_passes: u32,
    pub nonzero_coefficients: usize,
}

impl TileCoefficientPlanes {
    pub fn decode_arithmetic(
        component_layouts: &[TileComponentLayout],
        work_plan: &Tier1BoundWorkPlan<'_>,
        codeblock_style: crate::codestream::CodeblockStyle,
    ) -> Result<Self, Tier1DecodeError> {
        Self::decode_arithmetic_with_reconstruction(
            component_layouts,
            work_plan,
            codeblock_style,
            CoefficientReconstruction::Midpoint,
        )
    }

    pub fn decode_arithmetic_with_reconstruction(
        component_layouts: &[TileComponentLayout],
        work_plan: &Tier1BoundWorkPlan<'_>,
        codeblock_style: crate::codestream::CodeblockStyle,
        reconstruction: CoefficientReconstruction,
    ) -> Result<Self, Tier1DecodeError> {
        let mut components = component_layouts
            .iter()
            .enumerate()
            .map(|(component, layout)| ComponentSubbandPlanes::new(component as u16, layout))
            .collect::<Result<Vec<_>, _>>()?;
        let decoded = work_plan
            .decode_arithmetic_passes_with_reconstruction(codeblock_style, reconstruction)?;
        let decoded_codeblocks = decoded.codeblocks.len();
        let decoded_passes = decoded.decoded_passes;
        let nonzero_coefficients = decoded
            .codeblocks
            .iter()
            .map(|codeblock| codeblock.nonzero_coefficients)
            .sum();

        for decoded in &decoded.codeblocks {
            let component = components
                .get_mut(decoded.id.component as usize)
                .ok_or_else(|| {
                    Tier1DecodeError::new(format!(
                        "component {} outside tile component list",
                        decoded.id.component
                    ))
                })?;
            component.place_codeblock(&decoded)?;
        }

        Ok(Self {
            components,
            decoded_codeblocks,
            decoded_passes,
            nonzero_coefficients,
        })
    }

    pub fn component(&self, component: u16) -> Option<&ComponentSubbandPlanes> {
        self.components
            .iter()
            .find(|planes| planes.component == component)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentSubbandPlanes {
    pub component: u16,
    pub planes: Vec<ResolutionSubbandPlane>,
}

impl ComponentSubbandPlanes {
    pub fn new(component: u16, layout: &TileComponentLayout) -> Result<Self, Tier1DecodeError> {
        let mut planes = Vec::new();
        for resolution in &layout.resolutions {
            for subband in &resolution.subbands {
                planes.push(ResolutionSubbandPlane {
                    resolution: resolution.level,
                    layout: subband.clone(),
                    plane: SubbandCoefficientPlane::new(subband)?,
                });
            }
        }

        Ok(Self { component, planes })
    }

    pub fn place_codeblock(
        &mut self,
        codeblock: &DecodedCodeblock,
    ) -> Result<(), Tier1DecodeError> {
        if codeblock.id.component != self.component {
            return Err(Tier1DecodeError::new(format!(
                "codeblock component {} does not match component plane {}",
                codeblock.id.component, self.component
            )));
        }
        let target = self
            .planes
            .iter_mut()
            .find(|plane| {
                plane.resolution == codeblock.id.resolution
                    && plane.layout.kind == codeblock.id.subband
            })
            .ok_or_else(|| {
                Tier1DecodeError::new(format!(
                    "missing plane for component {}, resolution {}, subband {}",
                    codeblock.id.component, codeblock.id.resolution, codeblock.id.subband
                ))
            })?;

        target.plane.place_codeblock(&target.layout, codeblock)
    }

    pub fn plane(&self, resolution: u8, subband: SubbandKind) -> Option<&SubbandCoefficientPlane> {
        self.planes
            .iter()
            .find(|plane| plane.resolution == resolution && plane.layout.kind == subband)
            .map(|plane| &plane.plane)
    }

    pub fn inverse_5_3(&self) -> Result<PlaneI32, Tier1DecodeError> {
        let max_resolution = self
            .planes
            .iter()
            .map(|plane| plane.resolution)
            .max()
            .unwrap_or(0);
        self.inverse_5_3_until(max_resolution)
    }

    pub fn inverse_5_3_until(&self, max_resolution: u8) -> Result<PlaneI32, Tier1DecodeError> {
        let mut current = plane_to_i32(
            self.plane(0, SubbandKind::Ll)
                .ok_or_else(|| Tier1DecodeError::new("missing resolution 0 LL plane"))?,
        );
        let available_max_resolution = self
            .planes
            .iter()
            .map(|plane| plane.resolution)
            .max()
            .unwrap_or(0);
        if max_resolution > available_max_resolution {
            return Err(Tier1DecodeError::new(format!(
                "requested resolution {max_resolution}, but component only has {available_max_resolution}"
            )));
        }

        for resolution in 1..=max_resolution {
            let hl = self.require_plane(resolution, SubbandKind::Hl)?;
            let lh = self.require_plane(resolution, SubbandKind::Lh)?;
            let hh = self.require_plane(resolution, SubbandKind::Hh)?;
            current = inverse_5_3_level(
                Band2D {
                    width: current.width,
                    height: current.height,
                    samples: &current.samples,
                },
                band(hl),
                band(lh),
                band(hh),
            )
            .map_err(|err| Tier1DecodeError::new(err.message))?;
        }

        Ok(current)
    }

    pub fn inverse_9_7(&self) -> Result<PlaneF32, Tier1DecodeError> {
        let max_resolution = self
            .planes
            .iter()
            .map(|plane| plane.resolution)
            .max()
            .unwrap_or(0);
        self.inverse_9_7_until(max_resolution)
    }

    pub fn inverse_9_7_until(&self, max_resolution: u8) -> Result<PlaneF32, Tier1DecodeError> {
        let mut current = plane_to_f32(
            self.plane(0, SubbandKind::Ll)
                .ok_or_else(|| Tier1DecodeError::new("missing resolution 0 LL plane"))?,
        );
        let available_max_resolution = self
            .planes
            .iter()
            .map(|plane| plane.resolution)
            .max()
            .unwrap_or(0);
        if max_resolution > available_max_resolution {
            return Err(Tier1DecodeError::new(format!(
                "requested resolution {max_resolution}, but component only has {available_max_resolution}"
            )));
        }

        for resolution in 1..=max_resolution {
            let hl = plane_to_f32(self.require_plane(resolution, SubbandKind::Hl)?);
            let lh = plane_to_f32(self.require_plane(resolution, SubbandKind::Lh)?);
            let hh = plane_to_f32(self.require_plane(resolution, SubbandKind::Hh)?);
            current = inverse_9_7_level(
                Band2DF32 {
                    width: current.width,
                    height: current.height,
                    samples: &current.samples,
                },
                band_f32(&hl),
                band_f32(&lh),
                band_f32(&hh),
            )
            .map_err(|err| Tier1DecodeError::new(err.message))?;
        }

        Ok(current)
    }

    fn require_plane(
        &self,
        resolution: u8,
        subband: SubbandKind,
    ) -> Result<&SubbandCoefficientPlane, Tier1DecodeError> {
        self.plane(resolution, subband).ok_or_else(|| {
            Tier1DecodeError::new(format!(
                "missing component {} resolution {} {} plane",
                self.component, resolution, subband
            ))
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolutionSubbandPlane {
    pub resolution: u8,
    pub layout: SubbandLayout,
    pub plane: SubbandCoefficientPlane,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComponentFloatSubbandPlanes {
    pub component: u16,
    pub planes: Vec<ResolutionFloatSubbandPlane>,
}

impl ComponentFloatSubbandPlanes {
    pub fn from_component_with_scales(
        component: &ComponentSubbandPlanes,
        mut scale_for: impl FnMut(u8, SubbandKind) -> f32,
    ) -> Self {
        let planes = component
            .planes
            .iter()
            .map(|plane| {
                let scale = scale_for(plane.resolution, plane.layout.kind);
                ResolutionFloatSubbandPlane {
                    resolution: plane.resolution,
                    layout: plane.layout.clone(),
                    plane: FloatSubbandPlane {
                        kind: plane.layout.kind,
                        width: plane.plane.width,
                        height: plane.plane.height,
                        samples: plane
                            .plane
                            .coefficients
                            .iter()
                            .map(|coefficient| *coefficient as f32 * scale)
                            .collect(),
                    },
                }
            })
            .collect();

        Self {
            component: component.component,
            planes,
        }
    }

    /// Dequantize Tier-1 midpoint coefficients stored with eight fractional bits.
    pub fn from_component_q8_with_qcd(
        component: &ComponentSubbandPlanes,
        qcd: &QcdSegment,
        precision: u8,
        decomposition_levels: u8,
    ) -> Result<Self, Tier1DecodeError> {
        let mut result = Self::from_component_with_qcd(component, qcd, precision, decomposition_levels)?;
        for plane in &mut result.planes {
            for sample in &mut plane.plane.samples { *sample /= 256.0; }
        }
        Ok(result)
    }

    pub fn from_component_with_qcd(
        component: &ComponentSubbandPlanes,
        qcd: &QcdSegment,
        precision: u8,
        decomposition_levels: u8,
    ) -> Result<Self, Tier1DecodeError> {
        let scale_map = QcdScaleMap::new(qcd, precision, decomposition_levels)?;
        let mut planes = Vec::with_capacity(component.planes.len());
        for plane in &component.planes {
            let scale = scale_map.scale_for(plane.resolution, plane.layout.kind)?;
            planes.push(ResolutionFloatSubbandPlane {
                resolution: plane.resolution,
                layout: plane.layout.clone(),
                plane: FloatSubbandPlane {
                    kind: plane.layout.kind,
                    width: plane.plane.width,
                    height: plane.plane.height,
                    samples: plane
                        .plane
                        .coefficients
                        .iter()
                        .map(|coefficient| *coefficient as f32 * scale)
                        .collect(),
                },
            });
        }

        Ok(Self {
            component: component.component,
            planes,
        })
    }

    pub fn plane(&self, resolution: u8, subband: SubbandKind) -> Option<&FloatSubbandPlane> {
        self.planes
            .iter()
            .find(|plane| plane.resolution == resolution && plane.layout.kind == subband)
            .map(|plane| &plane.plane)
    }

    pub fn inverse_9_7(&self) -> Result<PlaneF32, Tier1DecodeError> {
        let max_resolution = self
            .planes
            .iter()
            .map(|plane| plane.resolution)
            .max()
            .unwrap_or(0);
        self.inverse_9_7_until(max_resolution)
    }

    pub fn inverse_9_7_until(&self, max_resolution: u8) -> Result<PlaneF32, Tier1DecodeError> {
        let ll = self
            .plane(0, SubbandKind::Ll)
            .ok_or_else(|| Tier1DecodeError::new("missing resolution 0 LL plane"))?;
        let mut current = PlaneF32 {
            width: ll.width,
            height: ll.height,
            samples: ll.samples.clone(),
        };
        let available_max_resolution = self
            .planes
            .iter()
            .map(|plane| plane.resolution)
            .max()
            .unwrap_or(0);
        if max_resolution > available_max_resolution {
            return Err(Tier1DecodeError::new(format!(
                "requested resolution {max_resolution}, but component only has {available_max_resolution}"
            )));
        }

        for resolution in 1..=max_resolution {
            let hl = self.require_plane(resolution, SubbandKind::Hl)?;
            let lh = self.require_plane(resolution, SubbandKind::Lh)?;
            let hh = self.require_plane(resolution, SubbandKind::Hh)?;
            current = inverse_9_7_level(
                Band2DF32 {
                    width: current.width,
                    height: current.height,
                    samples: &current.samples,
                },
                float_band(hl),
                float_band(lh),
                float_band(hh),
            )
            .map_err(|err| Tier1DecodeError::new(err.message))?;
        }

        Ok(current)
    }

    fn require_plane(
        &self,
        resolution: u8,
        subband: SubbandKind,
    ) -> Result<&FloatSubbandPlane, Tier1DecodeError> {
        self.plane(resolution, subband).ok_or_else(|| {
            Tier1DecodeError::new(format!(
                "missing component {} resolution {} {} float plane",
                self.component, resolution, subband
            ))
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolutionFloatSubbandPlane {
    pub resolution: u8,
    pub layout: SubbandLayout,
    pub plane: FloatSubbandPlane,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FloatSubbandPlane {
    pub kind: SubbandKind,
    pub width: u32,
    pub height: u32,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QcdScaleMap {
    style: QuantizationStyle,
    steps: Vec<QuantizationStep>,
    precision: u8,
    decomposition_levels: u8,
}

impl QcdScaleMap {
    fn new(
        qcd: &QcdSegment,
        precision: u8,
        decomposition_levels: u8,
    ) -> Result<Self, Tier1DecodeError> {
        if precision == 0 || precision > 31 {
            return Err(Tier1DecodeError::new(format!(
                "cannot build QCD dequantization scales for precision {precision}"
            )));
        }
        Ok(Self {
            style: qcd.style,
            steps: qcd
                .steps()
                .map_err(|err| Tier1DecodeError::new(format!("QCD step parse failed: {err}")))?,
            precision,
            decomposition_levels,
        })
    }

    fn scale_for(&self, resolution: u8, subband: SubbandKind) -> Result<f32, Tier1DecodeError> {
        let base_scale = match self.style {
            QuantizationStyle::NoQuantization => 1.0,
            QuantizationStyle::ScalarExpounded => {
                let step = self.expounded_step(resolution, subband)?;
                quantization_step_scale(step, self.precision, subband)
            }
            QuantizationStyle::ScalarDerived => {
                let step = self.derived_step(resolution, subband)?;
                quantization_step_scale(step, self.precision, subband)
            }
            QuantizationStyle::Unknown(value) => {
                return Err(Tier1DecodeError::new(format!(
                    "unsupported QCD quantization style {value}"
                )));
            }
        };
        // Match the 9/7 normalization used by FFmpeg/OpenJPEG: after the
        // inverse 1D pass has DC gain 1/K, LL carries K^2 per decomposition.
        let k: f32 = 1.2301741;
        let nl = self.decomposition_levels as i32;
        let r = resolution as i32;
        let k_power = match subband {
            SubbandKind::Ll => 2 * nl,
            SubbandKind::Hl | SubbandKind::Lh => 2 * (nl - r) - 1,
            SubbandKind::Hh => 2 * (nl - r) - 4,
        };
        Ok(base_scale * k.powi(k_power))
    }

    fn expounded_step(
        &self,
        resolution: u8,
        subband: SubbandKind,
    ) -> Result<QuantizationStep, Tier1DecodeError> {
        let index = qcd_step_index(resolution, subband).ok_or_else(|| {
            Tier1DecodeError::new(format!(
                "resolution {resolution} subband {subband} has no QCD step index"
            ))
        })?;
        self.steps.get(index).copied().ok_or_else(|| {
            Tier1DecodeError::new(format!(
                "QCD has {} step(s), missing index {index} for R{resolution} {subband} with {} decomposition level(s)",
                self.steps.len(),
                self.decomposition_levels
            ))
        })
    }

    fn derived_step(
        &self,
        resolution: u8,
        subband: SubbandKind,
    ) -> Result<QuantizationStep, Tier1DecodeError> {
        let base = self.steps.first().copied().ok_or_else(|| {
            Tier1DecodeError::new("scalar derived QCD has no LL quantization step")
        })?;
        let subband_decomposition_levels = match (resolution, subband) {
            (0, SubbandKind::Ll) => self.decomposition_levels,
            (1.., SubbandKind::Hl | SubbandKind::Lh | SubbandKind::Hh)
                if resolution <= self.decomposition_levels =>
            {
                self.decomposition_levels - resolution + 1
            }
            _ => {
                return Err(Tier1DecodeError::new(format!(
                    "resolution {resolution} subband {subband} cannot derive a QCD step for {} decomposition level(s)",
                    self.decomposition_levels
                )));
            }
        };
        let exponent = i32::from(base.exponent) + i32::from(subband_decomposition_levels)
            - i32::from(self.decomposition_levels);
        if !(0..=31).contains(&exponent) {
            return Err(Tier1DecodeError::new(format!(
                "derived QCD exponent {exponent} is outside the 5-bit range for R{resolution} {subband}"
            )));
        }

        Ok(QuantizationStep {
            exponent: exponent as u8,
            mantissa: base.mantissa,
            raw: 0,
        })
    }
}

fn qcd_step_index(resolution: u8, subband: SubbandKind) -> Option<usize> {
    match (resolution, subband) {
        (0, SubbandKind::Ll) => Some(0),
        (1.., SubbandKind::Hl) => Some(1 + (usize::from(resolution) - 1) * 3),
        (1.., SubbandKind::Lh) => Some(2 + (usize::from(resolution) - 1) * 3),
        (1.., SubbandKind::Hh) => Some(3 + (usize::from(resolution) - 1) * 3),
        _ => None,
    }
}

fn quantization_step_scale(step: QuantizationStep, precision: u8, subband: SubbandKind) -> f32 {
    let nominal_range_bits = i32::from(precision) + subband_gain_bits(subband);
    2.0f32.powi(nominal_range_bits - i32::from(step.exponent))
        * (1.0 + f32::from(step.mantissa) / 2048.0)
}

fn subband_gain_bits(subband: SubbandKind) -> i32 {
    match subband {
        SubbandKind::Ll => 0,
        SubbandKind::Hl | SubbandKind::Lh => 1,
        SubbandKind::Hh => 2,
    }
}

fn plane_to_i32(plane: &SubbandCoefficientPlane) -> PlaneI32 {
    PlaneI32 {
        width: plane.width,
        height: plane.height,
        samples: plane.coefficients.clone(),
    }
}

fn plane_to_f32(plane: &SubbandCoefficientPlane) -> PlaneF32 {
    PlaneF32 {
        width: plane.width,
        height: plane.height,
        samples: plane
            .coefficients
            .iter()
            .map(|coefficient| *coefficient as f32)
            .collect(),
    }
}

fn band(plane: &SubbandCoefficientPlane) -> Band2D<'_> {
    Band2D {
        width: plane.width,
        height: plane.height,
        samples: &plane.coefficients,
    }
}

fn band_f32(plane: &PlaneF32) -> Band2DF32<'_> {
    Band2DF32 {
        width: plane.width,
        height: plane.height,
        samples: &plane.samples,
    }
}

fn float_band(plane: &FloatSubbandPlane) -> Band2DF32<'_> {
    Band2DF32 {
        width: plane.width,
        height: plane.height,
        samples: &plane.samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codestream::{ResolutionLayout, SubbandLayout};
    use crate::tier1::{CodeblockCompressedSegment, CodeblockId, CompressedCodeblock};

    #[test]
    fn places_codeblocks_into_component_subband_planes() {
        let component_layout = TileComponentLayout {
            width: 8,
            height: 4,
            resolutions: vec![ResolutionLayout {
                level: 0,
                width: 8,
                height: 4,
                precincts_x: 1,
                precincts_y: 1,
                precinct_width: 32768,
                precinct_height: 32768,
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
        let mut component = ComponentSubbandPlanes::new(0, &component_layout).expect("component");
        let input = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 0,
                subband: SubbandKind::Ll,
                codeblock_x: 1,
                codeblock_y: 0,
            },
            width: 4,
            height: 4,
            magnitude_bitplanes: 8,
            zero_bit_planes: 0,
            segments: Vec::new(),
        };
        let mut decoded = DecodedCodeblock::new(&input).expect("decoded");
        decoded.set_coefficient(0, 0, 7).expect("coefficient");
        decoded.set_coefficient(3, 3, 9).expect("coefficient");

        component.place_codeblock(&decoded).expect("place");

        let plane = component.plane(0, SubbandKind::Ll).expect("plane");
        assert_eq!(plane.coefficient(4, 0), Some(7));
        assert_eq!(plane.coefficient(7, 3), Some(9));
        assert_eq!(plane.coefficient(3, 0), Some(0));
    }

    #[test]
    fn decodes_arithmetic_work_plan_into_tile_planes() {
        let component_layout = TileComponentLayout {
            width: 1,
            height: 1,
            resolutions: vec![ResolutionLayout {
                level: 0,
                width: 1,
                height: 1,
                precincts_x: 1,
                precincts_y: 1,
                precinct_width: 32768,
                precinct_height: 32768,
                subbands: vec![SubbandLayout {
                    kind: SubbandKind::Ll,
                    width: 1,
                    height: 1,
                    codeblock_width: 1,
                    codeblock_height: 1,
                    codeblocks_x: 1,
                    codeblocks_y: 1,
                }],
            }],
        };
        let codeblock = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 0,
                subband: SubbandKind::Ll,
                codeblock_x: 0,
                codeblock_y: 0,
            },
            width: 1,
            height: 1,
            magnitude_bitplanes: 8,
            zero_bit_planes: 0,
            segments: vec![CodeblockCompressedSegment {
                packet_index: 0,
                layer: 0,
                offset: 0,
                byte_len: 3,
                coding_passes: 1,
            }],
        };
        let tile_data = [0x00, 0x00, 0x00];
        let bound = crate::tier1::Tier1BoundWorkPlan {
            codeblocks: vec![codeblock.bind_segments(&tile_data).expect("bound")],
        };

        let planes = TileCoefficientPlanes::decode_arithmetic(
            &[component_layout],
            &bound,
            crate::codestream::CodeblockStyle::from_raw(0),
        )
        .expect("decode planes");

        assert_eq!(planes.decoded_codeblocks, 1);
        assert_eq!(planes.decoded_passes, 1);
        assert_eq!(
            planes
                .component(0)
                .and_then(|component| component.plane(0, SubbandKind::Ll))
                .and_then(|plane| plane.coefficient(0, 0)),
            Some(191)
        );
    }

    #[test]
    fn reconstructs_component_plane_from_one_5_3_level() {
        let component_layout = TileComponentLayout {
            width: 5,
            height: 4,
            resolutions: vec![
                ResolutionLayout {
                    level: 0,
                    width: 3,
                    height: 2,
                    precincts_x: 1,
                    precincts_y: 1,
                    precinct_width: 32768,
                    precinct_height: 32768,
                    subbands: vec![SubbandLayout {
                        kind: SubbandKind::Ll,
                        width: 3,
                        height: 2,
                        codeblock_width: 3,
                        codeblock_height: 2,
                        codeblocks_x: 1,
                        codeblocks_y: 1,
                    }],
                },
                ResolutionLayout {
                    level: 1,
                    width: 5,
                    height: 4,
                    precincts_x: 1,
                    precincts_y: 1,
                    precinct_width: 32768,
                    precinct_height: 32768,
                    subbands: vec![
                        SubbandLayout {
                            kind: SubbandKind::Hl,
                            width: 2,
                            height: 2,
                            codeblock_width: 2,
                            codeblock_height: 2,
                            codeblocks_x: 1,
                            codeblocks_y: 1,
                        },
                        SubbandLayout {
                            kind: SubbandKind::Lh,
                            width: 3,
                            height: 2,
                            codeblock_width: 3,
                            codeblock_height: 2,
                            codeblocks_x: 1,
                            codeblocks_y: 1,
                        },
                        SubbandLayout {
                            kind: SubbandKind::Hh,
                            width: 2,
                            height: 2,
                            codeblock_width: 2,
                            codeblock_height: 2,
                            codeblocks_x: 1,
                            codeblocks_y: 1,
                        },
                    ],
                },
            ],
        };
        let mut component = ComponentSubbandPlanes::new(0, &component_layout).expect("component");
        component
            .planes
            .iter_mut()
            .find(|plane| plane.resolution == 0 && plane.layout.kind == SubbandKind::Ll)
            .expect("ll")
            .plane
            .coefficients = vec![4, -3, 12, 1, 5, 9];
        component
            .planes
            .iter_mut()
            .find(|plane| plane.resolution == 1 && plane.layout.kind == SubbandKind::Hl)
            .expect("hl")
            .plane
            .coefficients = vec![7, -5, 2, 3];
        component
            .planes
            .iter_mut()
            .find(|plane| plane.resolution == 1 && plane.layout.kind == SubbandKind::Lh)
            .expect("lh")
            .plane
            .coefficients = vec![6, 8, -4, 0, 11, -7];
        component
            .planes
            .iter_mut()
            .find(|plane| plane.resolution == 1 && plane.layout.kind == SubbandKind::Hh)
            .expect("hh")
            .plane
            .coefficients = vec![1, -2, 4, -3];

        let plane = component.inverse_5_3().expect("inverse");

        assert_eq!(plane.width, 5);
        assert_eq!(plane.height, 4);
        assert_eq!(
            plane.samples,
            vec![
                -3, 1, -8, 0, 15, 3, 7, 3, 4, 9, -1, 0, -1, 8, 9, -3, 8, 10, 7, 3
            ]
        );
    }

    #[test]
    fn reconstructs_component_plane_from_one_9_7_level() {
        let mut component = one_level_component();

        set_plane(&mut component, 0, SubbandKind::Ll, vec![4, -3, 12, 1, 5, 9]);
        set_plane(&mut component, 1, SubbandKind::Hl, vec![7, -5, 2, 3]);
        set_plane(
            &mut component,
            1,
            SubbandKind::Lh,
            vec![6, 8, -4, 0, 11, -7],
        );
        set_plane(&mut component, 1, SubbandKind::Hh, vec![1, -2, 4, -3]);

        let plane = component.inverse_9_7().expect("inverse");

        assert_eq!(plane.width, 5);
        assert_eq!(plane.height, 4);
        assert_f32_slices_close(
            &plane.samples,
            &[
                -4.9983344,
                -0.7467388,
                -9.613315,
                -0.2599589,
                15.541199,
                2.1961172,
                9.223418,
                4.208291,
                2.3557115,
                7.369725,
                -0.33785442,
                -3.324371,
                -4.599637,
                8.222093,
                6.7605853,
                -5.3024955,
                8.784266,
                10.593021,
                7.070576,
                -1.0562062,
            ],
        );
        assert_f32_close(plane.sample(4, 3).unwrap(), -1.0562062);
        assert_eq!(plane.sample(5, 3), None);
    }

    #[test]
    fn inverse_9_7_until_rejects_unavailable_resolution() {
        let component = one_level_component();

        let err = component
            .inverse_9_7_until(2)
            .expect_err("resolution must be unavailable");

        assert!(err.message.contains("requested resolution 2"));
    }

    #[test]
    fn float_subband_component_applies_plane_scales_before_9_7() {
        let mut component = one_level_component();

        set_plane(&mut component, 0, SubbandKind::Ll, vec![4, -3, 12, 1, 5, 9]);
        set_plane(&mut component, 1, SubbandKind::Hl, vec![7, -5, 2, 3]);
        set_plane(
            &mut component,
            1,
            SubbandKind::Lh,
            vec![6, 8, -4, 0, 11, -7],
        );
        set_plane(&mut component, 1, SubbandKind::Hh, vec![1, -2, 4, -3]);

        let float_component =
            ComponentFloatSubbandPlanes::from_component_with_scales(&component, |_, subband| {
                match subband {
                    SubbandKind::Ll => 1.0,
                    SubbandKind::Hl => 0.5,
                    SubbandKind::Lh => 2.0,
                    SubbandKind::Hh => -1.0,
                }
            });

        assert_eq!(
            float_component
                .plane(1, SubbandKind::Hl)
                .expect("HL")
                .samples,
            vec![3.5, -2.5, 1.0, 1.5]
        );
        assert_eq!(
            float_component
                .plane(1, SubbandKind::Hh)
                .expect("HH")
                .samples,
            vec![-1.0, 2.0, -4.0, 3.0]
        );

        let reconstructed = float_component.inverse_9_7().expect("inverse");

        assert_eq!(reconstructed.width, 5);
        assert_eq!(reconstructed.height, 4);
        assert_eq!(reconstructed.samples.len(), 20);
    }

    #[test]
    fn unit_scaled_float_subband_component_matches_integer_9_7_path() {
        let mut component = one_level_component();

        set_plane(&mut component, 0, SubbandKind::Ll, vec![4, -3, 12, 1, 5, 9]);
        set_plane(&mut component, 1, SubbandKind::Hl, vec![7, -5, 2, 3]);
        set_plane(
            &mut component,
            1,
            SubbandKind::Lh,
            vec![6, 8, -4, 0, 11, -7],
        );
        set_plane(&mut component, 1, SubbandKind::Hh, vec![1, -2, 4, -3]);

        let float_component =
            ComponentFloatSubbandPlanes::from_component_with_scales(&component, |_, _| 1.0);
        let expected = component.inverse_9_7().expect("integer inverse");
        let reconstructed = float_component.inverse_9_7().expect("float inverse");

        assert_f32_slices_close(&reconstructed.samples, &expected.samples);
    }

    #[test]
    fn qcd_scaled_float_subband_component_applies_expounded_steps() {
        let mut component = one_level_component();

        set_plane(&mut component, 0, SubbandKind::Ll, vec![4, -3, 12, 1, 5, 9]);
        set_plane(&mut component, 1, SubbandKind::Hl, vec![7, -5, 2, 3]);
        set_plane(
            &mut component,
            1,
            SubbandKind::Lh,
            vec![6, 8, -4, 0, 11, -7],
        );
        set_plane(&mut component, 1, SubbandKind::Hh, vec![1, -2, 4, -3]);
        let qcd = qcd_expounded(&[
            step_raw(12, 0),
            step_raw(13, 1024),
            step_raw(14, 0),
            step_raw(14, 0),
        ]);

        let float_component =
            ComponentFloatSubbandPlanes::from_component_with_qcd(&component, &qcd, 12, 1)
                .expect("QCD scales");

        const K: f32 = 1.2301741;

        assert_f32_slices_close(
            float_component
                .plane(0, SubbandKind::Ll)
                .expect("LL")
                .samples
                .as_slice(),
            &[
                4.0 * K.powi(2),
                -3.0 * K.powi(2),
                12.0 * K.powi(2),
                1.0 * K.powi(2),
                5.0 * K.powi(2),
                9.0 * K.powi(2),
            ],
        );
        assert_f32_slices_close(
            float_component
                .plane(1, SubbandKind::Hl)
                .expect("HL")
                .samples
                .as_slice(),
            &[7.0 * 1.5 / K, -5.0 * 1.5 / K, 2.0 * 1.5 / K, 3.0 * 1.5 / K],
        );
        assert_f32_slices_close(
            float_component
                .plane(1, SubbandKind::Lh)
                .expect("LH")
                .samples
                .as_slice(),
            &[
                6.0 * 0.5 / K,
                8.0 * 0.5 / K,
                -4.0 * 0.5 / K,
                0.0,
                11.0 * 0.5 / K,
                -7.0 * 0.5 / K,
            ],
        );
        assert_f32_slices_close(
            float_component
                .plane(1, SubbandKind::Hh)
                .expect("HH")
                .samples
                .as_slice(),
            &[
                1.0 / K.powi(4),
                -2.0 / K.powi(4),
                4.0 / K.powi(4),
                -3.0 / K.powi(4),
            ],
        );
    }

    #[test]
    fn qcd_scaled_float_subband_component_rejects_missing_expounded_step() {
        let component = one_level_component();
        let qcd = qcd_expounded(&[step_raw(12, 0)]);

        let err = ComponentFloatSubbandPlanes::from_component_with_qcd(&component, &qcd, 12, 1)
            .expect_err("missing steps must fail");

        assert!(err.message.contains("missing index 1"));
    }

    #[test]
    fn qcd_scaled_float_subband_component_derives_steps_from_ll() {
        let mut component = one_level_component();

        set_plane(&mut component, 0, SubbandKind::Ll, vec![4, -3, 12, 1, 5, 9]);
        set_plane(&mut component, 1, SubbandKind::Hl, vec![7, -5, 2, 3]);
        set_plane(
            &mut component,
            1,
            SubbandKind::Lh,
            vec![6, 8, -4, 0, 11, -7],
        );
        set_plane(&mut component, 1, SubbandKind::Hh, vec![1, -2, 4, -3]);
        let qcd = qcd_derived(step_raw(12, 1024));

        let float_component =
            ComponentFloatSubbandPlanes::from_component_with_qcd(&component, &qcd, 12, 1)
                .expect("QCD scales");

        const K: f32 = 1.2301741;

        assert_f32_slices_close(
            float_component
                .plane(0, SubbandKind::Ll)
                .expect("LL")
                .samples
                .as_slice(),
            &[
                6.0 * K.powi(2),
                -4.5 * K.powi(2),
                18.0 * K.powi(2),
                1.5 * K.powi(2),
                7.5 * K.powi(2),
                13.5 * K.powi(2),
            ],
        );
        assert_f32_slices_close(
            float_component
                .plane(1, SubbandKind::Hl)
                .expect("HL")
                .samples
                .as_slice(),
            &[7.0 * 3.0 / K, -5.0 * 3.0 / K, 2.0 * 3.0 / K, 3.0 * 3.0 / K],
        );
        assert_f32_slices_close(
            float_component
                .plane(1, SubbandKind::Hh)
                .expect("HH")
                .samples
                .as_slice(),
            &[
                1.0 * 6.0 / K.powi(4),
                -2.0 * 6.0 / K.powi(4),
                4.0 * 6.0 / K.powi(4),
                -3.0 * 6.0 / K.powi(4),
            ],
        );
    }

    fn one_level_component() -> ComponentSubbandPlanes {
        let component_layout = TileComponentLayout {
            width: 5,
            height: 4,
            resolutions: vec![
                ResolutionLayout {
                    level: 0,
                    width: 3,
                    height: 2,
                    precincts_x: 1,
                    precincts_y: 1,
                    precinct_width: 32768,
                    precinct_height: 32768,
                    subbands: vec![SubbandLayout {
                        kind: SubbandKind::Ll,
                        width: 3,
                        height: 2,
                        codeblock_width: 3,
                        codeblock_height: 2,
                        codeblocks_x: 1,
                        codeblocks_y: 1,
                    }],
                },
                ResolutionLayout {
                    level: 1,
                    width: 5,
                    height: 4,
                    precincts_x: 1,
                    precincts_y: 1,
                    precinct_width: 32768,
                    precinct_height: 32768,
                    subbands: vec![
                        SubbandLayout {
                            kind: SubbandKind::Hl,
                            width: 2,
                            height: 2,
                            codeblock_width: 2,
                            codeblock_height: 2,
                            codeblocks_x: 1,
                            codeblocks_y: 1,
                        },
                        SubbandLayout {
                            kind: SubbandKind::Lh,
                            width: 3,
                            height: 2,
                            codeblock_width: 3,
                            codeblock_height: 2,
                            codeblocks_x: 1,
                            codeblocks_y: 1,
                        },
                        SubbandLayout {
                            kind: SubbandKind::Hh,
                            width: 2,
                            height: 2,
                            codeblock_width: 2,
                            codeblock_height: 2,
                            codeblocks_x: 1,
                            codeblocks_y: 1,
                        },
                    ],
                },
            ],
        };
        ComponentSubbandPlanes::new(0, &component_layout).expect("component")
    }

    fn set_plane(
        component: &mut ComponentSubbandPlanes,
        resolution: u8,
        subband: SubbandKind,
        coefficients: Vec<i32>,
    ) {
        component
            .planes
            .iter_mut()
            .find(|plane| plane.resolution == resolution && plane.layout.kind == subband)
            .expect("plane")
            .plane
            .coefficients = coefficients;
    }

    fn qcd_expounded(steps: &[u16]) -> QcdSegment {
        QcdSegment {
            guard_bits: 2,
            style: QuantizationStyle::ScalarExpounded,
            raw_steps: steps
                .iter()
                .flat_map(|step| step.to_be_bytes())
                .collect::<Vec<_>>(),
        }
    }

    fn qcd_derived(step: u16) -> QcdSegment {
        QcdSegment {
            guard_bits: 2,
            style: QuantizationStyle::ScalarDerived,
            raw_steps: step.to_be_bytes().to_vec(),
        }
    }

    fn step_raw(exponent: u8, mantissa: u16) -> u16 {
        (u16::from(exponent) << 11) | (mantissa & 0x07ff)
    }

    fn assert_f32_slices_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.0001,
                "sample {index}: got {actual}, expected {expected}"
            );
        }
    }

    fn assert_f32_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 0.0001,
            "got {actual}, expected {expected}"
        );
    }
}
