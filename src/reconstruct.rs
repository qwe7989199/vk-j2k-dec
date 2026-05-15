use crate::codestream::{SubbandKind, SubbandLayout, TileComponentLayout};
use crate::dwt::{Band2D, Band2DF32, PlaneF32, PlaneI32, inverse_5_3_level, inverse_9_7_level};
use crate::tier1::{
    DecodedCodeblock, SubbandCoefficientPlane, Tier1BoundWorkPlan, Tier1DecodeError,
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
        let mut components = component_layouts
            .iter()
            .enumerate()
            .map(|(component, layout)| ComponentSubbandPlanes::new(component as u16, layout))
            .collect::<Result<Vec<_>, _>>()?;
        let mut decoded_codeblocks = 0usize;
        let mut decoded_passes = 0u32;
        let mut nonzero_coefficients = 0usize;

        for codeblock in &work_plan.codeblocks {
            let decoded = codeblock.decode_arithmetic_passes(codeblock_style)?;
            let component = components
                .get_mut(decoded.id.component as usize)
                .ok_or_else(|| {
                    Tier1DecodeError::new(format!(
                        "component {} outside tile component list",
                        decoded.id.component
                    ))
                })?;
            nonzero_coefficients += decoded
                .coefficients
                .iter()
                .filter(|coefficient| **coefficient != 0)
                .count();
            decoded_passes += decoded.decoded_passes;
            component.place_codeblock(&decoded)?;
            decoded_codeblocks += 1;
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
                -2, 1, -8, 0, 16, 4, 7, 3, 4, 10, -2, -1, -1, 8, 10, -4, 7, 9, 7, 4
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
                -4.4902697,
                -0.65487915,
                -7.339456,
                -0.48657295,
                10.977664,
                1.3847177,
                7.50584,
                3.533594,
                1.0696449,
                5.0084,
                0.011556479,
                -3.2508037,
                -4.2915163,
                6.1909294,
                4.2609515,
                -4.837547,
                7.402795,
                7.862847,
                4.539947,
                -1.3311605,
            ],
        );
        assert_f32_close(plane.sample(4, 3).unwrap(), -1.3311605);
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
