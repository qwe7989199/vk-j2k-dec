#[cfg(test)]
mod tests {
    use super::*;
    use crate::codestream::{CodeblockStyle, SubbandKind, SubbandLayout};
    use crate::packet::{CodeblockSegmentRecord, PacketParseRecord};

    #[derive(Debug, Clone)]
    struct ScriptedDecisionReader {
        bits: Vec<bool>,
        contexts: Vec<usize>,
    }

    impl ScriptedDecisionReader {
        fn new(bits: &[bool]) -> Self {
            Self {
                bits: bits.iter().rev().copied().collect(),
                contexts: Vec::new(),
            }
        }
    }

    impl Tier1DecisionReader for ScriptedDecisionReader {
        fn decode_context_bit(&mut self, context: usize) -> Result<bool, Tier1DecodeError> {
            self.contexts.push(context);
            self.bits
                .pop()
                .ok_or_else(|| Tier1DecodeError::new("scripted bit reader exhausted"))
        }
    }

    #[test]
    fn builds_work_plan_from_packet_segments() {
        let summary = TilePacketParseSummary {
            consumed_bytes: 30,
            packets: vec![
                PacketParseRecord {
                    offset: 0,
                    layer: 0,
                    resolution: 0,
                    component: 0,
                    present: true,
                    header_bytes: 1,
                    body_bytes: 10,
                    segments: vec![CodeblockSegmentRecord {
                        component: 0,
                        resolution: 0,
                        subband: SubbandKind::Ll,
                        codeblock_x: 0,
                        codeblock_y: 0,
                        codeblock_width: 64,
                        codeblock_height: 64,
                        newly_included: true,
                        zero_bit_planes: Some(2),
                        coding_passes: 1,
                        body_offset: 1,
                        byte_len: 10,
                    }],
                },
                PacketParseRecord {
                    offset: 11,
                    layer: 1,
                    resolution: 0,
                    component: 0,
                    present: true,
                    header_bytes: 1,
                    body_bytes: 7,
                    segments: vec![CodeblockSegmentRecord {
                        component: 0,
                        resolution: 0,
                        subband: SubbandKind::Ll,
                        codeblock_x: 0,
                        codeblock_y: 0,
                        codeblock_width: 64,
                        codeblock_height: 64,
                        newly_included: false,
                        zero_bit_planes: None,
                        coding_passes: 2,
                        body_offset: 12,
                        byte_len: 7,
                    }],
                },
            ],
        };

        let plan = Tier1WorkPlan::from_packet_summary(&summary).expect("plan");

        assert_eq!(plan.codeblock_count(), 1);
        assert_eq!(plan.segment_count(), 2);
        assert_eq!(plan.total_bytes(), 17);
        assert_eq!(plan.total_coding_passes(), 3);
        assert_eq!(plan.codeblocks[0].width, 64);
        assert_eq!(plan.codeblocks[0].height, 64);
        assert_eq!(
            plan.dimension_bounds(),
            Some(CodeblockDimensionBounds {
                min_width: 64,
                max_width: 64,
                min_height: 64,
                max_height: 64,
            })
        );
        assert_eq!(plan.codeblocks[0].zero_bit_planes, 2);

        let decode_stubs = plan.allocate_decode_stubs().expect("decode stubs");
        assert_eq!(decode_stubs.codeblocks.len(), 1);
        assert_eq!(decode_stubs.decoded_passes, 0);
        assert_eq!(decode_stubs.pending_passes, 3);
        assert_eq!(decode_stubs.coefficient_samples(), 4096);

        let tile_data = [0xaa; 19];
        let bound = plan.bind_segment_data(&tile_data).expect("bound segments");
        assert_eq!(bound.codeblock_count(), 1);
        assert_eq!(bound.segment_count(), 2);
        assert_eq!(bound.total_bytes(), 17);
        assert_eq!(bound.codeblocks[0].total_coding_passes(), 3);
        assert_eq!(bound.codeblocks[0].segments[0].data.len(), 10);
        assert_eq!(bound.codeblocks[0].segments[1].data.len(), 7);

        let execution = bound.codeblocks[0]
            .execution_plan(CodeblockStyle::from_raw(0))
            .expect("execution plan");
        assert_eq!(execution.segments, 2);
        assert_eq!(execution.passes.len(), 3);
        assert_eq!(execution.passes[0].segment_index, 0);
        assert_eq!(execution.passes[1].segment_index, 1);
        assert_eq!(execution.passes[1].segment_pass_index, 0);
        assert_eq!(execution.passes[2].segment_pass_index, 1);
        assert_eq!(execution.passes[2].coding_mode, Tier1CodingMode::Arithmetic);

        let summary = bound
            .execution_summary(CodeblockStyle::from_raw(0))
            .expect("execution summary");
        assert_eq!(
            summary,
            Tier1ExecutionSummary {
                codeblocks: 1,
                segments: 2,
                bytes: 17,
                passes: 3,
                arithmetic_passes: 3,
                raw_bypass_passes: 0,
            }
        );
    }

    #[test]
    fn execution_plan_marks_selective_bypass_passes() {
        let input = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 1,
                subband: SubbandKind::Hl,
                codeblock_x: 0,
                codeblock_y: 0,
            },
            width: 8,
            height: 8,
            magnitude_bitplanes: 8,
            zero_bit_planes: 1,
            segments: vec![CodeblockCompressedSegment {
                packet_index: 0,
                layer: 0,
                offset: 0,
                byte_len: 12,
                coding_passes: 12,
            }],
        };
        let tile_data = [0xaa; 12];
        let bound = input.bind_segments(&tile_data).expect("bound");

        let execution = bound
            .execution_plan(CodeblockStyle::from_raw(0x01))
            .expect("execution plan");

        assert_eq!(execution.passes.len(), 12);
        assert_eq!(execution.passes[9].coding_mode, Tier1CodingMode::Arithmetic);
        assert_eq!(
            execution.passes[10].pass.kind,
            Tier1PassKind::SignificancePropagation
        );
        assert_eq!(execution.passes[10].coding_mode, Tier1CodingMode::RawBypass);
        assert_eq!(
            execution.passes[11].pass.kind,
            Tier1PassKind::MagnitudeRefinement
        );
        assert_eq!(execution.passes[11].coding_mode, Tier1CodingMode::RawBypass);
    }

    #[test]
    fn allocates_decoded_codeblock_planes() {
        let input = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 2,
                subband: SubbandKind::Hh,
                codeblock_x: 1,
                codeblock_y: 2,
            },
            width: 17,
            height: 9,
            magnitude_bitplanes: 8,
            zero_bit_planes: 3,
            segments: Vec::new(),
        };

        let decoded = DecodedCodeblock::new(&input).expect("decoded block");

        assert_eq!(decoded.coefficients.len(), 153);
        assert_eq!(decoded.coefficient(16, 8), Some(0));
        assert_eq!(decoded.coefficient(17, 8), None);
        assert_eq!(decoded.state.significant_count(), 0);
    }

    #[test]
    fn updates_decoded_codeblock_state() {
        let input = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 0,
                subband: SubbandKind::Ll,
                codeblock_x: 0,
                codeblock_y: 0,
            },
            width: 4,
            height: 4,
            magnitude_bitplanes: 8,
            zero_bit_planes: 0,
            segments: Vec::new(),
        };
        let mut decoded = DecodedCodeblock::new(&input).expect("decoded block");

        decoded.set_coefficient(2, 1, -7).expect("coefficient");
        decoded
            .state
            .mark_significant(2, 1, true)
            .expect("significant");
        decoded.state.mark_magnitude_refined(2, 1).expect("refined");
        decoded
            .state
            .mark_visited_in_bitplane(2, 1)
            .expect("visited");

        assert_eq!(decoded.coefficient(2, 1), Some(-7));
        assert_eq!(decoded.state.is_significant(2, 1), Some(true));
        assert_eq!(decoded.state.sign(2, 1), Some(true));
        assert_eq!(decoded.state.is_magnitude_refined(2, 1), Some(true));
        assert_eq!(decoded.state.is_visited_in_bitplane(2, 1), Some(true));
        assert_eq!(decoded.state.significant_count(), 1);
        decoded.state.clear_bitplane_visits();
        assert_eq!(decoded.state.is_visited_in_bitplane(2, 1), Some(false));
    }

    #[test]
    fn places_decoded_codeblock_into_subband_plane() {
        let layout = SubbandLayout {
            kind: SubbandKind::Hl,
            width: 5,
            height: 3,
            codeblock_width: 4,
            codeblock_height: 2,
            codeblocks_x: 2,
            codeblocks_y: 2,
        };
        let input = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 1,
                subband: SubbandKind::Hl,
                codeblock_x: 1,
                codeblock_y: 1,
            },
            width: 1,
            height: 1,
            magnitude_bitplanes: 8,
            zero_bit_planes: 0,
            segments: Vec::new(),
        };
        let mut codeblock = DecodedCodeblock::new(&input).expect("decoded");
        codeblock.set_coefficient(0, 0, 42).expect("coefficient");
        let mut plane = SubbandCoefficientPlane::new(&layout).expect("plane");

        plane
            .place_codeblock(&layout, &codeblock)
            .expect("place codeblock");

        assert_eq!(plane.coefficient(4, 2), Some(42));
        assert_eq!(plane.coefficient(3, 2), Some(0));
        assert_eq!(plane.coefficient(5, 2), None);
    }

    #[test]
    fn coefficient_scan_uses_four_row_stripes() {
        let coordinates = Tier1CoefficientScan::new(3, 5).collect::<Vec<_>>();

        assert_eq!(
            coordinates,
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (0, 4),
                (1, 4),
                (2, 4),
            ]
        );
    }

    #[test]
    fn significance_neighbors_count_orientation_groups() {
        let input = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 0,
                subband: SubbandKind::Ll,
                codeblock_x: 0,
                codeblock_y: 0,
            },
            width: 5,
            height: 5,
            magnitude_bitplanes: 8,
            zero_bit_planes: 0,
            segments: Vec::new(),
        };
        let mut decoded = DecodedCodeblock::new(&input).expect("decoded block");

        decoded.state.mark_significant(1, 2, false).expect("left");
        decoded.state.mark_significant(3, 2, false).expect("right");
        decoded.state.mark_significant(2, 1, false).expect("top");
        decoded
            .state
            .mark_significant(1, 1, false)
            .expect("diagonal");

        let neighbors = decoded
            .state
            .significance_neighbors(2, 2)
            .expect("neighbors");
        assert_eq!(
            neighbors,
            SignificanceNeighbors {
                horizontal: 2,
                vertical: 1,
                diagonal: 1,
            }
        );
        assert!(neighbors.has_any());
        assert_eq!(decoded.state.significance_neighbors(5, 2), None);
    }

    #[test]
    fn significance_contexts_follow_subband_orientation() {
        let input = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 1,
                subband: SubbandKind::Lh,
                codeblock_x: 0,
                codeblock_y: 0,
            },
            width: 5,
            height: 5,
            magnitude_bitplanes: 8,
            zero_bit_planes: 0,
            segments: Vec::new(),
        };
        let mut decoded = DecodedCodeblock::new(&input).expect("decoded block");
        decoded.state.mark_significant(1, 2, false).expect("left");

        assert_eq!(decoded.state.significance_context(SubbandKind::Lh, 2, 2), Some(5));
        assert_eq!(decoded.state.significance_context(SubbandKind::Ll, 2, 2), Some(5));
        assert_eq!(decoded.state.significance_context(SubbandKind::Hl, 2, 2), Some(3));

        decoded.state.mark_significant(2, 1, false).expect("top");
        assert_eq!(decoded.state.significance_context(SubbandKind::Hh, 2, 2), Some(2));
        decoded
            .state
            .mark_significant(1, 1, false)
            .expect("diagonal");
        assert_eq!(decoded.state.significance_context(SubbandKind::Hh, 2, 2), Some(5));
    }

    #[test]
    fn sign_context_predicts_context_and_xor_bit() {
        let input = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 1,
                subband: SubbandKind::Hl,
                codeblock_x: 0,
                codeblock_y: 0,
            },
            width: 5,
            height: 5,
            magnitude_bitplanes: 8,
            zero_bit_planes: 0,
            segments: Vec::new(),
        };
        let mut decoded = DecodedCodeblock::new(&input).expect("decoded block");
        decoded.state.mark_significant(1, 2, false).expect("positive left");
        decoded.state.mark_significant(2, 1, true).expect("negative top");

        assert_eq!(
            decoded.sign_context(2, 2),
            Some(SignCodingContext {
                context: 11,
                xor_bit: false,
            })
        );

        decoded.state.mark_significant(3, 2, true).expect("negative right");
        decoded.state.mark_significant(2, 3, true).expect("negative bottom");
        assert_eq!(
            decoded.sign_context(2, 2),
            Some(SignCodingContext {
                context: 10,
                xor_bit: true,
            })
        );
    }

    #[test]
    fn magnitude_refinement_context_tracks_first_refinement() {
        let input = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 1,
                subband: SubbandKind::Hh,
                codeblock_x: 0,
                codeblock_y: 0,
            },
            width: 4,
            height: 4,
            magnitude_bitplanes: 8,
            zero_bit_planes: 0,
            segments: Vec::new(),
        };
        let mut decoded = DecodedCodeblock::new(&input).expect("decoded block");

        assert_eq!(
            decoded.magnitude_refinement_context(2, 2),
            Some(TIER1_MAGNITUDE_REFINEMENT_CONTEXT_FIRST)
        );
        decoded.state.mark_significant(1, 2, false).expect("neighbor");
        assert_eq!(
            decoded.magnitude_refinement_context(2, 2),
            Some(TIER1_MAGNITUDE_REFINEMENT_CONTEXT_FIRST_WITH_NEIGHBORS)
        );
        decoded.state.mark_magnitude_refined(2, 2).expect("refined");
        assert_eq!(
            decoded.magnitude_refinement_context(2, 2),
            Some(TIER1_MAGNITUDE_REFINEMENT_CONTEXT_REFINED)
        );
    }

    #[test]
    fn coding_operation_filters_by_pass_and_current_bitplane_visit() {
        let input = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 1,
                subband: SubbandKind::Lh,
                codeblock_x: 0,
                codeblock_y: 0,
            },
            width: 4,
            height: 4,
            magnitude_bitplanes: 8,
            zero_bit_planes: 0,
            segments: Vec::new(),
        };
        let mut decoded = DecodedCodeblock::new(&input).expect("decoded block");

        assert_eq!(
            decoded.coding_operation(Tier1PassKind::Cleanup, 2, 2),
            Some(Tier1CodingOperation::Cleanup)
        );
        assert_eq!(
            decoded.coding_operation(Tier1PassKind::SignificancePropagation, 2, 2),
            None
        );

        decoded.state.mark_significant(1, 2, false).expect("neighbor");
        assert_eq!(
            decoded.coding_operation(Tier1PassKind::SignificancePropagation, 2, 2),
            Some(Tier1CodingOperation::Significance)
        );
        decoded
            .state
            .mark_visited_in_bitplane(2, 2)
            .expect("visited");
        assert_eq!(
            decoded.coding_operation(Tier1PassKind::Cleanup, 2, 2),
            None
        );

        decoded.state.clear_bitplane_visits();
        decoded.state.mark_significant(2, 2, false).expect("current");
        assert_eq!(
            decoded.coding_operation(Tier1PassKind::MagnitudeRefinement, 2, 2),
            Some(Tier1CodingOperation::MagnitudeRefinement)
        );
    }

    #[test]
    fn significance_decision_updates_coefficient_and_state() {
        let input = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 1,
                subband: SubbandKind::Hl,
                codeblock_x: 0,
                codeblock_y: 0,
            },
            width: 4,
            height: 4,
            magnitude_bitplanes: 8,
            zero_bit_planes: 0,
            segments: Vec::new(),
        };
        let mut decoded = DecodedCodeblock::new(&input).expect("decoded block");

        decoded
            .apply_significance_decision(1, 1, 5, false, None)
            .expect("insignificant");
        assert_eq!(decoded.coefficient(1, 1), Some(0));
        assert_eq!(decoded.state.is_visited_in_bitplane(1, 1), Some(true));
        assert_eq!(decoded.state.is_significant(1, 1), Some(false));

        decoded.state.clear_bitplane_visits();
        decoded
            .apply_significance_decision(1, 1, 5, true, Some(true))
            .expect("significant");
        assert_eq!(decoded.coefficient(1, 1), Some(-48));
        assert_eq!(decoded.state.is_significant(1, 1), Some(true));
        assert_eq!(decoded.state.sign(1, 1), Some(true));
        assert_eq!(decoded.state.is_visited_in_bitplane(1, 1), Some(true));
        assert!(decoded
            .apply_significance_decision(2, 2, -1, true, Some(false))
            .is_err());
    }

    #[test]
    fn magnitude_refinement_bit_updates_existing_magnitude() {
        let input = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 1,
                subband: SubbandKind::Hh,
                codeblock_x: 0,
                codeblock_y: 0,
            },
            width: 4,
            height: 4,
            magnitude_bitplanes: 8,
            zero_bit_planes: 0,
            segments: Vec::new(),
        };
        let mut decoded = DecodedCodeblock::new(&input).expect("decoded block");

        assert!(decoded
            .apply_magnitude_refinement_bit(2, 2, 2, true)
            .is_err());
        decoded
            .apply_significance_decision(2, 2, 5, true, Some(false))
            .expect("significant");
        decoded.state.clear_bitplane_visits();
        decoded
            .apply_magnitude_refinement_bit(2, 2, 2, true)
            .expect("refine");
        assert_eq!(decoded.coefficient(2, 2), Some(50));
        assert_eq!(decoded.state.is_magnitude_refined(2, 2), Some(true));
        assert_eq!(decoded.state.is_visited_in_bitplane(2, 2), Some(true));
    }

    #[test]
    fn exact_reconstruction_tracks_decoded_bits_without_half_bin() {
        let input = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 1,
                subband: SubbandKind::Hh,
                codeblock_x: 0,
                codeblock_y: 0,
            },
            width: 4,
            height: 4,
            magnitude_bitplanes: 8,
            zero_bit_planes: 0,
            segments: Vec::new(),
        };
        let mut decoded =
            DecodedCodeblock::new_with_reconstruction(&input, CoefficientReconstruction::Exact)
                .expect("decoded block");

        decoded
            .apply_significance_decision(2, 2, 5, true, Some(true))
            .expect("significant");
        assert_eq!(decoded.coefficient(2, 2), Some(-32));

        decoded.state.clear_bitplane_visits();
        decoded
            .apply_magnitude_refinement_bit(2, 2, 4, false)
            .expect("zero refinement");
        assert_eq!(decoded.coefficient(2, 2), Some(-32));

        decoded.state.clear_bitplane_visits();
        decoded
            .apply_magnitude_refinement_bit(2, 2, 3, true)
            .expect("one refinement");
        assert_eq!(decoded.coefficient(2, 2), Some(-40));
    }

    #[test]
    fn decode_pass_applies_scripted_cleanup_and_refinement_bits() {
        let input = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 1,
                subband: SubbandKind::Lh,
                codeblock_x: 0,
                codeblock_y: 0,
            },
            width: 2,
            height: 2,
            magnitude_bitplanes: 8,
            zero_bit_planes: 0,
            segments: Vec::new(),
        };
        let mut decoded = DecodedCodeblock::new(&input).expect("decoded block");
        let cleanup = Tier1Pass {
            index: 0,
            bitplane: 4,
            kind: Tier1PassKind::Cleanup,
        };
        let mut cleanup_reader =
            ScriptedDecisionReader::new(&[false, true, true, false, false]);

        let summary = decoded
            .decode_pass(cleanup, &mut cleanup_reader)
            .expect("cleanup pass");

        assert_eq!(summary.pass, cleanup);
        assert_eq!(summary.significance_bits, 4);
        assert_eq!(summary.sign_bits, 1);
        assert_eq!(summary.became_significant, 1);
        assert_eq!(summary.visited, 4);
        assert_eq!(decoded.coefficient(0, 1), Some(-24));
        assert_eq!(decoded.decoded_passes, 1);
        assert_eq!(cleanup_reader.contexts, vec![0, 0, 9, 1, 5]);

        let significance = Tier1Pass {
            index: 1,
            bitplane: 3,
            kind: Tier1PassKind::SignificancePropagation,
        };
        let mut significance_reader = ScriptedDecisionReader::new(&[true, true, false, false]);
        let summary = decoded
            .decode_pass(significance, &mut significance_reader)
            .expect("significance pass");
        assert_eq!(summary.significance_bits, 3);
        assert_eq!(summary.sign_bits, 1);
        assert_eq!(decoded.coefficient(0, 0), Some(12));

        let refinement = Tier1Pass {
            index: 2,
            bitplane: 3,
            kind: Tier1PassKind::MagnitudeRefinement,
        };
        let mut refinement_reader = ScriptedDecisionReader::new(&[true]);
        let summary = decoded
            .decode_pass(refinement, &mut refinement_reader)
            .expect("refinement pass");
        assert_eq!(summary.refinement_bits, 1);
        assert_eq!(decoded.coefficient(0, 1), Some(-28));
        assert_eq!(decoded.coefficient(0, 0), Some(12));
        assert_eq!(decoded.decoded_passes, 3);
    }

    #[test]
    fn bound_codeblock_decodes_all_arithmetic_passes() {
        let input = CompressedCodeblock {
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
        let bound = input.bind_segments(&tile_data).expect("bound");

        let decoded = bound
            .decode_arithmetic_passes(CodeblockStyle::from_raw(0))
            .expect("decode");

        assert_eq!(decoded.id, input.id);
        assert_eq!(decoded.decoded_passes, 1);
        assert_eq!(decoded.coefficient(0, 0), Some(191));
        assert_eq!(decoded.state.is_visited_in_bitplane(0, 0), Some(true));

        let exact = bound
            .decode_arithmetic_passes_with_reconstruction(
                CodeblockStyle::from_raw(0),
                CoefficientReconstruction::Exact,
            )
            .expect("exact decode");
        assert_eq!(exact.coefficient(0, 0), Some(128));
    }

    #[test]
    fn codeblock_schedule_uses_magnitude_and_zero_bitplanes() {
        let schedule = build_pass_schedule_for_codeblock(13, 1, 7);

        assert_eq!(schedule[0].bitplane, 11);
        assert_eq!(schedule[1].bitplane, 10);
        assert_eq!(schedule[2].bitplane, 10);
        assert_eq!(schedule[3].bitplane, 10);
        assert_eq!(schedule[6].bitplane, 9);
    }

    #[test]
    fn bound_work_plan_decodes_arithmetic_codeblocks_and_rejects_raw_bypass() {
        let codeblock = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 1,
                subband: SubbandKind::Hl,
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
        let bound = Tier1BoundWorkPlan {
            codeblocks: vec![codeblock.bind_segments(&tile_data).expect("bound")],
        };

        let result = bound
            .decode_arithmetic_passes(CodeblockStyle::from_raw(0))
            .expect("decode work plan");
        assert_eq!(result.codeblocks.len(), 1);
        assert_eq!(result.decoded_passes, 1);
        assert_eq!(result.pending_passes, 0);

        let raw_codeblock = CompressedCodeblock {
            id: CodeblockId {
                component: 0,
                resolution: 1,
                subband: SubbandKind::Hl,
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
                coding_passes: 12,
            }],
        };
        let raw_bound = raw_codeblock.bind_segments(&tile_data).expect("raw bound");
        let err = raw_bound
            .decode_arithmetic_passes(CodeblockStyle::from_raw(0x01))
            .expect_err("raw bypass is not handled by arithmetic-only path");
        assert!(err.message.contains("raw bypass"));
    }

    #[test]
    fn builds_tier1_pass_schedule() {
        let schedule = build_pass_schedule(2, 7);

        assert_eq!(
            schedule.iter().map(|pass| pass.kind).collect::<Vec<_>>(),
            vec![
                Tier1PassKind::Cleanup,
                Tier1PassKind::SignificancePropagation,
                Tier1PassKind::MagnitudeRefinement,
                Tier1PassKind::Cleanup,
                Tier1PassKind::SignificancePropagation,
                Tier1PassKind::MagnitudeRefinement,
                Tier1PassKind::Cleanup,
            ]
        );
        assert_eq!(schedule[0].bitplane, 28);
        assert_eq!(schedule[1].bitplane, 27);
        assert_eq!(schedule[2].bitplane, 27);
        assert_eq!(schedule[3].bitplane, 27);
        assert_eq!(schedule[6].bitplane, 26);
    }

    #[test]
    fn mq_context_transitions_follow_state_table() {
        let mut mq = MqDecoder::new();

        assert_eq!(
            mq.context(0),
            Some(MqContext {
                state: 4,
                mps: false,
            })
        );
        assert_eq!(
            mq.context(TIER1_RUN_LENGTH_CONTEXT),
            Some(MqContext {
                state: 3,
                mps: false,
            })
        );
        assert_eq!(
            mq.context(TIER1_UNIFORM_CONTEXT),
            Some(MqContext {
                state: 46,
                mps: false,
            })
        );

        mq.set_context(0, 0, false).expect("context");
        mq.transition_mps(0).expect("mps");
        assert_eq!(
            mq.context(0),
            Some(MqContext {
                state: 1,
                mps: false,
            })
        );

        mq.transition_lps(0).expect("lps");
        assert_eq!(
            mq.context(0),
            Some(MqContext {
                state: 6,
                mps: false,
            })
        );

        mq.set_context(1, 0, false).expect("context");
        mq.transition_lps(1).expect("lps");
        assert_eq!(
            mq.context(1),
            Some(MqContext {
                state: 1,
                mps: true,
            })
        );
    }

    #[test]
    fn mq_rejects_invalid_context_or_state() {
        let mut mq = MqDecoder::new();

        assert!(mq.set_context(0, 47, false).is_err());
        assert!(mq.transition_mps(19).is_err());
        assert!(mq.decode_bit(19).is_err());
        assert_eq!(mq.context(19), None);
    }

    #[test]
    fn mq_initializes_input_registers() {
        let mq = MqDecoder::with_data(&[0x12, 0x34]);
        let registers = mq.input_registers();

        assert_eq!(registers.byte_pos, 1);
        assert_eq!(registers.a, 0x8000);
        assert_eq!(registers.c, 0x091a_0000);
        assert_eq!(registers.ct, 1);
    }

    #[test]
    fn mq_renormalization_refills_input() {
        let mut mq = MqDecoder::with_data(&[0x12, 0x34]);

        mq.renormalize_input_once();
        let after_first = mq.input_registers();
        assert_eq!(after_first.byte_pos, 1);
        assert_eq!(after_first.a, 0x10000);
        assert_eq!(after_first.c, 0x1234_0000);
        assert_eq!(after_first.ct, 0);

        mq.renormalize_input_once();
        let after_refill = mq.input_registers();
        assert_eq!(after_refill.byte_pos, 2);
        assert_eq!(after_refill.a, 0x20000);
        assert_eq!(after_refill.c, 0x2469_fe00);
        assert_eq!(after_refill.ct, 7);
    }

    #[test]
    fn mq_decode_bit_updates_context_and_registers() {
        let mut mq = MqDecoder::with_data(&[0x00, 0x00, 0x00]);

        let bit = mq.decode_bit(0).expect("decode bit");
        let context = mq.context(0).expect("context");
        let registers = mq.input_registers();

        assert!(bit);
        assert_eq!(context.state, 29);
        assert!(!context.mps);
        assert_eq!(registers.a & 0x8000, 0x8000);
    }

    #[test]
    fn raw_bypass_reader_reads_bits_msb_first() {
        let mut reader = RawBypassBitReader::new(&[0b1010_0101, 0b0100_0000]);

        assert!(reader.read_bit().expect("bit"));
        assert!(!reader.read_bit().expect("bit"));
        assert_eq!(reader.read_bits(6).expect("bits"), 0b100101);
        assert!(!reader.read_bit().expect("bit"));
        assert!(reader.read_bit().expect("bit"));
        assert_eq!(reader.consumed_bytes(), 2);
    }

    #[test]
    fn raw_bypass_reader_byte_aligns() {
        let mut reader = RawBypassBitReader::new(&[0b1000_0000, 0b0100_0000]);

        assert!(reader.read_bit().expect("bit"));
        reader.byte_align();
        assert!(!reader.read_bit().expect("bit"));
        assert!(reader.read_bit().expect("bit"));
        assert_eq!(reader.consumed_bytes(), 2);
    }
}
