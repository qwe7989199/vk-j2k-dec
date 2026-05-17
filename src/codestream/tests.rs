#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_header() {
        let data = minimal_codestream();
        let codestream = Codestream::parse(&data).expect("valid codestream");

        let siz = codestream.siz.expect("SIZ");
        assert_eq!(siz.width, 1920);
        assert_eq!(siz.height, 1080);
        assert_eq!(siz.components.len(), 3);
        assert_eq!(siz.components[0].precision, 12);
        assert_eq!(codestream.byte_len, data.len());

        let cod = codestream.cod.expect("COD");
        assert_eq!(cod.progression_order, ProgressionOrder::Lrcp);
        assert_eq!(cod.layers, 1);
        assert_eq!(cod.codeblock_width, 64);
        assert_eq!(cod.codeblock_height, 64);
        assert_eq!(cod.transform, WaveletTransform::Irreversible97);

        let qcd = codestream.qcd.expect("QCD");
        assert_eq!(qcd.guard_bits, 2);
        assert_eq!(qcd.style, QuantizationStyle::ScalarExpounded);
        assert_eq!(
            qcd.steps().expect("quantization steps"),
            vec![QuantizationStep {
                exponent: 2,
                mantissa: 0x0234,
                raw: 0x1234,
            }]
        );

        assert_eq!(codestream.tile_parts.len(), 1);
        assert!(matches!(
            codestream.markers.last().map(|record| record.marker),
            Some(Marker::Eoc)
        ));
    }

    #[test]
    fn accepts_minimal_first_profile() {
        let data = minimal_codestream();
        let codestream = Codestream::parse(&data).expect("valid codestream");
        let report = codestream.first_profile_report();

        assert!(report.is_supported(), "{:?}", report.issues);
    }

    #[test]
    fn parses_component_quantization_override() {
        let mut data = minimal_codestream();
        let sot = data
            .windows(2)
            .position(|bytes| bytes == [0xff, 0x90])
            .expect("SOT marker");
        let qcc = [
            0xff, 0x5d, // QCC
            0x00, 0x06, // Lqcc
            0x01, // component 1, one-byte Cqcc because Csiz < 257
            0x42, // guard bits 2, scalar expounded
            0x23, 0x45,
        ];
        data.splice(sot..sot, qcc);

        let codestream = Codestream::parse(&data).expect("valid codestream");

        assert_eq!(codestream.qccs.len(), 1);
        assert_eq!(codestream.qccs[0].component, 1);
        assert_eq!(
            codestream
                .quantization_for_component(1)
                .expect("component override")
                .steps()
                .expect("steps")[0]
                .raw,
            0x2345
        );
        assert_eq!(
            codestream
                .quantization_for_component(0)
                .expect("default qcd")
                .steps()
                .expect("steps")[0]
                .raw,
            0x1234
        );
        assert!(codestream.first_profile_report().is_supported());
    }

    #[test]
    fn builds_tile_layout_for_minimal_profile() {
        let data = minimal_codestream();
        let codestream = Codestream::parse(&data).expect("valid codestream");
        let layouts = codestream.tile_layouts().expect("tile layout");

        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].components.len(), 3);
        assert_eq!(layouts[0].components[0].resolutions.len(), 6);

        let codeblocks: u32 = layouts[0].components[0]
            .resolutions
            .iter()
            .flat_map(|resolution| &resolution.subbands)
            .map(SubbandLayout::codeblock_count)
            .sum();
        assert_eq!(codeblocks, 577);

        let full_res_hl = layouts[0].components[0].resolutions[5]
            .subbands
            .iter()
            .find(|subband| subband.kind == SubbandKind::Hl)
            .expect("HL");
        assert_eq!(
            full_res_hl.codeblock_rect(14, 8),
            Some(CodeblockRect {
                x0: 896,
                y0: 512,
                x1: 960,
                y1: 540,
            })
        );
        assert_eq!(
            full_res_hl.codeblock_rect(14, 8).map(CodeblockRect::height),
            Some(28)
        );
        assert_eq!(full_res_hl.codeblock_rect(15, 8), None);
    }

    #[test]
    fn builds_lrcp_packet_sequence() {
        let data = minimal_codestream();
        let codestream = Codestream::parse(&data).expect("valid codestream");
        let packets = codestream.packet_sequence().expect("packet sequence");

        assert_eq!(packets.len(), 18);
        assert_eq!(packets[0].layer, 0);
        assert_eq!(packets[0].resolution, 0);
        assert_eq!(packets[0].component, 0);
        assert_eq!(packets[17].resolution, 5);
        assert_eq!(packets[17].component, 2);
    }

    #[test]
    fn flags_unsupported_progression() {
        let mut data = minimal_codestream();
        data[56] = 2;
        let codestream = Codestream::parse(&data).expect("valid codestream");
        let report = codestream.first_profile_report();

        assert!(!report.is_supported());
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.message.contains("progression order"))
        );
    }

    #[test]
    fn formats_codeblock_style_flags() {
        let style = CodeblockStyle::from_raw(0x01 | 0x04 | 0x20);

        assert!(style.selective_arithmetic_bypass());
        assert!(style.terminate_each_pass());
        assert!(style.segmentation_symbols());
        assert_eq!(style.to_string(), "0x25 (bypass, terminate, segmark)");
    }

    #[test]
    fn parses_qcd_no_quantization_steps() {
        let qcd = QcdSegment {
            guard_bits: 2,
            style: QuantizationStyle::NoQuantization,
            raw_steps: vec![0x28, 0x30],
        };

        assert_eq!(
            qcd.steps().expect("quantization steps"),
            vec![
                QuantizationStep {
                    exponent: 5,
                    mantissa: 0,
                    raw: 0x28,
                },
                QuantizationStep {
                    exponent: 6,
                    mantissa: 0,
                    raw: 0x30,
                },
            ]
        );
    }

    #[test]
    fn rejects_odd_sized_quantization_step_payloads() {
        let qcd = QcdSegment {
            guard_bits: 2,
            style: QuantizationStyle::ScalarDerived,
            raw_steps: vec![0x12],
        };

        assert!(qcd.steps().expect_err("must fail").message.contains("odd"));
    }

    #[test]
    fn rejects_missing_soc() {
        let err = Codestream::parse(&[0xff, 0x51]).expect_err("must fail");
        assert!(err.message.contains("SOC"));
    }

    #[test]
    fn scans_embedded_codestreams() {
        let first = minimal_codestream();
        let second = minimal_codestream();
        let mut data = vec![0, 1, 2, 3];
        data.extend_from_slice(&first);
        data.extend_from_slice(&[4, 5, 6]);
        data.extend_from_slice(&second);

        let hits = scan_codestreams(&data, 8);

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].offset, 4);
        assert_eq!(hits[0].codestream.byte_len, first.len());
        assert_eq!(hits[1].offset, 4 + first.len() + 3);
    }

    fn minimal_codestream() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xff, 0x4f]);

        data.extend_from_slice(&[0xff, 0x51]);
        data.extend_from_slice(&47u16.to_be_bytes());
        data.extend_from_slice(&0u16.to_be_bytes());
        data.extend_from_slice(&1920u32.to_be_bytes());
        data.extend_from_slice(&1080u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&1920u32.to_be_bytes());
        data.extend_from_slice(&1080u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&3u16.to_be_bytes());
        for _ in 0..3 {
            data.extend_from_slice(&[11, 1, 1]);
        }

        data.extend_from_slice(&[0xff, 0x52]);
        data.extend_from_slice(&12u16.to_be_bytes());
        data.extend_from_slice(&[0, 0]);
        data.extend_from_slice(&1u16.to_be_bytes());
        data.extend_from_slice(&[1, 5, 4, 4, 0, 0]);

        data.extend_from_slice(&[0xff, 0x5c]);
        data.extend_from_slice(&5u16.to_be_bytes());
        data.extend_from_slice(&[0x42, 0x12, 0x34]);

        data.extend_from_slice(&[0xff, 0x90]);
        data.extend_from_slice(&10u16.to_be_bytes());
        data.extend_from_slice(&0u16.to_be_bytes());
        data.extend_from_slice(&14u32.to_be_bytes());
        data.extend_from_slice(&[0, 1]);
        data.extend_from_slice(&[0xff, 0x93]);
        data.extend_from_slice(&[0xff, 0xd9]);

        data
    }
}
