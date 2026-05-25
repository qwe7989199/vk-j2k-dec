fn parse_siz(marker_offset: usize, segment: &[u8]) -> Result<SizSegment, ParseError> {
    let mut pos = 0;
    let capabilities = read_u16(segment, &mut pos, marker_offset + 4)?;
    let width = read_u32(segment, &mut pos, marker_offset + 6)?;
    let height = read_u32(segment, &mut pos, marker_offset + 10)?;
    let x_origin = read_u32(segment, &mut pos, marker_offset + 14)?;
    let y_origin = read_u32(segment, &mut pos, marker_offset + 18)?;
    let tile_width = read_u32(segment, &mut pos, marker_offset + 22)?;
    let tile_height = read_u32(segment, &mut pos, marker_offset + 26)?;
    let tile_x_origin = read_u32(segment, &mut pos, marker_offset + 30)?;
    let tile_y_origin = read_u32(segment, &mut pos, marker_offset + 34)?;
    let component_count = read_u16(segment, &mut pos, marker_offset + 38)? as usize;

    let expected = 36 + component_count * 3;
    if segment.len() != expected {
        return Err(ParseError::new(
            marker_offset,
            format!(
                "SIZ length does not match component count: expected {expected}, got {}",
                segment.len()
            ),
        ));
    }

    let mut components = Vec::with_capacity(component_count);
    for index in 0..component_count {
        let base_offset = marker_offset + 40 + index * 3;
        let sample_spec = read_u8(segment, &mut pos, base_offset)?;
        let x_subsampling = read_u8(segment, &mut pos, base_offset + 1)?;
        let y_subsampling = read_u8(segment, &mut pos, base_offset + 2)?;
        components.push(ComponentSpec {
            precision: (sample_spec & 0x7f) + 1,
            is_signed: sample_spec & 0x80 != 0,
            x_subsampling,
            y_subsampling,
        });
    }

    Ok(SizSegment {
        capabilities,
        width,
        height,
        x_origin,
        y_origin,
        tile_width,
        tile_height,
        tile_x_origin,
        tile_y_origin,
        components,
    })
}

fn parse_cod(marker_offset: usize, segment: &[u8]) -> Result<CodSegment, ParseError> {
    let mut pos = 0;
    let coding_style = read_u8(segment, &mut pos, marker_offset + 4)?;
    let progression_order = ProgressionOrder::from(read_u8(segment, &mut pos, marker_offset + 5)?);
    let layers = read_u16(segment, &mut pos, marker_offset + 6)?;
    let multiple_component_transform = read_u8(segment, &mut pos, marker_offset + 8)?;
    let decomposition_levels = read_u8(segment, &mut pos, marker_offset + 9)?;
    let codeblock_width_exp = read_u8(segment, &mut pos, marker_offset + 10)?;
    let codeblock_height_exp = read_u8(segment, &mut pos, marker_offset + 11)?;
    let codeblock_style = read_u8(segment, &mut pos, marker_offset + 12)?;
    let transform = WaveletTransform::from(read_u8(segment, &mut pos, marker_offset + 13)?);

    let codeblock_width = 1u16
        .checked_shl((codeblock_width_exp + 2).into())
        .ok_or_else(|| ParseError::new(marker_offset + 10, "invalid codeblock width exponent"))?;
    let codeblock_height = 1u16
        .checked_shl((codeblock_height_exp + 2).into())
        .ok_or_else(|| ParseError::new(marker_offset + 11, "invalid codeblock height exponent"))?;

    let precinct_sizes = if coding_style & 0x01 != 0 {
        segment[pos..].to_vec()
    } else {
        Vec::new()
    };

    Ok(CodSegment {
        coding_style,
        progression_order,
        layers,
        multiple_component_transform,
        decomposition_levels,
        codeblock_width,
        codeblock_height,
        codeblock_style,
        transform,
        precinct_sizes,
    })
}

fn parse_qcd(marker_offset: usize, segment: &[u8]) -> Result<QcdSegment, ParseError> {
    let mut pos = 0;
    let sqcd = read_u8(segment, &mut pos, marker_offset + 4)?;
    Ok(QcdSegment {
        guard_bits: sqcd >> 5,
        style: QuantizationStyle::from(sqcd & 0x1f),
        raw_steps: segment[pos..].to_vec(),
    })
}

fn parse_qcc(
    marker_offset: usize,
    segment: &[u8],
    component_index_bytes: usize,
) -> Result<QccSegment, ParseError> {
    let mut pos = 0;
    let component = match component_index_bytes {
        1 => u16::from(read_u8(segment, &mut pos, marker_offset + 4)?),
        2 => read_u16(segment, &mut pos, marker_offset + 4)?,
        _ => {
            return Err(ParseError::new(
                marker_offset,
                format!("unsupported QCC component index width {component_index_bytes}"),
            ));
        }
    };
    let sqcc = read_u8(segment, &mut pos, marker_offset + 4 + component_index_bytes)?;

    Ok(QccSegment {
        component,
        quantization: QcdSegment {
            guard_bits: sqcc >> 5,
            style: QuantizationStyle::from(sqcc & 0x1f),
            raw_steps: segment[pos..].to_vec(),
        },
    })
}

fn parse_sot(marker_offset: usize, segment: &[u8]) -> Result<SotSegment, ParseError> {
    if segment.len() != 8 {
        return Err(ParseError::new(
            marker_offset,
            "SOT payload must be 8 bytes",
        ));
    }

    let mut pos = 0;
    Ok(SotSegment {
        marker_offset,
        tile_index: read_u16(segment, &mut pos, marker_offset + 4)?,
        tile_part_length: read_u32(segment, &mut pos, marker_offset + 6)?,
        tile_part_index: read_u8(segment, &mut pos, marker_offset + 10)?,
        tile_part_count: read_u8(segment, &mut pos, marker_offset + 11)?,
        data_start: 0,
        data_end: 0,
    })
}

fn parse_com(marker_offset: usize, segment: &[u8]) -> Result<CommentSegment, ParseError> {
    let mut pos = 0;
    let registration = read_u16(segment, &mut pos, marker_offset + 4)?;
    Ok(CommentSegment {
        registration,
        bytes: segment[pos..].to_vec(),
    })
}

pub fn component_layout(
    component: &ComponentSpec,
    cod: &CodSegment,
    tile_x0: u32,
    tile_y0: u32,
    tile_x1: u32,
    tile_y1: u32,
) -> TileComponentLayout {
    let x0 = ceil_div_u32(tile_x0, component.x_subsampling.into());
    let y0 = ceil_div_u32(tile_y0, component.y_subsampling.into());
    let x1 = ceil_div_u32(tile_x1, component.x_subsampling.into());
    let y1 = ceil_div_u32(tile_y1, component.y_subsampling.into());
    let width = x1.saturating_sub(x0);
    let height = y1.saturating_sub(y0);

    let resolution_count = cod.decomposition_levels + 1;
    let mut resolutions = Vec::with_capacity(resolution_count as usize);
    for level in 0..resolution_count {
        let discard_levels = cod.decomposition_levels - level;
        let scale = 1u32 << discard_levels;
        let resolution_width = ceil_div_u32(width, scale);
        let resolution_height = ceil_div_u32(height, scale);
        let previous_width = if level == 0 {
            0
        } else {
            ceil_div_u32(width, scale * 2)
        };
        let previous_height = if level == 0 {
            0
        } else {
            ceil_div_u32(height, scale * 2)
        };
        let (precinct_exp_x, precinct_exp_y) = precinct_exponents(cod, level);
        let precinct_width = 1u32 << precinct_exp_x;
        let precinct_height = 1u32 << precinct_exp_y;
        let subbands = resolution_subbands(
            level,
            resolution_width,
            resolution_height,
            previous_width,
            previous_height,
            cod.codeblock_width.into(),
            cod.codeblock_height.into(),
        );
        resolutions.push(ResolutionLayout {
            level,
            width: resolution_width,
            height: resolution_height,
            precincts_x: ceil_div_u32(resolution_width.max(1), precinct_width),
            precincts_y: ceil_div_u32(resolution_height.max(1), precinct_height),
            subbands,
        });
    }

    TileComponentLayout {
        width,
        height,
        resolutions,
    }
}

fn resolution_subbands(
    level: u8,
    width: u32,
    height: u32,
    previous_width: u32,
    previous_height: u32,
    codeblock_width: u32,
    codeblock_height: u32,
) -> Vec<SubbandLayout> {
    if level == 0 {
        return vec![subband_layout(
            SubbandKind::Ll,
            width,
            height,
            codeblock_width,
            codeblock_height,
        )];
    }

    let high_width = width.saturating_sub(previous_width);
    let high_height = height.saturating_sub(previous_height);
    vec![
        subband_layout(
            SubbandKind::Hl,
            high_width,
            previous_height,
            codeblock_width,
            codeblock_height,
        ),
        subband_layout(
            SubbandKind::Lh,
            previous_width,
            high_height,
            codeblock_width,
            codeblock_height,
        ),
        subband_layout(
            SubbandKind::Hh,
            high_width,
            high_height,
            codeblock_width,
            codeblock_height,
        ),
    ]
}

fn subband_layout(
    kind: SubbandKind,
    width: u32,
    height: u32,
    codeblock_width: u32,
    codeblock_height: u32,
) -> SubbandLayout {
    SubbandLayout {
        kind,
        width,
        height,
        codeblock_width,
        codeblock_height,
        codeblocks_x: ceil_div_u32(width, codeblock_width),
        codeblocks_y: ceil_div_u32(height, codeblock_height),
    }
}

fn precinct_exponents(cod: &CodSegment, level: u8) -> (u8, u8) {
    if cod.precinct_sizes.is_empty() {
        return (15, 15);
    }

    let byte = cod
        .precinct_sizes
        .get(level as usize)
        .copied()
        .unwrap_or_else(|| *cod.precinct_sizes.last().expect("non-empty precinct list"));
    (byte & 0x0f, byte >> 4)
}

fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
    if divisor == 0 {
        return 0;
    }
    value.div_ceil(divisor)
}

fn read_u8(data: &[u8], pos: &mut usize, offset: usize) -> Result<u8, ParseError> {
    let value = *data
        .get(*pos)
        .ok_or_else(|| ParseError::new(offset, "unexpected end of segment"))?;
    *pos += 1;
    Ok(value)
}

fn read_u16(data: &[u8], pos: &mut usize, offset: usize) -> Result<u16, ParseError> {
    if *pos + 2 > data.len() {
        return Err(ParseError::new(offset, "unexpected end of segment"));
    }
    let value = u16::from_be_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(value)
}

fn read_u32(data: &[u8], pos: &mut usize, offset: usize) -> Result<u32, ParseError> {
    if *pos + 4 > data.len() {
        return Err(ParseError::new(offset, "unexpected end of segment"));
    }
    let value = u32::from_be_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(value)
}
