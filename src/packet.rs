use std::fmt;

use crate::codestream::{SubbandKind, TileLayout};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketBitError {
    pub offset: usize,
    pub message: String,
}

impl PacketBitError {
    fn new(offset: usize, message: impl Into<String>) -> Self {
        Self {
            offset,
            message: message.into(),
        }
    }
}

impl fmt::Display for PacketBitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "packet bitstream error at byte {}: {}",
            self.offset, self.message
        )
    }
}

impl std::error::Error for PacketBitError {}

#[derive(Debug, Clone)]
pub struct PacketHeaderBitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    current_byte: u8,
    bits_left: u8,
    previous_raw_ff: bool,
}

impl<'a> PacketHeaderBitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            current_byte: 0,
            bits_left: 0,
            previous_raw_ff: false,
        }
    }

    pub fn read_bit(&mut self) -> Result<bool, PacketBitError> {
        if self.bits_left == 0 {
            self.load_next_byte()?;
        }

        self.bits_left -= 1;
        Ok(((self.current_byte >> self.bits_left) & 1) != 0)
    }

    pub fn read_bits(&mut self, count: u8) -> Result<u32, PacketBitError> {
        let mut value = 0;
        for _ in 0..count {
            value = (value << 1) | u32::from(self.read_bit()?);
        }
        Ok(value)
    }

    pub fn byte_align(&mut self) {
        self.bits_left = 0;
    }

    pub fn consumed_bytes(&self) -> usize {
        self.byte_pos
    }

    fn load_next_byte(&mut self) -> Result<(), PacketBitError> {
        let offset = self.byte_pos;
        let raw = *self
            .data
            .get(self.byte_pos)
            .ok_or_else(|| PacketBitError::new(offset, "unexpected end of packet header"))?;
        self.byte_pos += 1;

        if self.previous_raw_ff {
            if raw & 0x80 != 0 {
                return Err(PacketBitError::new(
                    offset,
                    "expected stuffed zero bit after 0xff byte",
                ));
            }
            self.current_byte = raw & 0x7f;
            self.bits_left = 7;
        } else {
            self.current_byte = raw;
            self.bits_left = 8;
        }

        self.previous_raw_ff = raw == 0xff;
        Ok(())
    }
}

pub const INITIAL_CODEBLOCK_LBLOCK: u8 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeblockHeaderState {
    pub included: bool,
    pub lblock: u8,
    pub zero_bit_planes: Option<u32>,
}

impl Default for CodeblockHeaderState {
    fn default() -> Self {
        Self {
            included: false,
            lblock: INITIAL_CODEBLOCK_LBLOCK,
            zero_bit_planes: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeblockPacketHeader {
    pub newly_included: bool,
    pub zero_bit_planes: Option<u32>,
    pub coding_passes: u16,
    pub byte_len: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePacketHeaderState {
    pub tile_index: u16,
    pub subbands: Vec<SubbandPacketHeaderState>,
}

impl TilePacketHeaderState {
    pub fn from_tile_layout(tile: &TileLayout) -> Self {
        let mut subbands = Vec::new();
        for (component_index, component) in tile.components.iter().enumerate() {
            for resolution in &component.resolutions {
                for subband in &resolution.subbands {
                    subbands.push(SubbandPacketHeaderState::new(
                        component_index as u16,
                        resolution.level,
                        subband.kind,
                        subband.codeblocks_x,
                        subband.codeblocks_y,
                    ));
                }
            }
        }

        Self {
            tile_index: tile.tile_index,
            subbands,
        }
    }

    pub fn codeblock_count(&self) -> usize {
        self.subbands
            .iter()
            .map(SubbandPacketHeaderState::codeblock_count)
            .sum()
    }

    pub fn subband_mut(
        &mut self,
        component: u16,
        resolution: u8,
        kind: SubbandKind,
    ) -> Option<&mut SubbandPacketHeaderState> {
        self.subbands.iter_mut().find(|subband| {
            subband.component == component
                && subband.resolution == resolution
                && subband.kind == kind
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubbandPacketHeaderState {
    pub component: u16,
    pub resolution: u8,
    pub kind: SubbandKind,
    pub codeblocks_x: u32,
    pub codeblocks_y: u32,
    pub inclusion_tree: TagTree,
    pub zero_bit_plane_tree: TagTree,
    pub codeblocks: Vec<CodeblockHeaderState>,
}

impl SubbandPacketHeaderState {
    fn new(
        component: u16,
        resolution: u8,
        kind: SubbandKind,
        codeblocks_x: u32,
        codeblocks_y: u32,
    ) -> Self {
        let codeblock_count = codeblocks_x.saturating_mul(codeblocks_y) as usize;
        Self {
            component,
            resolution,
            kind,
            codeblocks_x,
            codeblocks_y,
            inclusion_tree: TagTree::new(codeblocks_x.max(1), codeblocks_y.max(1)),
            zero_bit_plane_tree: TagTree::new(codeblocks_x.max(1), codeblocks_y.max(1)),
            codeblocks: vec![CodeblockHeaderState::default(); codeblock_count],
        }
    }

    pub fn codeblock_count(&self) -> usize {
        self.codeblocks.len()
    }

    pub fn codeblock_state_mut(
        &mut self,
        codeblock_x: u32,
        codeblock_y: u32,
    ) -> Option<&mut CodeblockHeaderState> {
        if codeblock_x >= self.codeblocks_x || codeblock_y >= self.codeblocks_y {
            return None;
        }

        let index = codeblock_y * self.codeblocks_x + codeblock_x;
        self.codeblocks.get_mut(index as usize)
    }

    pub fn decode_codeblock(
        &mut self,
        reader: &mut PacketHeaderBitReader<'_>,
        layer: u16,
        codeblock_x: u32,
        codeblock_y: u32,
    ) -> Result<Option<CodeblockPacketHeader>, PacketBitError> {
        if codeblock_x >= self.codeblocks_x || codeblock_y >= self.codeblocks_y {
            return Err(PacketBitError::new(
                reader.consumed_bytes(),
                format!(
                    "codeblock {codeblock_x},{codeblock_y} outside {}x{} subband",
                    self.codeblocks_x, self.codeblocks_y
                ),
            ));
        }

        let index = (codeblock_y * self.codeblocks_x + codeblock_x) as usize;
        decode_codeblock_packet_header(
            reader,
            &mut self.codeblocks[index],
            &mut self.inclusion_tree,
            &mut self.zero_bit_plane_tree,
            layer,
            codeblock_x,
            codeblock_y,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePacketParseSummary {
    pub packets: Vec<PacketParseRecord>,
    pub consumed_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketParseRecord {
    pub offset: usize,
    pub layer: u16,
    pub resolution: u8,
    pub component: u16,
    pub present: bool,
    pub header_bytes: usize,
    pub body_bytes: usize,
    pub segments: Vec<CodeblockSegmentRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeblockSegmentRecord {
    pub component: u16,
    pub resolution: u8,
    pub subband: SubbandKind,
    pub codeblock_x: u32,
    pub codeblock_y: u32,
    pub codeblock_width: u32,
    pub codeblock_height: u32,
    pub newly_included: bool,
    pub zero_bit_planes: Option<u32>,
    pub coding_passes: u16,
    pub body_offset: usize,
    pub byte_len: usize,
}

impl TilePacketParseSummary {
    pub fn packet_count(&self) -> usize {
        self.packets.len()
    }

    pub fn included_codeblocks(&self) -> usize {
        self.packets
            .iter()
            .map(|packet| packet.segments.len())
            .sum()
    }

    pub fn header_bytes(&self) -> usize {
        self.packets.iter().map(|packet| packet.header_bytes).sum()
    }

    pub fn body_bytes(&self) -> usize {
        self.packets.iter().map(|packet| packet.body_bytes).sum()
    }
}

pub fn parse_lrcp_tile_packets(
    tile_data: &[u8],
    tile_layout: &TileLayout,
    layers: u16,
) -> Result<TilePacketParseSummary, PacketBitError> {
    let mut state = TilePacketHeaderState::from_tile_layout(tile_layout);
    let mut packets = Vec::new();
    let mut offset = 0;

    for layer in 0..layers {
        for component in &tile_layout.components {
            for resolution in &component.resolutions {
                if resolution.precincts_x != 1 || resolution.precincts_y != 1 {
                    return Err(PacketBitError::new(
                        offset,
                        "packet parser currently supports only one precinct per resolution",
                    ));
                }
            }
        }

        let max_resolutions = tile_layout
            .components
            .iter()
            .map(|component| component.resolutions.len())
            .max()
            .unwrap_or(0);

        for resolution_index in 0..max_resolutions {
            for (component_index, component) in tile_layout.components.iter().enumerate() {
                let Some(resolution) = component.resolutions.get(resolution_index) else {
                    continue;
                };

                let packet_start = offset;
                let mut reader = PacketHeaderBitReader::new(&tile_data[offset..]);
                let present = reader.read_bit()?;
                let mut body_bytes = 0usize;
                let mut segments = Vec::new();

                if present {
                    for subband in &resolution.subbands {
                        let subband_state = state
                            .subband_mut(component_index as u16, resolution.level, subband.kind)
                            .ok_or_else(|| {
                                PacketBitError::new(
                                    offset + reader.consumed_bytes(),
                                    "missing packet header subband state",
                                )
                            })?;

                        for codeblock_y in 0..subband.codeblocks_y {
                            for codeblock_x in 0..subband.codeblocks_x {
                                if let Some(header) = subband_state.decode_codeblock(
                                    &mut reader,
                                    layer,
                                    codeblock_x,
                                    codeblock_y,
                                )? {
                                    let body_offset = body_bytes;
                                    body_bytes = body_bytes
                                        .checked_add(header.byte_len as usize)
                                        .ok_or_else(|| {
                                        PacketBitError::new(
                                            offset + reader.consumed_bytes(),
                                            "packet body length overflow",
                                        )
                                    })?;
                                    segments.push(CodeblockSegmentRecord {
                                        component: component_index as u16,
                                        resolution: resolution.level,
                                        subband: subband.kind,
                                        codeblock_x,
                                        codeblock_y,
                                        codeblock_width: codeblock_extent(
                                            subband.width,
                                            subband.codeblock_width,
                                            codeblock_x,
                                        ),
                                        codeblock_height: codeblock_extent(
                                            subband.height,
                                            subband.codeblock_height,
                                            codeblock_y,
                                        ),
                                        newly_included: header.newly_included,
                                        zero_bit_planes: header.zero_bit_planes,
                                        coding_passes: header.coding_passes,
                                        body_offset,
                                        byte_len: header.byte_len as usize,
                                    });
                                }
                            }
                        }
                    }
                }

                reader.byte_align();
                let header_bytes = reader.consumed_bytes();
                for segment in &mut segments {
                    segment.body_offset += packet_start + header_bytes;
                }
                offset = offset
                    .checked_add(header_bytes)
                    .and_then(|value| value.checked_add(body_bytes))
                    .ok_or_else(|| PacketBitError::new(packet_start, "packet length overflow"))?;
                if offset > tile_data.len() {
                    return Err(PacketBitError::new(
                        packet_start,
                        "packet extends past tile-part payload",
                    ));
                }

                packets.push(PacketParseRecord {
                    offset: packet_start,
                    layer,
                    resolution: resolution.level,
                    component: component_index as u16,
                    present,
                    header_bytes,
                    body_bytes,
                    segments,
                });
            }
        }
    }

    Ok(TilePacketParseSummary {
        packets,
        consumed_bytes: offset,
    })
}

pub fn parse_cprl_tile_packets(
    tile_data: &[u8],
    tile_layout: &TileLayout,
    layers: u16,
) -> Result<TilePacketParseSummary, PacketBitError> {
    let mut state = TilePacketHeaderState::from_tile_layout(tile_layout);
    let mut packets = Vec::new();
    let mut offset = 0;

    let max_resolutions = tile_layout
        .components
        .iter()
        .map(|c| c.resolutions.len())
        .max()
        .unwrap_or(0);

    for (component_index, component) in tile_layout.components.iter().enumerate() {
        // Build precinct grid for each resolution
        // Iterate positions by resolution order: for each precinct at each resolution
        for resolution_index in 0..component.resolutions.len() {
            let resolution = &component.resolutions[resolution_index];
            for precinct_y in 0..resolution.precincts_y {
                for precinct_x in 0..resolution.precincts_x {
                    for layer in 0..layers {
                        let packet_start = offset;
                        let mut reader = PacketHeaderBitReader::new(&tile_data[offset..]);
                        let present = reader.read_bit()?;
                        let mut body_bytes = 0usize;
                        let mut segments = Vec::new();

                        if present {
                            for subband in &resolution.subbands {
                                let subband_state = state
                                    .subband_mut(component_index as u16, resolution.level, subband.kind)
                                    .ok_or_else(|| {
                                        PacketBitError::new(
                                            offset + reader.consumed_bytes(),
                                            "missing packet header subband state",
                                        )
                                    })?;

                                let (cb_x_start, cb_x_end, cb_y_start, cb_y_end) =
                                    precinct_codeblock_range(subband, precinct_x, precinct_y, resolution);

                                for codeblock_y in cb_y_start..cb_y_end {
                                    for codeblock_x in cb_x_start..cb_x_end {
                                        if let Some(header) = subband_state.decode_codeblock(
                                            &mut reader,
                                            layer,
                                            codeblock_x,
                                            codeblock_y,
                                        )? {
                                            let body_offset = body_bytes;
                                            body_bytes = body_bytes
                                                .checked_add(header.byte_len as usize)
                                                .ok_or_else(|| {
                                                    PacketBitError::new(
                                                        offset + reader.consumed_bytes(),
                                                        "packet body length overflow",
                                                    )
                                                })?;
                                            segments.push(CodeblockSegmentRecord {
                                                component: component_index as u16,
                                                resolution: resolution.level,
                                                subband: subband.kind,
                                                codeblock_x,
                                                codeblock_y,
                                                codeblock_width: codeblock_extent(
                                                    subband.width,
                                                    subband.codeblock_width,
                                                    codeblock_x,
                                                ),
                                                codeblock_height: codeblock_extent(
                                                    subband.height,
                                                    subband.codeblock_height,
                                                    codeblock_y,
                                                ),
                                                newly_included: header.newly_included,
                                                zero_bit_planes: header.zero_bit_planes,
                                                coding_passes: header.coding_passes,
                                                body_offset,
                                                byte_len: header.byte_len as usize,
                                            });
                                        }
                                    }
                                }
                            }
                        }

                        reader.byte_align();
                        let header_bytes = reader.consumed_bytes();
                        for segment in &mut segments {
                            segment.body_offset += packet_start + header_bytes;
                        }
                        offset = offset
                            .checked_add(header_bytes)
                            .and_then(|v| v.checked_add(body_bytes))
                            .ok_or_else(|| PacketBitError::new(packet_start, "packet length overflow"))?;
                        if offset > tile_data.len() {
                            return Err(PacketBitError::new(
                                packet_start,
                                "packet extends past tile-part payload",
                            ));
                        }

                        packets.push(PacketParseRecord {
                            offset: packet_start,
                            layer,
                            resolution: resolution.level,
                            component: component_index as u16,
                            present,
                            header_bytes,
                            body_bytes,
                            segments,
                        });
                    }
                }
            }
        }
    }

    Ok(TilePacketParseSummary {
        packets,
        consumed_bytes: offset,
    })
}

pub fn parse_tile_packets(
    tile_data: &[u8],
    tile_layout: &TileLayout,
    layers: u16,
    progression: &crate::codestream::ProgressionOrder,
) -> Result<TilePacketParseSummary, PacketBitError> {
    use crate::codestream::ProgressionOrder;
    match progression {
        ProgressionOrder::Lrcp => parse_lrcp_tile_packets(tile_data, tile_layout, layers),
        ProgressionOrder::Cprl => parse_cprl_tile_packets(tile_data, tile_layout, layers),
        other => Err(PacketBitError::new(
            0,
            format!("unsupported progression order {other}"),
        )),
    }
}

fn precinct_codeblock_range(
    subband: &crate::codestream::SubbandLayout,
    precinct_x: u32,
    precinct_y: u32,
    resolution: &crate::codestream::ResolutionLayout,
) -> (u32, u32, u32, u32) {
    if resolution.precincts_x == 1 && resolution.precincts_y == 1 {
        return (0, subband.codeblocks_x, 0, subband.codeblocks_y);
    }
    let cblk_per_precinct_x = subband.codeblocks_x.div_ceil(resolution.precincts_x);
    let cblk_per_precinct_y = subband.codeblocks_y.div_ceil(resolution.precincts_y);
    let cb_x_start = precinct_x * cblk_per_precinct_x;
    let cb_x_end = ((precinct_x + 1) * cblk_per_precinct_x).min(subband.codeblocks_x);
    let cb_y_start = precinct_y * cblk_per_precinct_y;
    let cb_y_end = ((precinct_y + 1) * cblk_per_precinct_y).min(subband.codeblocks_y);
    (cb_x_start, cb_x_end, cb_y_start, cb_y_end)
}

fn codeblock_extent(subband_extent: u32, nominal_extent: u32, codeblock_index: u32) -> u32 {
    let start = codeblock_index.saturating_mul(nominal_extent);
    subband_extent.saturating_sub(start).min(nominal_extent)
}

pub fn decode_codeblock_packet_header(
    reader: &mut PacketHeaderBitReader<'_>,
    state: &mut CodeblockHeaderState,
    inclusion_tree: &mut TagTree,
    zero_bit_plane_tree: &mut TagTree,
    layer: u16,
    codeblock_x: u32,
    codeblock_y: u32,
) -> Result<Option<CodeblockPacketHeader>, PacketBitError> {
    let newly_included;
    let included_in_packet = if state.included {
        newly_included = false;
        reader.read_bit()?
    } else {
        newly_included = true;
        inclusion_tree.decode(codeblock_x, codeblock_y, u32::from(layer) + 1, reader)?
    };

    if !included_in_packet {
        return Ok(None);
    }

    let zero_bit_planes = if newly_included {
        let value = decode_zero_bit_planes(reader, zero_bit_plane_tree, codeblock_x, codeblock_y)?;
        state.included = true;
        state.zero_bit_planes = Some(value);
        Some(value)
    } else {
        None
    };

    let coding_passes = read_codeblock_pass_count(reader)?;
    read_lblock_increments(reader, &mut state.lblock)?;
    let byte_len = read_codeblock_data_length(reader, state.lblock, coding_passes)?;

    Ok(Some(CodeblockPacketHeader {
        newly_included,
        zero_bit_planes,
        coding_passes,
        byte_len,
    }))
}

pub fn read_codeblock_pass_count(
    reader: &mut PacketHeaderBitReader<'_>,
) -> Result<u16, PacketBitError> {
    if !reader.read_bit()? {
        return Ok(1);
    }
    if !reader.read_bit()? {
        return Ok(2);
    }

    let short = reader.read_bits(2)? as u16;
    if short < 3 {
        return Ok(3 + short);
    }

    let medium = reader.read_bits(5)? as u16;
    if medium < 31 {
        return Ok(6 + medium);
    }

    Ok(37 + reader.read_bits(7)? as u16)
}

pub fn read_lblock_increments(
    reader: &mut PacketHeaderBitReader<'_>,
    lblock: &mut u8,
) -> Result<(), PacketBitError> {
    while reader.read_bit()? {
        *lblock = lblock.checked_add(1).ok_or_else(|| {
            PacketBitError::new(reader.consumed_bytes(), "codeblock Lblock overflow")
        })?;
    }
    Ok(())
}

pub fn read_codeblock_data_length(
    reader: &mut PacketHeaderBitReader<'_>,
    lblock: u8,
    coding_passes: u16,
) -> Result<u32, PacketBitError> {
    let pass_bits = 15 - coding_passes.leading_zeros() as u8;
    let length_bits = lblock.checked_add(pass_bits).ok_or_else(|| {
        PacketBitError::new(
            reader.consumed_bytes(),
            "codeblock length bit count overflow",
        )
    })?;

    if length_bits > 31 {
        return Err(PacketBitError::new(
            reader.consumed_bytes(),
            format!("codeblock length uses too many bits: {length_bits}"),
        ));
    }

    reader.read_bits(length_bits)
}

fn decode_zero_bit_planes(
    reader: &mut PacketHeaderBitReader<'_>,
    zero_bit_plane_tree: &mut TagTree,
    codeblock_x: u32,
    codeblock_y: u32,
) -> Result<u32, PacketBitError> {
    for threshold in 1..=u32::MAX {
        if zero_bit_plane_tree.decode(codeblock_x, codeblock_y, threshold, reader)? {
            return Ok(threshold - 1);
        }
    }

    unreachable!("u32 threshold loop must return or error")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagTree {
    levels: Vec<TagTreeLevel>,
}

impl TagTree {
    pub fn new(width: u32, height: u32) -> Self {
        let mut levels = Vec::new();
        let mut level_width = width.max(1);
        let mut level_height = height.max(1);

        loop {
            levels.push(TagTreeLevel::new(level_width, level_height));
            if level_width == 1 && level_height == 1 {
                break;
            }
            level_width = level_width.div_ceil(2);
            level_height = level_height.div_ceil(2);
        }

        Self { levels }
    }

    pub fn width(&self) -> u32 {
        self.levels[0].width
    }

    pub fn height(&self) -> u32 {
        self.levels[0].height
    }

    pub fn decode(
        &mut self,
        leaf_x: u32,
        leaf_y: u32,
        threshold: u32,
        reader: &mut PacketHeaderBitReader<'_>,
    ) -> Result<bool, PacketBitError> {
        if leaf_x >= self.width() || leaf_y >= self.height() {
            return Err(PacketBitError::new(
                reader.consumed_bytes(),
                format!(
                    "tag-tree leaf {leaf_x},{leaf_y} outside {}x{} tree",
                    self.width(),
                    self.height()
                ),
            ));
        }

        let mut path = Vec::with_capacity(self.levels.len());
        let mut x = leaf_x;
        let mut y = leaf_y;
        for level_index in 0..self.levels.len() {
            let node_index = self.levels[level_index].node_index(x, y);
            path.push((level_index, node_index));
            x /= 2;
            y /= 2;
        }

        let mut lower_bound = 0;
        for (level_index, node_index) in path.into_iter().rev() {
            let node = &mut self.levels[level_index].nodes[node_index];
            if node.value < lower_bound {
                node.value = lower_bound;
            }

            while node.value < threshold && !node.known {
                if reader.read_bit()? {
                    node.known = true;
                } else {
                    node.value += 1;
                }
            }
            lower_bound = node.value;
        }

        let leaf_index = self.levels[0].node_index(leaf_x, leaf_y);
        Ok(self.levels[0].nodes[leaf_index].value < threshold)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TagTreeLevel {
    width: u32,
    height: u32,
    nodes: Vec<TagTreeNode>,
}

impl TagTreeLevel {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            nodes: vec![TagTreeNode::default(); width.saturating_mul(height) as usize],
        }
    }

    fn node_index(&self, x: u32, y: u32) -> usize {
        y.saturating_mul(self.width).saturating_add(x) as usize
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct TagTreeNode {
    value: u32,
    known: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codestream::{ResolutionLayout, SubbandLayout, TileComponentLayout, TileLayout};

    #[test]
    fn reads_bits_msb_first() {
        let mut reader = PacketHeaderBitReader::new(&[0b1010_0000]);

        assert!(reader.read_bit().expect("bit"));
        assert!(!reader.read_bit().expect("bit"));
        assert_eq!(reader.read_bits(2).expect("bits"), 0b10);
        assert_eq!(reader.consumed_bytes(), 1);
    }

    #[test]
    fn skips_stuffed_zero_after_ff() {
        let mut reader = PacketHeaderBitReader::new(&[0xff, 0x55]);

        assert_eq!(reader.read_bits(8).expect("bits"), 0xff);
        assert!(reader.read_bit().expect("stuffed payload bit"));
        assert!(!reader.read_bit().expect("stuffed payload bit"));
        assert!(reader.read_bit().expect("stuffed payload bit"));
        assert_eq!(reader.consumed_bytes(), 2);
    }

    #[test]
    fn rejects_missing_stuffed_zero_after_ff() {
        let mut reader = PacketHeaderBitReader::new(&[0xff, 0x80]);

        assert_eq!(reader.read_bits(8).expect("bits"), 0xff);
        let err = reader.read_bit().expect_err("must fail");
        assert!(err.message.contains("stuffed zero"));
    }

    #[test]
    fn byte_align_discards_remaining_bits() {
        let mut reader = PacketHeaderBitReader::new(&[0b1000_0000, 0b0100_0000]);

        assert!(reader.read_bit().expect("bit"));
        reader.byte_align();
        assert!(!reader.read_bit().expect("bit"));
        assert!(reader.read_bit().expect("bit"));
        assert_eq!(reader.consumed_bytes(), 2);
    }

    #[test]
    fn tag_tree_decodes_single_leaf_zero_value() {
        let mut tree = TagTree::new(1, 1);
        let data = pack_bits(&[true]);
        let mut reader = PacketHeaderBitReader::new(&data);

        assert!(tree.decode(0, 0, 1, &mut reader).expect("decode"));
    }

    #[test]
    fn tag_tree_can_revisit_partially_known_leaf() {
        let mut tree = TagTree::new(1, 1);
        let data = pack_bits(&[false, false, true]);
        let mut reader = PacketHeaderBitReader::new(&data);

        assert!(!tree.decode(0, 0, 2, &mut reader).expect("decode"));
        assert!(tree.decode(0, 0, 3, &mut reader).expect("decode"));
    }

    #[test]
    fn tag_tree_shares_parent_state_between_leaves() {
        let mut tree = TagTree::new(2, 1);
        let data = pack_bits(&[true, true, true]);
        let mut reader = PacketHeaderBitReader::new(&data);

        assert!(tree.decode(0, 0, 1, &mut reader).expect("decode"));
        assert!(tree.decode(1, 0, 1, &mut reader).expect("decode"));
    }

    #[test]
    fn tag_tree_rejects_out_of_range_leaf() {
        let mut tree = TagTree::new(1, 1);
        let mut reader = PacketHeaderBitReader::new(&[0]);

        let err = tree.decode(1, 0, 1, &mut reader).expect_err("must fail");
        assert!(err.message.contains("outside"));
    }

    #[test]
    fn reads_codeblock_pass_count_variants() {
        let data = pack_bits(&[
            false, // 1 pass
            true, false, // 2 passes
            true, true, false, true, // 4 passes
        ]);
        let mut reader = PacketHeaderBitReader::new(&data);

        assert_eq!(read_codeblock_pass_count(&mut reader).expect("passes"), 1);
        assert_eq!(read_codeblock_pass_count(&mut reader).expect("passes"), 2);
        assert_eq!(read_codeblock_pass_count(&mut reader).expect("passes"), 4);
    }

    #[test]
    fn decodes_new_codeblock_packet_header() {
        let data = pack_bits(&[
            true,  // inclusion tag-tree value 0
            true,  // zero bit-plane tag-tree value 0
            false, // 1 coding pass
            false, // no Lblock increment
            true, false, true, // byte length 5 with Lblock 3
        ]);
        let mut reader = PacketHeaderBitReader::new(&data);
        let mut state = CodeblockHeaderState::default();
        let mut inclusion_tree = TagTree::new(1, 1);
        let mut zero_bit_plane_tree = TagTree::new(1, 1);

        let header = decode_codeblock_packet_header(
            &mut reader,
            &mut state,
            &mut inclusion_tree,
            &mut zero_bit_plane_tree,
            0,
            0,
            0,
        )
        .expect("header")
        .expect("included");

        assert!(header.newly_included);
        assert_eq!(header.zero_bit_planes, Some(0));
        assert_eq!(header.coding_passes, 1);
        assert_eq!(header.byte_len, 5);
        assert!(state.included);
        assert_eq!(state.lblock, INITIAL_CODEBLOCK_LBLOCK);
    }

    #[test]
    fn decodes_existing_codeblock_packet_header() {
        let data = pack_bits(&[
            true, // included in this packet
            true, false, // 2 coding passes
            false, // no Lblock increment
            true, false, false, true, // byte length 9 with Lblock 3 + floor(log2(2))
        ]);
        let mut reader = PacketHeaderBitReader::new(&data);
        let mut state = CodeblockHeaderState {
            included: true,
            lblock: INITIAL_CODEBLOCK_LBLOCK,
            zero_bit_planes: Some(0),
        };
        let mut inclusion_tree = TagTree::new(1, 1);
        let mut zero_bit_plane_tree = TagTree::new(1, 1);

        let header = decode_codeblock_packet_header(
            &mut reader,
            &mut state,
            &mut inclusion_tree,
            &mut zero_bit_plane_tree,
            1,
            0,
            0,
        )
        .expect("header")
        .expect("included");

        assert!(!header.newly_included);
        assert_eq!(header.zero_bit_planes, None);
        assert_eq!(header.coding_passes, 2);
        assert_eq!(header.byte_len, 9);
    }

    #[test]
    fn decodes_absent_existing_codeblock_packet_header() {
        let data = pack_bits(&[false]);
        let mut reader = PacketHeaderBitReader::new(&data);
        let mut state = CodeblockHeaderState {
            included: true,
            lblock: INITIAL_CODEBLOCK_LBLOCK,
            zero_bit_planes: Some(0),
        };
        let mut inclusion_tree = TagTree::new(1, 1);
        let mut zero_bit_plane_tree = TagTree::new(1, 1);

        let header = decode_codeblock_packet_header(
            &mut reader,
            &mut state,
            &mut inclusion_tree,
            &mut zero_bit_plane_tree,
            1,
            0,
            0,
        )
        .expect("header");

        assert_eq!(header, None);
        assert!(state.included);
    }

    #[test]
    fn builds_tile_packet_header_state_from_layout() {
        let tile = one_subband_tile_layout(2, 1);

        let mut state = TilePacketHeaderState::from_tile_layout(&tile);

        assert_eq!(state.tile_index, 7);
        assert_eq!(state.subbands.len(), 1);
        assert_eq!(state.codeblock_count(), 2);
        let subband = state.subband_mut(0, 0, SubbandKind::Ll).expect("subband");
        assert!(subband.codeblock_state_mut(1, 0).is_some());
        assert!(subband.codeblock_state_mut(2, 0).is_none());
    }

    #[test]
    fn parses_simple_lrcp_tile_packet() {
        let tile = one_subband_tile_layout(1, 1);
        let header = pack_bits(&[
            true,  // packet present
            true,  // inclusion tag-tree value 0
            true,  // zero bit-plane tag-tree value 0
            false, // 1 coding pass
            false, // no Lblock increment
            true, false, true, // byte length 5
        ]);
        let mut data = header;
        data.extend_from_slice(&[1, 2, 3, 4, 5]);

        let summary = parse_lrcp_tile_packets(&data, &tile, 1).expect("packets");

        assert_eq!(summary.packet_count(), 1);
        assert_eq!(summary.consumed_bytes, 6);
        assert_eq!(summary.header_bytes(), 1);
        assert_eq!(summary.body_bytes(), 5);
        assert_eq!(summary.included_codeblocks(), 1);
        assert_eq!(summary.packets[0].segments.len(), 1);
        assert_eq!(summary.packets[0].segments[0].body_offset, 1);
        assert_eq!(summary.packets[0].segments[0].byte_len, 5);
        assert_eq!(summary.packets[0].segments[0].coding_passes, 1);
    }

    fn one_subband_tile_layout(codeblocks_x: u32, codeblocks_y: u32) -> TileLayout {
        TileLayout {
            tile_index: 7,
            x0: 0,
            y0: 0,
            x1: 128,
            y1: 64,
            components: vec![TileComponentLayout {
                width: 128,
                height: 64,
                resolutions: vec![ResolutionLayout {
                    level: 0,
                    width: 128,
                    height: 64,
                    precincts_x: 1,
                    precincts_y: 1,
                    subbands: vec![SubbandLayout {
                        kind: SubbandKind::Ll,
                        width: 128,
                        height: 64,
                        codeblock_width: 64,
                        codeblock_height: 64,
                        codeblocks_x,
                        codeblocks_y,
                    }],
                }],
            }],
        }
    }

    fn pack_bits(bits: &[bool]) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut current = 0;
        for (index, bit) in bits.iter().enumerate() {
            current <<= 1;
            current |= u8::from(*bit);
            if index % 8 == 7 {
                bytes.push(current);
                current = 0;
            }
        }

        let remaining = bits.len() % 8;
        if remaining != 0 {
            current <<= 8 - remaining;
            bytes.push(current);
        }

        bytes
    }
}
