#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Codestream {
    pub byte_len: usize,
    pub siz: Option<SizSegment>,
    pub cod: Option<CodSegment>,
    pub qcd: Option<QcdSegment>,
    pub comments: Vec<CommentSegment>,
    pub tile_parts: Vec<SotSegment>,
    pub markers: Vec<MarkerRecord>,
}

impl Codestream {
    pub fn parse(data: &[u8]) -> Result<Self, ParseError> {
        Parser::new(data).parse()
    }

    pub fn first_profile_report(&self) -> ProfileReport {
        let mut issues = Vec::new();

        let Some(siz) = &self.siz else {
            issues.push(ProfileIssue::new("missing SIZ marker"));
            return ProfileReport { issues };
        };

        let Some(cod) = &self.cod else {
            issues.push(ProfileIssue::new("missing COD marker"));
            return ProfileReport { issues };
        };

        if self.qcd.is_none() {
            issues.push(ProfileIssue::new("missing QCD marker"));
        }

        if siz.components.len() != 3 {
            issues.push(ProfileIssue::new(format!(
                "expected 3 components, found {}",
                siz.components.len()
            )));
        }

        for (index, component) in siz.components.iter().enumerate() {
            if !matches!(component.precision, 8 | 10 | 12 | 16) {
                issues.push(ProfileIssue::new(format!(
                    "component {index} precision {} is outside 8/10/12/16-bit target",
                    component.precision
                )));
            }
            if component.is_signed {
                issues.push(ProfileIssue::new(format!(
                    "component {index} uses signed samples"
                )));
            }
            if component.x_subsampling != 1 || component.y_subsampling != 1 {
                issues.push(ProfileIssue::new(format!(
                    "component {index} has unsupported subsampling {}x{}",
                    component.x_subsampling, component.y_subsampling
                )));
            }
        }

        if cod.progression_order != ProgressionOrder::Lrcp {
            issues.push(ProfileIssue::new(format!(
                "progression order {} is not in the first LRCP-only target",
                cod.progression_order
            )));
        }

        if !matches!(
            cod.transform,
            WaveletTransform::Irreversible97 | WaveletTransform::Reversible53
        ) {
            issues.push(ProfileIssue::new(format!(
                "unsupported wavelet transform {}",
                cod.transform
            )));
        }

        if cod.multiple_component_transform > 1 {
            issues.push(ProfileIssue::new(format!(
                "unsupported multiple component transform value {}",
                cod.multiple_component_transform
            )));
        }

        if cod.codeblock_width > 64 || cod.codeblock_height > 64 {
            issues.push(ProfileIssue::new(format!(
                "codeblock {}x{} exceeds first target limit of 64x64",
                cod.codeblock_width, cod.codeblock_height
            )));
        }

        if cod.codeblock_style != 0 {
            issues.push(ProfileIssue::new(format!(
                "codeblock style {} is outside the first style-0 target",
                cod.codeblock_style_flags()
            )));
        }

        for record in &self.markers {
            match record.marker {
                Marker::Coc | Marker::Qcc | Marker::Rgn | Marker::Pod | Marker::Ppm => {
                    issues.push(ProfileIssue::new(format!(
                        "{} marker at 0x{:08x} is outside the first target",
                        record.marker, record.offset
                    )));
                }
                _ => {}
            }
        }

        ProfileReport { issues }
    }

    pub fn tile_layouts(&self) -> Result<Vec<TileLayout>, LayoutError> {
        let siz = self
            .siz
            .as_ref()
            .ok_or_else(|| LayoutError::new("missing SIZ marker"))?;
        let cod = self
            .cod
            .as_ref()
            .ok_or_else(|| LayoutError::new("missing COD marker"))?;

        let tile_count_x =
            ceil_div_u32(siz.width.saturating_sub(siz.tile_x_origin), siz.tile_width);
        let tile_count_y = ceil_div_u32(
            siz.height.saturating_sub(siz.tile_y_origin),
            siz.tile_height,
        );
        let mut layouts = Vec::with_capacity(tile_count_x.saturating_mul(tile_count_y) as usize);

        for tile_y in 0..tile_count_y {
            for tile_x in 0..tile_count_x {
                let x0 = siz
                    .tile_x_origin
                    .saturating_add(tile_x.saturating_mul(siz.tile_width))
                    .max(siz.x_origin);
                let y0 = siz
                    .tile_y_origin
                    .saturating_add(tile_y.saturating_mul(siz.tile_height))
                    .max(siz.y_origin);
                let x1 = siz
                    .tile_x_origin
                    .saturating_add((tile_x + 1).saturating_mul(siz.tile_width))
                    .min(siz.width);
                let y1 = siz
                    .tile_y_origin
                    .saturating_add((tile_y + 1).saturating_mul(siz.tile_height))
                    .min(siz.height);

                let components = siz
                    .components
                    .iter()
                    .map(|component| component_layout(component, cod, x0, y0, x1, y1))
                    .collect();

                layouts.push(TileLayout {
                    tile_index: (tile_y * tile_count_x + tile_x) as u16,
                    x0,
                    y0,
                    x1,
                    y1,
                    components,
                });
            }
        }

        Ok(layouts)
    }

    pub fn packet_sequence(&self) -> Result<Vec<PacketLayout>, LayoutError> {
        let cod = self
            .cod
            .as_ref()
            .ok_or_else(|| LayoutError::new("missing COD marker"))?;
        if cod.progression_order != ProgressionOrder::Lrcp {
            return Err(LayoutError::new(format!(
                "packet sequence only supports LRCP for now, got {}",
                cod.progression_order
            )));
        }

        let layouts = self.tile_layouts()?;
        let mut packets = Vec::new();
        for tile in layouts {
            for layer in 0..cod.layers {
                for resolution in 0..=cod.decomposition_levels {
                    for (component_index, component) in tile.components.iter().enumerate() {
                        let Some(resolution_layout) =
                            component.resolutions.get(resolution as usize)
                        else {
                            continue;
                        };
                        for precinct_y in 0..resolution_layout.precincts_y {
                            for precinct_x in 0..resolution_layout.precincts_x {
                                packets.push(PacketLayout {
                                    tile_index: tile.tile_index,
                                    layer,
                                    resolution,
                                    component: component_index as u16,
                                    precinct_x,
                                    precinct_y,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(packets)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodestreamScanHit {
    pub offset: usize,
    pub codestream: Codestream,
}

pub fn scan_codestreams(data: &[u8], max_hits: usize) -> Vec<CodestreamScanHit> {
    let mut hits = Vec::new();
    let mut pos = 0;

    while pos + 4 <= data.len() && hits.len() < max_hits {
        let Some(relative) = find_soc_siz(&data[pos..]) else {
            break;
        };

        let offset = pos + relative;
        match Codestream::parse(&data[offset..]) {
            Ok(codestream) => {
                let byte_len = codestream.byte_len.max(4);
                hits.push(CodestreamScanHit { offset, codestream });
                pos = offset.saturating_add(byte_len);
            }
            Err(_) => {
                pos = offset + 2;
            }
        }
    }

    hits
}

fn find_soc_siz(data: &[u8]) -> Option<usize> {
    data.windows(4)
        .position(|bytes| bytes == [0xff, 0x4f, 0xff, 0x51])
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileReport {
    pub issues: Vec<ProfileIssue>,
}

impl ProfileReport {
    pub fn is_supported(&self) -> bool {
        self.issues.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileIssue {
    pub message: String,
}

impl ProfileIssue {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutError {
    pub message: String,
}

impl LayoutError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LayoutError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkerRecord {
    pub offset: usize,
    pub marker: Marker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Marker {
    Soc,
    Siz,
    Cod,
    Coc,
    Qcd,
    Qcc,
    Rgn,
    Pod,
    Ppm,
    Tlm,
    Plm,
    Sot,
    Sod,
    Eoc,
    Com,
    Unknown(u16),
}

impl Marker {
    fn from_code(code: u16) -> Self {
        match code {
            0xff4f => Self::Soc,
            0xff51 => Self::Siz,
            0xff52 => Self::Cod,
            0xff53 => Self::Coc,
            0xff5c => Self::Qcd,
            0xff5d => Self::Qcc,
            0xff5e => Self::Rgn,
            0xff5f => Self::Pod,
            0xff60 => Self::Ppm,
            0xff55 => Self::Tlm,
            0xff57 => Self::Plm,
            0xff90 => Self::Sot,
            0xff93 => Self::Sod,
            0xffd9 => Self::Eoc,
            0xff64 => Self::Com,
            other => Self::Unknown(other),
        }
    }

    pub fn code(self) -> u16 {
        match self {
            Self::Soc => 0xff4f,
            Self::Siz => 0xff51,
            Self::Cod => 0xff52,
            Self::Coc => 0xff53,
            Self::Qcd => 0xff5c,
            Self::Qcc => 0xff5d,
            Self::Rgn => 0xff5e,
            Self::Pod => 0xff5f,
            Self::Ppm => 0xff60,
            Self::Tlm => 0xff55,
            Self::Plm => 0xff57,
            Self::Sot => 0xff90,
            Self::Sod => 0xff93,
            Self::Eoc => 0xffd9,
            Self::Com => 0xff64,
            Self::Unknown(code) => code,
        }
    }

    fn has_segment(self) -> bool {
        !matches!(self, Self::Soc | Self::Sod | Self::Eoc)
    }
}

impl fmt::Display for Marker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Soc => "SOC",
            Self::Siz => "SIZ",
            Self::Cod => "COD",
            Self::Coc => "COC",
            Self::Qcd => "QCD",
            Self::Qcc => "QCC",
            Self::Rgn => "RGN",
            Self::Pod => "POD",
            Self::Ppm => "PPM",
            Self::Tlm => "TLM",
            Self::Plm => "PLM",
            Self::Sot => "SOT",
            Self::Sod => "SOD",
            Self::Eoc => "EOC",
            Self::Com => "COM",
            Self::Unknown(_) => "UNKNOWN",
        };

        write!(f, "{name}(0x{:04x})", self.code())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SizSegment {
    pub capabilities: u16,
    pub width: u32,
    pub height: u32,
    pub x_origin: u32,
    pub y_origin: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub tile_x_origin: u32,
    pub tile_y_origin: u32,
    pub components: Vec<ComponentSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentSpec {
    pub precision: u8,
    pub is_signed: bool,
    pub x_subsampling: u8,
    pub y_subsampling: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodSegment {
    pub coding_style: u8,
    pub progression_order: ProgressionOrder,
    pub layers: u16,
    pub multiple_component_transform: u8,
    pub decomposition_levels: u8,
    pub codeblock_width: u16,
    pub codeblock_height: u16,
    pub codeblock_style: u8,
    pub transform: WaveletTransform,
    pub precinct_sizes: Vec<u8>,
}

impl CodSegment {
    pub fn codeblock_style_flags(&self) -> CodeblockStyle {
        CodeblockStyle {
            raw: self.codeblock_style,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressionOrder {
    Lrcp,
    Rlcp,
    Rpcl,
    Pcrl,
    Cprl,
    Unknown(u8),
}

impl From<u8> for ProgressionOrder {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Lrcp,
            1 => Self::Rlcp,
            2 => Self::Rpcl,
            3 => Self::Pcrl,
            4 => Self::Cprl,
            other => Self::Unknown(other),
        }
    }
}

impl fmt::Display for ProgressionOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lrcp => f.write_str("LRCP"),
            Self::Rlcp => f.write_str("RLCP"),
            Self::Rpcl => f.write_str("RPCL"),
            Self::Pcrl => f.write_str("PCRL"),
            Self::Cprl => f.write_str("CPRL"),
            Self::Unknown(value) => write!(f, "unknown({value})"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletTransform {
    Irreversible97,
    Reversible53,
    Unknown(u8),
}

impl From<u8> for WaveletTransform {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Irreversible97,
            1 => Self::Reversible53,
            other => Self::Unknown(other),
        }
    }
}

impl fmt::Display for WaveletTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Irreversible97 => f.write_str("irreversible 9/7"),
            Self::Reversible53 => f.write_str("reversible 5/3"),
            Self::Unknown(value) => write!(f, "unknown({value})"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodeblockStyle {
    raw: u8,
}

impl CodeblockStyle {
    pub fn from_raw(raw: u8) -> Self {
        Self { raw }
    }

    pub fn raw(self) -> u8 {
        self.raw
    }

    pub fn selective_arithmetic_bypass(self) -> bool {
        self.raw & 0x01 != 0
    }

    pub fn reset_contexts(self) -> bool {
        self.raw & 0x02 != 0
    }

    pub fn terminate_each_pass(self) -> bool {
        self.raw & 0x04 != 0
    }

    pub fn vertically_causal(self) -> bool {
        self.raw & 0x08 != 0
    }

    pub fn predictable_termination(self) -> bool {
        self.raw & 0x10 != 0
    }

    pub fn segmentation_symbols(self) -> bool {
        self.raw & 0x20 != 0
    }
}

impl fmt::Display for CodeblockStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut names = Vec::new();
        if self.selective_arithmetic_bypass() {
            names.push("bypass");
        }
        if self.reset_contexts() {
            names.push("reset");
        }
        if self.terminate_each_pass() {
            names.push("terminate");
        }
        if self.vertically_causal() {
            names.push("vcausal");
        }
        if self.predictable_termination() {
            names.push("predictable");
        }
        if self.segmentation_symbols() {
            names.push("segmark");
        }
        if names.is_empty() {
            names.push("none");
        }
        write!(f, "0x{:02x} ({})", self.raw, names.join(", "))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QcdSegment {
    pub guard_bits: u8,
    pub style: QuantizationStyle,
    pub raw_steps: Vec<u8>,
}

impl QcdSegment {
    pub fn steps(&self) -> Result<Vec<QuantizationStep>, QuantizationError> {
        match self.style {
            QuantizationStyle::NoQuantization => Ok(self
                .raw_steps
                .iter()
                .copied()
                .map(|raw| QuantizationStep {
                    exponent: raw >> 3,
                    mantissa: 0,
                    raw: u16::from(raw),
                })
                .collect()),
            QuantizationStyle::ScalarDerived | QuantizationStyle::ScalarExpounded => {
                if !self.raw_steps.len().is_multiple_of(2) {
                    return Err(QuantizationError::new(format!(
                        "{} QCD payload has odd step byte count {}",
                        self.style,
                        self.raw_steps.len()
                    )));
                }
                Ok(self
                    .raw_steps
                    .chunks_exact(2)
                    .map(|bytes| {
                        let raw = u16::from_be_bytes([bytes[0], bytes[1]]);
                        QuantizationStep {
                            exponent: (raw >> 11) as u8,
                            mantissa: raw & 0x07ff,
                            raw,
                        }
                    })
                    .collect())
            }
            QuantizationStyle::Unknown(value) => Err(QuantizationError::new(format!(
                "unsupported QCD quantization style {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationStep {
    pub exponent: u8,
    pub mantissa: u16,
    pub raw: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizationError {
    pub message: String,
}

impl QuantizationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for QuantizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for QuantizationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationStyle {
    NoQuantization,
    ScalarDerived,
    ScalarExpounded,
    Unknown(u8),
}

impl From<u8> for QuantizationStyle {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::NoQuantization,
            1 => Self::ScalarDerived,
            2 => Self::ScalarExpounded,
            other => Self::Unknown(other),
        }
    }
}

impl fmt::Display for QuantizationStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoQuantization => f.write_str("none"),
            Self::ScalarDerived => f.write_str("scalar derived"),
            Self::ScalarExpounded => f.write_str("scalar expounded"),
            Self::Unknown(value) => write!(f, "unknown({value})"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SotSegment {
    pub marker_offset: usize,
    pub tile_index: u16,
    pub tile_part_length: u32,
    pub tile_part_index: u8,
    pub tile_part_count: u8,
    pub data_start: usize,
    pub data_end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommentSegment {
    pub registration: u16,
    pub bytes: Vec<u8>,
}

impl CommentSegment {
    pub fn text(&self) -> Option<&str> {
        std::str::from_utf8(&self.bytes).ok()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileLayout {
    pub tile_index: u16,
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub components: Vec<TileComponentLayout>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileComponentLayout {
    pub width: u32,
    pub height: u32,
    pub resolutions: Vec<ResolutionLayout>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolutionLayout {
    pub level: u8,
    pub width: u32,
    pub height: u32,
    pub precincts_x: u32,
    pub precincts_y: u32,
    pub subbands: Vec<SubbandLayout>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketLayout {
    pub tile_index: u16,
    pub layer: u16,
    pub resolution: u8,
    pub component: u16,
    pub precinct_x: u32,
    pub precinct_y: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubbandLayout {
    pub kind: SubbandKind,
    pub width: u32,
    pub height: u32,
    pub codeblock_width: u32,
    pub codeblock_height: u32,
    pub codeblocks_x: u32,
    pub codeblocks_y: u32,
}

impl SubbandLayout {
    pub fn codeblock_count(&self) -> u32 {
        self.codeblocks_x * self.codeblocks_y
    }

    pub fn codeblock_rect(&self, codeblock_x: u32, codeblock_y: u32) -> Option<CodeblockRect> {
        if codeblock_x >= self.codeblocks_x || codeblock_y >= self.codeblocks_y {
            return None;
        }
        let x0 = codeblock_x.saturating_mul(self.codeblock_width);
        let y0 = codeblock_y.saturating_mul(self.codeblock_height);
        let x1 = x0.saturating_add(self.codeblock_width).min(self.width);
        let y1 = y0.saturating_add(self.codeblock_height).min(self.height);
        Some(CodeblockRect { x0, y0, x1, y1 })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodeblockRect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl CodeblockRect {
    pub fn width(self) -> u32 {
        self.x1.saturating_sub(self.x0)
    }

    pub fn height(self) -> u32 {
        self.y1.saturating_sub(self.y0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SubbandKind {
    Ll,
    Hl,
    Lh,
    Hh,
}

impl fmt::Display for SubbandKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ll => f.write_str("LL"),
            Self::Hl => f.write_str("HL"),
            Self::Lh => f.write_str("LH"),
            Self::Hh => f.write_str("HH"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub offset: usize,
    pub message: String,
}

impl ParseError {
    fn new(offset: usize, message: impl Into<String>) -> Self {
        Self {
            offset,
            message: message.into(),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error at byte {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for ParseError {}
