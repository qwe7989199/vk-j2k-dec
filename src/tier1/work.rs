const MQ_STATE_COUNT: usize = 47;
const MQ_CONTEXT_COUNT: usize = 19;
const LEGACY_MAGNITUDE_BITPLANES: u32 = 31;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CodeblockId {
    pub component: u16,
    pub resolution: u8,
    pub subband: SubbandKind,
    pub codeblock_x: u32,
    pub codeblock_y: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedCodeblock {
    pub id: CodeblockId,
    pub width: u32,
    pub height: u32,
    pub magnitude_bitplanes: u32,
    pub zero_bit_planes: u32,
    pub segments: Vec<CodeblockCompressedSegment>,
}

impl CompressedCodeblock {
    pub fn total_bytes(&self) -> usize {
        self.segments.iter().map(|segment| segment.byte_len).sum()
    }

    pub fn total_coding_passes(&self) -> u32 {
        self.segments
            .iter()
            .map(|segment| u32::from(segment.coding_passes))
            .sum()
    }

    pub fn pass_schedule(&self) -> Vec<Tier1Pass> {
        build_pass_schedule_for_codeblock(
            self.magnitude_bitplanes,
            self.zero_bit_planes,
            self.total_coding_passes(),
        )
    }

    pub fn bind_segments<'a>(
        &self,
        tile_data: &'a [u8],
    ) -> Result<BoundCompressedCodeblock<'a>, Tier1DecodeError> {
        let mut segments = Vec::with_capacity(self.segments.len());
        for segment in &self.segments {
            let data = tile_data
                .get(segment.offset..segment.offset.saturating_add(segment.byte_len))
                .ok_or_else(|| {
                    Tier1DecodeError::new(format!(
                        "codeblock segment outside tile payload: offset {}, length {}",
                        segment.offset, segment.byte_len
                    ))
                })?;
            segments.push(BoundCodeblockSegment {
                packet_index: segment.packet_index,
                layer: segment.layer,
                coding_passes: segment.coding_passes,
                data,
            });
        }

        Ok(BoundCompressedCodeblock {
            id: self.id,
            width: self.width,
            height: self.height,
            magnitude_bitplanes: self.magnitude_bitplanes,
            zero_bit_planes: self.zero_bit_planes,
            segments,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundCompressedCodeblock<'a> {
    pub id: CodeblockId,
    pub width: u32,
    pub height: u32,
    pub magnitude_bitplanes: u32,
    pub zero_bit_planes: u32,
    pub segments: Vec<BoundCodeblockSegment<'a>>,
}

impl BoundCompressedCodeblock<'_> {
    pub fn total_bytes(&self) -> usize {
        self.segments.iter().map(|segment| segment.data.len()).sum()
    }

    pub fn total_coding_passes(&self) -> u32 {
        self.segments
            .iter()
            .map(|segment| u32::from(segment.coding_passes))
            .sum()
    }

    pub fn execution_plan(
        &self,
        codeblock_style: CodeblockStyle,
    ) -> Result<CodeblockExecutionPlan, Tier1DecodeError> {
        let schedule = build_pass_schedule_for_codeblock(
            self.magnitude_bitplanes,
            self.zero_bit_planes,
            self.total_coding_passes(),
        );
        let mut passes = Vec::with_capacity(schedule.len());
        let mut schedule_index = 0usize;

        for (segment_index, segment) in self.segments.iter().enumerate() {
            for segment_pass_index in 0..segment.coding_passes {
                let pass = *schedule.get(schedule_index).ok_or_else(|| {
                    Tier1DecodeError::new(format!(
                        "codeblock {:?} has more segment passes than scheduled passes",
                        self.id
                    ))
                })?;
                passes.push(CodeblockExecutionPass {
                    pass,
                    segment_index,
                    segment_pass_index,
                    packet_index: segment.packet_index,
                    layer: segment.layer,
                    coding_mode: coding_mode_for_pass(pass, codeblock_style),
                });
                schedule_index += 1;
            }
        }

        if schedule_index != schedule.len() {
            return Err(Tier1DecodeError::new(format!(
                "codeblock {:?} scheduled {} passes but segments contain {}",
                self.id,
                schedule.len(),
                schedule_index
            )));
        }

        Ok(CodeblockExecutionPlan {
            id: self.id,
            width: self.width,
            height: self.height,
            segments: self.segments.len(),
            bytes: self.total_bytes(),
            passes,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundCodeblockSegment<'a> {
    pub packet_index: usize,
    pub layer: u16,
    pub coding_passes: u16,
    pub data: &'a [u8],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeblockExecutionPlan {
    pub id: CodeblockId,
    pub width: u32,
    pub height: u32,
    pub segments: usize,
    pub bytes: usize,
    pub passes: Vec<CodeblockExecutionPass>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodeblockExecutionPass {
    pub pass: Tier1Pass,
    pub segment_index: usize,
    pub segment_pass_index: u16,
    pub packet_index: usize,
    pub layer: u16,
    pub coding_mode: Tier1CodingMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier1CodingMode {
    Arithmetic,
    RawBypass,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeblockCompressedSegment {
    pub packet_index: usize,
    pub layer: u16,
    pub offset: usize,
    pub byte_len: usize,
    pub coding_passes: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier1PassKind {
    Cleanup,
    SignificancePropagation,
    MagnitudeRefinement,
}

impl std::fmt::Display for Tier1PassKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cleanup => f.write_str("cleanup"),
            Self::SignificancePropagation => f.write_str("significance"),
            Self::MagnitudeRefinement => f.write_str("refinement"),
        }
    }
}

impl std::fmt::Display for Tier1CodingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Arithmetic => f.write_str("mq"),
            Self::RawBypass => f.write_str("raw"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tier1Pass {
    pub index: u32,
    pub bitplane: i32,
    pub kind: Tier1PassKind,
}

pub fn build_pass_schedule(zero_bit_planes: u32, pass_count: u32) -> Vec<Tier1Pass> {
    build_pass_schedule_for_codeblock(LEGACY_MAGNITUDE_BITPLANES, zero_bit_planes, pass_count)
}

pub fn build_pass_schedule_for_codeblock(
    magnitude_bitplanes: u32,
    zero_bit_planes: u32,
    pass_count: u32,
) -> Vec<Tier1Pass> {
    let mut passes = Vec::with_capacity(pass_count as usize);
    let first_bitplane = magnitude_bitplanes as i32 - zero_bit_planes as i32 - 1;

    for index in 0..pass_count {
        let (bitplane, kind) = if index == 0 {
            (first_bitplane, Tier1PassKind::Cleanup)
        } else {
            let bitplane = first_bitplane - 1 - ((index - 1) / 3) as i32;
            match (index - 1) % 3 {
                0 => (bitplane, Tier1PassKind::SignificancePropagation),
                1 => (bitplane, Tier1PassKind::MagnitudeRefinement),
                _ => (bitplane, Tier1PassKind::Cleanup),
            }
        };

        passes.push(Tier1Pass {
            index,
            bitplane,
            kind,
        });
    }

    passes
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tier1CoefficientScan {
    width: u32,
    height: u32,
    stripe_y: u32,
    x: u32,
    y_in_stripe: u32,
}

impl Tier1CoefficientScan {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            stripe_y: 0,
            x: 0,
            y_in_stripe: 0,
        }
    }
}

impl Iterator for Tier1CoefficientScan {
    type Item = (u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        while self.stripe_y < self.height {
            let stripe_height = 4.min(self.height - self.stripe_y);
            if self.x < self.width && self.y_in_stripe < stripe_height {
                let coordinate = (self.x, self.stripe_y + self.y_in_stripe);
                self.y_in_stripe += 1;
                if self.y_in_stripe == stripe_height {
                    self.y_in_stripe = 0;
                    self.x += 1;
                }
                return Some(coordinate);
            }

            self.stripe_y += 4;
            self.x = 0;
            self.y_in_stripe = 0;
        }

        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QuantizationBitplanes {
    guard_bits: u8,
    steps: Vec<QuantizationStep>,
}

impl QuantizationBitplanes {
    fn new(qcd: &QcdSegment) -> Result<Self, Tier1PlanError> {
        Ok(Self {
            guard_bits: qcd.guard_bits,
            steps: qcd
                .steps()
                .map_err(|err| Tier1PlanError::new(format!("QCD step parse failed: {err}")))?,
        })
    }

    fn magnitude_bitplanes(
        &self,
        resolution: u8,
        subband: SubbandKind,
    ) -> Result<u32, Tier1PlanError> {
        let step = if self.steps.len() == 1 {
            self.steps[0]
        } else {
            let index = qcd_step_index(resolution, subband).ok_or_else(|| {
                Tier1PlanError::new(format!(
                    "resolution {resolution} subband {subband} has no QCD step index"
                ))
            })?;
            *self.steps.get(index).ok_or_else(|| {
                Tier1PlanError::new(format!(
                    "QCD has {} steps, missing index {index} for R{resolution} {subband}",
                    self.steps.len()
                ))
            })?
        };

        let magnitude_bitplanes = u32::from(step.exponent) + u32::from(self.guard_bits);
        magnitude_bitplanes.checked_sub(1).ok_or_else(|| {
            Tier1PlanError::new(format!(
                "QCD exponent {} and guard bits {} produce no magnitude bitplanes",
                step.exponent, self.guard_bits
            ))
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

fn magnitude_bitplanes_for_segment(
    segment: &CodeblockSegmentRecord,
    quantization: Option<&QuantizationBitplanes>,
) -> Result<u32, Tier1PlanError> {
    quantization.map_or(Ok(LEGACY_MAGNITUDE_BITPLANES), |quantization| {
        quantization.magnitude_bitplanes(segment.resolution, segment.subband)
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tier1WorkPlan {
    pub codeblocks: Vec<CompressedCodeblock>,
}

impl Tier1WorkPlan {
    pub fn from_packet_summary(summary: &TilePacketParseSummary) -> Result<Self, Tier1PlanError> {
        Self::from_packet_summary_with_qcd(summary, None)
    }

    pub fn from_packet_summary_with_qcd(
        summary: &TilePacketParseSummary,
        qcd: Option<&QcdSegment>,
    ) -> Result<Self, Tier1PlanError> {
        Self::from_packet_summary_with_quantization(summary, qcd, &[])
    }

    pub fn from_packet_summary_with_quantization(
        summary: &TilePacketParseSummary,
        qcd: Option<&QcdSegment>,
        qccs: &[QccSegment],
    ) -> Result<Self, Tier1PlanError> {
        let mut codeblocks = Vec::<CompressedCodeblock>::new();
        let default_quantization = qcd.map(QuantizationBitplanes::new).transpose()?;
        let qcc_quantizations = qccs
            .iter()
            .map(|qcc| {
                QuantizationBitplanes::new(&qcc.quantization)
                    .map(|quantization| (qcc.component, quantization))
            })
            .collect::<Result<Vec<_>, _>>()?;

        for (packet_index, packet) in summary.packets.iter().enumerate() {
            for segment in &packet.segments {
                let id = CodeblockId {
                    component: segment.component,
                    resolution: segment.resolution,
                    subband: segment.subband,
                    codeblock_x: segment.codeblock_x,
                    codeblock_y: segment.codeblock_y,
                };

                let index = match codeblocks.binary_search_by(|codeblock| codeblock.id.cmp(&id)) {
                    Ok(index) => index,
                    Err(index) => {
                        let zero_bit_planes = segment.zero_bit_planes.ok_or_else(|| {
                            Tier1PlanError::new(format!(
                                "first segment for codeblock {:?} is missing zero bit-plane count",
                                id
                            ))
                        })?;
                        let quantization = qcc_quantizations
                            .iter()
                            .rev()
                            .find(|(component, _)| *component == segment.component)
                            .map(|(_, quantization)| quantization)
                            .or(default_quantization.as_ref());
                        let magnitude_bitplanes =
                            magnitude_bitplanes_for_segment(segment, quantization)?;
                        codeblocks.insert(
                            index,
                            CompressedCodeblock {
                                id,
                                width: segment.codeblock_width,
                                height: segment.codeblock_height,
                                magnitude_bitplanes,
                                zero_bit_planes,
                                segments: Vec::new(),
                            },
                        );
                        index
                    }
                };

                if segment.newly_included && !codeblocks[index].segments.is_empty() {
                    return Err(Tier1PlanError::new(format!(
                        "codeblock {:?} was marked newly included more than once",
                        id
                    )));
                }
                if codeblocks[index].width != segment.codeblock_width
                    || codeblocks[index].height != segment.codeblock_height
                {
                    return Err(Tier1PlanError::new(format!(
                        "codeblock {:?} changed dimensions from {}x{} to {}x{}",
                        id,
                        codeblocks[index].width,
                        codeblocks[index].height,
                        segment.codeblock_width,
                        segment.codeblock_height
                    )));
                }

                codeblocks[index].segments.push(CodeblockCompressedSegment {
                    packet_index,
                    layer: packet.layer,
                    offset: segment.body_offset,
                    byte_len: segment.byte_len,
                    coding_passes: segment.coding_passes,
                });
            }
        }

        Ok(Self { codeblocks })
    }

    pub fn codeblock_count(&self) -> usize {
        self.codeblocks.len()
    }

    pub fn segment_count(&self) -> usize {
        self.codeblocks
            .iter()
            .map(|codeblock| codeblock.segments.len())
            .sum()
    }

    pub fn total_bytes(&self) -> usize {
        self.codeblocks
            .iter()
            .map(CompressedCodeblock::total_bytes)
            .sum()
    }

    pub fn total_coding_passes(&self) -> u32 {
        self.codeblocks
            .iter()
            .map(CompressedCodeblock::total_coding_passes)
            .sum()
    }

    pub fn pass_kind_counts(&self) -> Tier1PassKindCounts {
        let mut counts = Tier1PassKindCounts::default();
        for codeblock in &self.codeblocks {
            for pass in codeblock.pass_schedule() {
                match pass.kind {
                    Tier1PassKind::Cleanup => counts.cleanup += 1,
                    Tier1PassKind::SignificancePropagation => counts.significance += 1,
                    Tier1PassKind::MagnitudeRefinement => counts.refinement += 1,
                }
            }
        }
        counts
    }

    pub fn dimension_bounds(&self) -> Option<CodeblockDimensionBounds> {
        let first = self.codeblocks.first()?;
        let mut bounds = CodeblockDimensionBounds {
            min_width: first.width,
            max_width: first.width,
            min_height: first.height,
            max_height: first.height,
        };

        for codeblock in &self.codeblocks[1..] {
            bounds.min_width = bounds.min_width.min(codeblock.width);
            bounds.max_width = bounds.max_width.max(codeblock.width);
            bounds.min_height = bounds.min_height.min(codeblock.height);
            bounds.max_height = bounds.max_height.max(codeblock.height);
        }

        Some(bounds)
    }

    pub fn allocate_decode_stubs(&self) -> Result<Tier1DecodeResult, Tier1DecodeError> {
        let mut codeblocks = Vec::with_capacity(self.codeblocks.len());
        for codeblock in &self.codeblocks {
            codeblocks.push(DecodedCodeblock::new(codeblock)?);
        }

        Ok(Tier1DecodeResult {
            codeblocks,
            decoded_passes: 0,
            pending_passes: self.total_coding_passes(),
        })
    }

    pub fn bind_segment_data<'a>(
        &self,
        tile_data: &'a [u8],
    ) -> Result<Tier1BoundWorkPlan<'a>, Tier1DecodeError> {
        let mut codeblocks = Vec::with_capacity(self.codeblocks.len());
        for codeblock in &self.codeblocks {
            codeblocks.push(codeblock.bind_segments(tile_data)?);
        }
        Ok(Tier1BoundWorkPlan { codeblocks })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tier1BoundWorkPlan<'a> {
    pub codeblocks: Vec<BoundCompressedCodeblock<'a>>,
}

impl Tier1BoundWorkPlan<'_> {
    pub fn codeblock_count(&self) -> usize {
        self.codeblocks.len()
    }

    pub fn segment_count(&self) -> usize {
        self.codeblocks
            .iter()
            .map(|codeblock| codeblock.segments.len())
            .sum()
    }

    pub fn total_bytes(&self) -> usize {
        self.codeblocks
            .iter()
            .map(BoundCompressedCodeblock::total_bytes)
            .sum()
    }

    pub fn execution_summary(
        &self,
        codeblock_style: CodeblockStyle,
    ) -> Result<Tier1ExecutionSummary, Tier1DecodeError> {
        let mut summary = Tier1ExecutionSummary {
            codeblocks: self.codeblock_count(),
            segments: self.segment_count(),
            bytes: self.total_bytes(),
            ..Tier1ExecutionSummary::default()
        };

        for codeblock in &self.codeblocks {
            let plan = codeblock.execution_plan(codeblock_style)?;
            summary.passes += plan.passes.len() as u32;
            for pass in plan.passes {
                match pass.coding_mode {
                    Tier1CodingMode::Arithmetic => summary.arithmetic_passes += 1,
                    Tier1CodingMode::RawBypass => summary.raw_bypass_passes += 1,
                }
            }
        }

        Ok(summary)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tier1PassKindCounts {
    pub cleanup: u32,
    pub significance: u32,
    pub refinement: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tier1ExecutionSummary {
    pub codeblocks: usize,
    pub segments: usize,
    pub bytes: usize,
    pub passes: u32,
    pub arithmetic_passes: u32,
    pub raw_bypass_passes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodeblockDimensionBounds {
    pub min_width: u32,
    pub max_width: u32,
    pub min_height: u32,
    pub max_height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tier1PlanError {
    pub message: String,
}

impl Tier1PlanError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Tier1PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Tier1PlanError {}
