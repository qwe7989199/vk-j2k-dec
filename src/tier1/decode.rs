#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedCodeblock {
    pub id: CodeblockId,
    pub width: u32,
    pub height: u32,
    pub coefficients: Vec<i32>,
    pub state: CodeblockStatePlanes,
    pub reconstruction: CoefficientReconstruction,
    pub decoded_passes: u32,
    pub source_bytes: usize,
    pub decoder_bytes_consumed: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tier1DecodeResult {
    pub codeblocks: Vec<DecodedCodeblock>,
    pub decoded_passes: u32,
    pub pending_passes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubbandCoefficientPlane {
    pub kind: SubbandKind,
    pub width: u32,
    pub height: u32,
    pub coefficients: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoefficientReconstruction {
    Exact,
    Midpoint,
}

impl SubbandCoefficientPlane {
    pub fn new(layout: &SubbandLayout) -> Result<Self, Tier1DecodeError> {
        Ok(Self {
            kind: layout.kind,
            width: layout.width,
            height: layout.height,
            coefficients: vec![0; checked_sample_count(layout.width, layout.height)?],
        })
    }

    pub fn place_codeblock(
        &mut self,
        layout: &SubbandLayout,
        codeblock: &DecodedCodeblock,
    ) -> Result<(), Tier1DecodeError> {
        if layout.kind != self.kind || layout.width != self.width || layout.height != self.height {
            return Err(Tier1DecodeError::new(format!(
                "layout {} {}x{} does not match coefficient plane {} {}x{}",
                layout.kind, layout.width, layout.height, self.kind, self.width, self.height
            )));
        }
        if codeblock.id.subband != self.kind {
            return Err(Tier1DecodeError::new(format!(
                "codeblock subband {} does not match plane {}",
                codeblock.id.subband, self.kind
            )));
        }
        let rect = layout
            .codeblock_rect(codeblock.id.codeblock_x, codeblock.id.codeblock_y)
            .ok_or_else(|| {
                Tier1DecodeError::new(format!(
                    "codeblock {},{} outside {} subband grid",
                    codeblock.id.codeblock_x, codeblock.id.codeblock_y, self.kind
                ))
            })?;
        if rect.width() != codeblock.width || rect.height() != codeblock.height {
            return Err(Tier1DecodeError::new(format!(
                "codeblock dimensions {}x{} do not match subband rect {}x{}",
                codeblock.width,
                codeblock.height,
                rect.width(),
                rect.height()
            )));
        }

        for y in 0..codeblock.height as usize {
            let source_offset = y * codeblock.width as usize;
            let target_offset = (rect.y0 as usize + y) * self.width as usize + rect.x0 as usize;
            let source =
                &codeblock.coefficients[source_offset..source_offset + codeblock.width as usize];
            let target =
                &mut self.coefficients[target_offset..target_offset + codeblock.width as usize];
            target.copy_from_slice(source);
        }

        Ok(())
    }

    pub fn coefficient(&self, x: u32, y: u32) -> Option<i32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.coefficients[y as usize * self.width as usize + x as usize])
    }
}

impl Tier1DecodeResult {
    pub fn coefficient_samples(&self) -> usize {
        self.codeblocks
            .iter()
            .map(|codeblock| codeblock.coefficients.len())
            .sum()
    }
}

impl DecodedCodeblock {
    pub fn new(input: &CompressedCodeblock) -> Result<Self, Tier1DecodeError> {
        Self::new_with_reconstruction(input, CoefficientReconstruction::Midpoint)
    }

    pub fn new_with_reconstruction(
        input: &CompressedCodeblock,
        reconstruction: CoefficientReconstruction,
    ) -> Result<Self, Tier1DecodeError> {
        let sample_count = checked_sample_count(input.width, input.height)?;
        Ok(Self {
            id: input.id,
            width: input.width,
            height: input.height,
            coefficients: vec![0; sample_count],
            state: CodeblockStatePlanes::new(input.width, input.height)?,
            reconstruction,
            decoded_passes: 0,
            source_bytes: 0,
            decoder_bytes_consumed: 0,
        })
    }

    pub fn new_bound(input: &BoundCompressedCodeblock<'_>) -> Result<Self, Tier1DecodeError> {
        Self::new_bound_with_reconstruction(input, CoefficientReconstruction::Midpoint)
    }

    pub fn new_bound_with_reconstruction(
        input: &BoundCompressedCodeblock<'_>,
        reconstruction: CoefficientReconstruction,
    ) -> Result<Self, Tier1DecodeError> {
        let sample_count = checked_sample_count(input.width, input.height)?;
        Ok(Self {
            id: input.id,
            width: input.width,
            height: input.height,
            coefficients: vec![0; sample_count],
            state: CodeblockStatePlanes::new(input.width, input.height)?,
            reconstruction,
            decoded_passes: 0,
            source_bytes: input.total_bytes(),
            decoder_bytes_consumed: 0,
        })
    }

    pub fn coefficient(&self, x: u32, y: u32) -> Option<i32> {
        self.index(x, y).map(|index| self.coefficients[index])
    }

    pub fn set_coefficient(&mut self, x: u32, y: u32, value: i32) -> Result<(), Tier1DecodeError> {
        let index = self.index(x, y).ok_or_else(|| {
            Tier1DecodeError::new(format!(
                "coefficient coordinate {x},{y} outside {}x{} codeblock",
                self.width, self.height
            ))
        })?;
        self.coefficients[index] = value;
        Ok(())
    }

    pub fn apply_significance_decision(
        &mut self,
        x: u32,
        y: u32,
        bitplane: i32,
        is_significant: bool,
        is_negative: Option<bool>,
    ) -> Result<(), Tier1DecodeError> {
        self.index(x, y).ok_or_else(|| {
            Tier1DecodeError::new(format!(
                "coefficient coordinate {x},{y} outside {}x{} codeblock",
                self.width, self.height
            ))
        })?;
        self.state.mark_visited_in_bitplane(x, y)?;
        if !is_significant {
            return Ok(());
        }

        let is_negative = is_negative.ok_or_else(|| {
            Tier1DecodeError::new("significant coefficient decision is missing sign bit")
        })?;
        let magnitude =
            significance_reconstruction_magnitude(self.id, bitplane, self.reconstruction)?;
        let value = if is_negative { -magnitude } else { magnitude };
        self.set_coefficient(x, y, value)?;
        self.state.mark_significant(x, y, is_negative)
    }

    pub fn apply_magnitude_refinement_bit(
        &mut self,
        x: u32,
        y: u32,
        bitplane: i32,
        bit: bool,
    ) -> Result<(), Tier1DecodeError> {
        let index = self.index(x, y).ok_or_else(|| {
            Tier1DecodeError::new(format!(
                "coefficient coordinate {x},{y} outside {}x{} codeblock",
                self.width, self.height
            ))
        })?;
        if !self.state.is_significant(x, y).unwrap_or(false) {
            return Err(Tier1DecodeError::new(format!(
                "cannot refine insignificant coefficient {x},{y}"
            )));
        }

        let magnitude_delta =
            refinement_reconstruction_delta(bitplane, bit, self.reconstruction)?;
        let current = self.coefficients[index];
        let delta = if current < 0 {
            -magnitude_delta
        } else {
            magnitude_delta
        };
        self.coefficients[index] = current.checked_add(delta).ok_or_else(|| {
            Tier1DecodeError::new(format!("coefficient refinement overflow at {x},{y}"))
        })?;
        self.state.mark_magnitude_refined(x, y)?;
        self.state.mark_visited_in_bitplane(x, y)
    }

    pub fn coefficient_scan(&self) -> Tier1CoefficientScan {
        Tier1CoefficientScan::new(self.width, self.height)
    }

    pub fn significance_context(&self, x: u32, y: u32) -> Option<usize> {
        self.state.significance_context(self.id.subband, x, y)
    }

    pub fn sign_context(&self, x: u32, y: u32) -> Option<SignCodingContext> {
        self.state.sign_context_for(x, y)
    }

    pub fn magnitude_refinement_context(&self, x: u32, y: u32) -> Option<usize> {
        self.state.magnitude_refinement_context(x, y)
    }

    pub fn coding_operation(
        &self,
        pass_kind: Tier1PassKind,
        x: u32,
        y: u32,
    ) -> Option<Tier1CodingOperation> {
        self.index(x, y)?;
        if self.state.is_visited_in_bitplane(x, y)? {
            return None;
        }

        match pass_kind {
            Tier1PassKind::SignificancePropagation => {
                let significant = self.state.is_significant(x, y)?;
                let neighbors = self.state.significance_neighbors(x, y)?;
                (!significant && neighbors.has_any()).then_some(Tier1CodingOperation::Significance)
            }
            Tier1PassKind::MagnitudeRefinement => self
                .state
                .is_significant(x, y)?
                .then_some(Tier1CodingOperation::MagnitudeRefinement),
            Tier1PassKind::Cleanup => (!self.state.is_significant(x, y)?)
                .then_some(Tier1CodingOperation::Cleanup),
        }
    }

    pub fn decode_pass(
        &mut self,
        pass: Tier1Pass,
        reader: &mut impl Tier1DecisionReader,
    ) -> Result<Tier1DecodedPassSummary, Tier1DecodeError> {
        if pass.index == 0 || pass.kind == Tier1PassKind::SignificancePropagation {
            self.state.clear_bitplane_visits();
        }
        if pass.kind == Tier1PassKind::Cleanup {
            return self.decode_cleanup_pass(pass, reader);
        }

        let mut summary = Tier1DecodedPassSummary {
            pass,
            ..Tier1DecodedPassSummary::default()
        };
        let coordinates = self.coefficient_scan().collect::<Vec<_>>();
        for (x, y) in coordinates {
            let Some(operation) = self.coding_operation(pass.kind, x, y) else {
                continue;
            };
            match operation {
                Tier1CodingOperation::Significance | Tier1CodingOperation::Cleanup => {
                    let context = self.significance_context(x, y).ok_or_else(|| {
                        Tier1DecodeError::new(format!(
                            "missing significance context for coefficient {x},{y}"
                        ))
                    })?;
                    let is_significant = reader.decode_context_bit(context)?;
                    summary.significance_bits += 1;
                    if is_significant {
                        let sign_context = self.sign_context(x, y).ok_or_else(|| {
                            Tier1DecodeError::new(format!(
                                "missing sign context for coefficient {x},{y}"
                            ))
                        })?;
                        let sign_bit = reader.decode_context_bit(sign_context.context)?
                            ^ sign_context.xor_bit;
                        summary.sign_bits += 1;
                        summary.became_significant += 1;
                        self.apply_significance_decision(
                            x,
                            y,
                            pass.bitplane,
                            true,
                            Some(sign_bit),
                        )?;
                    } else {
                        self.apply_significance_decision(x, y, pass.bitplane, false, None)?;
                    }
                }
                Tier1CodingOperation::MagnitudeRefinement => {
                    let context = self.magnitude_refinement_context(x, y).ok_or_else(|| {
                        Tier1DecodeError::new(format!(
                            "missing magnitude refinement context for coefficient {x},{y}"
                        ))
                    })?;
                    let bit = reader.decode_context_bit(context)?;
                    summary.refinement_bits += 1;
                    self.apply_magnitude_refinement_bit(x, y, pass.bitplane, bit)?;
                }
            }
            summary.visited += 1;
        }

        self.decoded_passes += 1;
        Ok(summary)
    }

    fn decode_cleanup_pass(
        &mut self,
        pass: Tier1Pass,
        reader: &mut impl Tier1DecisionReader,
    ) -> Result<Tier1DecodedPassSummary, Tier1DecodeError> {
        let mut summary = Tier1DecodedPassSummary {
            pass,
            ..Tier1DecodedPassSummary::default()
        };

        let mut stripe_y = 0;
        while stripe_y < self.height {
            let stripe_height = 4.min(self.height - stripe_y);
            for x in 0..self.width {
                if stripe_height == 4 && self.cleanup_run_length_candidate(x, stripe_y) {
                    let any_significant = reader.decode_context_bit(TIER1_RUN_LENGTH_CONTEXT)?;
                    summary.run_length_bits += 1;
                    if !any_significant {
                        for y_offset in 0..4 {
                            self.apply_significance_decision(
                                x,
                                stripe_y + y_offset,
                                pass.bitplane,
                                false,
                                None,
                            )?;
                            summary.significance_bits += 1;
                            summary.visited += 1;
                        }
                        continue;
                    }

                    let run = (u8::from(reader.decode_context_bit(TIER1_UNIFORM_CONTEXT)?) << 1)
                        | u8::from(reader.decode_context_bit(TIER1_UNIFORM_CONTEXT)?);
                    summary.uniform_bits += 2;
                    for y_offset in 0..run {
                        self.apply_significance_decision(
                            x,
                            stripe_y + u32::from(y_offset),
                            pass.bitplane,
                            false,
                            None,
                        )?;
                        summary.significance_bits += 1;
                        summary.visited += 1;
                    }

                    let y = stripe_y + u32::from(run);
                    self.decode_significant_sample(x, y, pass.bitplane, reader, &mut summary)?;

                    for y_offset in u32::from(run) + 1..stripe_height {
                        self.decode_cleanup_sample(
                            x,
                            stripe_y + y_offset,
                            pass.bitplane,
                            reader,
                            &mut summary,
                        )?;
                    }
                } else {
                    for y_offset in 0..stripe_height {
                        self.decode_cleanup_sample(
                            x,
                            stripe_y + y_offset,
                            pass.bitplane,
                            reader,
                            &mut summary,
                        )?;
                    }
                }
            }
            stripe_y += 4;
        }

        self.decoded_passes += 1;
        Ok(summary)
    }

    fn decode_cleanup_sample(
        &mut self,
        x: u32,
        y: u32,
        bitplane: i32,
        reader: &mut impl Tier1DecisionReader,
        summary: &mut Tier1DecodedPassSummary,
    ) -> Result<(), Tier1DecodeError> {
        if self.coding_operation(Tier1PassKind::Cleanup, x, y) != Some(Tier1CodingOperation::Cleanup)
        {
            return Ok(());
        }

        let context = self.significance_context(x, y).ok_or_else(|| {
            Tier1DecodeError::new(format!("missing cleanup context for coefficient {x},{y}"))
        })?;
        let is_significant = reader.decode_context_bit(context)?;
        if is_significant {
            self.decode_significant_sample(x, y, bitplane, reader, summary)?;
        } else {
            summary.significance_bits += 1;
            self.apply_significance_decision(x, y, bitplane, false, None)?;
            summary.visited += 1;
        }
        Ok(())
    }

    fn decode_significant_sample(
        &mut self,
        x: u32,
        y: u32,
        bitplane: i32,
        reader: &mut impl Tier1DecisionReader,
        summary: &mut Tier1DecodedPassSummary,
    ) -> Result<(), Tier1DecodeError> {
        let sign_context = self.sign_context(x, y).ok_or_else(|| {
            Tier1DecodeError::new(format!("missing sign context for coefficient {x},{y}"))
        })?;
        let sign_bit = reader.decode_context_bit(sign_context.context)? ^ sign_context.xor_bit;
        summary.sign_bits += 1;
        summary.became_significant += 1;
        summary.significance_bits += 1;
        summary.visited += 1;
        self.apply_significance_decision(x, y, bitplane, true, Some(sign_bit))
    }

    fn cleanup_run_length_candidate(&self, x: u32, stripe_y: u32) -> bool {
        (0..4).all(|y_offset| {
            let y = stripe_y + y_offset;
            self.coding_operation(Tier1PassKind::Cleanup, x, y)
                == Some(Tier1CodingOperation::Cleanup)
                && self.significance_context(x, y) == Some(0)
        })
    }

    fn index(&self, x: u32, y: u32) -> Option<usize> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some((y * self.width + x) as usize)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SignificanceNeighbors {
    pub horizontal: u8,
    pub vertical: u8,
    pub diagonal: u8,
}

impl SignificanceNeighbors {
    pub fn total(self) -> u8 {
        self.horizontal + self.vertical + self.diagonal
    }

    pub fn has_any(self) -> bool {
        self.total() != 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeblockStatePlanes {
    width: u32,
    height: u32,
    significant: Vec<bool>,
    sign: Vec<bool>,
    magnitude_refined: Vec<bool>,
    visited_in_bitplane: Vec<bool>,
}

impl BoundCompressedCodeblock<'_> {
    pub fn decode_arithmetic_passes(
        &self,
        codeblock_style: CodeblockStyle,
    ) -> Result<DecodedCodeblock, Tier1DecodeError> {
        self.decode_arithmetic_passes_with_reconstruction(
            codeblock_style,
            CoefficientReconstruction::Midpoint,
        )
    }

    pub fn decode_arithmetic_passes_with_reconstruction(
        &self,
        codeblock_style: CodeblockStyle,
        reconstruction: CoefficientReconstruction,
    ) -> Result<DecodedCodeblock, Tier1DecodeError> {
        let execution = self.execution_plan(codeblock_style)?;
        if let Some(raw_pass) = execution
            .passes
            .iter()
            .find(|pass| pass.coding_mode == Tier1CodingMode::RawBypass)
        {
            return Err(Tier1DecodeError::new(format!(
                "codeblock {:?} contains raw bypass pass {}",
                self.id, raw_pass.pass.index
            )));
        }

        let mut data = Vec::with_capacity(self.total_bytes());
        for segment in &self.segments {
            data.extend_from_slice(segment.data);
        }

        let mut decoded = DecodedCodeblock::new_bound_with_reconstruction(self, reconstruction)?;
        let mut mq = MqDecoder::with_data(&data);
        for pass in execution.passes {
            decoded.decode_pass(pass.pass, &mut mq)?;
        }
        decoded.decoder_bytes_consumed = mq.consumed_bytes();
        Ok(decoded)
    }
}

impl Tier1BoundWorkPlan<'_> {
    pub fn decode_arithmetic_passes(
        &self,
        codeblock_style: CodeblockStyle,
    ) -> Result<Tier1DecodeResult, Tier1DecodeError> {
        self.decode_arithmetic_passes_with_reconstruction(
            codeblock_style,
            CoefficientReconstruction::Midpoint,
        )
    }

    pub fn decode_arithmetic_passes_with_reconstruction(
        &self,
        codeblock_style: CodeblockStyle,
        reconstruction: CoefficientReconstruction,
    ) -> Result<Tier1DecodeResult, Tier1DecodeError> {
        let mut codeblocks = Vec::with_capacity(self.codeblocks.len());
        let mut decoded_passes = 0;
        for codeblock in &self.codeblocks {
            let decoded =
                codeblock.decode_arithmetic_passes_with_reconstruction(codeblock_style, reconstruction)?;
            decoded_passes += decoded.decoded_passes;
            codeblocks.push(decoded);
        }
        Ok(Tier1DecodeResult {
            codeblocks,
            decoded_passes,
            pending_passes: 0,
        })
    }
}

impl CodeblockStatePlanes {
    pub fn new(width: u32, height: u32) -> Result<Self, Tier1DecodeError> {
        let sample_count = checked_sample_count(width, height)?;
        Ok(Self {
            width,
            height,
            significant: vec![false; sample_count],
            sign: vec![false; sample_count],
            magnitude_refined: vec![false; sample_count],
            visited_in_bitplane: vec![false; sample_count],
        })
    }

    pub fn is_significant(&self, x: u32, y: u32) -> Option<bool> {
        self.index(x, y).map(|index| self.significant[index])
    }

    pub fn sign(&self, x: u32, y: u32) -> Option<bool> {
        self.index(x, y).map(|index| self.sign[index])
    }

    pub fn is_magnitude_refined(&self, x: u32, y: u32) -> Option<bool> {
        self.index(x, y).map(|index| self.magnitude_refined[index])
    }

    pub fn is_visited_in_bitplane(&self, x: u32, y: u32) -> Option<bool> {
        self.index(x, y).map(|index| self.visited_in_bitplane[index])
    }

    pub fn clear_bitplane_visits(&mut self) {
        self.visited_in_bitplane.fill(false);
    }

    pub fn mark_significant(
        &mut self,
        x: u32,
        y: u32,
        is_negative: bool,
    ) -> Result<(), Tier1DecodeError> {
        let index = self.index(x, y).ok_or_else(|| {
            Tier1DecodeError::new(format!(
                "state coordinate {x},{y} outside {}x{} codeblock",
                self.width, self.height
            ))
        })?;
        self.significant[index] = true;
        self.sign[index] = is_negative;
        Ok(())
    }

    pub fn mark_magnitude_refined(&mut self, x: u32, y: u32) -> Result<(), Tier1DecodeError> {
        let index = self.index(x, y).ok_or_else(|| {
            Tier1DecodeError::new(format!(
                "state coordinate {x},{y} outside {}x{} codeblock",
                self.width, self.height
            ))
        })?;
        self.magnitude_refined[index] = true;
        Ok(())
    }

    pub fn mark_visited_in_bitplane(&mut self, x: u32, y: u32) -> Result<(), Tier1DecodeError> {
        let index = self.index(x, y).ok_or_else(|| {
            Tier1DecodeError::new(format!(
                "state coordinate {x},{y} outside {}x{} codeblock",
                self.width, self.height
            ))
        })?;
        self.visited_in_bitplane[index] = true;
        Ok(())
    }

    pub fn significance_neighbors(&self, x: u32, y: u32) -> Option<SignificanceNeighbors> {
        self.index(x, y)?;

        let mut neighbors = SignificanceNeighbors::default();
        for y_delta in -1i32..=1 {
            for x_delta in -1i32..=1 {
                if x_delta == 0 && y_delta == 0 {
                    continue;
                }
                let Some(nx) = x.checked_add_signed(x_delta) else {
                    continue;
                };
                let Some(ny) = y.checked_add_signed(y_delta) else {
                    continue;
                };
                if !self.is_significant(nx, ny).unwrap_or(false) {
                    continue;
                }

                match (x_delta, y_delta) {
                    (-1 | 1, 0) => neighbors.horizontal += 1,
                    (0, -1 | 1) => neighbors.vertical += 1,
                    _ => neighbors.diagonal += 1,
                }
            }
        }

        Some(neighbors)
    }

    pub fn significance_context(&self, subband: SubbandKind, x: u32, y: u32) -> Option<usize> {
        let neighbors = self.significance_neighbors(x, y)?;
        Some(significance_context_label(subband, neighbors))
    }

    pub fn sign_context_for(&self, x: u32, y: u32) -> Option<SignCodingContext> {
        self.index(x, y)?;
        let horizontal = self.sign_contribution(x, y, [(-1, 0), (1, 0)]);
        let vertical = self.sign_contribution(x, y, [(0, -1), (0, 1)]);
        Some(SignCodingContext::from_contributions(horizontal, vertical))
    }

    pub fn magnitude_refinement_context(&self, x: u32, y: u32) -> Option<usize> {
        let is_refined = self.is_magnitude_refined(x, y)?;
        if is_refined {
            return Some(TIER1_MAGNITUDE_REFINEMENT_CONTEXT_REFINED);
        }

        let neighbors = self.significance_neighbors(x, y)?;
        if neighbors.has_any() {
            Some(TIER1_MAGNITUDE_REFINEMENT_CONTEXT_FIRST_WITH_NEIGHBORS)
        } else {
            Some(TIER1_MAGNITUDE_REFINEMENT_CONTEXT_FIRST)
        }
    }

    pub fn significant_count(&self) -> usize {
        self.significant.iter().filter(|value| **value).count()
    }

    fn index(&self, x: u32, y: u32) -> Option<usize> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some((y * self.width + x) as usize)
    }

    fn sign_contribution(&self, x: u32, y: u32, offsets: [(i32, i32); 2]) -> i8 {
        let mut has_positive = false;
        let mut has_negative = false;

        for (x_delta, y_delta) in offsets {
            let Some(nx) = x.checked_add_signed(x_delta) else {
                continue;
            };
            let Some(ny) = y.checked_add_signed(y_delta) else {
                continue;
            };
            if !self.is_significant(nx, ny).unwrap_or(false) {
                continue;
            }
            if self.sign(nx, ny).unwrap_or(false) {
                has_negative = true;
            } else {
                has_positive = true;
            }
        }

        match (has_positive, has_negative) {
            (true, false) => 1,
            (false, true) => -1,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignCodingContext {
    pub context: usize,
    pub xor_bit: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier1CodingOperation {
    Significance,
    MagnitudeRefinement,
    Cleanup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tier1DecodedPassSummary {
    pub pass: Tier1Pass,
    pub significance_bits: u32,
    pub sign_bits: u32,
    pub refinement_bits: u32,
    pub run_length_bits: u32,
    pub uniform_bits: u32,
    pub became_significant: u32,
    pub visited: u32,
}

impl Default for Tier1DecodedPassSummary {
    fn default() -> Self {
        Self {
            pass: Tier1Pass {
                index: 0,
                bitplane: 0,
                kind: Tier1PassKind::Cleanup,
            },
            significance_bits: 0,
            sign_bits: 0,
            refinement_bits: 0,
            run_length_bits: 0,
            uniform_bits: 0,
            became_significant: 0,
            visited: 0,
        }
    }
}

pub trait Tier1DecisionReader {
    fn decode_context_bit(&mut self, context: usize) -> Result<bool, Tier1DecodeError>;
}

impl SignCodingContext {
    fn from_contributions(horizontal: i8, vertical: i8) -> Self {
        match (horizontal, vertical) {
            (1, 1) => Self::new(13, false),
            (1, 0) => Self::new(12, false),
            (1, -1) => Self::new(11, false),
            (0, 1) => Self::new(10, false),
            (0, 0) => Self::new(9, false),
            (0, -1) => Self::new(10, true),
            (-1, 1) => Self::new(11, true),
            (-1, 0) => Self::new(12, true),
            (-1, -1) => Self::new(13, true),
            _ => unreachable!("sign contributions are clamped to -1, 0, or 1"),
        }
    }

    const fn new(context: usize, xor_bit: bool) -> Self {
        Self { context, xor_bit }
    }
}

pub const TIER1_SIGNIFICANCE_CONTEXTS: std::ops::RangeInclusive<usize> = 0..=8;
pub const TIER1_SIGN_CONTEXTS: std::ops::RangeInclusive<usize> = 9..=13;
pub const TIER1_MAGNITUDE_REFINEMENT_CONTEXT_FIRST: usize = 14;
pub const TIER1_MAGNITUDE_REFINEMENT_CONTEXT_FIRST_WITH_NEIGHBORS: usize = 15;
pub const TIER1_MAGNITUDE_REFINEMENT_CONTEXT_REFINED: usize = 16;
pub const TIER1_RUN_LENGTH_CONTEXT: usize = 17;
pub const TIER1_UNIFORM_CONTEXT: usize = 18;

fn significance_context_label(subband: SubbandKind, neighbors: SignificanceNeighbors) -> usize {
    match subband {
        SubbandKind::Ll | SubbandKind::Lh => {
            lh_or_ll_significance_context(neighbors.horizontal, neighbors.vertical, neighbors.diagonal)
        }
        SubbandKind::Hl => {
            lh_or_ll_significance_context(neighbors.vertical, neighbors.horizontal, neighbors.diagonal)
        }
        SubbandKind::Hh => hh_significance_context(neighbors),
    }
}

fn lh_or_ll_significance_context(primary: u8, secondary: u8, diagonal: u8) -> usize {
    match (primary, secondary, diagonal) {
        (2, _, _) => 8,
        (1, 1..=2, _) => 7,
        (1, 0, 1..=4) => 6,
        (1, 0, 0) => 5,
        (0, 2, _) => 4,
        (0, 1, _) => 3,
        (0, 0, 2..=4) => 2,
        (0, 0, 1) => 1,
        _ => 0,
    }
}

fn hh_significance_context(neighbors: SignificanceNeighbors) -> usize {
    let hv = neighbors.horizontal + neighbors.vertical;
    match (hv, neighbors.diagonal) {
        (_, 3..=4) => 8,
        (1..=4, 2) => 7,
        (0, 2) => 6,
        (2..=4, 1) => 5,
        (1, 1) => 4,
        (0, 1) => 3,
        (2..=4, 0) => 2,
        (1, 0) => 1,
        _ => 0,
    }
}

fn coefficient_bit_value(bitplane: i32) -> Result<i32, Tier1DecodeError> {
    let bitplane = u32::try_from(bitplane).map_err(|_| {
        Tier1DecodeError::new(format!("negative coefficient bitplane {bitplane} cannot be decoded"))
    })?;
    1i32.checked_shl(bitplane).ok_or_else(|| {
        Tier1DecodeError::new(format!(
            "coefficient bitplane {bitplane} exceeds i32 staging precision"
        ))
    })
}

fn coefficient_one_plus_half(bitplane: i32) -> Result<i32, Tier1DecodeError> {
    let one = coefficient_bit_value(bitplane)?;
    Ok(one | (one >> 1))
}

fn significance_reconstruction_magnitude(
    codeblock: CodeblockId,
    bitplane: i32,
    reconstruction: CoefficientReconstruction,
) -> Result<i32, Tier1DecodeError> {
    match reconstruction {
        CoefficientReconstruction::Exact => coefficient_bit_value(bitplane),
        CoefficientReconstruction::Midpoint => {
            let magnitude = coefficient_one_plus_half(bitplane)?;
            if codeblock.resolution == 0 && codeblock.subband == SubbandKind::Ll {
                // FFmpeg's reversible path lands one integer below the generic half-bin
                // midpoint for the LL band. Keep the compatibility adjustment isolated
                // until the dequantization staging is modeled explicitly.
                return Ok(magnitude.saturating_sub(1));
            }
            Ok(magnitude)
        }
    }
}

fn coefficient_positive_half(bitplane: i32) -> Result<i32, Tier1DecodeError> {
    Ok(coefficient_bit_value(bitplane)? >> 1)
}

fn refinement_reconstruction_delta(
    bitplane: i32,
    bit: bool,
    reconstruction: CoefficientReconstruction,
) -> Result<i32, Tier1DecodeError> {
    match reconstruction {
        CoefficientReconstruction::Exact => {
            if bit {
                coefficient_bit_value(bitplane)
            } else {
                Ok(0)
            }
        }
        CoefficientReconstruction::Midpoint => {
            let poshalf = coefficient_positive_half(bitplane)?;
            if bit { Ok(poshalf) } else { Ok(-poshalf) }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tier1DecodeError {
    pub message: String,
}

impl Tier1DecodeError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Tier1DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Tier1DecodeError {}
