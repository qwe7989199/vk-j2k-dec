fn checked_sample_count(width: u32, height: u32) -> Result<usize, Tier1DecodeError> {
    width
        .checked_mul(height)
        .map(|count| count as usize)
        .ok_or_else(|| {
            Tier1DecodeError::new(format!("codeblock dimensions overflow: {width}x{height}"))
        })
}

fn coding_mode_for_pass(pass: Tier1Pass, codeblock_style: CodeblockStyle) -> Tier1CodingMode {
    if codeblock_style.selective_arithmetic_bypass()
        && pass.index >= 10
        && matches!(
            pass.kind,
            Tier1PassKind::SignificancePropagation | Tier1PassKind::MagnitudeRefinement
        )
    {
        Tier1CodingMode::RawBypass
    } else {
        Tier1CodingMode::Arithmetic
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawBypassBitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    current_byte: u8,
    bits_left: u8,
}

impl<'a> RawBypassBitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            current_byte: 0,
            bits_left: 0,
        }
    }

    pub fn read_bit(&mut self) -> Result<bool, Tier1DecodeError> {
        if self.bits_left == 0 {
            self.current_byte = *self.data.get(self.byte_pos).ok_or_else(|| {
                Tier1DecodeError::new(format!(
                    "raw bypass bitstream ended at byte {}",
                    self.byte_pos
                ))
            })?;
            self.byte_pos += 1;
            self.bits_left = 8;
        }

        self.bits_left -= 1;
        Ok(((self.current_byte >> self.bits_left) & 1) != 0)
    }

    pub fn read_bits(&mut self, count: u8) -> Result<u32, Tier1DecodeError> {
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
}

impl Tier1DecisionReader for RawBypassBitReader<'_> {
    fn decode_context_bit(&mut self, _context: usize) -> Result<bool, Tier1DecodeError> {
        self.read_bit()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MqDecoder {
    contexts: [MqContext; MQ_CONTEXT_COUNT],
    input: MqInput,
}

impl MqDecoder {
    pub fn new() -> Self {
        let mut decoder = Self {
            contexts: [MqContext::default(); MQ_CONTEXT_COUNT],
            input: MqInput::default(),
        };
        decoder.reset_contexts();
        decoder
    }

    pub fn with_data(data: &[u8]) -> Self {
        let mut decoder = Self::new();
        decoder.reset_input(data);
        decoder
    }

    pub fn reset_input(&mut self, data: &[u8]) {
        self.input.reset(data);
    }

    pub fn reset_contexts(&mut self) {
        self.contexts = [MqContext::default(); MQ_CONTEXT_COUNT];
        self.contexts[0].state = 4;
        self.contexts[TIER1_RUN_LENGTH_CONTEXT].state = 3;
        self.contexts[TIER1_UNIFORM_CONTEXT].state = 46;
    }

    pub fn input_registers(&self) -> MqInputRegisters {
        self.input.registers()
    }

    pub fn consumed_bytes(&self) -> usize {
        self.input.byte_pos
    }

    pub fn renormalize_input_once(&mut self) {
        self.input.renormalize_once();
    }

    pub fn decode_bit(&mut self, context_index: usize) -> Result<bool, Tier1DecodeError> {
        let context = *self.contexts.get(context_index).ok_or_else(|| {
            Tier1DecodeError::new(format!(
                "MQ context index {context_index} outside context array"
            ))
        })?;
        let qe = u32::from(MQ_STATES[context.state as usize].qe);
        self.input.a = self.input.a.saturating_sub(qe);

        let decoded = if (self.input.c >> 16) < qe {
            self.exchange_lps(context_index, qe)?
        } else {
            self.input.c = self.input.c.wrapping_sub(qe << 16);
            if self.input.a & 0x8000 == 0 {
                self.exchange_mps(context_index, qe)?
            } else {
                context.mps
            }
        };

        Ok(decoded)
    }

    pub fn context(&self, index: usize) -> Option<MqContext> {
        self.contexts.get(index).copied()
    }

    pub fn set_context(
        &mut self,
        index: usize,
        state: u8,
        mps: bool,
    ) -> Result<(), Tier1DecodeError> {
        if state as usize >= MQ_STATE_COUNT {
            return Err(Tier1DecodeError::new(format!(
                "MQ state index {state} outside table"
            )));
        }
        let context = self.contexts.get_mut(index).ok_or_else(|| {
            Tier1DecodeError::new(format!("MQ context index {index} outside context array"))
        })?;
        context.state = state;
        context.mps = mps;
        Ok(())
    }

    pub fn transition_mps(&mut self, index: usize) -> Result<(), Tier1DecodeError> {
        let context = self.contexts.get_mut(index).ok_or_else(|| {
            Tier1DecodeError::new(format!("MQ context index {index} outside context array"))
        })?;
        let entry = MQ_STATES[context.state as usize];
        context.state = entry.next_mps;
        Ok(())
    }

    pub fn transition_lps(&mut self, index: usize) -> Result<(), Tier1DecodeError> {
        let context = self.contexts.get_mut(index).ok_or_else(|| {
            Tier1DecodeError::new(format!("MQ context index {index} outside context array"))
        })?;
        let entry = MQ_STATES[context.state as usize];
        if entry.switch_mps {
            context.mps = !context.mps;
        }
        context.state = entry.next_lps;
        Ok(())
    }

    fn exchange_mps(&mut self, context_index: usize, qe: u32) -> Result<bool, Tier1DecodeError> {
        let context = *self.contexts.get(context_index).ok_or_else(|| {
            Tier1DecodeError::new(format!(
                "MQ context index {context_index} outside context array"
            ))
        })?;

        let decoded = if self.input.a < qe {
            self.transition_lps(context_index)?;
            !context.mps
        } else {
            self.transition_mps(context_index)?;
            context.mps
        };

        self.input.renormalize();
        Ok(decoded)
    }

    fn exchange_lps(&mut self, context_index: usize, qe: u32) -> Result<bool, Tier1DecodeError> {
        let context = *self.contexts.get(context_index).ok_or_else(|| {
            Tier1DecodeError::new(format!(
                "MQ context index {context_index} outside context array"
            ))
        })?;

        let decoded = if self.input.a < qe {
            self.input.a = qe;
            self.transition_mps(context_index)?;
            context.mps
        } else {
            self.input.a = qe;
            self.transition_lps(context_index)?;
            !context.mps
        };

        self.input.renormalize();
        Ok(decoded)
    }
}

impl Tier1DecisionReader for MqDecoder {
    fn decode_context_bit(&mut self, context: usize) -> Result<bool, Tier1DecodeError> {
        self.decode_bit(context)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct MqInput {
    data: Vec<u8>,
    byte_pos: usize,
    c: u32,
    a: u32,
    ct: u32,
}

impl MqInput {
    fn reset(&mut self, data: &[u8]) {
        self.data.clear();
        self.data.extend_from_slice(data);
        self.data.extend_from_slice(&[0xff, 0xff]);
        self.byte_pos = 0;
        self.c = if data.is_empty() {
            0xff << 16
        } else {
            u32::from(self.byte_at(0)) << 16
        };
        self.ct = 0;
        self.byte_in();
        self.c <<= 7;
        self.ct = self.ct.saturating_sub(7);
        self.a = 0x8000;
    }

    fn byte_in(&mut self) {
        let current = self.byte_at(self.byte_pos);
        let next = self.byte_at(self.byte_pos.saturating_add(1));
        if current == 0xff {
            if next > 0x8f {
                self.c = self.c.wrapping_add(0xff00);
                self.ct = 8;
            } else {
                self.byte_pos = self.byte_pos.saturating_add(1);
                self.c = self.c.wrapping_add(u32::from(next) << 9);
                self.ct = 7;
            }
        } else {
            self.byte_pos = self.byte_pos.saturating_add(1);
            self.c = self.c.wrapping_add(u32::from(next) << 8);
            self.ct = 8;
        }
    }

    fn renormalize_once(&mut self) {
        if self.ct == 0 {
            self.byte_in();
        }
        self.a <<= 1;
        self.c <<= 1;
        self.ct = self.ct.saturating_sub(1);
    }

    fn renormalize(&mut self) {
        while self.a < 0x8000 {
            self.renormalize_once();
        }
    }

    fn byte_at(&self, index: usize) -> u8 {
        self.data.get(index).copied().unwrap_or(0xff)
    }

    fn registers(&self) -> MqInputRegisters {
        MqInputRegisters {
            byte_pos: self.byte_pos,
            c: self.c,
            a: self.a,
            ct: self.ct,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MqInputRegisters {
    pub byte_pos: usize,
    pub c: u32,
    pub a: u32,
    pub ct: u32,
}

impl Default for MqDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MqContext {
    pub state: u8,
    pub mps: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MqState {
    pub qe: u16,
    pub next_mps: u8,
    pub next_lps: u8,
    pub switch_mps: bool,
}

pub const MQ_STATES: [MqState; MQ_STATE_COUNT] = [
    MqState {
        qe: 0x5601,
        next_mps: 1,
        next_lps: 1,
        switch_mps: true,
    },
    MqState {
        qe: 0x3401,
        next_mps: 2,
        next_lps: 6,
        switch_mps: false,
    },
    MqState {
        qe: 0x1801,
        next_mps: 3,
        next_lps: 9,
        switch_mps: false,
    },
    MqState {
        qe: 0x0ac1,
        next_mps: 4,
        next_lps: 12,
        switch_mps: false,
    },
    MqState {
        qe: 0x0521,
        next_mps: 5,
        next_lps: 29,
        switch_mps: false,
    },
    MqState {
        qe: 0x0221,
        next_mps: 38,
        next_lps: 33,
        switch_mps: false,
    },
    MqState {
        qe: 0x5601,
        next_mps: 7,
        next_lps: 6,
        switch_mps: true,
    },
    MqState {
        qe: 0x5401,
        next_mps: 8,
        next_lps: 14,
        switch_mps: false,
    },
    MqState {
        qe: 0x4801,
        next_mps: 9,
        next_lps: 14,
        switch_mps: false,
    },
    MqState {
        qe: 0x3801,
        next_mps: 10,
        next_lps: 14,
        switch_mps: false,
    },
    MqState {
        qe: 0x3001,
        next_mps: 11,
        next_lps: 17,
        switch_mps: false,
    },
    MqState {
        qe: 0x2401,
        next_mps: 12,
        next_lps: 18,
        switch_mps: false,
    },
    MqState {
        qe: 0x1c01,
        next_mps: 13,
        next_lps: 20,
        switch_mps: false,
    },
    MqState {
        qe: 0x1601,
        next_mps: 29,
        next_lps: 21,
        switch_mps: false,
    },
    MqState {
        qe: 0x5601,
        next_mps: 15,
        next_lps: 14,
        switch_mps: true,
    },
    MqState {
        qe: 0x5401,
        next_mps: 16,
        next_lps: 14,
        switch_mps: false,
    },
    MqState {
        qe: 0x5101,
        next_mps: 17,
        next_lps: 15,
        switch_mps: false,
    },
    MqState {
        qe: 0x4801,
        next_mps: 18,
        next_lps: 16,
        switch_mps: false,
    },
    MqState {
        qe: 0x3801,
        next_mps: 19,
        next_lps: 17,
        switch_mps: false,
    },
    MqState {
        qe: 0x3401,
        next_mps: 20,
        next_lps: 18,
        switch_mps: false,
    },
    MqState {
        qe: 0x3001,
        next_mps: 21,
        next_lps: 19,
        switch_mps: false,
    },
    MqState {
        qe: 0x2801,
        next_mps: 22,
        next_lps: 19,
        switch_mps: false,
    },
    MqState {
        qe: 0x2401,
        next_mps: 23,
        next_lps: 20,
        switch_mps: false,
    },
    MqState {
        qe: 0x2201,
        next_mps: 24,
        next_lps: 21,
        switch_mps: false,
    },
    MqState {
        qe: 0x1c01,
        next_mps: 25,
        next_lps: 22,
        switch_mps: false,
    },
    MqState {
        qe: 0x1801,
        next_mps: 26,
        next_lps: 23,
        switch_mps: false,
    },
    MqState {
        qe: 0x1601,
        next_mps: 27,
        next_lps: 24,
        switch_mps: false,
    },
    MqState {
        qe: 0x1401,
        next_mps: 28,
        next_lps: 25,
        switch_mps: false,
    },
    MqState {
        qe: 0x1201,
        next_mps: 29,
        next_lps: 26,
        switch_mps: false,
    },
    MqState {
        qe: 0x1101,
        next_mps: 30,
        next_lps: 27,
        switch_mps: false,
    },
    MqState {
        qe: 0x0ac1,
        next_mps: 31,
        next_lps: 28,
        switch_mps: false,
    },
    MqState {
        qe: 0x09c1,
        next_mps: 32,
        next_lps: 29,
        switch_mps: false,
    },
    MqState {
        qe: 0x08a1,
        next_mps: 33,
        next_lps: 30,
        switch_mps: false,
    },
    MqState {
        qe: 0x0521,
        next_mps: 34,
        next_lps: 31,
        switch_mps: false,
    },
    MqState {
        qe: 0x0441,
        next_mps: 35,
        next_lps: 32,
        switch_mps: false,
    },
    MqState {
        qe: 0x02a1,
        next_mps: 36,
        next_lps: 33,
        switch_mps: false,
    },
    MqState {
        qe: 0x0221,
        next_mps: 37,
        next_lps: 34,
        switch_mps: false,
    },
    MqState {
        qe: 0x0141,
        next_mps: 38,
        next_lps: 35,
        switch_mps: false,
    },
    MqState {
        qe: 0x0111,
        next_mps: 39,
        next_lps: 36,
        switch_mps: false,
    },
    MqState {
        qe: 0x0085,
        next_mps: 40,
        next_lps: 37,
        switch_mps: false,
    },
    MqState {
        qe: 0x0049,
        next_mps: 41,
        next_lps: 38,
        switch_mps: false,
    },
    MqState {
        qe: 0x0025,
        next_mps: 42,
        next_lps: 39,
        switch_mps: false,
    },
    MqState {
        qe: 0x0015,
        next_mps: 43,
        next_lps: 40,
        switch_mps: false,
    },
    MqState {
        qe: 0x0009,
        next_mps: 44,
        next_lps: 41,
        switch_mps: false,
    },
    MqState {
        qe: 0x0005,
        next_mps: 45,
        next_lps: 42,
        switch_mps: false,
    },
    MqState {
        qe: 0x0001,
        next_mps: 45,
        next_lps: 43,
        switch_mps: false,
    },
    MqState {
        qe: 0x5601,
        next_mps: 46,
        next_lps: 46,
        switch_mps: false,
    },
];
