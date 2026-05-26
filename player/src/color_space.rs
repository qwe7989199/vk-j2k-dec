#[derive(Clone, Copy)]
pub enum ColorSpace {
    Passthrough,
    Xyz,
    Bt709,
    Bt2020,
}

pub enum TransferFunction {
    Linear,
    Srgb,
    Pq,
}

pub struct ColorConfig {
    pub matrix: [f32; 9],
    pub offset: [f32; 3],
    pub input_scale: f32,
    pub transfer_func: TransferFunction,
}

impl ColorConfig {
    pub fn for_colorspace(cs: ColorSpace, precision: u8) -> Self {
        let max_val = ((1u32 << precision) - 1) as f32;
        let input_scale = 1.0 / max_val;
        let level_shift = -((1u32 << (precision - 1)) as f32);

        match cs {
            ColorSpace::Passthrough => Self {
                matrix: IDENTITY,
                offset: [0.0, 0.0, 0.0],
                input_scale,
                transfer_func: TransferFunction::Srgb,
            },
            ColorSpace::Xyz => Self {
                matrix: XYZ_TO_SRGB,
                offset: [level_shift, level_shift, level_shift],
                input_scale,
                transfer_func: TransferFunction::Srgb,
            },
            ColorSpace::Bt709 => Self {
                matrix: BT709_TO_RGB,
                offset: [level_shift, 0.0, 0.0],
                input_scale,
                transfer_func: TransferFunction::Srgb,
            },
            ColorSpace::Bt2020 => Self {
                matrix: BT2020_TO_RGB,
                offset: [level_shift, 0.0, 0.0],
                input_scale,
                transfer_func: TransferFunction::Srgb,
            },
        }
    }

    pub fn transfer_func_value(&self) -> u32 {
        match self.transfer_func {
            TransferFunction::Linear => 0,
            TransferFunction::Srgb => 1,
            TransferFunction::Pq => 2,
        }
    }
}

const IDENTITY: [f32; 9] = [
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
];

const XYZ_TO_SRGB: [f32; 9] = [
     3.2404542, -1.5371385, -0.4985314,
    -0.9692660,  1.8760108,  0.0415560,
     0.0556434, -0.2040259,  1.0572252,
];

const BT709_TO_RGB: [f32; 9] = [
    1.0,  0.0,       1.5748,
    1.0, -0.1873,   -0.4681,
    1.0,  1.8556,    0.0,
];

const BT2020_TO_RGB: [f32; 9] = [
    1.0,  0.0,       1.4746,
    1.0, -0.1646,   -0.5714,
    1.0,  1.8814,    0.0,
];

pub fn parse_colorspace(s: &str) -> Option<ColorSpace> {
    match s.to_lowercase().as_str() {
        "rgb" | "passthrough" | "none" => Some(ColorSpace::Passthrough),
        "xyz" | "dci-p3" | "dcip3" => Some(ColorSpace::Xyz),
        "bt709" | "709" | "rec709" => Some(ColorSpace::Bt709),
        "bt2020" | "2020" | "rec2020" => Some(ColorSpace::Bt2020),
        _ => None,
    }
}
