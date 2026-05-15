#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorError {
    pub message: String,
}

impl ColorError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ColorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ColorError {}

pub fn inverse_rct_sample(y: i32, db: i32, dr: i32) -> (i32, i32, i32) {
    let g = y - floor_div_pow2(db + dr, 2);
    let r = dr + g;
    let b = db + g;
    (r, g, b)
}

pub fn inverse_ict_sample(y: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
    (
        y + 1.402 * cr,
        y - 0.344_136 * cb - 0.714_136 * cr,
        y + 1.772 * cb,
    )
}

pub fn level_shift_and_clamp(value: i32, precision: u8) -> Result<u16, ColorError> {
    if precision == 0 || precision > 16 {
        return Err(ColorError::new(format!(
            "precision {precision} cannot be packed to u16"
        )));
    }
    let shift = 1i32 << (precision - 1);
    let max = (1i32 << precision) - 1;
    Ok((value + shift).clamp(0, max) as u16)
}

pub fn level_shift_and_clamp_float(value: f32, precision: u8) -> Result<u16, ColorError> {
    if precision == 0 || precision > 16 {
        return Err(ColorError::new(format!(
            "precision {precision} cannot be packed to u16"
        )));
    }
    let shift = (1u32 << (precision - 1)) as f32;
    let max = ((1u32 << precision) - 1) as f32;
    Ok((value + shift).round().clamp(0.0, max) as u16)
}

fn floor_div_pow2(value: i32, shift: u32) -> i32 {
    value.div_euclid(1i32 << shift)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_rct_reconstructs_rgb_sample() {
        let r = 130;
        let g = 90;
        let b = 40;
        let y = floor_div_pow2(r + 2 * g + b, 2);
        let db = b - g;
        let dr = r - g;

        assert_eq!(inverse_rct_sample(y, db, dr), (r, g, b));
    }

    #[test]
    fn inverse_ict_reconstructs_reference_values() {
        let (r, g, b) = inverse_ict_sample(100.0, 10.0, -5.0);

        assert!((r - 92.99).abs() < 0.001);
        assert!((g - 100.129_32).abs() < 0.001);
        assert!((b - 117.72).abs() < 0.001);
    }

    #[test]
    fn level_shift_clamps_to_precision() {
        assert_eq!(level_shift_and_clamp(-2048, 12).expect("pack"), 0);
        assert_eq!(level_shift_and_clamp(0, 12).expect("pack"), 2048);
        assert_eq!(level_shift_and_clamp(4096, 12).expect("pack"), 4095);
        assert!(level_shift_and_clamp(0, 17).is_err());
    }
}
