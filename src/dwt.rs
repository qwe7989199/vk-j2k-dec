#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DwtError {
    pub message: String,
}

impl DwtError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for DwtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DwtError {}

pub fn inverse_5_3_1d(low: &[i32], high: &[i32]) -> Result<Vec<i32>, DwtError> {
    if low.len() < high.len() || low.len().saturating_sub(high.len()) > 1 {
        return Err(DwtError::new(format!(
            "invalid 5/3 band lengths: low {}, high {}",
            low.len(),
            high.len()
        )));
    }
    if low.is_empty() {
        return Ok(Vec::new());
    }
    if high.is_empty() {
        return Ok(low.to_vec());
    }

    let mut low = low.to_vec();
    let mut high = high.to_vec();

    for index in 0..low.len() {
        let left = if index == 0 { high[0] } else { high[index - 1] };
        let right = if index < high.len() {
            high[index]
        } else {
            high[high.len() - 1]
        };
        low[index] = checked_i64_to_i32(
            i64::from(low[index]) - floor_div_pow2_i64(i64::from(left) + i64::from(right) + 2, 2),
        )?;
    }

    for index in 0..high.len() {
        let left = low[index];
        let right = low.get(index + 1).copied().unwrap_or(left);
        high[index] = checked_i64_to_i32(
            i64::from(high[index]) + floor_div_pow2_i64(i64::from(left) + i64::from(right), 1),
        )?;
    }

    let mut output = Vec::with_capacity(low.len() + high.len());
    for index in 0..high.len() {
        output.push(low[index]);
        output.push(high[index]);
    }
    if low.len() > high.len() {
        output.push(low[low.len() - 1]);
    }

    Ok(output)
}

pub fn inverse_9_7_1d(low: &[f32], high: &[f32]) -> Result<Vec<f32>, DwtError> {
    if low.len() < high.len() || low.len().saturating_sub(high.len()) > 1 {
        return Err(DwtError::new(format!(
            "invalid 9/7 band lengths: low {}, high {}",
            low.len(),
            high.len()
        )));
    }
    if low.is_empty() {
        return Ok(Vec::new());
    }

    let unscaled_low = low.to_vec();
    let unscaled_high = (0..high.len())
        .map(|index| high[index] * DWT97_K)
        .collect::<Vec<_>>();

    let mut updated_low = Vec::with_capacity(low.len());
    for index in 0..low.len() {
        let left = if index == 0 {
            high_at_f32(&unscaled_high, 0)
        } else {
            high_at_f32(&unscaled_high, index - 1)
        };
        let right = high_at_f32(&unscaled_high, index);
        updated_low.push(unscaled_low[index] - DWT97_DELTA * (left + right));
    }

    let mut updated_high = Vec::with_capacity(high.len());
    for index in 0..high.len() {
        let left = updated_low[index];
        let right = if index + 1 < updated_low.len() {
            updated_low[index + 1]
        } else {
            updated_low[index]
        };
        updated_high.push(unscaled_high[index] - DWT97_GAMMA * (left + right));
    }

    for index in 0..updated_low.len() {
        let left = if index == 0 {
            high_at_f32(&updated_high, 0)
        } else {
            high_at_f32(&updated_high, index - 1)
        };
        let right = high_at_f32(&updated_high, index);
        updated_low[index] -= DWT97_BETA * (left + right);
    }

    for index in 0..updated_high.len() {
        let left = updated_low[index];
        let right = if index + 1 < updated_low.len() {
            updated_low[index + 1]
        } else {
            updated_low[index]
        };
        updated_high[index] -= DWT97_ALPHA * (left + right);
    }

    let mut output = Vec::with_capacity(low.len() + high.len());
    for index in 0..high.len() {
        output.push(updated_low[index]);
        output.push(updated_high[index]);
    }
    if low.len() > high.len() {
        output.push(updated_low[updated_low.len() - 1]);
    }

    Ok(output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Band2D<'a> {
    pub width: u32,
    pub height: u32,
    pub samples: &'a [i32],
}

impl Band2D<'_> {
    fn checked_len(&self, name: &str) -> Result<(), DwtError> {
        let expected = self.width as usize * self.height as usize;
        if self.samples.len() != expected {
            return Err(DwtError::new(format!(
                "{name} band has {} samples, expected {expected}",
                self.samples.len()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Band2DF32<'a> {
    pub width: u32,
    pub height: u32,
    pub samples: &'a [f32],
}

impl Band2DF32<'_> {
    fn checked_len(&self, name: &str) -> Result<(), DwtError> {
        let expected = self.width as usize * self.height as usize;
        if self.samples.len() != expected {
            return Err(DwtError::new(format!(
                "{name} band has {} samples, expected {expected}",
                self.samples.len()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneI32 {
    pub width: u32,
    pub height: u32,
    pub samples: Vec<i32>,
}

impl PlaneI32 {
    pub fn sample(&self, x: u32, y: u32) -> Option<i32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.samples[y as usize * self.width as usize + x as usize])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlaneF32 {
    pub width: u32,
    pub height: u32,
    pub samples: Vec<f32>,
}

impl PlaneF32 {
    pub fn sample(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.samples[y as usize * self.width as usize + x as usize])
    }
}

pub fn inverse_5_3_level(
    ll: Band2D<'_>,
    hl: Band2D<'_>,
    lh: Band2D<'_>,
    hh: Band2D<'_>,
) -> Result<PlaneI32, DwtError> {
    ll.checked_len("LL")?;
    hl.checked_len("HL")?;
    lh.checked_len("LH")?;
    hh.checked_len("HH")?;
    if ll.width != lh.width || hl.width != hh.width {
        return Err(DwtError::new(
            "low/high vertical bands have mismatched widths",
        ));
    }
    if ll.height != hl.height || lh.height != hh.height {
        return Err(DwtError::new(
            "low/high horizontal bands have mismatched heights",
        ));
    }

    let width = ll.width + hl.width;
    let height = ll.height + lh.height;
    let mut low_columns = vec![0; width as usize * ll.height as usize];
    let mut high_columns = vec![0; width as usize * lh.height as usize];

    // Match FFmpeg's integer 5/3 order. Horizontal and vertical lifting are not
    // exactly commutative once each 1D pass rounds intermediate values.
    for y in 0..ll.height as usize {
        let low_offset = y * ll.width as usize;
        let high_offset = y * hl.width as usize;
        let row = inverse_5_3_1d(
            &ll.samples[low_offset..low_offset + ll.width as usize],
            &hl.samples[high_offset..high_offset + hl.width as usize],
        )?;
        low_columns[y * width as usize..(y + 1) * width as usize].copy_from_slice(&row);
    }

    for y in 0..lh.height as usize {
        let low_offset = y * lh.width as usize;
        let high_offset = y * hh.width as usize;
        let row = inverse_5_3_1d(
            &lh.samples[low_offset..low_offset + lh.width as usize],
            &hh.samples[high_offset..high_offset + hh.width as usize],
        )?;
        high_columns[y * width as usize..(y + 1) * width as usize].copy_from_slice(&row);
    }

    let mut samples = vec![0; width as usize * height as usize];
    for x in 0..width as usize {
        let low_column = (0..ll.height as usize)
            .map(|y| low_columns[y * width as usize + x])
            .collect::<Vec<_>>();
        let high_column = (0..lh.height as usize)
            .map(|y| high_columns[y * width as usize + x])
            .collect::<Vec<_>>();
        let column = inverse_5_3_1d(&low_column, &high_column)?;
        for (y, value) in column.into_iter().enumerate() {
            samples[y * width as usize + x] = value;
        }
    }

    Ok(PlaneI32 {
        width,
        height,
        samples,
    })
}

pub fn inverse_9_7_level(
    ll: Band2DF32<'_>,
    hl: Band2DF32<'_>,
    lh: Band2DF32<'_>,
    hh: Band2DF32<'_>,
) -> Result<PlaneF32, DwtError> {
    ll.checked_len("LL")?;
    hl.checked_len("HL")?;
    lh.checked_len("LH")?;
    hh.checked_len("HH")?;
    if ll.width != lh.width || hl.width != hh.width {
        return Err(DwtError::new(
            "low/high vertical bands have mismatched widths",
        ));
    }
    if ll.height != hl.height || lh.height != hh.height {
        return Err(DwtError::new(
            "low/high horizontal bands have mismatched heights",
        ));
    }

    let width = ll.width + hl.width;
    let height = ll.height + lh.height;
    let mut low_rows = vec![0.0; ll.width as usize * height as usize];
    let mut high_rows = vec![0.0; hl.width as usize * height as usize];

    for x in 0..ll.width as usize {
        let low_column = (0..ll.height as usize)
            .map(|y| ll.samples[y * ll.width as usize + x])
            .collect::<Vec<_>>();
        let high_column = (0..lh.height as usize)
            .map(|y| lh.samples[y * lh.width as usize + x])
            .collect::<Vec<_>>();
        let column = inverse_9_7_1d(&low_column, &high_column)?;
        for (y, value) in column.into_iter().enumerate() {
            low_rows[y * ll.width as usize + x] = value;
        }
    }

    for x in 0..hl.width as usize {
        let low_column = (0..hl.height as usize)
            .map(|y| hl.samples[y * hl.width as usize + x])
            .collect::<Vec<_>>();
        let high_column = (0..hh.height as usize)
            .map(|y| hh.samples[y * hh.width as usize + x])
            .collect::<Vec<_>>();
        let column = inverse_9_7_1d(&low_column, &high_column)?;
        for (y, value) in column.into_iter().enumerate() {
            high_rows[y * hl.width as usize + x] = value;
        }
    }

    let mut samples = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height as usize {
        let low_offset = y * ll.width as usize;
        let high_offset = y * hl.width as usize;
        samples.extend(inverse_9_7_1d(
            &low_rows[low_offset..low_offset + ll.width as usize],
            &high_rows[high_offset..high_offset + hl.width as usize],
        )?);
    }

    Ok(PlaneF32 {
        width,
        height,
        samples,
    })
}

const DWT97_ALPHA: f32 = -1.586_134_3;
const DWT97_BETA: f32 = -0.052_980_12;
const DWT97_GAMMA: f32 = 0.882_911_1;
const DWT97_DELTA: f32 = 0.443_506_87;
const DWT97_K: f32 = 1.230_174_1;

#[cfg(test)]
fn floor_div_pow2(value: i32, shift: u32) -> i32 {
    value.div_euclid(1i32 << shift)
}

fn floor_div_pow2_i64(value: i64, shift: u32) -> i64 {
    value.div_euclid(1i64 << shift)
}

fn checked_i64_to_i32(value: i64) -> Result<i32, DwtError> {
    i32::try_from(value).map_err(|_| {
        DwtError::new(format!(
            "inverse 5/3 intermediate value {value} exceeds i32 coefficient staging"
        ))
    })
}

fn high_at_f32(high: &[f32], index: usize) -> f32 {
    if high.is_empty() {
        0.0
    } else {
        high[index.min(high.len() - 1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_5_3_roundtrips_even_length_signal() {
        let original = vec![12, -3, 7, 4, 18, 6];
        let (low, high) = forward_5_3_1d(&original);

        assert_eq!(inverse_5_3_1d(&low, &high).expect("inverse"), original);
    }

    #[test]
    fn inverse_5_3_roundtrips_odd_length_signal() {
        let original = vec![5, 9, -2, 3, 11];
        let (low, high) = forward_5_3_1d(&original);

        assert_eq!(inverse_5_3_1d(&low, &high).expect("inverse"), original);
    }

    #[test]
    fn inverse_5_3_rejects_impossible_band_lengths() {
        let err = inverse_5_3_1d(&[1], &[2, 3]).expect_err("must fail");

        assert!(err.message.contains("invalid"));
    }

    #[test]
    fn inverse_5_3_level_matches_horizontal_first_integer_decode() {
        let original = PlaneI32 {
            width: 5,
            height: 4,
            samples: vec![
                1, 5, 2, 8, 3, -4, 9, 7, 6, 0, 12, 11, -2, 4, 10, 3, 2, 1, 0, -1,
            ],
        };
        let (ll, hl, lh, hh) = forward_5_3_level(&original);

        let reconstructed = inverse_5_3_level(
            Band2D {
                width: ll.width,
                height: ll.height,
                samples: &ll.samples,
            },
            Band2D {
                width: hl.width,
                height: hl.height,
                samples: &hl.samples,
            },
            Band2D {
                width: lh.width,
                height: lh.height,
                samples: &lh.samples,
            },
            Band2D {
                width: hh.width,
                height: hh.height,
                samples: &hh.samples,
            },
        )
        .expect("inverse level");

        assert_eq!(
            reconstructed.samples,
            vec![
                1, 6, 2, 9, 3, -5, 9, 7, 6, 0, 12, 11, -1, 4, 10, 3, 1, 1, -1, -1
            ]
        );
        assert_eq!(reconstructed.width, original.width);
        assert_eq!(reconstructed.height, original.height);
        assert_eq!(reconstructed.sample(4, 3), Some(-1));
        assert_eq!(reconstructed.sample(5, 3), None);
    }

    #[test]
    fn inverse_9_7_roundtrips_even_length_signal() {
        let original = vec![12.25, -3.5, 7.0, 4.75, 18.125, 6.5];
        let (low, high) = forward_9_7_1d(&original);
        let reconstructed = inverse_9_7_1d(&low, &high).expect("inverse");

        assert_f32_slices_close(&reconstructed, &original);
    }

    #[test]
    fn inverse_9_7_roundtrips_odd_length_signal() {
        let original = vec![5.0, 9.5, -2.25, 3.75, 11.0];
        let (low, high) = forward_9_7_1d(&original);
        let reconstructed = inverse_9_7_1d(&low, &high).expect("inverse");

        assert_f32_slices_close(&reconstructed, &original);
    }

    #[test]
    fn inverse_9_7_rejects_impossible_band_lengths() {
        let err = inverse_9_7_1d(&[1.0], &[2.0, 3.0]).expect_err("must fail");

        assert!(err.message.contains("invalid"));
    }

    #[test]
    fn inverse_9_7_level_roundtrips_plane() {
        let original = PlaneF32 {
            width: 5,
            height: 4,
            samples: vec![
                1.25, 5.5, 2.0, 8.75, 3.0, -4.5, 9.25, 7.0, 6.5, 0.25, 12.0, 11.5, -2.25, 4.0,
                10.75, 3.125, 2.5, 1.0, 0.0, -1.5,
            ],
        };
        let (ll, hl, lh, hh) = forward_9_7_level(&original);

        let reconstructed = inverse_9_7_level(
            Band2DF32 {
                width: ll.width,
                height: ll.height,
                samples: &ll.samples,
            },
            Band2DF32 {
                width: hl.width,
                height: hl.height,
                samples: &hl.samples,
            },
            Band2DF32 {
                width: lh.width,
                height: lh.height,
                samples: &lh.samples,
            },
            Band2DF32 {
                width: hh.width,
                height: hh.height,
                samples: &hh.samples,
            },
        )
        .expect("inverse level");

        assert_eq!(reconstructed.width, original.width);
        assert_eq!(reconstructed.height, original.height);
        assert_f32_slices_close(&reconstructed.samples, &original.samples);
        assert_f32_close(reconstructed.sample(4, 3).unwrap(), -1.5);
        assert_eq!(reconstructed.sample(5, 3), None);
    }

    fn forward_5_3_1d(input: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let mut low = input.iter().step_by(2).copied().collect::<Vec<_>>();
        let mut high = input.iter().skip(1).step_by(2).copied().collect::<Vec<_>>();

        for index in 0..high.len() {
            let left = low[index];
            let right = low.get(index + 1).copied().unwrap_or(left);
            high[index] -= floor_div_pow2(left + right, 1);
        }

        for index in 0..low.len() {
            if high.is_empty() {
                continue;
            }
            let left = if index == 0 { high[0] } else { high[index - 1] };
            let right = if index < high.len() {
                high[index]
            } else {
                high[high.len() - 1]
            };
            low[index] += floor_div_pow2(left + right + 2, 2);
        }

        (low, high)
    }

    fn forward_5_3_level(input: &PlaneI32) -> (PlaneI32, PlaneI32, PlaneI32, PlaneI32) {
        let low_width = input.width.div_ceil(2);
        let high_width = input.width / 2;
        let low_height = input.height.div_ceil(2);
        let high_height = input.height / 2;

        let mut low_rows = vec![0; low_width as usize * input.height as usize];
        let mut high_rows = vec![0; high_width as usize * input.height as usize];
        for y in 0..input.height {
            let start = y as usize * input.width as usize;
            let row = &input.samples[start..start + input.width as usize];
            let (low, high) = forward_5_3_1d(row);
            let low_offset = y as usize * low_width as usize;
            let high_offset = y as usize * high_width as usize;
            low_rows[low_offset..low_offset + low.len()].copy_from_slice(&low);
            high_rows[high_offset..high_offset + high.len()].copy_from_slice(&high);
        }

        let mut ll = vec![0; low_width as usize * low_height as usize];
        let mut lh = vec![0; low_width as usize * high_height as usize];
        for x in 0..low_width as usize {
            let column = (0..input.height as usize)
                .map(|y| low_rows[y * low_width as usize + x])
                .collect::<Vec<_>>();
            let (low, high) = forward_5_3_1d(&column);
            for (y, value) in low.into_iter().enumerate() {
                ll[y * low_width as usize + x] = value;
            }
            for (y, value) in high.into_iter().enumerate() {
                lh[y * low_width as usize + x] = value;
            }
        }

        let mut hl = vec![0; high_width as usize * low_height as usize];
        let mut hh = vec![0; high_width as usize * high_height as usize];
        for x in 0..high_width as usize {
            let column = (0..input.height as usize)
                .map(|y| high_rows[y * high_width as usize + x])
                .collect::<Vec<_>>();
            let (low, high) = forward_5_3_1d(&column);
            for (y, value) in low.into_iter().enumerate() {
                hl[y * high_width as usize + x] = value;
            }
            for (y, value) in high.into_iter().enumerate() {
                hh[y * high_width as usize + x] = value;
            }
        }

        (
            PlaneI32 {
                width: low_width,
                height: low_height,
                samples: ll,
            },
            PlaneI32 {
                width: high_width,
                height: low_height,
                samples: hl,
            },
            PlaneI32 {
                width: low_width,
                height: high_height,
                samples: lh,
            },
            PlaneI32 {
                width: high_width,
                height: high_height,
                samples: hh,
            },
        )
    }

    fn forward_9_7_1d(input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut low = input.iter().step_by(2).copied().collect::<Vec<_>>();
        let mut high = input.iter().skip(1).step_by(2).copied().collect::<Vec<_>>();

        for index in 0..high.len() {
            let left = low[index];
            let right = low.get(index + 1).copied().unwrap_or(left);
            high[index] += DWT97_ALPHA * (left + right);
        }

        for index in 0..low.len() {
            let left = if index == 0 {
                high_at_f32(&high, 0)
            } else {
                high_at_f32(&high, index - 1)
            };
            let right = high_at_f32(&high, index);
            low[index] += DWT97_BETA * (left + right);
        }

        for index in 0..high.len() {
            let left = low[index];
            let right = low.get(index + 1).copied().unwrap_or(left);
            high[index] += DWT97_GAMMA * (left + right);
        }

        for index in 0..low.len() {
            let left = if index == 0 {
                high_at_f32(&high, 0)
            } else {
                high_at_f32(&high, index - 1)
            };
            let right = high_at_f32(&high, index);
            low[index] += DWT97_DELTA * (left + right);
        }

        for value in &mut high {
            *value /= DWT97_K;
        }

        (low, high)
    }

    fn forward_9_7_level(input: &PlaneF32) -> (PlaneF32, PlaneF32, PlaneF32, PlaneF32) {
        let low_width = input.width.div_ceil(2);
        let high_width = input.width / 2;
        let low_height = input.height.div_ceil(2);
        let high_height = input.height / 2;

        let mut low_rows = vec![0.0; low_width as usize * input.height as usize];
        let mut high_rows = vec![0.0; high_width as usize * input.height as usize];
        for y in 0..input.height {
            let start = y as usize * input.width as usize;
            let row = &input.samples[start..start + input.width as usize];
            let (low, high) = forward_9_7_1d(row);
            let low_offset = y as usize * low_width as usize;
            let high_offset = y as usize * high_width as usize;
            low_rows[low_offset..low_offset + low.len()].copy_from_slice(&low);
            high_rows[high_offset..high_offset + high.len()].copy_from_slice(&high);
        }

        let mut ll = vec![0.0; low_width as usize * low_height as usize];
        let mut lh = vec![0.0; low_width as usize * high_height as usize];
        for x in 0..low_width as usize {
            let column = (0..input.height as usize)
                .map(|y| low_rows[y * low_width as usize + x])
                .collect::<Vec<_>>();
            let (low, high) = forward_9_7_1d(&column);
            for (y, value) in low.into_iter().enumerate() {
                ll[y * low_width as usize + x] = value;
            }
            for (y, value) in high.into_iter().enumerate() {
                lh[y * low_width as usize + x] = value;
            }
        }

        let mut hl = vec![0.0; high_width as usize * low_height as usize];
        let mut hh = vec![0.0; high_width as usize * high_height as usize];
        for x in 0..high_width as usize {
            let column = (0..input.height as usize)
                .map(|y| high_rows[y * high_width as usize + x])
                .collect::<Vec<_>>();
            let (low, high) = forward_9_7_1d(&column);
            for (y, value) in low.into_iter().enumerate() {
                hl[y * high_width as usize + x] = value;
            }
            for (y, value) in high.into_iter().enumerate() {
                hh[y * high_width as usize + x] = value;
            }
        }

        (
            PlaneF32 {
                width: low_width,
                height: low_height,
                samples: ll,
            },
            PlaneF32 {
                width: high_width,
                height: low_height,
                samples: hl,
            },
            PlaneF32 {
                width: low_width,
                height: high_height,
                samples: lh,
            },
            PlaneF32 {
                width: high_width,
                height: high_height,
                samples: hh,
            },
        )
    }

    fn assert_f32_slices_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.0001,
                "sample {index}: got {actual}, expected {expected}"
            );
        }
    }

    fn assert_f32_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 0.0001,
            "got {actual}, expected {expected}"
        );
    }

    #[test]
    fn inverse_97_dc_gain_2d_6_levels() {
        use super::*;
        // Forward transform a constant signal through 6 levels to measure
        // what coefficient values the encoder produces.
        let signal = PlaneF32 {
            width: 3840,
            height: 2160,
            samples: vec![1000.0f32; 3840 * 2160],
        };
        // Do 6 levels of forward 2D DWT
        let mut current = signal;
        for _level in 0..6 {
            let (ll, _hl, _lh, _hh) = forward_9_7_level(&current);
            current = ll;
        }
        // The LL coefficient after 6 levels of forward DWT on constant 1000
        let ll_coeff = current.samples[current.samples.len() / 2];
        eprintln!("forward 6-level LL of constant 1000: {ll_coeff}");
        eprintln!("  ratio = {}", ll_coeff / 1000.0);
        eprintln!("  K^12 = {}", 1.2301741f64.powi(12));
        // Now inverse should give back 1000
        // dequant_scale should be such that: ll_coeff / dequant_scale * iDWT_gain = 1000
        assert!((ll_coeff / 1000.0 - DWT97_K.powi(12)).abs() < 0.05);
    }

    #[test]
    fn inverse_9_7_low_band_dc_gain_is_one_over_k() {
        let low = vec![-2047.0; 8];
        let high = vec![0.0; 8];
        let reconstructed = inverse_9_7_1d(&low, &high).expect("inverse");
        let expected = -2047.0 / DWT97_K;

        assert!(
            (reconstructed[reconstructed.len() / 2] - expected).abs() < 0.001,
            "got {}, expected {expected}",
            reconstructed[reconstructed.len() / 2]
        );
    }
}
