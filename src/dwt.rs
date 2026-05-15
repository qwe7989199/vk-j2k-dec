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
    let mut low_rows = vec![0; ll.width as usize * height as usize];
    let mut high_rows = vec![0; hl.width as usize * height as usize];

    for x in 0..ll.width as usize {
        let low_column = (0..ll.height as usize)
            .map(|y| ll.samples[y * ll.width as usize + x])
            .collect::<Vec<_>>();
        let high_column = (0..lh.height as usize)
            .map(|y| lh.samples[y * lh.width as usize + x])
            .collect::<Vec<_>>();
        let column = inverse_5_3_1d(&low_column, &high_column)?;
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
        let column = inverse_5_3_1d(&low_column, &high_column)?;
        for (y, value) in column.into_iter().enumerate() {
            high_rows[y * hl.width as usize + x] = value;
        }
    }

    let mut samples = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height as usize {
        let low_offset = y * ll.width as usize;
        let high_offset = y * hl.width as usize;
        samples.extend(inverse_5_3_1d(
            &low_rows[low_offset..low_offset + ll.width as usize],
            &high_rows[high_offset..high_offset + hl.width as usize],
        )?);
    }

    Ok(PlaneI32 {
        width,
        height,
        samples,
    })
}

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
    fn inverse_5_3_level_roundtrips_plane() {
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

        assert_eq!(reconstructed, original);
        assert_eq!(reconstructed.sample(4, 3), Some(-1));
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
}
