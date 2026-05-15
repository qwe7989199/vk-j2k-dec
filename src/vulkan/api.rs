pub struct VulkanDeviceSummary {
    pub name: String,
    pub api_version: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: VulkanDeviceType,
    pub queue_families: Vec<VulkanQueueFamilySummary>,
}

impl VulkanDeviceSummary {
    pub fn first_compute_queue_family(&self) -> Option<&VulkanQueueFamilySummary> {
        self.queue_families
            .iter()
            .filter(|family| family.compute)
            .min_by_key(|family| family.graphics)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanDeviceType {
    Other,
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
    Unknown(u32),
}

impl From<u32> for VulkanDeviceType {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Other,
            1 => Self::IntegratedGpu,
            2 => Self::DiscreteGpu,
            3 => Self::VirtualGpu,
            4 => Self::Cpu,
            other => Self::Unknown(other),
        }
    }
}

impl std::fmt::Display for VulkanDeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Other => f.write_str("other"),
            Self::IntegratedGpu => f.write_str("integrated GPU"),
            Self::DiscreteGpu => f.write_str("discrete GPU"),
            Self::VirtualGpu => f.write_str("virtual GPU"),
            Self::Cpu => f.write_str("CPU"),
            Self::Unknown(value) => write!(f, "unknown({value})"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanQueueFamilySummary {
    pub index: u32,
    pub queue_count: u32,
    pub graphics: bool,
    pub compute: bool,
    pub transfer: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanError {
    pub message: String,
}

impl VulkanError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for VulkanError {}

pub fn enumerate_devices() -> Result<Vec<VulkanDeviceSummary>, VulkanError> {
    let loader = VulkanLoader::open()?;
    let instance = loader.create_instance()?;
    let result = instance.enumerate_devices();
    drop(instance);
    result
}

fn with_compute_device<T>(
    run: impl FnOnce(&VulkanDevice, &VulkanDeviceSummary, u32) -> Result<T, VulkanError>,
) -> Result<T, VulkanError> {
    let loader = VulkanLoader::open()?;
    let instance = loader.create_instance()?;
    let (physical_device, summary) = instance.select_compute_device()?;
    let queue_family_index = summary
        .first_compute_queue_family()
        .ok_or_else(|| VulkanError::new("selected Vulkan device has no compute queue"))?
        .index;
    let device = instance.create_compute_device(physical_device, queue_family_index)?;
    run(&device, &summary, queue_family_index)
}

fn dispatch_storage_shader(
    device: &VulkanDevice,
    queue_family_index: u32,
    spirv: &[u32],
    descriptors: &[VkDescriptorBufferInfo],
    push_constants: &[u8],
    groups_x: u32,
    groups_y: u32,
) -> Result<(), VulkanError> {
    let descriptor_set_layout =
        device.create_storage_descriptor_set_layout(descriptors.len() as u32)?;
    let pipeline_layout =
        device.create_pipeline_layout(descriptor_set_layout.handle, push_constants.len() as u32)?;
    let shader_module = device.create_shader_module(spirv)?;
    let pipeline = device.create_compute_pipeline(shader_module.handle, pipeline_layout.handle)?;
    let descriptor_pool = device.create_descriptor_pool(descriptors.len() as u32)?;
    let descriptor_set =
        device.allocate_descriptor_set(descriptor_pool.handle, descriptor_set_layout.handle)?;
    device.update_storage_descriptor_set(descriptor_set, descriptors);
    let command_pool = device.create_command_pool(queue_family_index)?;
    let command_buffer = device.allocate_command_buffer(command_pool.handle)?;
    device.record_storage_dispatch(
        command_buffer,
        StorageDispatch {
            pipeline: pipeline.handle,
            pipeline_layout: pipeline_layout.handle,
            descriptor_set,
            push_constants,
            groups: [groups_x, groups_y, 1],
        },
    )?;
    device.submit_and_wait(command_buffer)
}

#[derive(Debug, Clone, PartialEq)]
pub struct DequantSmokeResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub input: Vec<i32>,
    pub output: Vec<f32>,
}

pub fn run_dequant_smoke(shader_path: &str) -> Result<DequantSmokeResult, VulkanError> {
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let input = vec![1i32, -2, 7, 13, -11, 0, 256, -128];
        let sample_count = input.len() as u32;
        let step_indices = vec![0u32; input.len()];
        let steps = vec![2.5f32];
        let output = vec![0f32; input.len()];

        let coeff_buffer = device.create_host_buffer(bytes_len::<i32>(input.len())?)?;
        let step_index_buffer = device.create_host_buffer(bytes_len::<u32>(step_indices.len())?)?;
        let step_buffer = device.create_host_buffer(bytes_len::<f32>(steps.len())?)?;
        let output_buffer = device.create_host_buffer(bytes_len::<f32>(output.len())?)?;

        coeff_buffer.write_slice(&input)?;
        step_index_buffer.write_slice(&step_indices)?;
        step_buffer.write_slice(&steps)?;
        output_buffer.write_slice(&output)?;
        dispatch_storage_shader(
            device,
            queue_family_index,
            &spirv,
            &[
                coeff_buffer.descriptor_info(),
                step_index_buffer.descriptor_info(),
                step_buffer.descriptor_info(),
                output_buffer.descriptor_info(),
            ],
            bytes_of(&sample_count),
            sample_count.div_ceil(256),
            1,
        )?;

        let output = output_buffer.read_vec::<f32>(input.len())?;
        let expected = input
            .iter()
            .map(|value| *value as f32 * steps[0])
            .collect::<Vec<_>>();
        for (index, (actual, expected)) in output.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 0.0001 {
                return Err(VulkanError::new(format!(
                    "dequant smoke output mismatch at {index}: got {actual}, expected {expected}"
                )));
            }
        }
        device.wait_idle()?;

        Ok(DequantSmokeResult {
            device_name: summary.name.clone(),
            queue_family_index,
            input,
            output,
        })
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColorSmokeResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub none_output: Vec<u32>,
    pub rct_output: Vec<u32>,
    pub ict_output: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Idwt53SmokeResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub row_output: Vec<i32>,
    pub column_output: Vec<i32>,
    pub level_output: Vec<i32>,
}

pub fn run_idwt53_smoke(shader_path: &str) -> Result<Idwt53SmokeResult, VulkanError> {
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let low_count = 3usize;
        let high_count = 2usize;
        let output_stride = low_count + high_count;
        let line_count = 2usize;
        let low = vec![4i32, -3, 12, 1, 5, 9];
        let high = vec![7i32, -5, 2, 3];
        let expected_line0 = inverse_5_3_1d(&low[..low_count], &high[..high_count])
            .map_err(|err| VulkanError::new(format!("CPU inverse 5/3 failed: {err}")))?;
        let expected_line1 = inverse_5_3_1d(&low[low_count..], &high[high_count..])
            .map_err(|err| VulkanError::new(format!("CPU inverse 5/3 failed: {err}")))?;
        let expected = expected_line0
            .into_iter()
            .chain(expected_line1)
            .collect::<Vec<_>>();
        let output_seed = vec![0i32; expected.len()];
        let push = Idwt53PushConstants {
            low_count: low_count as u32,
            high_count: high_count as u32,
            line_count: line_count as u32,
            low_line_stride: low_count as u32,
            low_sample_stride: 1,
            high_line_stride: high_count as u32,
            high_sample_stride: 1,
            output_line_stride: output_stride as u32,
            output_sample_stride: 1,
        };

        let low_buffer = device.create_host_buffer(bytes_len::<i32>(low.len())?)?;
        let high_buffer = device.create_host_buffer(bytes_len::<i32>(high.len())?)?;
        let output_buffer = device.create_host_buffer(bytes_len::<i32>(output_seed.len())?)?;
        low_buffer.write_slice(&low)?;
        high_buffer.write_slice(&high)?;
        output_buffer.write_slice(&output_seed)?;
        dispatch_storage_shader(
            device,
            queue_family_index,
            &spirv,
            &[
                low_buffer.descriptor_info(),
                high_buffer.descriptor_info(),
                output_buffer.descriptor_info(),
            ],
            bytes_of(&push),
            (output_stride as u32).div_ceil(16),
            (line_count as u32).div_ceil(16),
        )?;

        let row_output = output_buffer.read_vec::<i32>(output_seed.len())?;
        if row_output != expected {
            return Err(VulkanError::new(format!(
                "idwt53 row smoke output mismatch: got {row_output:?}, expected {expected:?}"
            )));
        }

        let vertical_width = 2usize;
        let low_vertical = vec![4i32, 1, -3, 5, 12, 9];
        let high_vertical = vec![7i32, 2, -5, 3];
        let column_expected = vec![0i32, 0, 5, 4, -4, 4, 0, 8, 14, 7];
        let column_seed = vec![0i32; column_expected.len()];
        let column_push = Idwt53PushConstants {
            low_count: low_count as u32,
            high_count: high_count as u32,
            line_count: vertical_width as u32,
            low_line_stride: 1,
            low_sample_stride: vertical_width as u32,
            high_line_stride: 1,
            high_sample_stride: vertical_width as u32,
            output_line_stride: 1,
            output_sample_stride: vertical_width as u32,
        };
        let low_vertical_buffer =
            device.create_host_buffer(bytes_len::<i32>(low_vertical.len())?)?;
        let high_vertical_buffer =
            device.create_host_buffer(bytes_len::<i32>(high_vertical.len())?)?;
        let column_output_buffer =
            device.create_host_buffer(bytes_len::<i32>(column_seed.len())?)?;
        low_vertical_buffer.write_slice(&low_vertical)?;
        high_vertical_buffer.write_slice(&high_vertical)?;
        column_output_buffer.write_slice(&column_seed)?;
        dispatch_storage_shader(
            device,
            queue_family_index,
            &spirv,
            &[
                low_vertical_buffer.descriptor_info(),
                high_vertical_buffer.descriptor_info(),
                column_output_buffer.descriptor_info(),
            ],
            bytes_of(&column_push),
            (output_stride as u32).div_ceil(16),
            (vertical_width as u32).div_ceil(16),
        )?;

        let column_output = column_output_buffer.read_vec::<i32>(column_seed.len())?;
        if column_output != column_expected {
            return Err(VulkanError::new(format!(
                "idwt53 column smoke output mismatch: got {column_output:?}, expected {column_expected:?}"
            )));
        }

        let ll = vec![4i32, -3, 12, 1, 5, 9];
        let hl = vec![7i32, -5, 2, 3];
        let lh = vec![6i32, 8, -4, 0, 11, -7];
        let hh = vec![1i32, -2, 4, -3];
        let expected_level = inverse_5_3_level(
            Band2D {
                width: 3,
                height: 2,
                samples: &ll,
            },
            Band2D {
                width: 2,
                height: 2,
                samples: &hl,
            },
            Band2D {
                width: 3,
                height: 2,
                samples: &lh,
            },
            Band2D {
                width: 2,
                height: 2,
                samples: &hh,
            },
        )
        .map_err(|err| VulkanError::new(format!("CPU inverse 5/3 level failed: {err}")))?;

        let low_rows_seed = vec![0i32; 3 * 4];
        let high_rows_seed = vec![0i32; 2 * 4];
        let level_seed = vec![0i32; expected_level.samples.len()];
        let ll_buffer = device.create_host_buffer(bytes_len::<i32>(ll.len())?)?;
        let hl_buffer = device.create_host_buffer(bytes_len::<i32>(hl.len())?)?;
        let lh_buffer = device.create_host_buffer(bytes_len::<i32>(lh.len())?)?;
        let hh_buffer = device.create_host_buffer(bytes_len::<i32>(hh.len())?)?;
        let low_rows_buffer = device.create_host_buffer(bytes_len::<i32>(low_rows_seed.len())?)?;
        let high_rows_buffer = device.create_host_buffer(bytes_len::<i32>(high_rows_seed.len())?)?;
        let level_output_buffer = device.create_host_buffer(bytes_len::<i32>(level_seed.len())?)?;
        ll_buffer.write_slice(&ll)?;
        hl_buffer.write_slice(&hl)?;
        lh_buffer.write_slice(&lh)?;
        hh_buffer.write_slice(&hh)?;
        low_rows_buffer.write_slice(&low_rows_seed)?;
        high_rows_buffer.write_slice(&high_rows_seed)?;
        level_output_buffer.write_slice(&level_seed)?;

        let vertical_low_push = Idwt53PushConstants {
            low_count: 2,
            high_count: 2,
            line_count: 3,
            low_line_stride: 1,
            low_sample_stride: 3,
            high_line_stride: 1,
            high_sample_stride: 3,
            output_line_stride: 1,
            output_sample_stride: 3,
        };
        dispatch_storage_shader(
            device,
            queue_family_index,
            &spirv,
            &[
                ll_buffer.descriptor_info(),
                lh_buffer.descriptor_info(),
                low_rows_buffer.descriptor_info(),
            ],
            bytes_of(&vertical_low_push),
            1,
            1,
        )?;

        let vertical_high_push = Idwt53PushConstants {
            low_count: 2,
            high_count: 2,
            line_count: 2,
            low_line_stride: 1,
            low_sample_stride: 2,
            high_line_stride: 1,
            high_sample_stride: 2,
            output_line_stride: 1,
            output_sample_stride: 2,
        };
        dispatch_storage_shader(
            device,
            queue_family_index,
            &spirv,
            &[
                hl_buffer.descriptor_info(),
                hh_buffer.descriptor_info(),
                high_rows_buffer.descriptor_info(),
            ],
            bytes_of(&vertical_high_push),
            1,
            1,
        )?;

        let horizontal_push = Idwt53PushConstants {
            low_count: 3,
            high_count: 2,
            line_count: 4,
            low_line_stride: 3,
            low_sample_stride: 1,
            high_line_stride: 2,
            high_sample_stride: 1,
            output_line_stride: 5,
            output_sample_stride: 1,
        };
        dispatch_storage_shader(
            device,
            queue_family_index,
            &spirv,
            &[
                low_rows_buffer.descriptor_info(),
                high_rows_buffer.descriptor_info(),
                level_output_buffer.descriptor_info(),
            ],
            bytes_of(&horizontal_push),
            1,
            1,
        )?;

        let level_output = level_output_buffer.read_vec::<i32>(level_seed.len())?;
        if level_output != expected_level.samples {
            return Err(VulkanError::new(format!(
                "idwt53 level smoke output mismatch: got {level_output:?}, expected {:?}",
                expected_level.samples
            )));
        }
        device.wait_idle()?;

        Ok(Idwt53SmokeResult {
            device_name: summary.name.clone(),
            queue_family_index,
            row_output,
            column_output,
            level_output,
        })
    })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Idwt53PushConstants {
    low_count: u32,
    high_count: u32,
    line_count: u32,
    low_line_stride: u32,
    low_sample_stride: u32,
    high_line_stride: u32,
    high_sample_stride: u32,
    output_line_stride: u32,
    output_sample_stride: u32,
}

pub fn run_color_smoke(shader_path: &str) -> Result<ColorSmokeResult, VulkanError> {
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let c0 = vec![-1.0f32, 0.0, 1.0, 2048.0];
        let c1 = vec![2.0f32, 3.0, 4.0, -2048.0];
        let c2 = vec![5.0f32, 6.0, 7.0, 123.0];
        let output_max = 4095u32;
        let level_shift = 2048.0f32;
        let none_output = run_color_case(
            device,
            queue_family_index,
            &spirv,
            ColorPushConstants {
                sample_count: c0.len() as u32,
                color_transform: 0,
                output_max: 4095,
                level_shift: 2048.0,
            },
            &c0,
            &c1,
            &c2,
        )?;

        let expected = c0
            .iter()
            .zip(&c1)
            .zip(&c2)
            .flat_map(|((r, g), b)| {
                [
                    (*r + level_shift)
                        .round()
                        .clamp(0.0, output_max as f32) as u32,
                    (*g + level_shift)
                        .round()
                        .clamp(0.0, output_max as f32) as u32,
                    (*b + level_shift)
                        .round()
                        .clamp(0.0, output_max as f32) as u32,
                ]
            })
            .collect::<Vec<_>>();
        if none_output != expected {
            return Err(VulkanError::new(format!(
                "color none smoke output mismatch: got {none_output:?}, expected {expected:?}"
            )));
        }

        let rct_rgb = [
            (10i32, -20i32, 30i32),
            (-5i32, 7i32, -9i32),
            (2048i32, -2048i32, 32i32),
        ];
        let mut rct_y = Vec::new();
        let mut rct_db = Vec::new();
        let mut rct_dr = Vec::new();
        let mut rct_expected = Vec::new();
        for (r, g, b) in rct_rgb {
            let y = (r + 2 * g + b).div_euclid(4);
            let db = b - g;
            let dr = r - g;
            let (out_r, out_g, out_b) = inverse_rct_sample(y, db, dr);
            rct_y.push(y as f32);
            rct_db.push(db as f32);
            rct_dr.push(dr as f32);
            rct_expected.extend([
                level_shift_and_clamp(out_r, 12)
                    .map_err(|err| VulkanError::new(err.message))? as u32,
                level_shift_and_clamp(out_g, 12)
                    .map_err(|err| VulkanError::new(err.message))? as u32,
                level_shift_and_clamp(out_b, 12)
                    .map_err(|err| VulkanError::new(err.message))? as u32,
            ]);
        }
        let rct_output = run_color_case(
            device,
            queue_family_index,
            &spirv,
            ColorPushConstants {
                sample_count: rct_y.len() as u32,
                color_transform: 1,
                output_max: 4095,
                level_shift: 2048.0,
            },
            &rct_y,
            &rct_db,
            &rct_dr,
        )?;
        if rct_output != rct_expected {
            return Err(VulkanError::new(format!(
                "color RCT smoke output mismatch: got {rct_output:?}, expected {rct_expected:?}"
            )));
        }

        let ict_y = vec![100.0f32, -20.0, 2040.0];
        let ict_cb = vec![10.0f32, -12.0, 30.0];
        let ict_cr = vec![-5.0f32, 7.0, -70.0];
        let ict_expected = ict_y
            .iter()
            .zip(&ict_cb)
            .zip(&ict_cr)
            .flat_map(|((y, cb), cr)| {
                let (r, g, b) = inverse_ict_sample(*y, *cb, *cr);
                [
                    level_shift_and_clamp_float(r, 12).unwrap() as u32,
                    level_shift_and_clamp_float(g, 12).unwrap() as u32,
                    level_shift_and_clamp_float(b, 12).unwrap() as u32,
                ]
            })
            .collect::<Vec<_>>();
        let ict_output = run_color_case(
            device,
            queue_family_index,
            &spirv,
            ColorPushConstants {
                sample_count: ict_y.len() as u32,
                color_transform: 2,
                output_max: 4095,
                level_shift: 2048.0,
            },
            &ict_y,
            &ict_cb,
            &ict_cr,
        )?;
        if ict_output != ict_expected {
            return Err(VulkanError::new(format!(
                "color ICT smoke output mismatch: got {ict_output:?}, expected {ict_expected:?}"
            )));
        }
        device.wait_idle()?;

        Ok(ColorSmokeResult {
            device_name: summary.name.clone(),
            queue_family_index,
            none_output,
            rct_output,
            ict_output,
        })
    })
}

fn run_color_case(
    device: &VulkanDevice,
    queue_family_index: u32,
    spirv: &[u32],
    push: ColorPushConstants,
    c0: &[f32],
    c1: &[f32],
    c2: &[f32],
) -> Result<Vec<u32>, VulkanError> {
    let output_seed = vec![0u32; c0.len() * 3];
    let c0_buffer = device.create_host_buffer(bytes_len::<f32>(c0.len())?)?;
    let c1_buffer = device.create_host_buffer(bytes_len::<f32>(c1.len())?)?;
    let c2_buffer = device.create_host_buffer(bytes_len::<f32>(c2.len())?)?;
    let output_buffer = device.create_host_buffer(bytes_len::<u32>(output_seed.len())?)?;
    c0_buffer.write_slice(c0)?;
    c1_buffer.write_slice(c1)?;
    c2_buffer.write_slice(c2)?;
    output_buffer.write_slice(&output_seed)?;
    dispatch_storage_shader(
        device,
        queue_family_index,
        spirv,
        &[
            c0_buffer.descriptor_info(),
            c1_buffer.descriptor_info(),
            c2_buffer.descriptor_info(),
            output_buffer.descriptor_info(),
        ],
        bytes_of(&push),
        push.sample_count.div_ceil(256),
        1,
    )?;
    output_buffer.read_vec::<u32>(output_seed.len())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ColorPushConstants {
    sample_count: u32,
    color_transform: u32,
    output_max: u32,
    level_shift: f32,
}
