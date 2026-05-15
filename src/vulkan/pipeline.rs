#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Idwt53ComponentsResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub planes: Vec<crate::dwt::PlaneI32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Idwt97ComponentsResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub planes: Vec<crate::dwt::PlaneF32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorComponentsResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub samples: Vec<u16>,
}

pub fn run_idwt53_components(
    shader_path: &str,
    components: &[&ComponentSubbandPlanes],
    target_resolution: u8,
) -> Result<Idwt53ComponentsResult, VulkanError> {
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let mut planes = Vec::with_capacity(components.len());
        for component in components {
            planes.push(run_idwt53_component_on_device(
                device,
                queue_family_index,
                &spirv,
                component,
                target_resolution,
            )?);
        }
        device.wait_idle()?;
        Ok(Idwt53ComponentsResult {
            device_name: summary.name.clone(),
            queue_family_index,
            planes,
        })
    })
}

pub fn run_idwt97_components(
    shader_path: &str,
    components: &[&ComponentSubbandPlanes],
    target_resolution: u8,
) -> Result<Idwt97ComponentsResult, VulkanError> {
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let mut planes = Vec::with_capacity(components.len());
        for component in components {
            planes.push(run_idwt97_component_on_device(
                device,
                queue_family_index,
                &spirv,
                component,
                target_resolution,
            )?);
        }
        device.wait_idle()?;
        Ok(Idwt97ComponentsResult {
            device_name: summary.name.clone(),
            queue_family_index,
            planes,
        })
    })
}

fn run_idwt53_component_on_device(
    device: &VulkanDevice,
    queue_family_index: u32,
    spirv: &[u32],
    component: &ComponentSubbandPlanes,
    target_resolution: u8,
) -> Result<crate::dwt::PlaneI32, VulkanError> {
    let ll = component
        .plane(0, SubbandKind::Ll)
        .ok_or_else(|| VulkanError::new("GPU IDWT missing resolution 0 LL plane"))?;
    let mut current_width = ll.width;
    let mut current_height = ll.height;
    let mut current_buffer = create_i32_buffer(device, &ll.coefficients)?;

    if target_resolution == 0 {
        return Ok(crate::dwt::PlaneI32 {
            width: current_width,
            height: current_height,
            samples: current_buffer.read_vec::<i32>(ll.coefficients.len())?,
        });
    }

    for resolution in 1..=target_resolution {
        let hl = component
            .plane(resolution, SubbandKind::Hl)
            .ok_or_else(|| VulkanError::new(format!("GPU IDWT missing r{resolution} HL plane")))?;
        let lh = component
            .plane(resolution, SubbandKind::Lh)
            .ok_or_else(|| VulkanError::new(format!("GPU IDWT missing r{resolution} LH plane")))?;
        let hh = component
            .plane(resolution, SubbandKind::Hh)
            .ok_or_else(|| VulkanError::new(format!("GPU IDWT missing r{resolution} HH plane")))?;
        if lh.width != current_width || hl.height != current_height {
            return Err(VulkanError::new(format!(
                "GPU IDWT r{resolution} low/high dimensions do not match current {}x{}: HL {}x{}, LH {}x{}",
                current_width, current_height, hl.width, hl.height, lh.width, lh.height
            )));
        }
        if hh.width != hl.width || hh.height != lh.height {
            return Err(VulkanError::new(format!(
                "GPU IDWT r{resolution} HH dimensions {}x{} do not match HL/LH high extents {}x{}",
                hh.width, hh.height, hl.width, lh.height
            )));
        }

        let output_width = current_width + hl.width;
        let output_height = current_height + lh.height;
        let low_rows = vec![0i32; checked_plane_len(current_width, output_height)?];
        let high_rows = vec![0i32; checked_plane_len(hl.width, output_height)?];
        let output_seed = vec![0i32; checked_plane_len(output_width, output_height)?];
        let hl_buffer = create_i32_buffer(device, &hl.coefficients)?;
        let lh_buffer = create_i32_buffer(device, &lh.coefficients)?;
        let hh_buffer = create_i32_buffer(device, &hh.coefficients)?;
        let low_rows_buffer = create_i32_buffer(device, &low_rows)?;
        let high_rows_buffer = create_i32_buffer(device, &high_rows)?;
        let output_buffer = create_i32_buffer(device, &output_seed)?;

        dispatch_idwt_lines(
            device,
            queue_family_index,
            spirv,
            &current_buffer,
            &lh_buffer,
            &low_rows_buffer,
            Idwt53PushConstants {
                low_count: current_height,
                high_count: lh.height,
                line_count: current_width,
                low_line_stride: 1,
                low_sample_stride: current_width,
                high_line_stride: 1,
                high_sample_stride: lh.width,
                output_line_stride: 1,
                output_sample_stride: current_width,
            },
        )?;
        dispatch_idwt_lines(
            device,
            queue_family_index,
            spirv,
            &hl_buffer,
            &hh_buffer,
            &high_rows_buffer,
            Idwt53PushConstants {
                low_count: current_height,
                high_count: hh.height,
                line_count: hl.width,
                low_line_stride: 1,
                low_sample_stride: hl.width,
                high_line_stride: 1,
                high_sample_stride: hh.width,
                output_line_stride: 1,
                output_sample_stride: hl.width,
            },
        )?;
        dispatch_idwt_lines(
            device,
            queue_family_index,
            spirv,
            &low_rows_buffer,
            &high_rows_buffer,
            &output_buffer,
            Idwt53PushConstants {
                low_count: current_width,
                high_count: hl.width,
                line_count: output_height,
                low_line_stride: current_width,
                low_sample_stride: 1,
                high_line_stride: hl.width,
                high_sample_stride: 1,
                output_line_stride: output_width,
                output_sample_stride: 1,
            },
        )?;

        current_width = output_width;
        current_height = output_height;
        current_buffer = output_buffer;
    }

    Ok(crate::dwt::PlaneI32 {
        width: current_width,
        height: current_height,
        samples: current_buffer.read_vec::<i32>(checked_plane_len(current_width, current_height)?)?,
    })
}

fn run_idwt97_component_on_device(
    device: &VulkanDevice,
    queue_family_index: u32,
    spirv: &[u32],
    component: &ComponentSubbandPlanes,
    target_resolution: u8,
) -> Result<crate::dwt::PlaneF32, VulkanError> {
    let ll = component
        .plane(0, SubbandKind::Ll)
        .ok_or_else(|| VulkanError::new("GPU IDWT 9/7 missing resolution 0 LL plane"))?;
    let mut current_width = ll.width;
    let mut current_height = ll.height;
    let ll_coefficients = ll
        .coefficients
        .iter()
        .map(|coefficient| *coefficient as f32)
        .collect::<Vec<_>>();
    let mut current_buffer = create_f32_buffer(device, &ll_coefficients)?;

    if target_resolution == 0 {
        return Ok(crate::dwt::PlaneF32 {
            width: current_width,
            height: current_height,
            samples: current_buffer.read_vec::<f32>(ll_coefficients.len())?,
        });
    }

    for resolution in 1..=target_resolution {
        let hl = component.plane(resolution, SubbandKind::Hl).ok_or_else(|| {
            VulkanError::new(format!("GPU IDWT 9/7 missing r{resolution} HL plane"))
        })?;
        let lh = component.plane(resolution, SubbandKind::Lh).ok_or_else(|| {
            VulkanError::new(format!("GPU IDWT 9/7 missing r{resolution} LH plane"))
        })?;
        let hh = component.plane(resolution, SubbandKind::Hh).ok_or_else(|| {
            VulkanError::new(format!("GPU IDWT 9/7 missing r{resolution} HH plane"))
        })?;
        if lh.width != current_width || hl.height != current_height {
            return Err(VulkanError::new(format!(
                "GPU IDWT 9/7 r{resolution} low/high dimensions do not match current {}x{}: HL {}x{}, LH {}x{}",
                current_width, current_height, hl.width, hl.height, lh.width, lh.height
            )));
        }
        if hh.width != hl.width || hh.height != lh.height {
            return Err(VulkanError::new(format!(
                "GPU IDWT 9/7 r{resolution} HH dimensions {}x{} do not match HL/LH high extents {}x{}",
                hh.width, hh.height, hl.width, lh.height
            )));
        }

        let output_width = current_width + hl.width;
        let output_height = current_height + lh.height;
        let low_rows = vec![0.0f32; checked_plane_len(current_width, output_height)?];
        let high_rows = vec![0.0f32; checked_plane_len(hl.width, output_height)?];
        let output_seed = vec![0.0f32; checked_plane_len(output_width, output_height)?];
        let hl_buffer = create_f32_buffer_from_i32(device, &hl.coefficients)?;
        let lh_buffer = create_f32_buffer_from_i32(device, &lh.coefficients)?;
        let hh_buffer = create_f32_buffer_from_i32(device, &hh.coefficients)?;
        let low_rows_buffer = create_f32_buffer(device, &low_rows)?;
        let high_rows_buffer = create_f32_buffer(device, &high_rows)?;
        let output_buffer = create_f32_buffer(device, &output_seed)?;

        dispatch_idwt_lines(
            device,
            queue_family_index,
            spirv,
            &current_buffer,
            &lh_buffer,
            &low_rows_buffer,
            Idwt53PushConstants {
                low_count: current_height,
                high_count: lh.height,
                line_count: current_width,
                low_line_stride: 1,
                low_sample_stride: current_width,
                high_line_stride: 1,
                high_sample_stride: lh.width,
                output_line_stride: 1,
                output_sample_stride: current_width,
            },
        )?;
        dispatch_idwt_lines(
            device,
            queue_family_index,
            spirv,
            &hl_buffer,
            &hh_buffer,
            &high_rows_buffer,
            Idwt53PushConstants {
                low_count: current_height,
                high_count: hh.height,
                line_count: hl.width,
                low_line_stride: 1,
                low_sample_stride: hl.width,
                high_line_stride: 1,
                high_sample_stride: hh.width,
                output_line_stride: 1,
                output_sample_stride: hl.width,
            },
        )?;
        dispatch_idwt_lines(
            device,
            queue_family_index,
            spirv,
            &low_rows_buffer,
            &high_rows_buffer,
            &output_buffer,
            Idwt53PushConstants {
                low_count: current_width,
                high_count: hl.width,
                line_count: output_height,
                low_line_stride: current_width,
                low_sample_stride: 1,
                high_line_stride: hl.width,
                high_sample_stride: 1,
                output_line_stride: output_width,
                output_sample_stride: 1,
            },
        )?;

        current_width = output_width;
        current_height = output_height;
        current_buffer = output_buffer;
    }

    Ok(crate::dwt::PlaneF32 {
        width: current_width,
        height: current_height,
        samples: current_buffer.read_vec::<f32>(checked_plane_len(current_width, current_height)?)?,
    })
}

fn dispatch_idwt_lines(
    device: &VulkanDevice,
    queue_family_index: u32,
    spirv: &[u32],
    low_buffer: &HostBuffer,
    high_buffer: &HostBuffer,
    output_buffer: &HostBuffer,
    push: Idwt53PushConstants,
) -> Result<(), VulkanError> {
    dispatch_storage_shader(
        device,
        queue_family_index,
        spirv,
        &[
            low_buffer.descriptor_info(),
            high_buffer.descriptor_info(),
            output_buffer.descriptor_info(),
        ],
        bytes_of(&push),
        (push.low_count + push.high_count).div_ceil(16),
        push.line_count.div_ceil(16),
    )
}

fn create_i32_buffer(device: &VulkanDevice, values: &[i32]) -> Result<HostBuffer, VulkanError> {
    let buffer = device.create_host_buffer(bytes_len::<i32>(values.len())?)?;
    buffer.write_slice(values)?;
    Ok(buffer)
}

pub fn run_color_components(
    shader_path: &str,
    component_planes: &[crate::dwt::PlaneI32],
    color_transform: u32,
    precision: u8,
) -> Result<ColorComponentsResult, VulkanError> {
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let samples = run_color_components_on_device(
            device,
            queue_family_index,
            &spirv,
            component_planes,
            color_transform,
            precision,
        )?;
        device.wait_idle()?;
        Ok(ColorComponentsResult {
            device_name: summary.name.clone(),
            queue_family_index,
            samples,
        })
    })
}

pub fn run_color_float_components(
    shader_path: &str,
    component_planes: &[crate::dwt::PlaneF32],
    color_transform: u32,
    precision: u8,
) -> Result<ColorComponentsResult, VulkanError> {
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let samples = run_color_float_components_on_device(
            device,
            queue_family_index,
            &spirv,
            component_planes,
            color_transform,
            precision,
        )?;
        device.wait_idle()?;
        Ok(ColorComponentsResult {
            device_name: summary.name.clone(),
            queue_family_index,
            samples,
        })
    })
}

fn run_color_components_on_device(
    device: &VulkanDevice,
    queue_family_index: u32,
    spirv: &[u32],
    component_planes: &[crate::dwt::PlaneI32],
    color_transform: u32,
    precision: u8,
) -> Result<Vec<u16>, VulkanError> {
    if component_planes.len() != 3 {
        return Err(VulkanError::new(format!(
            "GPU color conversion requires 3 component planes, got {}",
            component_planes.len()
        )));
    }
    if precision == 0 || precision > 16 {
        return Err(VulkanError::new(format!(
            "GPU color conversion cannot pack precision {precision}"
        )));
    }
    let width = component_planes[0].width;
    let height = component_planes[0].height;
    let sample_count = checked_plane_len(width, height)?;
    for (index, plane) in component_planes.iter().enumerate() {
        if plane.width != width || plane.height != height || plane.samples.len() != sample_count {
            return Err(VulkanError::new(format!(
                "GPU color component {index} has {}x{} / {} samples, expected {}x{} / {sample_count}",
                plane.width,
                plane.height,
                plane.samples.len(),
                width,
                height
            )));
        }
    }

    let c0 = component_planes[0]
        .samples
        .iter()
        .map(|sample| *sample as f32)
        .collect::<Vec<_>>();
    let c1 = component_planes[1]
        .samples
        .iter()
        .map(|sample| *sample as f32)
        .collect::<Vec<_>>();
    let c2 = component_planes[2]
        .samples
        .iter()
        .map(|sample| *sample as f32)
        .collect::<Vec<_>>();
    run_color_float_slices_on_device(
        device,
        queue_family_index,
        spirv,
        [&c0, &c1, &c2],
        color_transform,
        precision,
    )
}

fn run_color_float_components_on_device(
    device: &VulkanDevice,
    queue_family_index: u32,
    spirv: &[u32],
    component_planes: &[crate::dwt::PlaneF32],
    color_transform: u32,
    precision: u8,
) -> Result<Vec<u16>, VulkanError> {
    if component_planes.len() != 3 {
        return Err(VulkanError::new(format!(
            "GPU color conversion requires 3 float component planes, got {}",
            component_planes.len()
        )));
    }
    if precision == 0 || precision > 16 {
        return Err(VulkanError::new(format!(
            "GPU color conversion cannot pack precision {precision}"
        )));
    }
    let width = component_planes[0].width;
    let height = component_planes[0].height;
    let sample_count = checked_plane_len(width, height)?;
    for (index, plane) in component_planes.iter().enumerate() {
        if plane.width != width || plane.height != height || plane.samples.len() != sample_count {
            return Err(VulkanError::new(format!(
                "GPU color float component {index} has {}x{} / {} samples, expected {}x{} / {sample_count}",
                plane.width,
                plane.height,
                plane.samples.len(),
                width,
                height
            )));
        }
    }

    run_color_float_slices_on_device(
        device,
        queue_family_index,
        spirv,
        [
            &component_planes[0].samples,
            &component_planes[1].samples,
            &component_planes[2].samples,
        ],
        color_transform,
        precision,
    )
}

fn run_color_float_slices_on_device(
    device: &VulkanDevice,
    queue_family_index: u32,
    spirv: &[u32],
    components: [&[f32]; 3],
    color_transform: u32,
    precision: u8,
) -> Result<Vec<u16>, VulkanError> {
    let sample_count = components[0].len();
    for (index, component) in components.iter().enumerate() {
        if component.len() != sample_count {
            return Err(VulkanError::new(format!(
                "GPU color float slice {index} has {} samples, expected {sample_count}",
                component.len()
            )));
        }
    }
    let output_seed = vec![0u32; sample_count * 3];
    let c0_buffer = create_f32_buffer(device, components[0])?;
    let c1_buffer = create_f32_buffer(device, components[1])?;
    let c2_buffer = create_f32_buffer(device, components[2])?;
    let output_buffer = create_u32_buffer(device, &output_seed)?;
    let output_max = (1u32 << precision) - 1;
    let push = ColorPushConstants {
        sample_count: sample_count as u32,
        color_transform,
        output_max,
        level_shift: (1u32 << (precision - 1)) as f32,
    };
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
    let output = output_buffer.read_vec::<u32>(output_seed.len())?;
    output
        .into_iter()
        .map(|sample| {
            u16::try_from(sample).map_err(|_| {
                VulkanError::new(format!("GPU color output sample {sample} exceeds u16"))
            })
        })
        .collect()
}

fn create_f32_buffer(device: &VulkanDevice, values: &[f32]) -> Result<HostBuffer, VulkanError> {
    let buffer = device.create_host_buffer(bytes_len::<f32>(values.len())?)?;
    buffer.write_slice(values)?;
    Ok(buffer)
}

fn create_f32_buffer_from_i32(device: &VulkanDevice, values: &[i32]) -> Result<HostBuffer, VulkanError> {
    let floats = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
    create_f32_buffer(device, &floats)
}

fn create_u32_buffer(device: &VulkanDevice, values: &[u32]) -> Result<HostBuffer, VulkanError> {
    let buffer = device.create_host_buffer(bytes_len::<u32>(values.len())?)?;
    buffer.write_slice(values)?;
    Ok(buffer)
}

fn checked_plane_len(width: u32, height: u32) -> Result<usize, VulkanError> {
    width
        .checked_mul(height)
        .map(|count| count as usize)
        .ok_or_else(|| VulkanError::new(format!("GPU plane dimensions overflow: {width}x{height}")))
}
