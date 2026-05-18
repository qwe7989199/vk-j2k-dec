#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Idwt53ComponentsResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub planes: Vec<crate::dwt::PlaneI32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Idwt53ScatteredComponentsResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub planes: Vec<crate::dwt::PlaneI32>,
    pub upload_ms: f64,
    pub dispatch_ms: f64,
    pub readback_ms: f64,
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

#[derive(Debug, Clone, PartialEq)]
pub struct GpuRgb48ComponentsResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub width: u32,
    pub height: u32,
    pub samples: Vec<u16>,
    pub setup_ms: f64,
    pub reconstruct_ms: f64,
    pub reconstruct_upload_ms: f64,
    pub reconstruct_alloc_ms: f64,
    pub reconstruct_dispatch_ms: f64,
    pub pack_ms: f64,
    pub pack_shader_ms: f64,
    pub pack_copy_ms: f64,
    pub pack_readback_ms: f64,
}

struct GpuPlaneI32Buffer {
    width: u32,
    height: u32,
    buffer: HostBuffer,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct GpuDwtProfile {
    upload_ms: f64,
    alloc_ms: f64,
    dispatch_ms: f64,
}

struct GpuPackedRgb48Result {
    samples: Vec<u16>,
    shader_ms: f64,
    copy_ms: f64,
    readback_ms: f64,
}

pub struct VulkanRgb48Pipeline {
    idwt53_program: StorageComputeProgram,
    color_i32_program: Option<StorageComputeProgram>,
    color_program: StorageComputeProgram,
    context: VulkanComputeContext,
    device_name: String,
    queue_family_index: u32,
    setup_ms: f64,
}

impl VulkanRgb48Pipeline {
    pub fn new(idwt53_shader_path: &str, color_shader_path: &str) -> Result<Self, VulkanError> {
        let setup_start = std::time::Instant::now();
        let idwt53_spirv = read_spirv(idwt53_shader_path)?;
        let color_spirv = read_spirv(color_shader_path)?;
        let color_i32_spirv = resolved_i32_color_shader_path(color_shader_path)
            .map(|path| read_spirv(&path))
            .transpose()?;
        let (context, summary, queue_family_index) = create_compute_context()?;
        let idwt53_program = StorageComputeProgram::new(
            &context.device,
            &idwt53_spirv,
            3,
            std::mem::size_of::<Idwt53PushConstants>() as u32,
        )?;
        let color_program = StorageComputeProgram::new(
            &context.device,
            &color_spirv,
            4,
            std::mem::size_of::<ColorPushConstants>() as u32,
        )?;
        let color_i32_program = color_i32_spirv
            .as_deref()
            .map(|spirv| {
                StorageComputeProgram::new(
                    &context.device,
                    spirv,
                    4,
                    std::mem::size_of::<ColorPushConstants>() as u32,
                )
            })
            .transpose()?;
        Ok(Self {
            idwt53_program,
            color_i32_program,
            color_program,
            context,
            device_name: summary.name,
            queue_family_index,
            setup_ms: setup_start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub fn setup_ms(&self) -> f64 {
        self.setup_ms
    }

    pub fn decode_reversible53_rgb48(
        &self,
        components: &[&ComponentSubbandPlanes],
        target_resolution: u8,
        color_transform: u32,
        precision: u8,
    ) -> Result<GpuRgb48ComponentsResult, VulkanError> {
        if let Some(color_i32_program) = &self.color_i32_program {
            return self.decode_reversible53_rgb48_i32(
                components,
                target_resolution,
                color_i32_program,
                color_transform,
                precision,
            );
        }

        let stage_start = std::time::Instant::now();
        let mut planes = Vec::with_capacity(components.len());
        let dwt_profile = GpuDwtProfile::default();
        for component in components {
            let plane = run_idwt53_component_on_device(
                &self.context.device,
                self.queue_family_index,
                &self.idwt53_program,
                component,
                target_resolution,
            )?;
            planes.push(plane);
        }
        let reconstruct_ms = stage_start.elapsed().as_secs_f64() * 1000.0;

        let width = planes
            .first()
            .map(|plane| plane.width)
            .ok_or_else(|| VulkanError::new("GPU RGB48 pipeline requires at least one plane"))?;
        let height = planes[0].height;
        let stage_start = std::time::Instant::now();
        let samples = run_color_components_on_device_with_program(
            &self.context.device,
            self.queue_family_index,
            &self.color_program,
            &planes,
            color_transform,
            precision,
        )?;
        let pack_ms = stage_start.elapsed().as_secs_f64() * 1000.0;
        self.context.device.wait_idle()?;
        Ok(GpuRgb48ComponentsResult {
            device_name: self.device_name.clone(),
            queue_family_index: self.queue_family_index,
            width,
            height,
            samples,
            setup_ms: 0.0,
            reconstruct_ms,
            reconstruct_upload_ms: dwt_profile.upload_ms,
            reconstruct_alloc_ms: dwt_profile.alloc_ms,
            reconstruct_dispatch_ms: dwt_profile.dispatch_ms,
            pack_ms,
            pack_shader_ms: 0.0,
            pack_copy_ms: 0.0,
            pack_readback_ms: 0.0,
        })
    }

    fn decode_reversible53_rgb48_i32(
        &self,
        components: &[&ComponentSubbandPlanes],
        target_resolution: u8,
        color_i32_program: &StorageComputeProgram,
        color_transform: u32,
        precision: u8,
    ) -> Result<GpuRgb48ComponentsResult, VulkanError> {
        let stage_start = std::time::Instant::now();
        let mut planes = Vec::with_capacity(components.len());
        let mut dwt_profile = GpuDwtProfile::default();
        for component in components {
            planes.push(run_idwt53_component_buffer_on_device(
                &self.context.device,
                self.queue_family_index,
                &self.idwt53_program,
                component,
                target_resolution,
                &mut dwt_profile,
            )?);
        }
        let reconstruct_ms = stage_start.elapsed().as_secs_f64() * 1000.0;

        let width = planes
            .first()
            .map(|plane| plane.width)
            .ok_or_else(|| VulkanError::new("GPU RGB48 pipeline requires at least one plane"))?;
        let height = planes[0].height;
        let stage_start = std::time::Instant::now();
        let packed = run_color_i32_buffers_on_device_with_program(
            &self.context.device,
            self.queue_family_index,
            color_i32_program,
            &planes,
            color_transform,
            precision,
        )?;
        let pack_ms = stage_start.elapsed().as_secs_f64() * 1000.0;
        self.context.device.wait_idle()?;
        Ok(GpuRgb48ComponentsResult {
            device_name: self.device_name.clone(),
            queue_family_index: self.queue_family_index,
            width,
            height,
            samples: packed.samples,
            setup_ms: 0.0,
            reconstruct_ms,
            reconstruct_upload_ms: dwt_profile.upload_ms,
            reconstruct_alloc_ms: dwt_profile.alloc_ms,
            reconstruct_dispatch_ms: dwt_profile.dispatch_ms,
            pack_ms,
            pack_shader_ms: packed.shader_ms,
            pack_copy_ms: packed.copy_ms,
            pack_readback_ms: packed.readback_ms,
        })
    }
}

pub fn run_idwt53_components(
    shader_path: &str,
    components: &[&ComponentSubbandPlanes],
    target_resolution: u8,
) -> Result<Idwt53ComponentsResult, VulkanError> {
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let program = StorageComputeProgram::new(
            device,
            &spirv,
            3,
            std::mem::size_of::<Idwt53PushConstants>() as u32,
        )?;
        let mut planes = Vec::with_capacity(components.len());
        for component in components {
            planes.push(run_idwt53_component_on_device(
                device,
                queue_family_index,
                &program,
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

pub fn run_idwt53_scattered_components(
    shader_path: &str,
    planes: &[crate::gpu::GpuTier1ScatterPlane],
    plane_coefficients: &[i32],
    target_resolution: u8,
) -> Result<Idwt53ScatteredComponentsResult, VulkanError> {
    if plane_coefficients.is_empty() {
        return Err(VulkanError::new(
            "GPU scattered IDWT requires plane coefficients",
        ));
    }
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let program = StorageComputeProgram::new(
            device,
            &spirv,
            3,
            std::mem::size_of::<Idwt53OffsetPushConstants>() as u32,
        )?;
        let mut profile = GpuDwtProfile::default();
        let plane_buffer =
            create_i32_device_buffer(device, queue_family_index, plane_coefficients, &mut profile)?;
        let mut component_ids = planes
            .iter()
            .map(|plane| plane.component)
            .collect::<Vec<_>>();
        component_ids.sort_unstable();
        component_ids.dedup();
        let mut output_planes = Vec::with_capacity(component_ids.len());
        let readback_start = std::time::Instant::now();
        let mut readback_ms = 0.0;
        for component in component_ids {
            let output = run_idwt53_scattered_component_on_device(
                device,
                queue_family_index,
                &program,
                &plane_buffer,
                planes,
                component,
                target_resolution,
                &mut profile,
            )?;
            let output_len = checked_plane_len(output.width, output.height)?;
            let samples =
                read_device_i32_buffer(device, queue_family_index, &output.buffer, output_len)?;
            readback_ms += readback_start.elapsed().as_secs_f64() * 1000.0 - readback_ms;
            output_planes.push(crate::dwt::PlaneI32 {
                width: output.width,
                height: output.height,
                samples,
            });
        }
        device.wait_idle()?;
        Ok(Idwt53ScatteredComponentsResult {
            device_name: summary.name.clone(),
            queue_family_index,
            planes: output_planes,
            upload_ms: profile.upload_ms,
            dispatch_ms: profile.dispatch_ms,
            readback_ms,
        })
    })
}

pub fn run_idwt53_rgb48_components(
    idwt53_shader_path: &str,
    color_shader_path: &str,
    components: &[&ComponentSubbandPlanes],
    target_resolution: u8,
    color_transform: u32,
    precision: u8,
) -> Result<GpuRgb48ComponentsResult, VulkanError> {
    let pipeline = VulkanRgb48Pipeline::new(idwt53_shader_path, color_shader_path)?;
    let mut result = pipeline.decode_reversible53_rgb48(
        components,
        target_resolution,
        color_transform,
        precision,
    )?;
    result.setup_ms = pipeline.setup_ms();
    Ok(result)
}

pub fn run_idwt97_components(
    shader_path: &str,
    components: &[&ComponentSubbandPlanes],
    target_resolution: u8,
) -> Result<Idwt97ComponentsResult, VulkanError> {
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let program = StorageComputeProgram::new(
            device,
            &spirv,
            3,
            std::mem::size_of::<Idwt53PushConstants>() as u32,
        )?;
        let mut planes = Vec::with_capacity(components.len());
        for component in components {
            planes.push(run_idwt97_component_on_device(
                device,
                queue_family_index,
                &program,
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

pub fn run_idwt97_float_components(
    shader_path: &str,
    components: &[&ComponentFloatSubbandPlanes],
    target_resolution: u8,
) -> Result<Idwt97ComponentsResult, VulkanError> {
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let program = StorageComputeProgram::new(
            device,
            &spirv,
            3,
            std::mem::size_of::<Idwt53PushConstants>() as u32,
        )?;
        let mut planes = Vec::with_capacity(components.len());
        for component in components {
            planes.push(run_idwt97_float_component_on_device(
                device,
                queue_family_index,
                &program,
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
    program: &StorageComputeProgram,
    component: &ComponentSubbandPlanes,
    target_resolution: u8,
) -> Result<crate::dwt::PlaneI32, VulkanError> {
    let plane = run_idwt53_component_buffer_on_device(
        device,
        queue_family_index,
        program,
        component,
        target_resolution,
        &mut GpuDwtProfile::default(),
    )?;
    Ok(crate::dwt::PlaneI32 {
        width: plane.width,
        height: plane.height,
        samples: read_device_i32_buffer(
            device,
            queue_family_index,
            &plane.buffer,
            checked_plane_len(plane.width, plane.height)?,
        )?,
    })
}

fn run_idwt53_component_buffer_on_device(
    device: &VulkanDevice,
    queue_family_index: u32,
    program: &StorageComputeProgram,
    component: &ComponentSubbandPlanes,
    target_resolution: u8,
    profile: &mut GpuDwtProfile,
) -> Result<GpuPlaneI32Buffer, VulkanError> {
    let ll = component
        .plane(0, SubbandKind::Ll)
        .ok_or_else(|| VulkanError::new("GPU IDWT missing resolution 0 LL plane"))?;
    let mut current_width = ll.width;
    let mut current_height = ll.height;
    let mut current_buffer =
        create_i32_device_buffer(device, queue_family_index, &ll.coefficients, profile)?;

    if target_resolution == 0 {
        return Ok(GpuPlaneI32Buffer {
            width: current_width,
            height: current_height,
            buffer: current_buffer,
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
        let upload_start = std::time::Instant::now();
        let hl_upload = create_i32_device_upload(device, &hl.coefficients)?;
        let lh_upload = create_i32_device_upload(device, &lh.coefficients)?;
        let hh_upload = create_i32_device_upload(device, &hh.coefficients)?;
        copy_i32_device_uploads(
            device,
            queue_family_index,
            &[&hl_upload, &lh_upload, &hh_upload],
        )?;
        profile.upload_ms += upload_start.elapsed().as_secs_f64() * 1000.0;
        let hl_buffer = hl_upload.buffer;
        let lh_buffer = lh_upload.buffer;
        let hh_buffer = hh_upload.buffer;
        let alloc_start = std::time::Instant::now();
        let low_columns_buffer = create_empty_device_buffer::<i32>(
            device,
            checked_plane_len(output_width, current_height)?,
        )?;
        let high_columns_buffer =
            create_empty_device_buffer::<i32>(device, checked_plane_len(output_width, lh.height)?)?;
        let output_buffer = create_empty_device_buffer::<i32>(
            device,
            checked_plane_len(output_width, output_height)?,
        )?;
        profile.alloc_ms += alloc_start.elapsed().as_secs_f64() * 1000.0;

        let dispatch_start = std::time::Instant::now();
        dispatch_idwt_line_batch(
            device,
            queue_family_index,
            program,
            &[
                IdwtLineDispatch {
                    low_buffer: &current_buffer,
                    high_buffer: &hl_buffer,
                    output_buffer: &low_columns_buffer,
                    push: Idwt53PushConstants {
                        low_count: current_width,
                        high_count: hl.width,
                        line_count: current_height,
                        low_line_stride: current_width,
                        low_sample_stride: 1,
                        high_line_stride: hl.width,
                        high_sample_stride: 1,
                        output_line_stride: output_width,
                        output_sample_stride: 1,
                    },
                },
                IdwtLineDispatch {
                    low_buffer: &lh_buffer,
                    high_buffer: &hh_buffer,
                    output_buffer: &high_columns_buffer,
                    push: Idwt53PushConstants {
                        low_count: current_width,
                        high_count: hh.width,
                        line_count: lh.height,
                        low_line_stride: lh.width,
                        low_sample_stride: 1,
                        high_line_stride: hh.width,
                        high_sample_stride: 1,
                        output_line_stride: output_width,
                        output_sample_stride: 1,
                    },
                },
                IdwtLineDispatch {
                    low_buffer: &low_columns_buffer,
                    high_buffer: &high_columns_buffer,
                    output_buffer: &output_buffer,
                    push: Idwt53PushConstants {
                        low_count: current_height,
                        high_count: lh.height,
                        line_count: output_width,
                        low_line_stride: 1,
                        low_sample_stride: output_width,
                        high_line_stride: 1,
                        high_sample_stride: output_width,
                        output_line_stride: 1,
                        output_sample_stride: output_width,
                    },
                },
            ],
        )?;
        profile.dispatch_ms += dispatch_start.elapsed().as_secs_f64() * 1000.0;

        current_width = output_width;
        current_height = output_height;
        current_buffer = output_buffer;
    }

    Ok(GpuPlaneI32Buffer {
        width: current_width,
        height: current_height,
        buffer: current_buffer,
    })
}

fn run_idwt53_scattered_component_on_device(
    device: &VulkanDevice,
    queue_family_index: u32,
    program: &StorageComputeProgram,
    plane_buffer: &HostBuffer,
    planes: &[crate::gpu::GpuTier1ScatterPlane],
    component: u16,
    target_resolution: u8,
    profile: &mut GpuDwtProfile,
) -> Result<GpuPlaneI32Buffer, VulkanError> {
    let ll = require_scattered_plane(planes, component, 0, SubbandKind::Ll)?;
    let mut current_width = ll.width;
    let mut current_height = ll.height;
    let mut current_buffer: Option<HostBuffer> = None;

    if target_resolution == 0 {
        return Err(VulkanError::new(
            "GPU scattered IDWT currently requires target resolution above 0",
        ));
    }

    for resolution in 1..=target_resolution {
        let hl = require_scattered_plane(planes, component, resolution, SubbandKind::Hl)?;
        let lh = require_scattered_plane(planes, component, resolution, SubbandKind::Lh)?;
        let hh = require_scattered_plane(planes, component, resolution, SubbandKind::Hh)?;
        if lh.width != current_width || hl.height != current_height {
            return Err(VulkanError::new(format!(
                "GPU scattered IDWT r{resolution} low/high dimensions do not match current {}x{}: HL {}x{}, LH {}x{}",
                current_width, current_height, hl.width, hl.height, lh.width, lh.height
            )));
        }
        if hh.width != hl.width || hh.height != lh.height {
            return Err(VulkanError::new(format!(
                "GPU scattered IDWT r{resolution} HH dimensions {}x{} do not match HL/LH high extents {}x{}",
                hh.width, hh.height, hl.width, lh.height
            )));
        }

        let output_width = current_width + hl.width;
        let output_height = current_height + lh.height;
        let alloc_start = std::time::Instant::now();
        let low_columns_buffer = create_empty_device_buffer::<i32>(
            device,
            checked_plane_len(output_width, current_height)?,
        )?;
        let high_columns_buffer =
            create_empty_device_buffer::<i32>(device, checked_plane_len(output_width, lh.height)?)?;
        let output_buffer = create_empty_device_buffer::<i32>(
            device,
            checked_plane_len(output_width, output_height)?,
        )?;
        profile.alloc_ms += alloc_start.elapsed().as_secs_f64() * 1000.0;

        let current_source = current_buffer.as_ref().unwrap_or(plane_buffer);
        let current_offset = current_buffer
            .as_ref()
            .map(|_| 0)
            .unwrap_or(ll.plane_offset);
        let dispatch_start = std::time::Instant::now();
        dispatch_idwt_offset_line_batch(
            device,
            queue_family_index,
            program,
            &[
                IdwtOffsetLineDispatch {
                    low_buffer: current_source,
                    high_buffer: plane_buffer,
                    output_buffer: &low_columns_buffer,
                    push: Idwt53OffsetPushConstants {
                        low_count: current_width,
                        high_count: hl.width,
                        line_count: current_height,
                        low_line_stride: current_width,
                        low_sample_stride: 1,
                        high_line_stride: hl.width,
                        high_sample_stride: 1,
                        output_line_stride: output_width,
                        output_sample_stride: 1,
                        low_offset: current_offset,
                        high_offset: hl.plane_offset,
                        output_offset: 0,
                    },
                },
                IdwtOffsetLineDispatch {
                    low_buffer: plane_buffer,
                    high_buffer: plane_buffer,
                    output_buffer: &high_columns_buffer,
                    push: Idwt53OffsetPushConstants {
                        low_count: current_width,
                        high_count: hh.width,
                        line_count: lh.height,
                        low_line_stride: lh.width,
                        low_sample_stride: 1,
                        high_line_stride: hh.width,
                        high_sample_stride: 1,
                        output_line_stride: output_width,
                        output_sample_stride: 1,
                        low_offset: lh.plane_offset,
                        high_offset: hh.plane_offset,
                        output_offset: 0,
                    },
                },
                IdwtOffsetLineDispatch {
                    low_buffer: &low_columns_buffer,
                    high_buffer: &high_columns_buffer,
                    output_buffer: &output_buffer,
                    push: Idwt53OffsetPushConstants {
                        low_count: current_height,
                        high_count: lh.height,
                        line_count: output_width,
                        low_line_stride: 1,
                        low_sample_stride: output_width,
                        high_line_stride: 1,
                        high_sample_stride: output_width,
                        output_line_stride: 1,
                        output_sample_stride: output_width,
                        low_offset: 0,
                        high_offset: 0,
                        output_offset: 0,
                    },
                },
            ],
        )?;
        profile.dispatch_ms += dispatch_start.elapsed().as_secs_f64() * 1000.0;

        current_width = output_width;
        current_height = output_height;
        current_buffer = Some(output_buffer);
    }

    let buffer = current_buffer.ok_or_else(|| VulkanError::new("GPU scattered IDWT produced no output"))?;
    Ok(GpuPlaneI32Buffer {
        width: current_width,
        height: current_height,
        buffer,
    })
}

fn require_scattered_plane(
    planes: &[crate::gpu::GpuTier1ScatterPlane],
    component: u16,
    resolution: u8,
    subband: SubbandKind,
) -> Result<crate::gpu::GpuTier1ScatterPlane, VulkanError> {
    planes
        .iter()
        .copied()
        .find(|plane| {
            plane.component == component
                && plane.resolution == resolution
                && plane.subband == subband
        })
        .ok_or_else(|| {
            VulkanError::new(format!(
                "GPU scattered IDWT missing C{component} R{resolution} {subband} plane"
            ))
        })
}

fn run_idwt97_component_on_device(
    device: &VulkanDevice,
    queue_family_index: u32,
    program: &StorageComputeProgram,
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
        let hl_buffer = create_f32_buffer_from_i32(device, &hl.coefficients)?;
        let lh_buffer = create_f32_buffer_from_i32(device, &lh.coefficients)?;
        let hh_buffer = create_f32_buffer_from_i32(device, &hh.coefficients)?;
        let low_rows_buffer =
            create_empty_buffer::<f32>(device, checked_plane_len(current_width, output_height)?)?;
        let high_rows_buffer =
            create_empty_buffer::<f32>(device, checked_plane_len(hl.width, output_height)?)?;
        let output_buffer =
            create_empty_buffer::<f32>(device, checked_plane_len(output_width, output_height)?)?;

        dispatch_idwt_lines(
            device,
            queue_family_index,
            program,
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
            program,
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
            program,
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

fn run_idwt97_float_component_on_device(
    device: &VulkanDevice,
    queue_family_index: u32,
    program: &StorageComputeProgram,
    component: &ComponentFloatSubbandPlanes,
    target_resolution: u8,
) -> Result<crate::dwt::PlaneF32, VulkanError> {
    let ll = component
        .plane(0, SubbandKind::Ll)
        .ok_or_else(|| VulkanError::new("GPU IDWT 9/7 missing resolution 0 float LL plane"))?;
    let mut current_width = ll.width;
    let mut current_height = ll.height;
    let mut current_buffer = create_f32_buffer(device, &ll.samples)?;

    if target_resolution == 0 {
        return Ok(crate::dwt::PlaneF32 {
            width: current_width,
            height: current_height,
            samples: current_buffer.read_vec::<f32>(ll.samples.len())?,
        });
    }

    for resolution in 1..=target_resolution {
        let hl = component.plane(resolution, SubbandKind::Hl).ok_or_else(|| {
            VulkanError::new(format!(
                "GPU IDWT 9/7 missing r{resolution} float HL plane"
            ))
        })?;
        let lh = component.plane(resolution, SubbandKind::Lh).ok_or_else(|| {
            VulkanError::new(format!(
                "GPU IDWT 9/7 missing r{resolution} float LH plane"
            ))
        })?;
        let hh = component.plane(resolution, SubbandKind::Hh).ok_or_else(|| {
            VulkanError::new(format!(
                "GPU IDWT 9/7 missing r{resolution} float HH plane"
            ))
        })?;
        if lh.width != current_width || hl.height != current_height {
            return Err(VulkanError::new(format!(
                "GPU IDWT 9/7 r{resolution} float low/high dimensions do not match current {}x{}: HL {}x{}, LH {}x{}",
                current_width, current_height, hl.width, hl.height, lh.width, lh.height
            )));
        }
        if hh.width != hl.width || hh.height != lh.height {
            return Err(VulkanError::new(format!(
                "GPU IDWT 9/7 r{resolution} float HH dimensions {}x{} do not match HL/LH high extents {}x{}",
                hh.width, hh.height, hl.width, lh.height
            )));
        }

        let output_width = current_width + hl.width;
        let output_height = current_height + lh.height;
        let hl_buffer = create_f32_buffer(device, &hl.samples)?;
        let lh_buffer = create_f32_buffer(device, &lh.samples)?;
        let hh_buffer = create_f32_buffer(device, &hh.samples)?;
        let low_rows_buffer =
            create_empty_buffer::<f32>(device, checked_plane_len(current_width, output_height)?)?;
        let high_rows_buffer =
            create_empty_buffer::<f32>(device, checked_plane_len(hl.width, output_height)?)?;
        let output_buffer =
            create_empty_buffer::<f32>(device, checked_plane_len(output_width, output_height)?)?;

        dispatch_idwt_lines(
            device,
            queue_family_index,
            program,
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
            program,
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
            program,
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
    program: &StorageComputeProgram,
    low_buffer: &HostBuffer,
    high_buffer: &HostBuffer,
    output_buffer: &HostBuffer,
    push: Idwt53PushConstants,
) -> Result<(), VulkanError> {
    program.dispatch(
        device,
        queue_family_index,
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

struct IdwtLineDispatch<'a> {
    low_buffer: &'a HostBuffer,
    high_buffer: &'a HostBuffer,
    output_buffer: &'a HostBuffer,
    push: Idwt53PushConstants,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Idwt53OffsetPushConstants {
    low_count: u32,
    high_count: u32,
    line_count: u32,
    low_line_stride: u32,
    low_sample_stride: u32,
    high_line_stride: u32,
    high_sample_stride: u32,
    output_line_stride: u32,
    output_sample_stride: u32,
    low_offset: u32,
    high_offset: u32,
    output_offset: u32,
}

struct IdwtOffsetLineDispatch<'a> {
    low_buffer: &'a HostBuffer,
    high_buffer: &'a HostBuffer,
    output_buffer: &'a HostBuffer,
    push: Idwt53OffsetPushConstants,
}

fn dispatch_idwt_line_batch(
    device: &VulkanDevice,
    queue_family_index: u32,
    program: &StorageComputeProgram,
    dispatches: &[IdwtLineDispatch<'_>],
) -> Result<(), VulkanError> {
    let descriptors = dispatches
        .iter()
        .map(|dispatch| {
            [
                dispatch.low_buffer.descriptor_info(),
                dispatch.high_buffer.descriptor_info(),
                dispatch.output_buffer.descriptor_info(),
            ]
        })
        .collect::<Vec<_>>();
    let pushes = dispatches
        .iter()
        .map(|dispatch| dispatch.push)
        .collect::<Vec<_>>();
    let batches = dispatches
        .iter()
        .enumerate()
        .map(|(index, dispatch)| StorageDispatchBatch {
            descriptors: &descriptors[index],
            push_constants: bytes_of(&pushes[index]),
            groups: [
                (dispatch.push.low_count + dispatch.push.high_count).div_ceil(16),
                dispatch.push.line_count.div_ceil(16),
                1,
            ],
        })
        .collect::<Vec<_>>();
    program.dispatch_many(device, queue_family_index, &batches)
}

fn dispatch_idwt_offset_line_batch(
    device: &VulkanDevice,
    queue_family_index: u32,
    program: &StorageComputeProgram,
    dispatches: &[IdwtOffsetLineDispatch<'_>],
) -> Result<(), VulkanError> {
    let descriptors = dispatches
        .iter()
        .map(|dispatch| {
            [
                dispatch.low_buffer.descriptor_info(),
                dispatch.high_buffer.descriptor_info(),
                dispatch.output_buffer.descriptor_info(),
            ]
        })
        .collect::<Vec<_>>();
    let pushes = dispatches
        .iter()
        .map(|dispatch| dispatch.push)
        .collect::<Vec<_>>();
    let batches = dispatches
        .iter()
        .enumerate()
        .map(|(index, dispatch)| StorageDispatchBatch {
            descriptors: &descriptors[index],
            push_constants: bytes_of(&pushes[index]),
            groups: [
                (dispatch.push.low_count + dispatch.push.high_count).div_ceil(16),
                dispatch.push.line_count.div_ceil(16),
                1,
            ],
        })
        .collect::<Vec<_>>();
    program.dispatch_many(device, queue_family_index, &batches)
}

fn create_i32_upload_buffer(
    device: &VulkanDevice,
    values: &[i32],
) -> Result<HostBuffer, VulkanError> {
    let buffer = device.create_host_cached_buffer(
        bytes_len::<i32>(values.len())?,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    )?;
    buffer.write_slice(values)?;
    Ok(buffer)
}

struct I32DeviceUpload {
    staging: HostBuffer,
    buffer: HostBuffer,
    byte_len: u64,
}

fn create_i32_device_upload(
    device: &VulkanDevice,
    values: &[i32],
) -> Result<I32DeviceUpload, VulkanError> {
    let byte_len = bytes_len::<i32>(values.len())?;
    let staging = create_i32_upload_buffer(device, values)?;
    let buffer = device.create_device_local_buffer(
        byte_len,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    )?;
    Ok(I32DeviceUpload {
        staging,
        buffer,
        byte_len,
    })
}

fn copy_i32_device_uploads(
    device: &VulkanDevice,
    queue_family_index: u32,
    uploads: &[&I32DeviceUpload],
) -> Result<(), VulkanError> {
    let copies = uploads
        .iter()
        .map(|upload| BufferCopy {
            source: &upload.staging,
            destination: &upload.buffer,
            byte_len: upload.byte_len,
        })
        .collect::<Vec<_>>();
    device.copy_buffers(queue_family_index, &copies)
}

fn create_i32_device_buffer(
    device: &VulkanDevice,
    queue_family_index: u32,
    values: &[i32],
    profile: &mut GpuDwtProfile,
) -> Result<HostBuffer, VulkanError> {
    let upload_start = std::time::Instant::now();
    let upload = create_i32_device_upload(device, values)?;
    copy_i32_device_uploads(device, queue_family_index, &[&upload])?;
    profile.upload_ms += upload_start.elapsed().as_secs_f64() * 1000.0;
    Ok(upload.buffer)
}

pub fn run_color_components(
    shader_path: &str,
    component_planes: &[crate::dwt::PlaneI32],
    color_transform: u32,
    precision: u8,
) -> Result<ColorComponentsResult, VulkanError> {
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let program = StorageComputeProgram::new(
            device,
            &spirv,
            4,
            std::mem::size_of::<ColorPushConstants>() as u32,
        )?;
        let samples = run_color_components_on_device_with_program(
            device,
            queue_family_index,
            &program,
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
        let program = StorageComputeProgram::new(
            device,
            &spirv,
            4,
            std::mem::size_of::<ColorPushConstants>() as u32,
        )?;
        let samples = run_color_float_components_on_device_with_program(
            device,
            queue_family_index,
            &program,
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

fn run_color_components_on_device_with_program(
    device: &VulkanDevice,
    queue_family_index: u32,
    program: &StorageComputeProgram,
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
    run_color_float_slices_on_device_with_program(
        device,
        queue_family_index,
        program,
        [&c0, &c1, &c2],
        color_transform,
        precision,
    )
}

fn run_color_float_components_on_device_with_program(
    device: &VulkanDevice,
    queue_family_index: u32,
    program: &StorageComputeProgram,
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

    run_color_float_slices_on_device_with_program(
        device,
        queue_family_index,
        program,
        [
            &component_planes[0].samples,
            &component_planes[1].samples,
            &component_planes[2].samples,
        ],
        color_transform,
        precision,
    )
}

fn run_color_i32_buffers_on_device_with_program(
    device: &VulkanDevice,
    queue_family_index: u32,
    program: &StorageComputeProgram,
    component_planes: &[GpuPlaneI32Buffer],
    color_transform: u32,
    precision: u8,
) -> Result<GpuPackedRgb48Result, VulkanError> {
    if component_planes.len() != 3 {
        return Err(VulkanError::new(format!(
            "GPU integer color conversion requires 3 component planes, got {}",
            component_planes.len()
        )));
    }
    if color_transform > 1 {
        return Err(VulkanError::new(format!(
            "GPU integer color conversion supports only none/RCT, got mode {color_transform}"
        )));
    }
    if precision == 0 || precision > 16 {
        return Err(VulkanError::new(format!(
            "GPU integer color conversion cannot pack precision {precision}"
        )));
    }
    let width = component_planes[0].width;
    let height = component_planes[0].height;
    let sample_count = checked_plane_len(width, height)?;
    for (index, plane) in component_planes.iter().enumerate() {
        if plane.width != width || plane.height != height {
            return Err(VulkanError::new(format!(
                "GPU integer color component {index} has {}x{}, expected {}x{}",
                plane.width, plane.height, width, height
            )));
        }
    }

    let output_count = sample_count * 3;
    let pixel_pair_count = sample_count.div_ceil(2);
    let packed_output_count = pixel_pair_count * 3;
    let output_byte_len = bytes_len::<u32>(packed_output_count)?;
    let output_buffer = device.create_device_local_buffer(
        output_byte_len,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    )?;
    let output_max = (1u32 << precision) - 1;
    let push = ColorPushConstants {
        sample_count: sample_count as u32,
        color_transform,
        output_max,
        level_shift: (1u32 << (precision - 1)) as f32,
    };
    let shader_start = std::time::Instant::now();
    program.dispatch(
        device,
        queue_family_index,
        &[
            component_planes[0].buffer.descriptor_info(),
            component_planes[1].buffer.descriptor_info(),
            component_planes[2].buffer.descriptor_info(),
            output_buffer.descriptor_info(),
        ],
        bytes_of(&push),
        (pixel_pair_count as u32).div_ceil(256),
        1,
    )?;
    let shader_ms = shader_start.elapsed().as_secs_f64() * 1000.0;

    let staging_buffer =
        device.create_host_cached_buffer(output_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
    let copy_start = std::time::Instant::now();
    device.copy_buffer(
        queue_family_index,
        &output_buffer,
        &staging_buffer,
        output_byte_len,
    )?;
    let copy_ms = copy_start.elapsed().as_secs_f64() * 1000.0;

    let readback_start = std::time::Instant::now();
    let samples = staging_buffer.read_vec::<u16>(output_count)?;
    let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;
    Ok(GpuPackedRgb48Result {
        samples,
        shader_ms,
        copy_ms,
        readback_ms,
    })
}

fn run_color_float_slices_on_device_with_program(
    device: &VulkanDevice,
    queue_family_index: u32,
    program: &StorageComputeProgram,
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
    let c0_buffer = create_f32_buffer(device, components[0])?;
    let c1_buffer = create_f32_buffer(device, components[1])?;
    let c2_buffer = create_f32_buffer(device, components[2])?;
    let output_count = sample_count * 3;
    let output_buffer = create_empty_buffer::<u32>(device, output_count)?;
    let output_max = (1u32 << precision) - 1;
    let push = ColorPushConstants {
        sample_count: sample_count as u32,
        color_transform,
        output_max,
        level_shift: (1u32 << (precision - 1)) as f32,
    };
    program.dispatch(
        device,
        queue_family_index,
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
    let output = output_buffer.read_vec::<u32>(output_count)?;
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

fn create_empty_buffer<T>(device: &VulkanDevice, count: usize) -> Result<HostBuffer, VulkanError> {
    device.create_host_buffer(bytes_len::<T>(count)?)
}

fn create_empty_device_buffer<T>(
    device: &VulkanDevice,
    count: usize,
) -> Result<HostBuffer, VulkanError> {
    device.create_device_local_buffer(
        bytes_len::<T>(count)?,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    )
}

fn read_device_i32_buffer(
    device: &VulkanDevice,
    queue_family_index: u32,
    source: &HostBuffer,
    count: usize,
) -> Result<Vec<i32>, VulkanError> {
    let byte_len = bytes_len::<i32>(count)?;
    let staging = device.create_host_cached_buffer(byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
    device.copy_buffer(queue_family_index, source, &staging, byte_len)?;
    staging.read_vec::<i32>(count)
}

fn create_f32_buffer_from_i32(
    device: &VulkanDevice,
    values: &[i32],
) -> Result<HostBuffer, VulkanError> {
    let floats = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
    create_f32_buffer(device, &floats)
}

fn checked_plane_len(width: u32, height: u32) -> Result<usize, VulkanError> {
    width
        .checked_mul(height)
        .map(|count| count as usize)
        .ok_or_else(|| VulkanError::new(format!("GPU plane dimensions overflow: {width}x{height}")))
}

fn resolved_i32_color_shader_path(color_shader_path: &str) -> Option<String> {
    let path = std::path::Path::new(color_shader_path);
    let candidate = path.with_file_name("color_convert_i32.comp.spv");
    candidate
        .is_file()
        .then(|| candidate.to_string_lossy().into_owned())
}
