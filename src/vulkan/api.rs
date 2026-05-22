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

struct VulkanComputeContext {
    device: VulkanDevice,
    instance: VulkanInstance,
    loader: VulkanLoader,
}

fn create_compute_context() -> Result<(VulkanComputeContext, VulkanDeviceSummary, u32), VulkanError>
{
    let loader = VulkanLoader::open()?;
    let instance = loader.create_instance()?;
    let (physical_device, summary) = instance.select_compute_device()?;
    let queue_family_index = summary
        .first_compute_queue_family()
        .ok_or_else(|| VulkanError::new("selected Vulkan device has no compute queue"))?
        .index;
    let device = instance.create_compute_device(physical_device, queue_family_index)?;
    Ok((
        VulkanComputeContext {
            device,
            instance,
            loader,
        },
        summary,
        queue_family_index,
    ))
}

fn with_compute_device<T>(
    run: impl FnOnce(&VulkanDevice, &VulkanDeviceSummary, u32) -> Result<T, VulkanError>,
) -> Result<T, VulkanError> {
    let (context, summary, queue_family_index) = create_compute_context()?;
    let _keep_instance_alive = context.instance.instance;
    let _keep_loader_alive = context.loader.module;
    run(&context.device, &summary, queue_family_index)
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
    let program = StorageComputeProgram::new(
        device,
        spirv,
        descriptors.len() as u32,
        push_constants.len() as u32,
    )?;
    program.dispatch(
        device,
        queue_family_index,
        descriptors,
        push_constants,
        groups_x,
        groups_y,
    )
}

struct StorageDispatchBatch<'a> {
    descriptors: &'a [VkDescriptorBufferInfo],
    push_constants: &'a [u8],
    groups: [u32; 3],
}

const TIER1_CLEANUP_CODEBLOCKS_PER_WORKGROUP: u32 = 1;

fn tier1_cleanup_groups(codeblock_count: usize) -> u32 {
    (codeblock_count as u32).div_ceil(TIER1_CLEANUP_CODEBLOCKS_PER_WORKGROUP)
}

struct StorageComputeProgram {
    descriptor_set_layout: DescriptorSetLayoutHandle,
    pipeline_layout: PipelineLayoutHandle,
    _shader_module: ShaderModuleHandle,
    pipeline: PipelineHandle,
    binding_count: u32,
    push_constant_size: u32,
}

impl StorageComputeProgram {
    fn new(
        device: &VulkanDevice,
        spirv: &[u32],
        binding_count: u32,
        push_constant_size: u32,
    ) -> Result<Self, VulkanError> {
        let descriptor_set_layout = device.create_storage_descriptor_set_layout(binding_count)?;
        let pipeline_layout =
            device.create_pipeline_layout(descriptor_set_layout.handle, push_constant_size)?;
        let shader_module = device.create_shader_module(spirv)?;
        let pipeline = device.create_compute_pipeline(shader_module.handle, pipeline_layout.handle)?;
        Ok(Self {
            descriptor_set_layout,
            pipeline_layout,
            _shader_module: shader_module,
            pipeline,
            binding_count,
            push_constant_size,
        })
    }

    fn dispatch(
        &self,
        device: &VulkanDevice,
        queue_family_index: u32,
        descriptors: &[VkDescriptorBufferInfo],
        push_constants: &[u8],
        groups_x: u32,
        groups_y: u32,
    ) -> Result<(), VulkanError> {
        self.dispatch_many(
            device,
            queue_family_index,
            &[StorageDispatchBatch {
                descriptors,
                push_constants,
                groups: [groups_x, groups_y, 1],
            }],
        )
    }

    fn dispatch_repeated(
        &self,
        device: &VulkanDevice,
        queue_family_index: u32,
        descriptors: &[VkDescriptorBufferInfo],
        push_constants: &[u8],
        groups_x: u32,
        groups_y: u32,
        iterations: usize,
    ) -> Result<(), VulkanError> {
        if iterations == 0 {
            return Ok(());
        }
        if descriptors.len() as u32 != self.binding_count {
            return Err(VulkanError::new(format!(
                "compute program expected {} descriptors, got {}",
                self.binding_count,
                descriptors.len()
            )));
        }
        if push_constants.len() as u32 != self.push_constant_size {
            return Err(VulkanError::new(format!(
                "compute program expected {} push-constant bytes, got {}",
                self.push_constant_size,
                push_constants.len()
            )));
        }

        let descriptor_pool = device.create_descriptor_pool(self.binding_count, 1)?;
        let descriptor_set =
            device.allocate_descriptor_set(descriptor_pool.handle, self.descriptor_set_layout.handle)?;
        device.update_storage_descriptor_set(descriptor_set, descriptors);
        let command_pool = device.create_command_pool(queue_family_index)?;
        let command_buffer = device.allocate_command_buffer(command_pool.handle)?;
        let storage_dispatches = (0..iterations)
            .map(|_| StorageDispatch {
                pipeline: self.pipeline.handle,
                pipeline_layout: self.pipeline_layout.handle,
                descriptor_set,
                push_constants,
                groups: [groups_x, groups_y, 1],
            })
            .collect::<Vec<_>>();
        device.record_storage_dispatches(command_buffer, &storage_dispatches)?;
        device.submit_and_wait(command_buffer)
    }

    fn dispatch_many(
        &self,
        device: &VulkanDevice,
        queue_family_index: u32,
        dispatches: &[StorageDispatchBatch<'_>],
    ) -> Result<(), VulkanError> {
        if dispatches.is_empty() {
            return Ok(());
        }
        for dispatch in dispatches {
            if dispatch.descriptors.len() as u32 != self.binding_count {
                return Err(VulkanError::new(format!(
                    "compute program expected {} descriptors, got {}",
                    self.binding_count,
                    dispatch.descriptors.len()
                )));
            }
            if dispatch.push_constants.len() as u32 != self.push_constant_size {
                return Err(VulkanError::new(format!(
                    "compute program expected {} push-constant bytes, got {}",
                    self.push_constant_size,
                    dispatch.push_constants.len()
                )));
            }
        }

        let descriptor_pool = device.create_descriptor_pool(
            self.binding_count * dispatches.len() as u32,
            dispatches.len() as u32,
        )?;
        let mut descriptor_sets = Vec::with_capacity(dispatches.len());
        for dispatch in dispatches {
            let descriptor_set =
                device.allocate_descriptor_set(descriptor_pool.handle, self.descriptor_set_layout.handle)?;
            device.update_storage_descriptor_set(descriptor_set, dispatch.descriptors);
            descriptor_sets.push(descriptor_set);
        }
        let command_pool = device.create_command_pool(queue_family_index)?;
        let command_buffer = device.allocate_command_buffer(command_pool.handle)?;
        let storage_dispatches = dispatches
            .iter()
            .zip(&descriptor_sets)
            .map(|(dispatch, descriptor_set)| StorageDispatch {
                pipeline: self.pipeline.handle,
                pipeline_layout: self.pipeline_layout.handle,
                descriptor_set: *descriptor_set,
                push_constants: dispatch.push_constants,
                groups: dispatch.groups,
            })
            .collect::<Vec<_>>();
        device.record_storage_dispatches(command_buffer, &storage_dispatches)?;
        device.submit_and_wait(command_buffer)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DequantSmokeResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub input: Vec<i32>,
    pub output: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tier1PlanSmokeResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub output: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tier1MqSmokeResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub output: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tier1CleanupSmokeResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub output: Vec<u32>,
    pub coefficients: Vec<i32>,
    pub upload_ms: f64,
    pub dispatch_ms: f64,
    pub readback_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tier1CleanupBenchResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub iterations: usize,
    pub output: Vec<u32>,
    pub coefficients: Vec<i32>,
    pub setup_ms: f64,
    pub dispatch_total_ms: f64,
    pub dispatch_average_ms: f64,
    pub readback_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tier1ScatterSmokeResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub output: Vec<i32>,
    pub upload_ms: f64,
    pub dispatch_ms: f64,
    pub readback_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tier1CleanupScatterSmokeResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub output: Vec<u32>,
    pub plane_coefficients: Vec<i32>,
    pub upload_ms: f64,
    pub cleanup_dispatch_ms: f64,
    pub scatter_dispatch_ms: f64,
    pub readback_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tier1CleanupScatterBenchResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub iterations: usize,
    pub output: Vec<u32>,
    pub plane_coefficients: Vec<i32>,
    pub setup_ms: f64,
    pub dispatch_total_ms: f64,
    pub dispatch_average_ms: f64,
    pub readback_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tier1CleanupScatterIdwt53SmokeResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub output: Vec<u32>,
    pub planes: Vec<crate::dwt::PlaneI32>,
    pub upload_ms: f64,
    pub cleanup_dispatch_ms: f64,
    pub scatter_dispatch_ms: f64,
    pub idwt_dispatch_ms: f64,
    pub idwt_alloc_ms: f64,
    pub readback_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tier1CleanupScatterRgb48SmokeResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub output: Vec<u32>,
    pub samples: Vec<u16>,
    pub upload_ms: f64,
    pub cleanup_dispatch_ms: f64,
    pub scatter_dispatch_ms: f64,
    pub idwt_dispatch_ms: f64,
    pub idwt_alloc_ms: f64,
    pub color_shader_ms: f64,
    pub color_copy_ms: f64,
    pub color_readback_ms: f64,
    pub summary_readback_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tier1CleanupScatterRgb48BenchResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub iterations: usize,
    pub output: Vec<u32>,
    pub samples: Vec<u16>,
    pub setup_ms: f64,
    pub decode_total_ms: f64,
    pub decode_average_ms: f64,
    pub fused_dispatch_average_ms: f64,
    pub cleanup_dispatch_average_ms: f64,
    pub scatter_dispatch_average_ms: f64,
    pub idwt_dispatch_average_ms: f64,
    pub idwt_alloc_average_ms: f64,
    pub color_shader_average_ms: f64,
    pub color_copy_average_ms: f64,
    pub color_readback_average_ms: f64,
    pub summary_readback_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VulkanTier1Rgb48FrameResult {
    pub samples: Vec<u16>,
    pub upload_ms: f64,
    pub fused_dispatch_ms: f64,
    pub color_copy_ms: f64,
    pub color_readback_ms: f64,
    pub total_ms: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Tier1PlanSmokePushConstants {
    codeblock_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Tier1MqSmokePushConstants {
    codeblock_index: u32,
    bit_count: u32,
    context_index: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Tier1CleanupSmokePushConstants {
    codeblock_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Tier1ScatterSmokePushConstants {
    codeblock_count: u32,
}

pub fn run_tier1_plan_smoke(
    shader_path: &str,
    records: &[crate::gpu::GpuTier1CodeblockRecord],
    compressed_words: &[u32],
) -> Result<Tier1PlanSmokeResult, VulkanError> {
    if records.is_empty() {
        return Err(VulkanError::new(
            "Tier-1 plan smoke requires at least one record",
        ));
    }
    if compressed_words.is_empty() {
        return Err(VulkanError::new(
            "Tier-1 plan smoke requires compressed byte words",
        ));
    }
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let program = StorageComputeProgram::new(
            device,
            &spirv,
            3,
            std::mem::size_of::<Tier1PlanSmokePushConstants>() as u32,
        )?;
        let records_buffer =
            device.create_host_buffer(bytes_len::<crate::gpu::GpuTier1CodeblockRecord>(
                records.len(),
            )?)?;
        records_buffer.write_slice(records)?;
        let compressed_buffer =
            device.create_host_buffer(bytes_len::<u32>(compressed_words.len())?)?;
        compressed_buffer.write_slice(compressed_words)?;
        let output_len = records.len() * 4;
        let output_seed = vec![0u32; output_len];
        let output_buffer = device.create_host_buffer(bytes_len::<u32>(output_len)?)?;
        output_buffer.write_slice(&output_seed)?;
        let push = Tier1PlanSmokePushConstants {
            codeblock_count: records.len() as u32,
        };
        program.dispatch(
            device,
            queue_family_index,
            &[
                records_buffer.descriptor_info(),
                compressed_buffer.descriptor_info(),
                output_buffer.descriptor_info(),
            ],
            bytes_of(&push),
            (records.len() as u32).div_ceil(256),
            1,
        )?;
        let output = output_buffer.read_vec::<u32>(output_len)?;
        device.wait_idle()?;
        Ok(Tier1PlanSmokeResult {
            device_name: summary.name.clone(),
            queue_family_index,
            output,
        })
    })
}

pub fn run_tier1_mq_smoke(
    shader_path: &str,
    records: &[crate::gpu::GpuTier1CodeblockRecord],
    compressed_words: &[u32],
    codeblock_index: u32,
    bit_count: u32,
    context_index: u32,
) -> Result<Tier1MqSmokeResult, VulkanError> {
    if records.is_empty() {
        return Err(VulkanError::new(
            "Tier-1 MQ smoke requires at least one record",
        ));
    }
    if codeblock_index as usize >= records.len() {
        return Err(VulkanError::new(format!(
            "Tier-1 MQ smoke codeblock index {codeblock_index} outside {} records",
            records.len()
        )));
    }
    if compressed_words.is_empty() {
        return Err(VulkanError::new(
            "Tier-1 MQ smoke requires compressed byte words",
        ));
    }
    if bit_count == 0 || bit_count > 32 {
        return Err(VulkanError::new(format!(
            "Tier-1 MQ smoke bit count must be 1..32, got {bit_count}"
        )));
    }
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let program = StorageComputeProgram::new(
            device,
            &spirv,
            3,
            std::mem::size_of::<Tier1MqSmokePushConstants>() as u32,
        )?;
        let records_buffer =
            device.create_host_buffer(bytes_len::<crate::gpu::GpuTier1CodeblockRecord>(
                records.len(),
            )?)?;
        records_buffer.write_slice(records)?;
        let compressed_buffer =
            device.create_host_buffer(bytes_len::<u32>(compressed_words.len())?)?;
        compressed_buffer.write_slice(compressed_words)?;
        let output_len = 6usize;
        let output_seed = vec![0u32; output_len];
        let output_buffer = device.create_host_buffer(bytes_len::<u32>(output_len)?)?;
        output_buffer.write_slice(&output_seed)?;
        let push = Tier1MqSmokePushConstants {
            codeblock_index,
            bit_count,
            context_index,
        };
        program.dispatch(
            device,
            queue_family_index,
            &[
                records_buffer.descriptor_info(),
                compressed_buffer.descriptor_info(),
                output_buffer.descriptor_info(),
            ],
            bytes_of(&push),
            1,
            1,
        )?;
        let output = output_buffer.read_vec::<u32>(output_len)?;
        device.wait_idle()?;
        Ok(Tier1MqSmokeResult {
            device_name: summary.name.clone(),
            queue_family_index,
            output,
        })
    })
}

pub fn run_tier1_cleanup_smoke(
    shader_path: &str,
    records: &[crate::gpu::GpuTier1CodeblockRecord],
    compressed_words: &[u32],
) -> Result<Tier1CleanupSmokeResult, VulkanError> {
    validate_tier1_cleanup_inputs(records, compressed_words)?;
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let program = StorageComputeProgram::new(
            device,
            &spirv,
            4,
            std::mem::size_of::<Tier1CleanupSmokePushConstants>() as u32,
        )?;
        let upload_start = std::time::Instant::now();
        let records_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            records,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let compressed_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            compressed_words,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;

        let output_len = records.len() * 9;
        let output_byte_len = bytes_len::<u32>(output_len)?;
        let output_buffer = device.create_device_local_buffer(
            output_byte_len,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        )?;
        let coefficient_count = records
            .iter()
            .map(|record| record.coefficient_offset + record.coefficient_count)
            .max()
            .unwrap_or(0) as usize;
        let coefficient_byte_len = bytes_len::<i32>(coefficient_count)?;
        let coefficient_buffer = device.create_device_local_buffer(
            coefficient_byte_len,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        )?;
        let push = Tier1CleanupSmokePushConstants {
            codeblock_count: records.len() as u32,

        };
        let dispatch_start = std::time::Instant::now();
        program.dispatch(
            device,
            queue_family_index,
            &[
                records_buffer.descriptor_info(),
                compressed_buffer.descriptor_info(),
                output_buffer.descriptor_info(),
                coefficient_buffer.descriptor_info(),
            ],
            bytes_of(&push),
            tier1_cleanup_groups(records.len()),
            1,
        )?;
        let dispatch_ms = dispatch_start.elapsed().as_secs_f64() * 1000.0;

        let readback_start = std::time::Instant::now();
        let output_staging =
            device.create_host_cached_buffer(output_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
        let coefficient_staging = device
            .create_host_cached_buffer(coefficient_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
        device.copy_buffers(
            queue_family_index,
            &[
                BufferCopy {
                    source: &output_buffer,
                    destination: &output_staging,
                    byte_len: output_byte_len,
                },
                BufferCopy {
                    source: &coefficient_buffer,
                    destination: &coefficient_staging,
                    byte_len: coefficient_byte_len,
                },
            ],
        )?;
        let output = output_staging.read_vec::<u32>(output_len)?;
        let coefficients = coefficient_staging.read_vec::<i32>(coefficient_count)?;
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;
        device.wait_idle()?;
        Ok(Tier1CleanupSmokeResult {
            device_name: summary.name.clone(),
            queue_family_index,
            output,
            coefficients,
            upload_ms,
            dispatch_ms,
            readback_ms,
        })
    })
}

pub fn run_tier1_cleanup_bench(
    shader_path: &str,
    records: &[crate::gpu::GpuTier1CodeblockRecord],
    compressed_words: &[u32],
    iterations: usize,
) -> Result<Tier1CleanupBenchResult, VulkanError> {
    validate_tier1_cleanup_inputs(records, compressed_words)?;
    if iterations == 0 {
        return Err(VulkanError::new(
            "Tier-1 cleanup bench requires at least one iteration",
        ));
    }
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let program = StorageComputeProgram::new(
            device,
            &spirv,
            4,
            std::mem::size_of::<Tier1CleanupSmokePushConstants>() as u32,
        )?;
        let setup_start = std::time::Instant::now();
        let records_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            records,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let compressed_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            compressed_words,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let output_len = records.len() * 9;
        let output_byte_len = bytes_len::<u32>(output_len)?;
        let output_buffer = device.create_device_local_buffer(
            output_byte_len,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        )?;
        let coefficient_count = records
            .iter()
            .map(|record| record.coefficient_offset + record.coefficient_count)
            .max()
            .unwrap_or(0) as usize;
        let coefficient_byte_len = bytes_len::<i32>(coefficient_count)?;
        let coefficient_buffer = device.create_device_local_buffer(
            coefficient_byte_len,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        )?;
        let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;

        let push = Tier1CleanupSmokePushConstants {
            codeblock_count: records.len() as u32,

        };
        let descriptors = [
            records_buffer.descriptor_info(),
            compressed_buffer.descriptor_info(),
            output_buffer.descriptor_info(),
            coefficient_buffer.descriptor_info(),
        ];
        let dispatch_start = std::time::Instant::now();
        program.dispatch_repeated(
            device,
            queue_family_index,
            &descriptors,
            bytes_of(&push),
            tier1_cleanup_groups(records.len()),
            1,
            iterations,
        )?;
        let dispatch_total_ms = dispatch_start.elapsed().as_secs_f64() * 1000.0;

        let readback_start = std::time::Instant::now();
        let output_staging =
            device.create_host_cached_buffer(output_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
        let coefficient_staging = device
            .create_host_cached_buffer(coefficient_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
        device.copy_buffers(
            queue_family_index,
            &[
                BufferCopy {
                    source: &output_buffer,
                    destination: &output_staging,
                    byte_len: output_byte_len,
                },
                BufferCopy {
                    source: &coefficient_buffer,
                    destination: &coefficient_staging,
                    byte_len: coefficient_byte_len,
                },
            ],
        )?;
        let output = output_staging.read_vec::<u32>(output_len)?;
        let coefficients = coefficient_staging.read_vec::<i32>(coefficient_count)?;
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;
        device.wait_idle()?;
        Ok(Tier1CleanupBenchResult {
            device_name: summary.name.clone(),
            queue_family_index,
            iterations,
            output,
            coefficients,
            setup_ms,
            dispatch_total_ms,
            dispatch_average_ms: dispatch_total_ms / iterations as f64,
            readback_ms,
        })
    })
}

#[derive(Debug, Clone)]
pub struct Tier1CleanupPipelineBenchResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub iterations: usize,
    pub pipeline_depth: usize,
    pub setup_ms: f64,
    pub total_pipeline_ms: f64,
    pub average_frame_ms: f64,
    pub last_frame_output: Vec<u32>,
    pub last_frame_coefficients: Vec<i32>,
}

pub fn run_tier1_cleanup_pipeline_bench(
    shader_path: &str,
    records: &[crate::gpu::GpuTier1CodeblockRecord],
    compressed_words: &[u32],
    iterations: usize,
    pipeline_depth: usize,
) -> Result<Tier1CleanupPipelineBenchResult, VulkanError> {
    validate_tier1_cleanup_inputs(records, compressed_words)?;
    if iterations == 0 {
        return Err(VulkanError::new(
            "Tier-1 cleanup pipeline bench requires at least one iteration",
        ));
    }
    if pipeline_depth == 0 {
        return Err(VulkanError::new(
            "Pipeline depth must be at least 1",
        ));
    }
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let program = StorageComputeProgram::new(
            device,
            &spirv,
            4,
            std::mem::size_of::<Tier1CleanupSmokePushConstants>() as u32,
        )?;
        let setup_start = std::time::Instant::now();

        let records_byte_len = bytes_len::<crate::gpu::GpuTier1CodeblockRecord>(records.len())?;
        let compressed_byte_len = bytes_len::<u32>(compressed_words.len())?;
        let output_len = records.len() * 9;
        let output_byte_len = bytes_len::<u32>(output_len)?;
        let coefficient_count = records
            .iter()
            .map(|record| record.coefficient_offset + record.coefficient_count)
            .max()
            .unwrap_or(0) as usize;
        let coefficient_byte_len = bytes_len::<i32>(coefficient_count)?;

        struct PipelineSlot {
            records_staging: HostBuffer,
            records_device: HostBuffer,
            compressed_staging: HostBuffer,
            compressed_device: HostBuffer,
            output_staging: HostBuffer,
            output_device: HostBuffer,
            coefficient_staging: HostBuffer,
            coefficient_device: HostBuffer,
            _descriptor_pool: DescriptorPoolHandle,
            descriptor_set: VkDescriptorSet,
            _command_pool: CommandPoolHandle,
            command_buffer: VkCommandBuffer,
            active_fence: Option<FenceHandle>,
        }

        let mut slots = Vec::with_capacity(pipeline_depth);
        for _ in 0..pipeline_depth {
            let records_staging = device.create_host_cached_buffer(records_byte_len, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)?;
            let records_device = device.create_device_local_buffer(records_byte_len, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;

            let compressed_staging = device.create_host_cached_buffer(compressed_byte_len, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)?;
            let compressed_device = device.create_device_local_buffer(compressed_byte_len, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;

            let output_staging = device.create_host_cached_buffer(output_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
            let output_device = device.create_device_local_buffer(output_byte_len, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)?;

            let coefficient_staging = device.create_host_cached_buffer(coefficient_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
            let coefficient_device = device.create_device_local_buffer(coefficient_byte_len, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)?;

            let descriptor_pool = device.create_descriptor_pool(4, 1)?;
            let descriptor_set = device.allocate_descriptor_set(descriptor_pool.handle, program.descriptor_set_layout.handle)?;

            let descriptors = [
                records_device.descriptor_info(),
                compressed_device.descriptor_info(),
                output_device.descriptor_info(),
                coefficient_device.descriptor_info(),
            ];
            device.update_storage_descriptor_set(descriptor_set, &descriptors);

            let command_pool = device.create_command_pool(queue_family_index)?;
            let command_buffer = device.allocate_command_buffer(command_pool.handle)?;

            slots.push(PipelineSlot {
                records_staging,
                records_device,
                compressed_staging,
                compressed_device,
                output_staging,
                output_device,
                coefficient_staging,
                coefficient_device,
                _descriptor_pool: descriptor_pool,
                descriptor_set,
                _command_pool: command_pool,
                command_buffer,
                active_fence: None,
            });
        }

        let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;

        let push = Tier1CleanupSmokePushConstants {
            codeblock_count: records.len() as u32,
        };
        let push_bytes = bytes_of(&push);
        let groups = [tier1_cleanup_groups(records.len()), 1, 1];

        let mut last_output = Vec::new();
        let mut last_coefficients = Vec::new();

        let pipeline_start = std::time::Instant::now();

        for i in 0..iterations {
            let slot_idx = i % pipeline_depth;

            if i >= pipeline_depth {
                if let Some(fence) = slots[slot_idx].active_fence.take() {
                    vk_check(
                        unsafe { (device.functions.wait_for_fences)(device.device, 1, &fence.handle, 1, u64::MAX) },
                        "vkWaitForFences",
                    )?;
                    if i + 1 == iterations || i + 1 - pipeline_depth == iterations {
                        last_output = slots[slot_idx].output_staging.read_vec::<u32>(output_len)?;
                        last_coefficients = slots[slot_idx].coefficient_staging.read_vec::<i32>(coefficient_count)?;
                    } else {
                        let _ = slots[slot_idx].output_staging.read_vec::<u32>(output_len)?;
                        let _ = slots[slot_idx].coefficient_staging.read_vec::<i32>(coefficient_count)?;
                    }
                }
            }

            let slot = &mut slots[slot_idx];
            slot.records_staging.write_slice(records)?;
            slot.compressed_staging.write_slice(compressed_words)?;

            let begin_info = VkCommandBufferBeginInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: 0,
                p_inheritance_info: ptr::null(),
            };
            vk_check(
                unsafe { (device.functions.begin_command_buffer)(slot.command_buffer, &begin_info) },
                "vkBeginCommandBuffer",
            )?;

            let copy_records = VkBufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: records_byte_len,
            };
            unsafe {
                (device.functions.cmd_copy_buffer)(
                    slot.command_buffer,
                    slot.records_staging.buffer,
                    slot.records_device.buffer,
                    1,
                    &copy_records,
                );
            }
            let copy_compressed = VkBufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: compressed_byte_len,
            };
            unsafe {
                (device.functions.cmd_copy_buffer)(
                    slot.command_buffer,
                    slot.compressed_staging.buffer,
                    slot.compressed_device.buffer,
                    1,
                    &copy_compressed,
                );
            }

            let h2d_barrier = VkMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_TRANSFER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_SHADER_READ_BIT,
            };
            unsafe {
                (device.functions.cmd_pipeline_barrier)(
                    slot.command_buffer,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1,
                    &h2d_barrier,
                    0,
                    ptr::null(),
                    0,
                    ptr::null(),
                );
            }

            unsafe {
                (device.functions.cmd_bind_pipeline)(
                    slot.command_buffer,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    program.pipeline.handle,
                );
                (device.functions.cmd_bind_descriptor_sets)(
                    slot.command_buffer,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    program.pipeline_layout.handle,
                    0,
                    1,
                    &slot.descriptor_set,
                    0,
                    ptr::null(),
                );
                (device.functions.cmd_push_constants)(
                    slot.command_buffer,
                    program.pipeline_layout.handle,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    push_bytes.len() as u32,
                    push_bytes.as_ptr() as *const c_void,
                );
                (device.functions.cmd_dispatch)(
                    slot.command_buffer,
                    groups[0],
                    groups[1],
                    groups[2],
                );
            }

            let d2h_barrier = VkMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_TRANSFER_READ_BIT,
            };
            unsafe {
                (device.functions.cmd_pipeline_barrier)(
                    slot.command_buffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0,
                    1,
                    &d2h_barrier,
                    0,
                    ptr::null(),
                    0,
                    ptr::null(),
                );
            }

            let copy_output = VkBufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: output_byte_len,
            };
            unsafe {
                (device.functions.cmd_copy_buffer)(
                    slot.command_buffer,
                    slot.output_device.buffer,
                    slot.output_staging.buffer,
                    1,
                    &copy_output,
                );
            }
            let copy_coefficients = VkBufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: coefficient_byte_len,
            };
            unsafe {
                (device.functions.cmd_copy_buffer)(
                    slot.command_buffer,
                    slot.coefficient_device.buffer,
                    slot.coefficient_staging.buffer,
                    1,
                    &copy_coefficients,
                );
            }

            vk_check(
                unsafe { (device.functions.end_command_buffer)(slot.command_buffer) },
                "vkEndCommandBuffer",
            )?;

            let fence_info = VkFenceCreateInfo {
                s_type: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
            };
            let mut fence_handle = 0usize;
            vk_check(
                unsafe {
                    (device.functions.create_fence)(device.device, &fence_info, ptr::null(), &mut fence_handle)
                },
                "vkCreateFence",
            )?;
            let fence = FenceHandle {
                device: device.device,
                functions: device.functions,
                handle: fence_handle,
            };

            let submit_info = VkSubmitInfo {
                s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                p_next: ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: ptr::null(),
                p_wait_dst_stage_mask: ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &slot.command_buffer,
                signal_semaphore_count: 0,
                p_signal_semaphores: ptr::null(),
            };
            vk_check(
                unsafe { (device.functions.queue_submit)(device.queue, 1, &submit_info, fence.handle) },
                "vkQueueSubmit",
            )?;

            slot.active_fence = Some(fence);
        }

        for slot in &mut slots {
            if let Some(fence) = slot.active_fence.take() {
                vk_check(
                    unsafe { (device.functions.wait_for_fences)(device.device, 1, &fence.handle, 1, u64::MAX) },
                    "vkWaitForFences",
                )?;
                last_output = slot.output_staging.read_vec::<u32>(output_len)?;
                last_coefficients = slot.coefficient_staging.read_vec::<i32>(coefficient_count)?;
            }
        }

        let total_pipeline_ms = pipeline_start.elapsed().as_secs_f64() * 1000.0;
        device.wait_idle()?;

        Ok(Tier1CleanupPipelineBenchResult {
            device_name: summary.name.clone(),
            queue_family_index,
            iterations,
            pipeline_depth,
            setup_ms,
            total_pipeline_ms,
            average_frame_ms: total_pipeline_ms / iterations as f64,
            last_frame_output: last_output,
            last_frame_coefficients: last_coefficients,
        })
    })
}

fn validate_tier1_cleanup_inputs(
    records: &[crate::gpu::GpuTier1CodeblockRecord],
    compressed_words: &[u32],
) -> Result<(), VulkanError> {
    if records.is_empty() {
        return Err(VulkanError::new(
            "Tier-1 cleanup requires at least one record",
        ));
    }
    if compressed_words.is_empty() {
        return Err(VulkanError::new(
            "Tier-1 cleanup requires compressed byte words",
        ));
    }
    for (index, record) in records.iter().enumerate() {
        if record.coefficient_count > 4096 {
            return Err(VulkanError::new(format!(
                "Tier-1 cleanup supports up to 4096 coefficients, got {} for record {index}",
                record.coefficient_count
            )));
        }
    }
    Ok(())
}

fn validate_tier1_scatter_inputs(
    records: &[crate::gpu::GpuTier1ScatterRecord],
    flat_coefficient_count: usize,
    output_coefficients: usize,
) -> Result<u32, VulkanError> {
    if records.is_empty() {
        return Err(VulkanError::new(
            "Tier-1 scatter requires at least one record",
        ));
    }
    if flat_coefficient_count == 0 {
        return Err(VulkanError::new(
            "Tier-1 scatter requires flat coefficients",
        ));
    }
    if output_coefficients == 0 {
        return Err(VulkanError::new(
            "Tier-1 scatter requires at least one output coefficient",
        ));
    }
    let max_codeblock_coefficients = records
        .iter()
        .map(|record| record.coefficient_count)
        .max()
        .unwrap_or(0);
    if max_codeblock_coefficients == 0 {
        return Err(VulkanError::new(
            "Tier-1 scatter records must contain coefficients",
        ));
    }
    for (index, record) in records.iter().enumerate() {
        let source_end = record
            .coefficient_offset
            .checked_add(record.coefficient_count)
            .ok_or_else(|| VulkanError::new(format!("scatter record {index} source overflows")))?;
        if source_end as usize > flat_coefficient_count {
            return Err(VulkanError::new(format!(
                "scatter record {index} source {}..{} outside {} coefficients",
                record.coefficient_offset, source_end, flat_coefficient_count
            )));
        }
        let destination_end = record
            .plane_offset
            .checked_add(
                record
                    .plane_y0
                    .checked_add(record.height.saturating_sub(1))
                    .and_then(|y| y.checked_mul(record.plane_width))
                    .and_then(|row| row.checked_add(record.plane_x0))
                    .and_then(|offset| offset.checked_add(record.width))
                    .ok_or_else(|| {
                        VulkanError::new(format!("scatter record {index} destination overflows"))
                    })?,
            )
            .ok_or_else(|| {
                VulkanError::new(format!("scatter record {index} destination overflows"))
            })?;
        if destination_end as usize > output_coefficients {
            return Err(VulkanError::new(format!(
                "scatter record {index} destination exceeds {output_coefficients} coefficients"
            )));
        }
    }
    Ok(max_codeblock_coefficients)
}

pub fn run_tier1_scatter_smoke(
    shader_path: &str,
    records: &[crate::gpu::GpuTier1ScatterRecord],
    flat_coefficients: &[i32],
    output_coefficients: usize,
) -> Result<Tier1ScatterSmokeResult, VulkanError> {
    let max_codeblock_coefficients =
        validate_tier1_scatter_inputs(records, flat_coefficients.len(), output_coefficients)?;

    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let program = StorageComputeProgram::new(
            device,
            &spirv,
            3,
            std::mem::size_of::<Tier1ScatterSmokePushConstants>() as u32,
        )?;
        let upload_start = std::time::Instant::now();
        let records_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            records,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let flat_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            flat_coefficients,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let output_seed = vec![0i32; output_coefficients];
        let output_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            &output_seed,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        )?;
        let upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;

        let push = Tier1ScatterSmokePushConstants {
            codeblock_count: records.len() as u32,
        };
        let dispatch_start = std::time::Instant::now();
        program.dispatch(
            device,
            queue_family_index,
            &[
                records_buffer.descriptor_info(),
                flat_buffer.descriptor_info(),
                output_buffer.descriptor_info(),
            ],
            bytes_of(&push),
            max_codeblock_coefficients.div_ceil(256),
            records.len() as u32,
        )?;
        let dispatch_ms = dispatch_start.elapsed().as_secs_f64() * 1000.0;

        let readback_start = std::time::Instant::now();
        let output_byte_len = bytes_len::<i32>(output_coefficients)?;
        let output_staging =
            device.create_host_cached_buffer(output_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
        device.copy_buffer(
            queue_family_index,
            &output_buffer,
            &output_staging,
            output_byte_len,
        )?;
        let output = output_staging.read_vec::<i32>(output_coefficients)?;
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;
        device.wait_idle()?;
        Ok(Tier1ScatterSmokeResult {
            device_name: summary.name.clone(),
            queue_family_index,
            output,
            upload_ms,
            dispatch_ms,
            readback_ms,
        })
    })
}

pub fn run_tier1_cleanup_scatter_smoke(
    cleanup_shader_path: &str,
    scatter_shader_path: &str,
    cleanup_records: &[crate::gpu::GpuTier1CodeblockRecord],
    compressed_words: &[u32],
    scatter_records: &[crate::gpu::GpuTier1ScatterRecord],
    output_coefficients: usize,
) -> Result<Tier1CleanupScatterSmokeResult, VulkanError> {
    validate_tier1_cleanup_inputs(cleanup_records, compressed_words)?;
    let flat_coefficient_count = cleanup_records
        .iter()
        .map(|record| record.coefficient_offset + record.coefficient_count)
        .max()
        .unwrap_or(0) as usize;
    let max_scatter_coefficients =
        validate_tier1_scatter_inputs(scatter_records, flat_coefficient_count, output_coefficients)?;
    let cleanup_spirv = read_spirv(cleanup_shader_path)?;
    let scatter_spirv = read_spirv(scatter_shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let cleanup_program = StorageComputeProgram::new(
            device,
            &cleanup_spirv,
            4,
            std::mem::size_of::<Tier1CleanupSmokePushConstants>() as u32,
        )?;
        let scatter_program = StorageComputeProgram::new(
            device,
            &scatter_spirv,
            3,
            std::mem::size_of::<Tier1ScatterSmokePushConstants>() as u32,
        )?;

        let upload_start = std::time::Instant::now();
        let cleanup_records_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            cleanup_records,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let compressed_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            compressed_words,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let scatter_records_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            scatter_records,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;

        let output_len = cleanup_records.len() * 9;
        let output_byte_len = bytes_len::<u32>(output_len)?;
        let output_buffer = device.create_device_local_buffer(
            output_byte_len,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        )?;
        let flat_coefficient_byte_len = bytes_len::<i32>(flat_coefficient_count)?;
        let flat_coefficient_buffer = device.create_device_local_buffer(
            flat_coefficient_byte_len,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let plane_byte_len = bytes_len::<i32>(output_coefficients)?;
        let plane_buffer = device.create_device_local_buffer(
            plane_byte_len,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        )?;

        let cleanup_push = Tier1CleanupSmokePushConstants {
            codeblock_count: cleanup_records.len() as u32,

        };
        let cleanup_dispatch_start = std::time::Instant::now();
        cleanup_program.dispatch(
            device,
            queue_family_index,
            &[
                cleanup_records_buffer.descriptor_info(),
                compressed_buffer.descriptor_info(),
                output_buffer.descriptor_info(),
                flat_coefficient_buffer.descriptor_info(),
            ],
            bytes_of(&cleanup_push),
            tier1_cleanup_groups(cleanup_records.len()),
            1,
        )?;
        let cleanup_dispatch_ms = cleanup_dispatch_start.elapsed().as_secs_f64() * 1000.0;

        let scatter_push = Tier1ScatterSmokePushConstants {
            codeblock_count: scatter_records.len() as u32,
        };
        let scatter_dispatch_start = std::time::Instant::now();
        scatter_program.dispatch(
            device,
            queue_family_index,
            &[
                scatter_records_buffer.descriptor_info(),
                flat_coefficient_buffer.descriptor_info(),
                plane_buffer.descriptor_info(),
            ],
            bytes_of(&scatter_push),
            max_scatter_coefficients.div_ceil(256),
            scatter_records.len() as u32,
        )?;
        let scatter_dispatch_ms = scatter_dispatch_start.elapsed().as_secs_f64() * 1000.0;

        let readback_start = std::time::Instant::now();
        let output_staging =
            device.create_host_cached_buffer(output_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
        let plane_staging =
            device.create_host_cached_buffer(plane_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
        device.copy_buffers(
            queue_family_index,
            &[
                BufferCopy {
                    source: &output_buffer,
                    destination: &output_staging,
                    byte_len: output_byte_len,
                },
                BufferCopy {
                    source: &plane_buffer,
                    destination: &plane_staging,
                    byte_len: plane_byte_len,
                },
            ],
        )?;
        let output = output_staging.read_vec::<u32>(output_len)?;
        let plane_coefficients = plane_staging.read_vec::<i32>(output_coefficients)?;
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;
        device.wait_idle()?;

        Ok(Tier1CleanupScatterSmokeResult {
            device_name: summary.name.clone(),
            queue_family_index,
            output,
            plane_coefficients,
            upload_ms,
            cleanup_dispatch_ms,
            scatter_dispatch_ms,
            readback_ms,
        })
    })
}

pub fn run_tier1_cleanup_scatter_bench(
    cleanup_shader_path: &str,
    scatter_shader_path: &str,
    cleanup_records: &[crate::gpu::GpuTier1CodeblockRecord],
    compressed_words: &[u32],
    scatter_records: &[crate::gpu::GpuTier1ScatterRecord],
    output_coefficients: usize,
    iterations: usize,
) -> Result<Tier1CleanupScatterBenchResult, VulkanError> {
    validate_tier1_cleanup_inputs(cleanup_records, compressed_words)?;
    if iterations == 0 {
        return Err(VulkanError::new(
            "Tier-1 cleanup+scatter bench requires at least one iteration",
        ));
    }
    let flat_coefficient_count = cleanup_records
        .iter()
        .map(|record| record.coefficient_offset + record.coefficient_count)
        .max()
        .unwrap_or(0) as usize;
    let max_scatter_coefficients =
        validate_tier1_scatter_inputs(scatter_records, flat_coefficient_count, output_coefficients)?;
    let cleanup_spirv = read_spirv(cleanup_shader_path)?;
    let scatter_spirv = read_spirv(scatter_shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let cleanup_program = StorageComputeProgram::new(
            device,
            &cleanup_spirv,
            4,
            std::mem::size_of::<Tier1CleanupSmokePushConstants>() as u32,
        )?;
        let scatter_program = StorageComputeProgram::new(
            device,
            &scatter_spirv,
            3,
            std::mem::size_of::<Tier1ScatterSmokePushConstants>() as u32,
        )?;

        let setup_start = std::time::Instant::now();
        let cleanup_records_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            cleanup_records,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let compressed_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            compressed_words,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let scatter_records_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            scatter_records,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let output_len = cleanup_records.len() * 9;
        let output_byte_len = bytes_len::<u32>(output_len)?;
        let output_buffer = device.create_device_local_buffer(
            output_byte_len,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        )?;
        let flat_coefficient_buffer = device.create_device_local_buffer(
            bytes_len::<i32>(flat_coefficient_count)?,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let plane_byte_len = bytes_len::<i32>(output_coefficients)?;
        let plane_buffer = device.create_device_local_buffer(
            plane_byte_len,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        )?;

        let cleanup_descriptors = [
            cleanup_records_buffer.descriptor_info(),
            compressed_buffer.descriptor_info(),
            output_buffer.descriptor_info(),
            flat_coefficient_buffer.descriptor_info(),
        ];
        let scatter_descriptors = [
            scatter_records_buffer.descriptor_info(),
            flat_coefficient_buffer.descriptor_info(),
            plane_buffer.descriptor_info(),
        ];
        let cleanup_descriptor_pool = device.create_descriptor_pool(4, 1)?;
        let cleanup_descriptor_set = device.allocate_descriptor_set(
            cleanup_descriptor_pool.handle,
            cleanup_program.descriptor_set_layout.handle,
        )?;
        device.update_storage_descriptor_set(cleanup_descriptor_set, &cleanup_descriptors);
        let scatter_descriptor_pool = device.create_descriptor_pool(3, 1)?;
        let scatter_descriptor_set = device.allocate_descriptor_set(
            scatter_descriptor_pool.handle,
            scatter_program.descriptor_set_layout.handle,
        )?;
        device.update_storage_descriptor_set(scatter_descriptor_set, &scatter_descriptors);
        let command_pool = device.create_command_pool(queue_family_index)?;
        let command_buffer = device.allocate_command_buffer(command_pool.handle)?;
        let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;

        let cleanup_push = Tier1CleanupSmokePushConstants {
            codeblock_count: cleanup_records.len() as u32,

        };
        let scatter_push = Tier1ScatterSmokePushConstants {
            codeblock_count: scatter_records.len() as u32,
        };
        let mut dispatches = Vec::with_capacity(iterations * 2);
        for _ in 0..iterations {
            dispatches.push(StorageDispatch {
                pipeline: cleanup_program.pipeline.handle,
                pipeline_layout: cleanup_program.pipeline_layout.handle,
                descriptor_set: cleanup_descriptor_set,
                push_constants: bytes_of(&cleanup_push),
                groups: [tier1_cleanup_groups(cleanup_records.len()), 1, 1],
            });
            dispatches.push(StorageDispatch {
                pipeline: scatter_program.pipeline.handle,
                pipeline_layout: scatter_program.pipeline_layout.handle,
                descriptor_set: scatter_descriptor_set,
                push_constants: bytes_of(&scatter_push),
                groups: [max_scatter_coefficients.div_ceil(256), scatter_records.len() as u32, 1],
            });
        }

        let dispatch_start = std::time::Instant::now();
        device.record_storage_dispatches(command_buffer, &dispatches)?;
        device.submit_and_wait(command_buffer)?;
        let dispatch_total_ms = dispatch_start.elapsed().as_secs_f64() * 1000.0;

        let readback_start = std::time::Instant::now();
        let output_staging =
            device.create_host_cached_buffer(output_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
        let plane_staging =
            device.create_host_cached_buffer(plane_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
        device.copy_buffers(
            queue_family_index,
            &[
                BufferCopy {
                    source: &output_buffer,
                    destination: &output_staging,
                    byte_len: output_byte_len,
                },
                BufferCopy {
                    source: &plane_buffer,
                    destination: &plane_staging,
                    byte_len: plane_byte_len,
                },
            ],
        )?;
        let output = output_staging.read_vec::<u32>(output_len)?;
        let plane_coefficients = plane_staging.read_vec::<i32>(output_coefficients)?;
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;
        device.wait_idle()?;

        Ok(Tier1CleanupScatterBenchResult {
            device_name: summary.name.clone(),
            queue_family_index,
            iterations,
            output,
            plane_coefficients,
            setup_ms,
            dispatch_total_ms,
            dispatch_average_ms: dispatch_total_ms / iterations as f64,
            readback_ms,
        })
    })
}

pub fn run_tier1_cleanup_scatter_idwt53_smoke(
    cleanup_shader_path: &str,
    scatter_shader_path: &str,
    idwt53_shader_path: &str,
    cleanup_records: &[crate::gpu::GpuTier1CodeblockRecord],
    compressed_words: &[u32],
    scatter_records: &[crate::gpu::GpuTier1ScatterRecord],
    scatter_planes: &[crate::gpu::GpuTier1ScatterPlane],
    output_coefficients: usize,
    target_resolution: u8,
) -> Result<Tier1CleanupScatterIdwt53SmokeResult, VulkanError> {
    validate_tier1_cleanup_inputs(cleanup_records, compressed_words)?;
    let flat_coefficient_count = cleanup_records
        .iter()
        .map(|record| record.coefficient_offset + record.coefficient_count)
        .max()
        .unwrap_or(0) as usize;
    let max_scatter_coefficients =
        validate_tier1_scatter_inputs(scatter_records, flat_coefficient_count, output_coefficients)?;
    let cleanup_spirv = read_spirv(cleanup_shader_path)?;
    let scatter_spirv = read_spirv(scatter_shader_path)?;
    let idwt53_spirv = read_spirv(idwt53_shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let cleanup_program = StorageComputeProgram::new(
            device,
            &cleanup_spirv,
            4,
            std::mem::size_of::<Tier1CleanupSmokePushConstants>() as u32,
        )?;
        let scatter_program = StorageComputeProgram::new(
            device,
            &scatter_spirv,
            3,
            std::mem::size_of::<Tier1ScatterSmokePushConstants>() as u32,
        )?;
        let idwt53_program = StorageComputeProgram::new(
            device,
            &idwt53_spirv,
            3,
            std::mem::size_of::<Idwt53OffsetPushConstants>() as u32,
        )?;

        let upload_start = std::time::Instant::now();
        let cleanup_records_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            cleanup_records,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let compressed_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            compressed_words,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let scatter_records_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            scatter_records,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;

        let output_len = cleanup_records.len() * 9;
        let output_byte_len = bytes_len::<u32>(output_len)?;
        let output_buffer = device.create_device_local_buffer(
            output_byte_len,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        )?;
        let flat_coefficient_buffer = device.create_device_local_buffer(
            bytes_len::<i32>(flat_coefficient_count)?,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let plane_buffer = device.create_device_local_buffer(
            bytes_len::<i32>(output_coefficients)?,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;

        let cleanup_push = Tier1CleanupSmokePushConstants {
            codeblock_count: cleanup_records.len() as u32,

        };
        let cleanup_dispatch_start = std::time::Instant::now();
        cleanup_program.dispatch(
            device,
            queue_family_index,
            &[
                cleanup_records_buffer.descriptor_info(),
                compressed_buffer.descriptor_info(),
                output_buffer.descriptor_info(),
                flat_coefficient_buffer.descriptor_info(),
            ],
            bytes_of(&cleanup_push),
            tier1_cleanup_groups(cleanup_records.len()),
            1,
        )?;
        let cleanup_dispatch_ms = cleanup_dispatch_start.elapsed().as_secs_f64() * 1000.0;

        let scatter_push = Tier1ScatterSmokePushConstants {
            codeblock_count: scatter_records.len() as u32,
        };
        let scatter_dispatch_start = std::time::Instant::now();
        scatter_program.dispatch(
            device,
            queue_family_index,
            &[
                scatter_records_buffer.descriptor_info(),
                flat_coefficient_buffer.descriptor_info(),
                plane_buffer.descriptor_info(),
            ],
            bytes_of(&scatter_push),
            max_scatter_coefficients.div_ceil(256),
            scatter_records.len() as u32,
        )?;
        let scatter_dispatch_ms = scatter_dispatch_start.elapsed().as_secs_f64() * 1000.0;

        let mut component_ids = scatter_planes
            .iter()
            .map(|plane| plane.component)
            .collect::<Vec<_>>();
        component_ids.sort_unstable();
        component_ids.dedup();
        let mut dwt_profile = GpuDwtProfile::default();
        let mut output_buffers = Vec::with_capacity(component_ids.len());
        for component in component_ids {
            output_buffers.push(run_idwt53_scattered_component_on_device(
                device,
                queue_family_index,
                &idwt53_program,
                &plane_buffer,
                scatter_planes,
                component,
                target_resolution,
                &mut dwt_profile,
            )?);
        }

        let readback_start = std::time::Instant::now();
        let output_staging =
            device.create_host_cached_buffer(output_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
        let mut output_copies = vec![BufferCopy {
            source: &output_buffer,
            destination: &output_staging,
            byte_len: output_byte_len,
        }];
        let plane_staging = output_buffers
            .iter()
            .map(|plane| {
                device.create_host_cached_buffer(
                    bytes_len::<i32>(checked_plane_len(plane.width, plane.height)?)?,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (plane, staging) in output_buffers.iter().zip(&plane_staging) {
            output_copies.push(BufferCopy {
                source: &plane.buffer,
                destination: staging,
                byte_len: bytes_len::<i32>(checked_plane_len(plane.width, plane.height)?)?,
            });
        }
        device.copy_buffers(queue_family_index, &output_copies)?;
        let output = output_staging.read_vec::<u32>(output_len)?;
        let mut planes = Vec::with_capacity(output_buffers.len());
        for (plane, staging) in output_buffers.iter().zip(&plane_staging) {
            planes.push(crate::dwt::PlaneI32 {
                width: plane.width,
                height: plane.height,
                samples: staging.read_vec::<i32>(checked_plane_len(plane.width, plane.height)?)?,
            });
        }
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;
        device.wait_idle()?;

        Ok(Tier1CleanupScatterIdwt53SmokeResult {
            device_name: summary.name.clone(),
            queue_family_index,
            output,
            planes,
            upload_ms,
            cleanup_dispatch_ms,
            scatter_dispatch_ms,
            idwt_dispatch_ms: dwt_profile.dispatch_ms,
            idwt_alloc_ms: dwt_profile.alloc_ms,
            readback_ms,
        })
    })
}

pub fn run_tier1_cleanup_scatter_rgb48_smoke(
    cleanup_shader_path: &str,
    scatter_shader_path: &str,
    idwt53_shader_path: &str,
    color_shader_path: &str,
    cleanup_records: &[crate::gpu::GpuTier1CodeblockRecord],
    compressed_words: &[u32],
    scatter_records: &[crate::gpu::GpuTier1ScatterRecord],
    scatter_planes: &[crate::gpu::GpuTier1ScatterPlane],
    output_coefficients: usize,
    target_resolution: u8,
    color_transform: u32,
    precision: u8,
) -> Result<Tier1CleanupScatterRgb48SmokeResult, VulkanError> {
    validate_tier1_cleanup_inputs(cleanup_records, compressed_words)?;
    let flat_coefficient_count = cleanup_records
        .iter()
        .map(|record| record.coefficient_offset + record.coefficient_count)
        .max()
        .unwrap_or(0) as usize;
    let max_scatter_coefficients =
        validate_tier1_scatter_inputs(scatter_records, flat_coefficient_count, output_coefficients)?;
    let cleanup_spirv = read_spirv(cleanup_shader_path)?;
    let scatter_spirv = read_spirv(scatter_shader_path)?;
    let idwt53_spirv = read_spirv(idwt53_shader_path)?;
    let color_spirv = read_spirv(color_shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let cleanup_program = StorageComputeProgram::new(
            device,
            &cleanup_spirv,
            4,
            std::mem::size_of::<Tier1CleanupSmokePushConstants>() as u32,
        )?;
        let scatter_program = StorageComputeProgram::new(
            device,
            &scatter_spirv,
            3,
            std::mem::size_of::<Tier1ScatterSmokePushConstants>() as u32,
        )?;
        let idwt53_program = StorageComputeProgram::new(
            device,
            &idwt53_spirv,
            3,
            std::mem::size_of::<Idwt53OffsetPushConstants>() as u32,
        )?;
        let color_program = StorageComputeProgram::new(
            device,
            &color_spirv,
            4,
            std::mem::size_of::<ColorPushConstants>() as u32,
        )?;

        let upload_start = std::time::Instant::now();
        let cleanup_records_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            cleanup_records,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let compressed_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            compressed_words,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let scatter_records_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            scatter_records,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;

        let output_len = cleanup_records.len() * 9;
        let output_byte_len = bytes_len::<u32>(output_len)?;
        let output_buffer = device.create_device_local_buffer(
            output_byte_len,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        )?;
        let flat_coefficient_buffer = device.create_device_local_buffer(
            bytes_len::<i32>(flat_coefficient_count)?,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let plane_buffer = device.create_device_local_buffer(
            bytes_len::<i32>(output_coefficients)?,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;

        let cleanup_push = Tier1CleanupSmokePushConstants {
            codeblock_count: cleanup_records.len() as u32,

        };
        let cleanup_dispatch_start = std::time::Instant::now();
        cleanup_program.dispatch(
            device,
            queue_family_index,
            &[
                cleanup_records_buffer.descriptor_info(),
                compressed_buffer.descriptor_info(),
                output_buffer.descriptor_info(),
                flat_coefficient_buffer.descriptor_info(),
            ],
            bytes_of(&cleanup_push),
            tier1_cleanup_groups(cleanup_records.len()),
            1,
        )?;
        let cleanup_dispatch_ms = cleanup_dispatch_start.elapsed().as_secs_f64() * 1000.0;

        let scatter_push = Tier1ScatterSmokePushConstants {
            codeblock_count: scatter_records.len() as u32,
        };
        let scatter_dispatch_start = std::time::Instant::now();
        scatter_program.dispatch(
            device,
            queue_family_index,
            &[
                scatter_records_buffer.descriptor_info(),
                flat_coefficient_buffer.descriptor_info(),
                plane_buffer.descriptor_info(),
            ],
            bytes_of(&scatter_push),
            max_scatter_coefficients.div_ceil(256),
            scatter_records.len() as u32,
        )?;
        let scatter_dispatch_ms = scatter_dispatch_start.elapsed().as_secs_f64() * 1000.0;

        let mut component_ids = scatter_planes
            .iter()
            .map(|plane| plane.component)
            .collect::<Vec<_>>();
        component_ids.sort_unstable();
        component_ids.dedup();
        let mut dwt_profile = GpuDwtProfile::default();
        let mut output_buffers = Vec::with_capacity(component_ids.len());
        for component in component_ids {
            output_buffers.push(run_idwt53_scattered_component_on_device(
                device,
                queue_family_index,
                &idwt53_program,
                &plane_buffer,
                scatter_planes,
                component,
                target_resolution,
                &mut dwt_profile,
            )?);
        }

        let packed = run_color_i32_buffers_on_device_with_program(
            device,
            queue_family_index,
            &color_program,
            &output_buffers,
            color_transform,
            precision,
        )?;

        let summary_readback_start = std::time::Instant::now();
        let output_staging =
            device.create_host_cached_buffer(output_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
        device.copy_buffer(
            queue_family_index,
            &output_buffer,
            &output_staging,
            output_byte_len,
        )?;
        let output = output_staging.read_vec::<u32>(output_len)?;
        let summary_readback_ms = summary_readback_start.elapsed().as_secs_f64() * 1000.0;
        device.wait_idle()?;

        Ok(Tier1CleanupScatterRgb48SmokeResult {
            device_name: summary.name.clone(),
            queue_family_index,
            output,
            samples: packed.samples,
            upload_ms,
            cleanup_dispatch_ms,
            scatter_dispatch_ms,
            idwt_dispatch_ms: dwt_profile.dispatch_ms,
            idwt_alloc_ms: dwt_profile.alloc_ms,
            color_shader_ms: packed.shader_ms,
            color_copy_ms: packed.copy_ms,
            color_readback_ms: packed.readback_ms,
            summary_readback_ms,
        })
    })
}

struct ReusableScatteredIdwt53Component {
    width: u32,
    height: u32,
    levels: Vec<ReusableScatteredIdwt53Level>,
    descriptor_sets: Vec<VkDescriptorSet>,
    pushes: Vec<Idwt53OffsetPushConstants>,
    groups: Vec<[u32; 3]>,
    _descriptor_pool: DescriptorPoolHandle,
}

struct ReusableScatteredIdwt53Level {
    current_width: u32,
    current_height: u32,
    hl: crate::gpu::GpuTier1ScatterPlane,
    lh: crate::gpu::GpuTier1ScatterPlane,
    hh: crate::gpu::GpuTier1ScatterPlane,
    output_width: u32,
    low_columns_buffer: HostBuffer,
    high_columns_buffer: HostBuffer,
    output_buffer: HostBuffer,
}

impl ReusableScatteredIdwt53Component {
    fn new(
        device: &VulkanDevice,
        program: &StorageComputeProgram,
        plane_buffer: &HostBuffer,
        planes: &[crate::gpu::GpuTier1ScatterPlane],
        component: u16,
        target_resolution: u8,
    ) -> Result<Self, VulkanError> {
        let ll = require_scattered_plane(planes, component, 0, SubbandKind::Ll)?;
        let mut current_width = ll.width;
        let mut current_height = ll.height;
        if target_resolution == 0 {
            return Err(VulkanError::new(
                "GPU scattered IDWT currently requires target resolution above 0",
            ));
        }

        let mut levels = Vec::with_capacity(target_resolution as usize);
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
            levels.push(ReusableScatteredIdwt53Level {
                current_width,
                current_height,
                hl,
                lh,
                hh,
                output_width,
                low_columns_buffer,
                high_columns_buffer,
                output_buffer,
            });
            current_width = output_width;
            current_height = output_height;
        }

        let dispatch_count = levels.len() * 3;
        let descriptor_pool =
            device.create_descriptor_pool(3 * dispatch_count as u32, dispatch_count as u32)?;
        let mut descriptor_sets = Vec::with_capacity(dispatch_count);
        let mut pushes = Vec::with_capacity(dispatch_count);
        let mut groups = Vec::with_capacity(dispatch_count);
        for (index, level) in levels.iter().enumerate() {
            let current_source = if index == 0 {
                plane_buffer
            } else {
                &levels[index - 1].output_buffer
            };
            let current_offset = if index == 0 { ll.plane_offset } else { 0 };
            let level_dispatches = [
                IdwtOffsetLineDispatch {
                    low_buffer: current_source,
                    high_buffer: plane_buffer,
                    output_buffer: &level.low_columns_buffer,
                    push: Idwt53OffsetPushConstants {
                        low_count: level.current_width,
                        high_count: level.hl.width,
                        line_count: level.current_height,
                        low_line_stride: level.current_width,
                        low_sample_stride: 1,
                        high_line_stride: level.hl.width,
                        high_sample_stride: 1,
                        output_line_stride: level.output_width,
                        output_sample_stride: 1,
                        low_offset: current_offset,
                        high_offset: level.hl.plane_offset,
                        output_offset: 0,
                    },
                },
                IdwtOffsetLineDispatch {
                    low_buffer: plane_buffer,
                    high_buffer: plane_buffer,
                    output_buffer: &level.high_columns_buffer,
                    push: Idwt53OffsetPushConstants {
                        low_count: level.current_width,
                        high_count: level.hh.width,
                        line_count: level.lh.height,
                        low_line_stride: level.lh.width,
                        low_sample_stride: 1,
                        high_line_stride: level.hh.width,
                        high_sample_stride: 1,
                        output_line_stride: level.output_width,
                        output_sample_stride: 1,
                        low_offset: level.lh.plane_offset,
                        high_offset: level.hh.plane_offset,
                        output_offset: 0,
                    },
                },
                IdwtOffsetLineDispatch {
                    low_buffer: &level.low_columns_buffer,
                    high_buffer: &level.high_columns_buffer,
                    output_buffer: &level.output_buffer,
                    push: Idwt53OffsetPushConstants {
                        low_count: level.current_height,
                        high_count: level.lh.height,
                        line_count: level.output_width,
                        low_line_stride: 1,
                        low_sample_stride: level.output_width,
                        high_line_stride: 1,
                        high_sample_stride: level.output_width,
                        output_line_stride: 1,
                        output_sample_stride: level.output_width,
                        low_offset: 0,
                        high_offset: 0,
                        output_offset: 0,
                    },
                },
            ];
            for dispatch in level_dispatches {
                let descriptors = [
                    dispatch.low_buffer.descriptor_info(),
                    dispatch.high_buffer.descriptor_info(),
                    dispatch.output_buffer.descriptor_info(),
                ];
                let descriptor_set = device.allocate_descriptor_set(
                    descriptor_pool.handle,
                    program.descriptor_set_layout.handle,
                )?;
                device.update_storage_descriptor_set(descriptor_set, &descriptors);
                descriptor_sets.push(descriptor_set);
                pushes.push(dispatch.push);
                groups.push([
                    (dispatch.push.low_count + dispatch.push.high_count).div_ceil(16),
                    dispatch.push.line_count.div_ceil(16),
                    1,
                ]);
            }
        }
        Ok(Self {
            width: current_width,
            height: current_height,
            levels,
            descriptor_sets,
            pushes,
            groups,
            _descriptor_pool: descriptor_pool,
        })
    }

    fn extend_storage_dispatches<'a>(
        &'a self,
        program: &StorageComputeProgram,
        dispatches: &mut Vec<StorageDispatch<'a>>,
    ) {
        for ((push, descriptor_set), groups) in self
            .pushes
            .iter()
            .zip(&self.descriptor_sets)
            .zip(&self.groups)
        {
            dispatches.push(StorageDispatch {
                pipeline: program.pipeline.handle,
                pipeline_layout: program.pipeline_layout.handle,
                descriptor_set: *descriptor_set,
                push_constants: bytes_of(push),
                groups: *groups,
            });
        }
    }

}

struct ReusableRgb48Packer {
    output_count: usize,
    output_byte_len: u64,
    output_buffer: HostBuffer,
    staging_buffer: HostBuffer,
    descriptor_set: VkDescriptorSet,
    push: ColorPushConstants,
    groups: [u32; 3],
    _descriptor_pool: DescriptorPoolHandle,
}

impl ReusableRgb48Packer {
    fn new(
        device: &VulkanDevice,
        program: &StorageComputeProgram,
        component_planes: &[ReusableScatteredIdwt53Component],
        color_transform: u32,
        precision: u8,
    ) -> Result<Self, VulkanError> {
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
        let staging_buffer =
            device.create_host_cached_buffer(output_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
        let mut descriptors = Vec::with_capacity(4);
        for (index, plane) in component_planes.iter().enumerate() {
            let Some(output) = plane.levels.last().map(|level| &level.output_buffer) else {
                return Err(VulkanError::new(format!(
                    "GPU integer color component {index} has no IDWT output buffer"
                )));
            };
            descriptors.push(output.descriptor_info());
        }
        descriptors.push(output_buffer.descriptor_info());
        let descriptor_pool = device.create_descriptor_pool(4, 1)?;
        let descriptor_set =
            device.allocate_descriptor_set(descriptor_pool.handle, program.descriptor_set_layout.handle)?;
        device.update_storage_descriptor_set(descriptor_set, &descriptors);
        let output_max = (1u32 << precision) - 1;
        let push = ColorPushConstants {
            sample_count: sample_count as u32,
            color_transform,
            output_max,
            level_shift: (1u32 << (precision - 1)) as f32,
            output_shift: u32::from(16 - precision),
        };
        Ok(Self {
            output_count,
            output_byte_len,
            output_buffer,
            staging_buffer,
            descriptor_set,
            push,
            groups: [(pixel_pair_count as u32).div_ceil(256), 1, 1],
            _descriptor_pool: descriptor_pool,
        })
    }

    fn extend_storage_dispatches<'a>(
        &'a self,
        program: &StorageComputeProgram,
        dispatches: &mut Vec<StorageDispatch<'a>>,
    ) {
        dispatches.push(StorageDispatch {
            pipeline: program.pipeline.handle,
            pipeline_layout: program.pipeline_layout.handle,
            descriptor_set: self.descriptor_set,
            push_constants: bytes_of(&self.push),
            groups: self.groups,
        });
    }

    fn readback(
        &self,
        device: &VulkanDevice,
        queue_family_index: u32,
    ) -> Result<GpuPackedRgb48Result, VulkanError> {
        let copy_start = std::time::Instant::now();
        device.copy_buffer(
            queue_family_index,
            &self.output_buffer,
            &self.staging_buffer,
            self.output_byte_len,
        )?;
        let copy_ms = copy_start.elapsed().as_secs_f64() * 1000.0;

        let readback_start = std::time::Instant::now();
        let samples = self.staging_buffer.read_vec::<u16>(self.output_count)?;
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;
        Ok(GpuPackedRgb48Result {
            samples,
            shader_ms: 0.0,
            copy_ms,
            readback_ms,
        })
    }

    fn read_staging(&self) -> Result<GpuPackedRgb48Result, VulkanError> {
        let readback_start = std::time::Instant::now();
        let samples = self.staging_buffer.read_vec::<u16>(self.output_count)?;
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;
        Ok(GpuPackedRgb48Result {
            samples,
            shader_ms: 0.0,
            copy_ms: 0.0,
            readback_ms,
        })
    }
}

pub struct VulkanTier1Rgb48Pipeline {
    device_name: String,
    codeblock_count: usize,
    compressed_word_capacity: usize,
    setup_ms: f64,
    cleanup_records_buffer: HostBuffer,
    compressed_buffer: HostBuffer,
    _scatter_records_buffer: HostBuffer,
    _output_buffer: HostBuffer,
    _flat_coefficient_buffer: HostBuffer,
    _plane_buffer: HostBuffer,
    _cleanup_descriptor_pool: DescriptorPoolHandle,
    _scatter_descriptor_pool: DescriptorPoolHandle,
    _fused_command_pool: CommandPoolHandle,
    fused_command_buffer: VkCommandBuffer,
    rgb48_packer: ReusableRgb48Packer,
    _idwt_components: Vec<ReusableScatteredIdwt53Component>,
    _cleanup_program: StorageComputeProgram,
    _scatter_program: StorageComputeProgram,
    _idwt53_program: StorageComputeProgram,
    _color_program: StorageComputeProgram,
    _context: VulkanComputeContext,
}

impl VulkanTier1Rgb48Pipeline {
    pub fn new(
        cleanup_shader_path: &str,
        scatter_shader_path: &str,
        idwt53_shader_path: &str,
        color_shader_path: &str,
        codeblock_count: usize,
        compressed_word_capacity: usize,
        flat_coefficient_count: usize,
        scatter_records: &[crate::gpu::GpuTier1ScatterRecord],
        scatter_planes: &[crate::gpu::GpuTier1ScatterPlane],
        output_coefficients: usize,
        target_resolution: u8,
        color_transform: u32,
        precision: u8,
    ) -> Result<Self, VulkanError> {
        if codeblock_count == 0 {
            return Err(VulkanError::new(
                "persistent GPU Tier-1 RGB48 pipeline requires at least one codeblock",
            ));
        }
        if compressed_word_capacity == 0 {
            return Err(VulkanError::new(
                "persistent GPU Tier-1 RGB48 pipeline requires compressed word capacity",
            ));
        }
        let max_scatter_coefficients =
            validate_tier1_scatter_inputs(scatter_records, flat_coefficient_count, output_coefficients)?;
        let cleanup_spirv = read_spirv(cleanup_shader_path)?;
        let scatter_spirv = read_spirv(scatter_shader_path)?;
        let idwt53_spirv = read_spirv(idwt53_shader_path)?;
        let color_spirv = read_spirv(color_shader_path)?;
        let (context, summary, queue_family_index) = create_compute_context()?;
        let device = &context.device;
        let setup_start = std::time::Instant::now();
        let cleanup_program = StorageComputeProgram::new(
            device,
            &cleanup_spirv,
            4,
            std::mem::size_of::<Tier1CleanupSmokePushConstants>() as u32,
        )?;
        let scatter_program = StorageComputeProgram::new(
            device,
            &scatter_spirv,
            3,
            std::mem::size_of::<Tier1ScatterSmokePushConstants>() as u32,
        )?;
        let idwt53_program = StorageComputeProgram::new(
            device,
            &idwt53_spirv,
            3,
            std::mem::size_of::<Idwt53OffsetPushConstants>() as u32,
        )?;
        let color_program = StorageComputeProgram::new(
            device,
            &color_spirv,
            4,
            std::mem::size_of::<ColorPushConstants>() as u32,
        )?;

        let records_byte_len = bytes_len::<crate::gpu::GpuTier1CodeblockRecord>(codeblock_count)?;
        let compressed_byte_len = bytes_len::<u32>(compressed_word_capacity)?;
        let cleanup_records_buffer = device.create_host_cached_buffer(
            records_byte_len,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let compressed_buffer = device.create_host_cached_buffer(
            compressed_byte_len,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let scatter_records_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            scatter_records,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let output_len = codeblock_count * 9;
        let output_buffer = device.create_device_local_buffer(
            bytes_len::<u32>(output_len)?,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        )?;
        let flat_coefficient_buffer = device.create_device_local_buffer(
            bytes_len::<i32>(flat_coefficient_count)?,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let plane_buffer = device.create_device_local_buffer(
            bytes_len::<i32>(output_coefficients)?,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;

        let mut component_ids = scatter_planes
            .iter()
            .map(|plane| plane.component)
            .collect::<Vec<_>>();
        component_ids.sort_unstable();
        component_ids.dedup();
        let idwt_components = component_ids
            .iter()
            .map(|component| {
                ReusableScatteredIdwt53Component::new(
                    device,
                    &idwt53_program,
                    &plane_buffer,
                    scatter_planes,
                    *component,
                    target_resolution,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rgb48_packer = ReusableRgb48Packer::new(
            device,
            &color_program,
            &idwt_components,
            color_transform,
            precision,
        )?;

        let cleanup_push = Tier1CleanupSmokePushConstants {
            codeblock_count: codeblock_count as u32,

        };
        let scatter_push = Tier1ScatterSmokePushConstants {
            codeblock_count: scatter_records.len() as u32,
        };
        let cleanup_descriptors = [
            cleanup_records_buffer.descriptor_info(),
            compressed_buffer.descriptor_info(),
            output_buffer.descriptor_info(),
            flat_coefficient_buffer.descriptor_info(),
        ];
        let cleanup_descriptor_pool = device.create_descriptor_pool(4, 1)?;
        let cleanup_descriptor_set = device.allocate_descriptor_set(
            cleanup_descriptor_pool.handle,
            cleanup_program.descriptor_set_layout.handle,
        )?;
        device.update_storage_descriptor_set(cleanup_descriptor_set, &cleanup_descriptors);
        let scatter_descriptors = [
            scatter_records_buffer.descriptor_info(),
            flat_coefficient_buffer.descriptor_info(),
            plane_buffer.descriptor_info(),
        ];
        let scatter_descriptor_pool = device.create_descriptor_pool(3, 1)?;
        let scatter_descriptor_set = device.allocate_descriptor_set(
            scatter_descriptor_pool.handle,
            scatter_program.descriptor_set_layout.handle,
        )?;
        device.update_storage_descriptor_set(scatter_descriptor_set, &scatter_descriptors);

        let fused_command_pool = device.create_command_pool(queue_family_index)?;
        let fused_command_buffer = device.allocate_command_buffer(fused_command_pool.handle)?;
        let mut fused_dispatches = Vec::new();
        fused_dispatches.push(StorageDispatch {
            pipeline: cleanup_program.pipeline.handle,
            pipeline_layout: cleanup_program.pipeline_layout.handle,
            descriptor_set: cleanup_descriptor_set,
            push_constants: bytes_of(&cleanup_push),
            groups: [tier1_cleanup_groups(codeblock_count), 1, 1],
        });
        fused_dispatches.push(StorageDispatch {
            pipeline: scatter_program.pipeline.handle,
            pipeline_layout: scatter_program.pipeline_layout.handle,
            descriptor_set: scatter_descriptor_set,
            push_constants: bytes_of(&scatter_push),
            groups: [
                max_scatter_coefficients.div_ceil(256),
                scatter_records.len() as u32,
                1,
            ],
        });
        for component in &idwt_components {
            component.extend_storage_dispatches(&idwt53_program, &mut fused_dispatches);
        }
        rgb48_packer.extend_storage_dispatches(&color_program, &mut fused_dispatches);
        device.record_storage_dispatches_with_copies(
            fused_command_buffer,
            &fused_dispatches,
            &[BufferCopy {
                source: &rgb48_packer.output_buffer,
                destination: &rgb48_packer.staging_buffer,
                byte_len: rgb48_packer.output_byte_len,
            }],
        )?;

        Ok(Self {
            device_name: summary.name.clone(),
            codeblock_count,
            compressed_word_capacity,
            setup_ms: setup_start.elapsed().as_secs_f64() * 1000.0,
            cleanup_records_buffer,
            compressed_buffer,
            _scatter_records_buffer: scatter_records_buffer,
            _output_buffer: output_buffer,
            _flat_coefficient_buffer: flat_coefficient_buffer,
            _plane_buffer: plane_buffer,
            _cleanup_descriptor_pool: cleanup_descriptor_pool,
            _scatter_descriptor_pool: scatter_descriptor_pool,
            _fused_command_pool: fused_command_pool,
            fused_command_buffer,
            rgb48_packer,
            _idwt_components: idwt_components,
            _cleanup_program: cleanup_program,
            _scatter_program: scatter_program,
            _idwt53_program: idwt53_program,
            _color_program: color_program,
            _context: context,
        })
    }

    pub fn setup_ms(&self) -> f64 {
        self.setup_ms
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn decode_frame(
        &self,
        records: &[crate::gpu::GpuTier1CodeblockRecord],
        compressed_words: &[u32],
    ) -> Result<VulkanTier1Rgb48FrameResult, VulkanError> {
        if records.len() != self.codeblock_count {
            return Err(VulkanError::new(format!(
                "persistent GPU Tier-1 RGB48 pipeline expected {} records, got {}",
                self.codeblock_count,
                records.len()
            )));
        }
        if compressed_words.len() > self.compressed_word_capacity {
            return Err(VulkanError::new(format!(
                "persistent GPU Tier-1 RGB48 compressed words {} exceed capacity {}",
                compressed_words.len(),
                self.compressed_word_capacity
            )));
        }
        let device = &self._context.device;
        let total_start = std::time::Instant::now();
        let upload_start = std::time::Instant::now();
        self.cleanup_records_buffer.write_slice(records)?;
        self.compressed_buffer.write_slice(compressed_words)?;
        let upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;

        let dispatch_start = std::time::Instant::now();
        device.submit_and_wait(self.fused_command_buffer)?;
        let fused_dispatch_ms = dispatch_start.elapsed().as_secs_f64() * 1000.0;
        let packed = self.rgb48_packer.read_staging()?;
        Ok(VulkanTier1Rgb48FrameResult {
            samples: packed.samples,
            upload_ms,
            fused_dispatch_ms,
            color_copy_ms: packed.copy_ms,
            color_readback_ms: packed.readback_ms,
            total_ms: total_start.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

pub fn run_tier1_cleanup_scatter_rgb48_bench(
    cleanup_shader_path: &str,
    scatter_shader_path: &str,
    idwt53_shader_path: &str,
    color_shader_path: &str,
    cleanup_records: &[crate::gpu::GpuTier1CodeblockRecord],
    compressed_words: &[u32],
    scatter_records: &[crate::gpu::GpuTier1ScatterRecord],
    scatter_planes: &[crate::gpu::GpuTier1ScatterPlane],
    output_coefficients: usize,
    target_resolution: u8,
    color_transform: u32,
    precision: u8,
    iterations: usize,
) -> Result<Tier1CleanupScatterRgb48BenchResult, VulkanError> {
    validate_tier1_cleanup_inputs(cleanup_records, compressed_words)?;
    if iterations == 0 {
        return Err(VulkanError::new(
            "Tier-1 cleanup+scatter+RGB48 bench requires at least one iteration",
        ));
    }
    let flat_coefficient_count = cleanup_records
        .iter()
        .map(|record| record.coefficient_offset + record.coefficient_count)
        .max()
        .unwrap_or(0) as usize;
    let max_scatter_coefficients =
        validate_tier1_scatter_inputs(scatter_records, flat_coefficient_count, output_coefficients)?;
    let cleanup_spirv = read_spirv(cleanup_shader_path)?;
    let scatter_spirv = read_spirv(scatter_shader_path)?;
    let idwt53_spirv = read_spirv(idwt53_shader_path)?;
    let color_spirv = read_spirv(color_shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let setup_start = std::time::Instant::now();
        let cleanup_program = StorageComputeProgram::new(
            device,
            &cleanup_spirv,
            4,
            std::mem::size_of::<Tier1CleanupSmokePushConstants>() as u32,
        )?;
        let scatter_program = StorageComputeProgram::new(
            device,
            &scatter_spirv,
            3,
            std::mem::size_of::<Tier1ScatterSmokePushConstants>() as u32,
        )?;
        let idwt53_program = StorageComputeProgram::new(
            device,
            &idwt53_spirv,
            3,
            std::mem::size_of::<Idwt53OffsetPushConstants>() as u32,
        )?;
        let color_program = StorageComputeProgram::new(
            device,
            &color_spirv,
            4,
            std::mem::size_of::<ColorPushConstants>() as u32,
        )?;

        let cleanup_records_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            cleanup_records,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let compressed_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            compressed_words,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let scatter_records_buffer = create_device_buffer_from_slice(
            device,
            queue_family_index,
            scatter_records,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let output_len = cleanup_records.len() * 9;
        let output_byte_len = bytes_len::<u32>(output_len)?;
        let output_buffer = device.create_device_local_buffer(
            output_byte_len,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        )?;
        let flat_coefficient_buffer = device.create_device_local_buffer(
            bytes_len::<i32>(flat_coefficient_count)?,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let plane_buffer = device.create_device_local_buffer(
            bytes_len::<i32>(output_coefficients)?,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )?;
        let mut component_ids = scatter_planes
            .iter()
            .map(|plane| plane.component)
            .collect::<Vec<_>>();
        component_ids.sort_unstable();
        component_ids.dedup();
        let idwt_components = component_ids
            .iter()
            .map(|component| {
                ReusableScatteredIdwt53Component::new(
                    device,
                    &idwt53_program,
                    &plane_buffer,
                    scatter_planes,
                    *component,
                    target_resolution,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rgb48_packer = ReusableRgb48Packer::new(
            device,
            &color_program,
            &idwt_components,
            color_transform,
            precision,
        )?;
        let cleanup_push = Tier1CleanupSmokePushConstants {
            codeblock_count: cleanup_records.len() as u32,

        };
        let scatter_push = Tier1ScatterSmokePushConstants {
            codeblock_count: scatter_records.len() as u32,
        };
        let cleanup_descriptors = [
            cleanup_records_buffer.descriptor_info(),
            compressed_buffer.descriptor_info(),
            output_buffer.descriptor_info(),
            flat_coefficient_buffer.descriptor_info(),
        ];
        let cleanup_descriptor_pool = device.create_descriptor_pool(4, 1)?;
        let cleanup_descriptor_set = device.allocate_descriptor_set(
            cleanup_descriptor_pool.handle,
            cleanup_program.descriptor_set_layout.handle,
        )?;
        device.update_storage_descriptor_set(cleanup_descriptor_set, &cleanup_descriptors);
        let scatter_descriptors = [
            scatter_records_buffer.descriptor_info(),
            flat_coefficient_buffer.descriptor_info(),
            plane_buffer.descriptor_info(),
        ];
        let scatter_descriptor_pool = device.create_descriptor_pool(3, 1)?;
        let scatter_descriptor_set = device.allocate_descriptor_set(
            scatter_descriptor_pool.handle,
            scatter_program.descriptor_set_layout.handle,
        )?;
        device.update_storage_descriptor_set(scatter_descriptor_set, &scatter_descriptors);
        let fused_command_pool = device.create_command_pool(queue_family_index)?;
        let fused_command_buffer = device.allocate_command_buffer(fused_command_pool.handle)?;
        let mut fused_dispatches = Vec::new();
        fused_dispatches.push(StorageDispatch {
            pipeline: cleanup_program.pipeline.handle,
            pipeline_layout: cleanup_program.pipeline_layout.handle,
            descriptor_set: cleanup_descriptor_set,
            push_constants: bytes_of(&cleanup_push),
            groups: [tier1_cleanup_groups(cleanup_records.len()), 1, 1],
        });
        fused_dispatches.push(StorageDispatch {
            pipeline: scatter_program.pipeline.handle,
            pipeline_layout: scatter_program.pipeline_layout.handle,
            descriptor_set: scatter_descriptor_set,
            push_constants: bytes_of(&scatter_push),
            groups: [
                max_scatter_coefficients.div_ceil(256),
                scatter_records.len() as u32,
                1,
            ],
        });
        for component in &idwt_components {
            component.extend_storage_dispatches(&idwt53_program, &mut fused_dispatches);
        }
        rgb48_packer.extend_storage_dispatches(&color_program, &mut fused_dispatches);
        device.record_storage_dispatches(fused_command_buffer, &fused_dispatches)?;
        let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;

        let cleanup_dispatch_ms = 0.0;
        let scatter_dispatch_ms = 0.0;
        let idwt_dispatch_ms = 0.0;
        let idwt_alloc_ms = 0.0;
        let mut color_shader_ms = 0.0;
        let mut color_copy_ms = 0.0;
        let mut color_readback_ms = 0.0;
        let mut fused_dispatch_ms = 0.0;
        let mut samples = Vec::new();
        let decode_start = std::time::Instant::now();
        for _ in 0..iterations {
            let fused_dispatch_start = std::time::Instant::now();
            device.submit_and_wait(fused_command_buffer)?;
            fused_dispatch_ms += fused_dispatch_start.elapsed().as_secs_f64() * 1000.0;
            let packed = rgb48_packer.readback(device, queue_family_index)?;
            color_shader_ms += packed.shader_ms;
            color_copy_ms += packed.copy_ms;
            color_readback_ms += packed.readback_ms;
            samples = packed.samples;
        }
        let decode_total_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        let summary_readback_start = std::time::Instant::now();
        let output_staging =
            device.create_host_cached_buffer(output_byte_len, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
        device.copy_buffer(
            queue_family_index,
            &output_buffer,
            &output_staging,
            output_byte_len,
        )?;
        let output = output_staging.read_vec::<u32>(output_len)?;
        let summary_readback_ms = summary_readback_start.elapsed().as_secs_f64() * 1000.0;
        device.wait_idle()?;

        let iterations_f64 = iterations as f64;
        Ok(Tier1CleanupScatterRgb48BenchResult {
            device_name: summary.name.clone(),
            queue_family_index,
            iterations,
            output,
            samples,
            setup_ms,
            decode_total_ms,
            decode_average_ms: decode_total_ms / iterations_f64,
            fused_dispatch_average_ms: fused_dispatch_ms / iterations_f64,
            cleanup_dispatch_average_ms: cleanup_dispatch_ms / iterations_f64,
            scatter_dispatch_average_ms: scatter_dispatch_ms / iterations_f64,
            idwt_dispatch_average_ms: idwt_dispatch_ms / iterations_f64,
            idwt_alloc_average_ms: idwt_alloc_ms / iterations_f64,
            color_shader_average_ms: color_shader_ms / iterations_f64,
            color_copy_average_ms: color_copy_ms / iterations_f64,
            color_readback_average_ms: color_readback_ms / iterations_f64,
            summary_readback_ms,
        })
    })
}

fn create_device_buffer_from_slice<T: Copy>(
    device: &VulkanDevice,
    queue_family_index: u32,
    values: &[T],
    usage: VkFlags,
) -> Result<HostBuffer, VulkanError> {
    let byte_len = bytes_len::<T>(values.len())?;
    let staging = device.create_host_cached_buffer(byte_len, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)?;
    staging.write_slice(values)?;
    let buffer =
        device.create_device_local_buffer(byte_len, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;
    device.copy_buffer(queue_family_index, &staging, &buffer, byte_len)?;
    Ok(buffer)
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
    pub float_component_output: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Idwt53SmokeResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub row_output: Vec<i32>,
    pub column_output: Vec<i32>,
    pub level_output: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Idwt97SmokeResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub row_output: Vec<f32>,
    pub column_output: Vec<f32>,
    pub level_output: Vec<f32>,
    pub component_output: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrreversibleSmokeResult {
    pub device_name: String,
    pub queue_family_index: u32,
    pub cpu_samples: Vec<u16>,
    pub gpu_samples: Vec<u16>,
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

        let low_columns_seed = vec![0i32; 5 * 2];
        let high_columns_seed = vec![0i32; 5 * 2];
        let level_seed = vec![0i32; expected_level.samples.len()];
        let ll_buffer = device.create_host_buffer(bytes_len::<i32>(ll.len())?)?;
        let hl_buffer = device.create_host_buffer(bytes_len::<i32>(hl.len())?)?;
        let lh_buffer = device.create_host_buffer(bytes_len::<i32>(lh.len())?)?;
        let hh_buffer = device.create_host_buffer(bytes_len::<i32>(hh.len())?)?;
        let low_columns_buffer =
            device.create_host_buffer(bytes_len::<i32>(low_columns_seed.len())?)?;
        let high_columns_buffer =
            device.create_host_buffer(bytes_len::<i32>(high_columns_seed.len())?)?;
        let level_output_buffer = device.create_host_buffer(bytes_len::<i32>(level_seed.len())?)?;
        ll_buffer.write_slice(&ll)?;
        hl_buffer.write_slice(&hl)?;
        lh_buffer.write_slice(&lh)?;
        hh_buffer.write_slice(&hh)?;
        low_columns_buffer.write_slice(&low_columns_seed)?;
        high_columns_buffer.write_slice(&high_columns_seed)?;
        level_output_buffer.write_slice(&level_seed)?;

        let horizontal_low_push = Idwt53PushConstants {
            low_count: 3,
            high_count: 2,
            line_count: 2,
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
                ll_buffer.descriptor_info(),
                hl_buffer.descriptor_info(),
                low_columns_buffer.descriptor_info(),
            ],
            bytes_of(&horizontal_low_push),
            1,
            1,
        )?;

        let horizontal_high_push = Idwt53PushConstants {
            low_count: 3,
            high_count: 2,
            line_count: 2,
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
                lh_buffer.descriptor_info(),
                hh_buffer.descriptor_info(),
                high_columns_buffer.descriptor_info(),
            ],
            bytes_of(&horizontal_high_push),
            1,
            1,
        )?;

        let vertical_push = Idwt53PushConstants {
            low_count: 2,
            high_count: 2,
            line_count: 5,
            low_line_stride: 1,
            low_sample_stride: 5,
            high_line_stride: 1,
            high_sample_stride: 5,
            output_line_stride: 1,
            output_sample_stride: 5,
        };
        dispatch_storage_shader(
            device,
            queue_family_index,
            &spirv,
            &[
                low_columns_buffer.descriptor_info(),
                high_columns_buffer.descriptor_info(),
                level_output_buffer.descriptor_info(),
            ],
            bytes_of(&vertical_push),
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

pub fn run_idwt97_smoke(shader_path: &str) -> Result<Idwt97SmokeResult, VulkanError> {
    let spirv = read_spirv(shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let low_count = 3usize;
        let high_count = 2usize;
        let output_stride = low_count + high_count;
        let line_count = 2usize;
        let low = vec![4.25f32, -3.5, 12.75, 1.0, 5.5, 9.25];
        let high = vec![7.5f32, -5.25, 2.125, 3.75];
        let expected_line0 = inverse_9_7_1d(&low[..low_count], &high[..high_count])
            .map_err(|err| VulkanError::new(format!("CPU inverse 9/7 failed: {err}")))?;
        let expected_line1 = inverse_9_7_1d(&low[low_count..], &high[high_count..])
            .map_err(|err| VulkanError::new(format!("CPU inverse 9/7 failed: {err}")))?;
        let expected = expected_line0
            .into_iter()
            .chain(expected_line1)
            .collect::<Vec<_>>();
        let output_seed = vec![0.0f32; expected.len()];
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

        let low_buffer = device.create_host_buffer(bytes_len::<f32>(low.len())?)?;
        let high_buffer = device.create_host_buffer(bytes_len::<f32>(high.len())?)?;
        let output_buffer = device.create_host_buffer(bytes_len::<f32>(output_seed.len())?)?;
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

        let row_output = output_buffer.read_vec::<f32>(output_seed.len())?;
        compare_f32_slices("idwt97 row smoke", &row_output, &expected, 0.0001)?;

        let vertical_width = 2usize;
        let low_vertical = vec![4.25f32, 1.0, -3.5, 5.5, 12.75, 9.25];
        let high_vertical = vec![7.5f32, 2.125, -5.25, 3.75];
        let column_expected0 = inverse_9_7_1d(&[4.25, -3.5, 12.75], &[7.5, -5.25])
            .map_err(|err| VulkanError::new(format!("CPU inverse 9/7 column failed: {err}")))?;
        let column_expected1 = inverse_9_7_1d(&[1.0, 5.5, 9.25], &[2.125, 3.75])
            .map_err(|err| VulkanError::new(format!("CPU inverse 9/7 column failed: {err}")))?;
        let mut column_expected = vec![0.0f32; output_stride * vertical_width];
        for (y, value) in column_expected0.into_iter().enumerate() {
            column_expected[y * vertical_width] = value;
        }
        for (y, value) in column_expected1.into_iter().enumerate() {
            column_expected[y * vertical_width + 1] = value;
        }
        let column_seed = vec![0.0f32; column_expected.len()];
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
            device.create_host_buffer(bytes_len::<f32>(low_vertical.len())?)?;
        let high_vertical_buffer =
            device.create_host_buffer(bytes_len::<f32>(high_vertical.len())?)?;
        let column_output_buffer =
            device.create_host_buffer(bytes_len::<f32>(column_seed.len())?)?;
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

        let column_output = column_output_buffer.read_vec::<f32>(column_seed.len())?;
        compare_f32_slices(
            "idwt97 column smoke",
            &column_output,
            &column_expected,
            0.0001,
        )?;

        let ll = vec![4.25f32, -3.5, 12.75, 1.0, 5.5, 9.25];
        let hl = vec![7.5f32, -5.25, 2.125, 3.75];
        let lh = vec![6.5f32, 8.25, -4.75, 0.5, 11.0, -7.125];
        let hh = vec![1.25f32, -2.5, 4.5, -3.25];
        let expected_level = inverse_9_7_level(
            Band2DF32 {
                width: 3,
                height: 2,
                samples: &ll,
            },
            Band2DF32 {
                width: 2,
                height: 2,
                samples: &hl,
            },
            Band2DF32 {
                width: 3,
                height: 2,
                samples: &lh,
            },
            Band2DF32 {
                width: 2,
                height: 2,
                samples: &hh,
            },
        )
        .map_err(|err| VulkanError::new(format!("CPU inverse 9/7 level failed: {err}")))?;

        let low_rows_seed = vec![0.0f32; 3 * 4];
        let high_rows_seed = vec![0.0f32; 2 * 4];
        let level_seed = vec![0.0f32; expected_level.samples.len()];
        let ll_buffer = device.create_host_buffer(bytes_len::<f32>(ll.len())?)?;
        let hl_buffer = device.create_host_buffer(bytes_len::<f32>(hl.len())?)?;
        let lh_buffer = device.create_host_buffer(bytes_len::<f32>(lh.len())?)?;
        let hh_buffer = device.create_host_buffer(bytes_len::<f32>(hh.len())?)?;
        let low_rows_buffer = device.create_host_buffer(bytes_len::<f32>(low_rows_seed.len())?)?;
        let high_rows_buffer = device.create_host_buffer(bytes_len::<f32>(high_rows_seed.len())?)?;
        let level_output_buffer = device.create_host_buffer(bytes_len::<f32>(level_seed.len())?)?;
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

        let level_output = level_output_buffer.read_vec::<f32>(level_seed.len())?;
        compare_f32_slices(
            "idwt97 level smoke",
            &level_output,
            &expected_level.samples,
            0.0002,
        )?;

        let component = idwt97_smoke_component()?;
        let idwt97_program = StorageComputeProgram::new(
            device,
            &spirv,
            3,
            std::mem::size_of::<Idwt53PushConstants>() as u32,
        )?;
        let component_output_plane = run_idwt97_component_on_device(
            device,
            queue_family_index,
            &idwt97_program,
            &component,
            1,
        )?;
        let component_ll = vec![4.0f32, -3.0, 12.0, 1.0, 5.0, 9.0];
        let component_hl = vec![7.0f32, -5.0, 2.0, 3.0];
        let component_lh = vec![6.0f32, 8.0, -4.0, 0.0, 11.0, -7.0];
        let component_hh = vec![1.0f32, -2.0, 4.0, -3.0];
        let expected_component = inverse_9_7_level(
            Band2DF32 {
                width: 3,
                height: 2,
                samples: &component_ll,
            },
            Band2DF32 {
                width: 2,
                height: 2,
                samples: &component_hl,
            },
            Band2DF32 {
                width: 3,
                height: 2,
                samples: &component_lh,
            },
            Band2DF32 {
                width: 2,
                height: 2,
                samples: &component_hh,
            },
        )
        .map_err(|err| {
            VulkanError::new(format!("CPU inverse 9/7 component smoke failed: {err}"))
        })?;
        compare_f32_slices(
            "idwt97 component smoke",
            &component_output_plane.samples,
            &expected_component.samples,
            0.0002,
        )?;
        device.wait_idle()?;

        Ok(Idwt97SmokeResult {
            device_name: summary.name.clone(),
            queue_family_index,
            row_output,
            column_output,
            level_output,
            component_output: component_output_plane.samples,
        })
    })
}

fn idwt97_smoke_component() -> Result<ComponentSubbandPlanes, VulkanError> {
    let component_layout = TileComponentLayout {
        width: 5,
        height: 4,
        resolutions: vec![
            ResolutionLayout {
                level: 0,
                width: 3,
                height: 2,
                precincts_x: 1,
                precincts_y: 1,
                subbands: vec![SubbandLayout {
                    kind: SubbandKind::Ll,
                    width: 3,
                    height: 2,
                    codeblock_width: 3,
                    codeblock_height: 2,
                    codeblocks_x: 1,
                    codeblocks_y: 1,
                }],
            },
            ResolutionLayout {
                level: 1,
                width: 5,
                height: 4,
                precincts_x: 1,
                precincts_y: 1,
                subbands: vec![
                    SubbandLayout {
                        kind: SubbandKind::Hl,
                        width: 2,
                        height: 2,
                        codeblock_width: 2,
                        codeblock_height: 2,
                        codeblocks_x: 1,
                        codeblocks_y: 1,
                    },
                    SubbandLayout {
                        kind: SubbandKind::Lh,
                        width: 3,
                        height: 2,
                        codeblock_width: 3,
                        codeblock_height: 2,
                        codeblocks_x: 1,
                        codeblocks_y: 1,
                    },
                    SubbandLayout {
                        kind: SubbandKind::Hh,
                        width: 2,
                        height: 2,
                        codeblock_width: 2,
                        codeblock_height: 2,
                        codeblocks_x: 1,
                        codeblocks_y: 1,
                    },
                ],
            },
        ],
    };
    let mut component = ComponentSubbandPlanes::new(0, &component_layout)
        .map_err(|err| VulkanError::new(format!("IDWT 9/7 smoke component build failed: {err}")))?;
    set_component_plane(&mut component, 0, SubbandKind::Ll, vec![4, -3, 12, 1, 5, 9])?;
    set_component_plane(&mut component, 1, SubbandKind::Hl, vec![7, -5, 2, 3])?;
    set_component_plane(&mut component, 1, SubbandKind::Lh, vec![6, 8, -4, 0, 11, -7])?;
    set_component_plane(&mut component, 1, SubbandKind::Hh, vec![1, -2, 4, -3])?;
    Ok(component)
}

fn set_component_plane(
    component: &mut ComponentSubbandPlanes,
    resolution: u8,
    subband: SubbandKind,
    coefficients: Vec<i32>,
) -> Result<(), VulkanError> {
    let plane = component
        .planes
        .iter_mut()
        .find(|plane| plane.resolution == resolution && plane.layout.kind == subband)
        .ok_or_else(|| {
            VulkanError::new(format!(
                "IDWT 9/7 smoke missing component plane r{resolution} {subband}"
            ))
        })?;
    if plane.plane.coefficients.len() != coefficients.len() {
        return Err(VulkanError::new(format!(
            "IDWT 9/7 smoke r{resolution} {subband} has {} samples, expected {}",
            coefficients.len(),
            plane.plane.coefficients.len()
        )));
    }
    plane.plane.coefficients = coefficients;
    Ok(())
}

pub fn run_irreversible_smoke(
    idwt97_shader_path: &str,
    color_shader_path: &str,
) -> Result<IrreversibleSmokeResult, VulkanError> {
    let idwt97_spirv = read_spirv(idwt97_shader_path)?;
    let color_spirv = read_spirv(color_shader_path)?;
    with_compute_device(|device, summary, queue_family_index| {
        let idwt97_program = StorageComputeProgram::new(
            device,
            &idwt97_spirv,
            3,
            std::mem::size_of::<Idwt53PushConstants>() as u32,
        )?;
        let color_program = StorageComputeProgram::new(
            device,
            &color_spirv,
            4,
            std::mem::size_of::<ColorPushConstants>() as u32,
        )?;
        let components = vec![
            scaled_idwt97_smoke_component(1)?,
            scaled_idwt97_smoke_component(2)?,
            scaled_idwt97_smoke_component(-1)?,
        ];
        let float_components = components
            .iter()
            .enumerate()
            .map(|(component_index, component)| {
                ComponentFloatSubbandPlanes::from_component_with_scales(
                    component,
                    |resolution, subband| {
                        irreversible_smoke_subband_scale(component_index, resolution, subband)
                    },
                )
            })
            .collect::<Vec<_>>();

        let mut cpu_planes = Vec::with_capacity(float_components.len());
        let mut gpu_planes = Vec::with_capacity(float_components.len());
        for component in &float_components {
            cpu_planes.push(component.inverse_9_7().map_err(|err| {
                VulkanError::new(format!("CPU irreversible smoke IDWT failed: {err}"))
            })?);
            gpu_planes.push(run_idwt97_float_component_on_device(
                device,
                queue_family_index,
                &idwt97_program,
                component,
                1,
            )?);
        }

        for (component_index, (gpu, cpu)) in gpu_planes.iter().zip(&cpu_planes).enumerate() {
            compare_f32_slices(
                &format!("irreversible smoke component {component_index} IDWT"),
                &gpu.samples,
                &cpu.samples,
                0.0002,
            )?;
        }

        let cpu_samples =
            pack_rgb48_float_samples_cpu(&cpu_planes, 12, 1, WaveletTransform::Irreversible97)
                .map_err(|err| {
                    VulkanError::new(format!("CPU irreversible smoke color pack failed: {err}"))
                })?;
        let gpu_samples = run_color_float_components_on_device_with_program(
            device,
            queue_family_index,
            &color_program,
            &gpu_planes,
            2,
            12,
        )?;
        if gpu_samples != cpu_samples {
            return Err(VulkanError::new(format!(
                "irreversible smoke RGB48 mismatch: got {gpu_samples:?}, expected {cpu_samples:?}"
            )));
        }
        device.wait_idle()?;

        Ok(IrreversibleSmokeResult {
            device_name: summary.name.clone(),
            queue_family_index,
            cpu_samples,
            gpu_samples,
        })
    })
}

fn scaled_idwt97_smoke_component(scale: i32) -> Result<ComponentSubbandPlanes, VulkanError> {
    let mut component = idwt97_smoke_component()?;
    for plane in &mut component.planes {
        for coefficient in &mut plane.plane.coefficients {
            *coefficient = coefficient.checked_mul(scale).ok_or_else(|| {
                VulkanError::new(format!(
                    "IDWT 9/7 smoke coefficient {coefficient} overflows scale {scale}"
                ))
            })?;
        }
    }
    Ok(component)
}

fn irreversible_smoke_subband_scale(
    component_index: usize,
    resolution: u8,
    subband: SubbandKind,
) -> f32 {
    let component_scale = [1.0, 0.5, 1.5][component_index];
    let subband_scale = match (resolution, subband) {
        (0, SubbandKind::Ll) => 1.0,
        (_, SubbandKind::Hl) => 0.75,
        (_, SubbandKind::Lh) => 1.25,
        (_, SubbandKind::Hh) => 0.5,
        _ => 1.0,
    };
    component_scale * subband_scale
}

fn compare_f32_slices(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    epsilon: f32,
) -> Result<(), VulkanError> {
    if actual.len() != expected.len() {
        return Err(VulkanError::new(format!(
            "{label} length mismatch: got {}, expected {}",
            actual.len(),
            expected.len()
        )));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if (actual - expected).abs() > epsilon {
            return Err(VulkanError::new(format!(
                "{label} output mismatch at {index}: got {actual}, expected {expected}"
            )));
        }
    }
    Ok(())
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
                output_shift: 0,
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
                output_shift: 0,
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
                output_shift: 0,
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
        let float_component_planes = [
            PlaneF32 {
                width: 2,
                height: 1,
                samples: vec![-1.0, 0.4],
            },
            PlaneF32 {
                width: 2,
                height: 1,
                samples: vec![2.0, -2048.0],
            },
            PlaneF32 {
                width: 2,
                height: 1,
                samples: vec![5.0, 2048.0],
            },
        ];
        let color_program = StorageComputeProgram::new(
            device,
            &spirv,
            4,
            std::mem::size_of::<ColorPushConstants>() as u32,
        )?;
        let float_component_output = run_color_float_components_on_device_with_program(
            device,
            queue_family_index,
            &color_program,
            &float_component_planes,
            0,
            12,
        )?;
        let float_component_expected = vec![2047, 2050, 2053, 2048, 0, 4095];
        if float_component_output != float_component_expected {
            return Err(VulkanError::new(format!(
                "color float component smoke output mismatch: got {float_component_output:?}, expected {float_component_expected:?}"
            )));
        }
        device.wait_idle()?;

        Ok(ColorSmokeResult {
            device_name: summary.name.clone(),
            queue_family_index,
            none_output,
            rct_output,
            ict_output,
            float_component_output,
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
    output_shift: u32,
}
