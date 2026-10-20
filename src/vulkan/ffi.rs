const VK_SUCCESS: i32 = 0;
const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
const VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE: u32 = 6;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 28;
const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
const VK_STRUCTURE_TYPE_MEMORY_BARRIER: u32 = 44;
const VK_QUEUE_GRAPHICS_BIT: u32 = 0x0000_0001;
const VK_QUEUE_COMPUTE_BIT: u32 = 0x0000_0002;
const VK_QUEUE_TRANSFER_BIT: u32 = 0x0000_0004;
const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x0000_0001;
const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x0000_0002;
const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x0000_0020;
const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x0000_0001;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x0000_0002;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x0000_0004;
const VK_MEMORY_PROPERTY_HOST_CACHED_BIT: u32 = 0x0000_0008;
const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x0000_0020;
const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: u32 = 0x0000_0800;
const VK_PIPELINE_STAGE_TRANSFER_BIT: u32 = 0x0000_1000;
const VK_PIPELINE_STAGE_HOST_BIT: u32 = 0x0000_4000;
const VK_ACCESS_TRANSFER_READ_BIT: u32 = 0x0000_0800;
const VK_ACCESS_TRANSFER_WRITE_BIT: u32 = 0x0000_1000;
const VK_ACCESS_HOST_READ_BIT: u32 = 0x0000_2000;
const VK_ACCESS_HOST_WRITE_BIT: u32 = 0x0000_4000;
const VK_ACCESS_SHADER_READ_BIT: u32 = 0x0000_0020;
const VK_ACCESS_SHADER_WRITE_BIT: u32 = 0x0000_0040;
const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;
const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
const VK_WHOLE_SIZE: u64 = !0;
const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x0000_0002;
const VK_MEMORY_HEAP_DEVICE_LOCAL_BIT: u32 = 0x0000_0001;
const VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO: u32 = 11;
const VK_QUERY_TYPE_TIMESTAMP: u32 = 2;
const VK_QUERY_RESULT_64_BIT: u32 = 0x0000_0001;
const VK_QUERY_RESULT_WAIT_BIT: u32 = 0x0000_0002;
const VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT: u32 = 0x0000_0001;
const VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT: u32 = 0x0000_2000;

type VkInstance = usize;
type VkPhysicalDevice = usize;
type VkDevice = usize;
type VkQueue = usize;
type VkBuffer = usize;
type VkDeviceMemory = usize;
type VkShaderModule = usize;
type VkDescriptorSetLayout = usize;
type VkPipelineLayout = usize;
type VkPipeline = usize;
type VkDescriptorPool = usize;
type VkDescriptorSet = usize;
type VkCommandPool = usize;
type VkCommandBuffer = usize;
type VkFence = usize;
type VkQueryPool = usize;
type VkFlags = u32;
type VkBool32 = u32;

type VkGetInstanceProcAddr =
    unsafe extern "system" fn(VkInstance, *const c_char) -> Option<unsafe extern "system" fn()>;
type VkGetDeviceProcAddr =
    unsafe extern "system" fn(VkDevice, *const c_char) -> Option<unsafe extern "system" fn()>;
type VkCreateInstance =
    unsafe extern "system" fn(*const VkInstanceCreateInfo, *const c_void, *mut VkInstance) -> i32;
type VkDestroyInstance = unsafe extern "system" fn(VkInstance, *const c_void);
type VkEnumeratePhysicalDevices =
    unsafe extern "system" fn(VkInstance, *mut u32, *mut VkPhysicalDevice) -> i32;
type VkGetPhysicalDeviceProperties =
    unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceProperties);
type VkGetPhysicalDeviceQueueFamilyProperties =
    unsafe extern "system" fn(VkPhysicalDevice, *mut u32, *mut VkQueueFamilyProperties);
type VkGetPhysicalDeviceMemoryProperties =
    unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryProperties);
type VkCreateDevice = unsafe extern "system" fn(
    VkPhysicalDevice,
    *const VkDeviceCreateInfo,
    *const c_void,
    *mut VkDevice,
) -> i32;
type VkDestroyDevice = unsafe extern "system" fn(VkDevice, *const c_void);
type VkGetDeviceQueue = unsafe extern "system" fn(VkDevice, u32, u32, *mut VkQueue);
type VkCreateBuffer = unsafe extern "system" fn(
    VkDevice,
    *const VkBufferCreateInfo,
    *const c_void,
    *mut VkBuffer,
) -> i32;
type VkDestroyBuffer = unsafe extern "system" fn(VkDevice, VkBuffer, *const c_void);
type VkGetBufferMemoryRequirements =
    unsafe extern "system" fn(VkDevice, VkBuffer, *mut VkMemoryRequirements);
type VkAllocateMemory = unsafe extern "system" fn(
    VkDevice,
    *const VkMemoryAllocateInfo,
    *const c_void,
    *mut VkDeviceMemory,
) -> i32;
type VkFreeMemory = unsafe extern "system" fn(VkDevice, VkDeviceMemory, *const c_void);
type VkBindBufferMemory = unsafe extern "system" fn(VkDevice, VkBuffer, VkDeviceMemory, u64) -> i32;
type VkMapMemory =
    unsafe extern "system" fn(VkDevice, VkDeviceMemory, u64, u64, VkFlags, *mut *mut c_void) -> i32;
type VkUnmapMemory = unsafe extern "system" fn(VkDevice, VkDeviceMemory);
type VkInvalidateMappedMemoryRanges =
    unsafe extern "system" fn(VkDevice, u32, *const VkMappedMemoryRange) -> i32;
type VkFlushMappedMemoryRanges =
    unsafe extern "system" fn(VkDevice, u32, *const VkMappedMemoryRange) -> i32;
type VkCreateShaderModule = unsafe extern "system" fn(
    VkDevice,
    *const VkShaderModuleCreateInfo,
    *const c_void,
    *mut VkShaderModule,
) -> i32;
type VkDestroyShaderModule = unsafe extern "system" fn(VkDevice, VkShaderModule, *const c_void);
type VkCreateDescriptorSetLayout = unsafe extern "system" fn(
    VkDevice,
    *const VkDescriptorSetLayoutCreateInfo,
    *const c_void,
    *mut VkDescriptorSetLayout,
) -> i32;
type VkDestroyDescriptorSetLayout =
    unsafe extern "system" fn(VkDevice, VkDescriptorSetLayout, *const c_void);
type VkCreatePipelineLayout = unsafe extern "system" fn(
    VkDevice,
    *const VkPipelineLayoutCreateInfo,
    *const c_void,
    *mut VkPipelineLayout,
) -> i32;
type VkDestroyPipelineLayout = unsafe extern "system" fn(VkDevice, VkPipelineLayout, *const c_void);
type VkCreateComputePipelines = unsafe extern "system" fn(
    VkDevice,
    usize,
    u32,
    *const VkComputePipelineCreateInfo,
    *const c_void,
    *mut VkPipeline,
) -> i32;
type VkDestroyPipeline = unsafe extern "system" fn(VkDevice, VkPipeline, *const c_void);
type VkCreateDescriptorPool = unsafe extern "system" fn(
    VkDevice,
    *const VkDescriptorPoolCreateInfo,
    *const c_void,
    *mut VkDescriptorPool,
) -> i32;
type VkDestroyDescriptorPool = unsafe extern "system" fn(VkDevice, VkDescriptorPool, *const c_void);
type VkAllocateDescriptorSets = unsafe extern "system" fn(
    VkDevice,
    *const VkDescriptorSetAllocateInfo,
    *mut VkDescriptorSet,
) -> i32;
type VkUpdateDescriptorSets =
    unsafe extern "system" fn(VkDevice, u32, *const VkWriteDescriptorSet, u32, *const c_void);
type VkCreateCommandPool = unsafe extern "system" fn(
    VkDevice,
    *const VkCommandPoolCreateInfo,
    *const c_void,
    *mut VkCommandPool,
) -> i32;
type VkDestroyCommandPool = unsafe extern "system" fn(VkDevice, VkCommandPool, *const c_void);
type VkAllocateCommandBuffers = unsafe extern "system" fn(
    VkDevice,
    *const VkCommandBufferAllocateInfo,
    *mut VkCommandBuffer,
) -> i32;
type VkBeginCommandBuffer =
    unsafe extern "system" fn(VkCommandBuffer, *const VkCommandBufferBeginInfo) -> i32;
type VkEndCommandBuffer = unsafe extern "system" fn(VkCommandBuffer) -> i32;
type VkCmdBindPipeline = unsafe extern "system" fn(VkCommandBuffer, u32, VkPipeline);
type VkCmdBindDescriptorSets = unsafe extern "system" fn(
    VkCommandBuffer,
    u32,
    VkPipelineLayout,
    u32,
    u32,
    *const VkDescriptorSet,
    u32,
    *const u32,
);
type VkCmdPushConstants =
    unsafe extern "system" fn(VkCommandBuffer, VkPipelineLayout, VkFlags, u32, u32, *const c_void);
type VkCmdDispatch = unsafe extern "system" fn(VkCommandBuffer, u32, u32, u32);
type VkCmdCopyBuffer =
    unsafe extern "system" fn(VkCommandBuffer, VkBuffer, VkBuffer, u32, *const VkBufferCopy);
type VkCmdPipelineBarrier = unsafe extern "system" fn(
    VkCommandBuffer,
    u32,
    u32,
    VkFlags,
    u32,
    *const VkMemoryBarrier,
    u32,
    *const c_void,
    u32,
    *const c_void,
);
type VkCreateFence = unsafe extern "system" fn(
    VkDevice,
    *const VkFenceCreateInfo,
    *const c_void,
    *mut VkFence,
) -> i32;
type VkDestroyFence = unsafe extern "system" fn(VkDevice, VkFence, *const c_void);
type VkQueueSubmit = unsafe extern "system" fn(VkQueue, u32, *const VkSubmitInfo, VkFence) -> i32;
type VkWaitForFences =
    unsafe extern "system" fn(VkDevice, u32, *const VkFence, VkBool32, u64) -> i32;
type VkResetFences = unsafe extern "system" fn(VkDevice, u32, *const VkFence) -> i32;
type VkDeviceWaitIdle = unsafe extern "system" fn(VkDevice) -> i32;
type VkCreateQueryPool = unsafe extern "system" fn(
    VkDevice,
    *const VkQueryPoolCreateInfo,
    *const c_void,
    *mut VkQueryPool,
) -> i32;
type VkDestroyQueryPool = unsafe extern "system" fn(VkDevice, VkQueryPool, *const c_void);
type VkCmdResetQueryPool = unsafe extern "system" fn(VkCommandBuffer, VkQueryPool, u32, u32);
type VkCmdWriteTimestamp = unsafe extern "system" fn(VkCommandBuffer, u32, VkQueryPool, u32);
type VkGetQueryPoolResults = unsafe extern "system" fn(
    VkDevice,
    VkQueryPool,
    u32,
    u32,
    usize,
    *mut c_void,
    u64,
    VkFlags,
) -> i32;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
}

#[repr(C)]
struct VkApplicationInfo {
    s_type: u32,
    p_next: *const c_void,
    p_application_name: *const c_char,
    application_version: u32,
    p_engine_name: *const c_char,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
struct VkInstanceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: VkFlags,
    p_application_info: *const VkApplicationInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const c_char,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const c_char,
}

#[repr(C)]
struct VkDeviceQueueCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: VkFlags,
    queue_family_index: u32,
    queue_count: u32,
    p_queue_priorities: *const f32,
}

#[repr(C)]
struct VkDeviceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: VkFlags,
    queue_create_info_count: u32,
    p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const c_char,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const c_char,
    p_enabled_features: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryType {
    property_flags: VkFlags,
    heap_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryHeap {
    size: u64,
    flags: VkFlags,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkPhysicalDeviceMemoryProperties {
    memory_type_count: u32,
    memory_types: [VkMemoryType; 32],
    memory_heap_count: u32,
    memory_heaps: [VkMemoryHeap; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryRequirements {
    size: u64,
    alignment: u64,
    memory_type_bits: u32,
}

#[repr(C)]
struct VkMemoryAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    allocation_size: u64,
    memory_type_index: u32,
}

#[repr(C)]
struct VkBufferCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: VkFlags,
    size: u64,
    usage: VkFlags,
    sharing_mode: u32,
    queue_family_index_count: u32,
    p_queue_family_indices: *const u32,
}

#[repr(C)]
struct VkShaderModuleCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: VkFlags,
    code_size: usize,
    p_code: *const u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkDescriptorSetLayoutBinding {
    binding: u32,
    descriptor_type: u32,
    descriptor_count: u32,
    stage_flags: VkFlags,
    p_immutable_samplers: *const c_void,
}

#[repr(C)]
struct VkDescriptorSetLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: VkFlags,
    binding_count: u32,
    p_bindings: *const VkDescriptorSetLayoutBinding,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkPushConstantRange {
    stage_flags: VkFlags,
    offset: u32,
    size: u32,
}

#[repr(C)]
struct VkPipelineLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: VkFlags,
    set_layout_count: u32,
    p_set_layouts: *const VkDescriptorSetLayout,
    push_constant_range_count: u32,
    p_push_constant_ranges: *const VkPushConstantRange,
}

#[repr(C)]
struct VkPipelineShaderStageCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: VkFlags,
    stage: VkFlags,
    module: VkShaderModule,
    p_name: *const c_char,
    p_specialization_info: *const c_void,
}

#[repr(C)]
struct VkComputePipelineCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: VkFlags,
    stage: VkPipelineShaderStageCreateInfo,
    layout: VkPipelineLayout,
    base_pipeline_handle: VkPipeline,
    base_pipeline_index: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkDescriptorPoolSize {
    descriptor_type: u32,
    descriptor_count: u32,
}

#[repr(C)]
struct VkDescriptorPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: VkFlags,
    max_sets: u32,
    pool_size_count: u32,
    p_pool_sizes: *const VkDescriptorPoolSize,
}

#[repr(C)]
struct VkDescriptorSetAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    descriptor_pool: VkDescriptorPool,
    descriptor_set_count: u32,
    p_set_layouts: *const VkDescriptorSetLayout,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkDescriptorBufferInfo {
    buffer: VkBuffer,
    offset: u64,
    range: u64,
}

#[repr(C)]
struct VkWriteDescriptorSet {
    s_type: u32,
    p_next: *const c_void,
    dst_set: VkDescriptorSet,
    dst_binding: u32,
    dst_array_element: u32,
    descriptor_count: u32,
    descriptor_type: u32,
    p_image_info: *const c_void,
    p_buffer_info: *const VkDescriptorBufferInfo,
    p_texel_buffer_view: *const c_void,
}

#[repr(C)]
struct VkCommandPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: VkFlags,
    queue_family_index: u32,
}

#[repr(C)]
struct VkCommandBufferAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    command_pool: VkCommandPool,
    level: u32,
    command_buffer_count: u32,
}

#[repr(C)]
struct VkCommandBufferBeginInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: VkFlags,
    p_inheritance_info: *const c_void,
}

#[repr(C)]
struct VkSubmitInfo {
    s_type: u32,
    p_next: *const c_void,
    wait_semaphore_count: u32,
    p_wait_semaphores: *const usize,
    p_wait_dst_stage_mask: *const VkFlags,
    command_buffer_count: u32,
    p_command_buffers: *const VkCommandBuffer,
    signal_semaphore_count: u32,
    p_signal_semaphores: *const usize,
}

#[repr(C)]
struct VkMappedMemoryRange {
    s_type: u32,
    p_next: *const c_void,
    memory: VkDeviceMemory,
    offset: u64,
    size: u64,
}

#[repr(C)]
struct VkMemoryBarrier {
    s_type: u32,
    p_next: *const c_void,
    src_access_mask: VkFlags,
    dst_access_mask: VkFlags,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkBufferCopy {
    src_offset: u64,
    dst_offset: u64,
    size: u64,
}

#[repr(C)]
struct VkFenceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: VkFlags,
}

#[repr(C)]
struct VkQueryPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: VkFlags,
    query_type: u32,
    query_count: u32,
    pipeline_statistics: VkFlags,
}

// VkPhysicalDeviceLimits contains VkDeviceSize fields, so it is 8-byte aligned: without
// align(8) it would sit 4 bytes early inside VkPhysicalDeviceProperties and the driver would
// write past the end of the struct.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct VkPhysicalDeviceLimits {
    bytes: [u8; 504],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkPhysicalDeviceSparseProperties {
    residency_standard_2d_block_shape: VkBool32,
    residency_standard_2d_multisample_block_shape: VkBool32,
    residency_standard_3d_block_shape: VkBool32,
    residency_aligned_mip_size: VkBool32,
    residency_non_resident_strict: VkBool32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkPhysicalDeviceProperties {
    api_version: u32,
    driver_version: u32,
    vendor_id: u32,
    device_id: u32,
    device_type: u32,
    device_name: [c_char; 256],
    pipeline_cache_uuid: [u8; 16],
    limits: VkPhysicalDeviceLimits,
    sparse_properties: VkPhysicalDeviceSparseProperties,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkExtent3D {
    width: u32,
    height: u32,
    depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkQueueFamilyProperties {
    queue_flags: VkFlags,
    queue_count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: VkExtent3D,
}

