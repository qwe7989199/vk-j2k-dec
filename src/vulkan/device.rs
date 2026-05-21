#[derive(Clone, Copy)]
struct VulkanDeviceFunctions {
    destroy_device: VkDestroyDevice,
    get_device_queue: VkGetDeviceQueue,
    create_buffer: VkCreateBuffer,
    destroy_buffer: VkDestroyBuffer,
    get_buffer_memory_requirements: VkGetBufferMemoryRequirements,
    allocate_memory: VkAllocateMemory,
    free_memory: VkFreeMemory,
    bind_buffer_memory: VkBindBufferMemory,
    map_memory: VkMapMemory,
    unmap_memory: VkUnmapMemory,
    invalidate_mapped_memory_ranges: VkInvalidateMappedMemoryRanges,
    flush_mapped_memory_ranges: VkFlushMappedMemoryRanges,
    create_shader_module: VkCreateShaderModule,
    destroy_shader_module: VkDestroyShaderModule,
    create_descriptor_set_layout: VkCreateDescriptorSetLayout,
    destroy_descriptor_set_layout: VkDestroyDescriptorSetLayout,
    create_pipeline_layout: VkCreatePipelineLayout,
    destroy_pipeline_layout: VkDestroyPipelineLayout,
    create_compute_pipelines: VkCreateComputePipelines,
    destroy_pipeline: VkDestroyPipeline,
    create_descriptor_pool: VkCreateDescriptorPool,
    destroy_descriptor_pool: VkDestroyDescriptorPool,
    allocate_descriptor_sets: VkAllocateDescriptorSets,
    update_descriptor_sets: VkUpdateDescriptorSets,
    create_command_pool: VkCreateCommandPool,
    destroy_command_pool: VkDestroyCommandPool,
    allocate_command_buffers: VkAllocateCommandBuffers,
    begin_command_buffer: VkBeginCommandBuffer,
    end_command_buffer: VkEndCommandBuffer,
    cmd_bind_pipeline: VkCmdBindPipeline,
    cmd_bind_descriptor_sets: VkCmdBindDescriptorSets,
    cmd_push_constants: VkCmdPushConstants,
    cmd_dispatch: VkCmdDispatch,
    cmd_copy_buffer: VkCmdCopyBuffer,
    cmd_pipeline_barrier: VkCmdPipelineBarrier,
    create_fence: VkCreateFence,
    destroy_fence: VkDestroyFence,
    queue_submit: VkQueueSubmit,
    wait_for_fences: VkWaitForFences,
    device_wait_idle: VkDeviceWaitIdle,
}

struct VulkanDevice {
    device: VkDevice,
    queue: VkQueue,
    functions: VulkanDeviceFunctions,
    memory_properties: VkPhysicalDeviceMemoryProperties,
}

struct StorageDispatch<'a> {
    pipeline: VkPipeline,
    pipeline_layout: VkPipelineLayout,
    descriptor_set: VkDescriptorSet,
    push_constants: &'a [u8],
    groups: [u32; 3],
}

struct BufferCopy<'a> {
    source: &'a HostBuffer,
    destination: &'a HostBuffer,
    byte_len: u64,
}

impl VulkanDevice {
    fn create_host_buffer(&self, size: u64) -> Result<HostBuffer, VulkanError> {
        self.create_buffer(
            size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
    }

    fn create_host_cached_buffer(&self, size: u64, usage: VkFlags) -> Result<HostBuffer, VulkanError> {
        self.create_buffer(
            size,
            usage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        )
    }

    fn create_device_local_buffer(
        &self,
        size: u64,
        usage: VkFlags,
    ) -> Result<HostBuffer, VulkanError> {
        self.create_buffer(size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    }

    fn create_buffer(
        &self,
        size: u64,
        usage: VkFlags,
        memory_properties: VkFlags,
    ) -> Result<HostBuffer, VulkanError> {
        let create_info = VkBufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            size,
            usage,
            sharing_mode: 0,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };
        let mut buffer = 0usize;
        vk_check(
            unsafe {
                (self.functions.create_buffer)(self.device, &create_info, ptr::null(), &mut buffer)
            },
            "vkCreateBuffer",
        )?;

        let mut requirements = unsafe { std::mem::zeroed::<VkMemoryRequirements>() };
        unsafe {
            (self.functions.get_buffer_memory_requirements)(self.device, buffer, &mut requirements);
        }
        let memory_type_index = self.find_memory_type(
            requirements.memory_type_bits,
            memory_properties,
        )?;
        let allocate_info = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: requirements.size,
            memory_type_index,
        };
        let mut memory = 0usize;
        vk_check(
            unsafe {
                (self.functions.allocate_memory)(
                    self.device,
                    &allocate_info,
                    ptr::null(),
                    &mut memory,
                )
            },
            "vkAllocateMemory",
        )?;
        vk_check(
            unsafe { (self.functions.bind_buffer_memory)(self.device, buffer, memory, 0) },
            "vkBindBufferMemory",
        )?;

        Ok(HostBuffer {
            device: self.device,
            functions: self.functions,
            buffer,
            memory,
            size,
            memory_properties,
        })
    }

    fn create_storage_descriptor_set_layout(
        &self,
        binding_count: u32,
    ) -> Result<DescriptorSetLayoutHandle, VulkanError> {
        let bindings = (0..binding_count)
            .map(|binding| VkDescriptorSetLayoutBinding {
                binding,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            })
            .collect::<Vec<_>>();
        let create_info = VkDescriptorSetLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            binding_count,
            p_bindings: bindings.as_ptr(),
        };
        let mut handle = 0usize;
        vk_check(
            unsafe {
                (self.functions.create_descriptor_set_layout)(
                    self.device,
                    &create_info,
                    ptr::null(),
                    &mut handle,
                )
            },
            "vkCreateDescriptorSetLayout",
        )?;
        Ok(DescriptorSetLayoutHandle {
            device: self.device,
            functions: self.functions,
            handle,
        })
    }

    fn create_pipeline_layout(
        &self,
        descriptor_set_layout: VkDescriptorSetLayout,
        push_constant_size: u32,
    ) -> Result<PipelineLayoutHandle, VulkanError> {
        let push_constant = VkPushConstantRange {
            stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: push_constant_size,
        };
        let create_info = VkPipelineLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant,
        };
        let mut handle = 0usize;
        vk_check(
            unsafe {
                (self.functions.create_pipeline_layout)(
                    self.device,
                    &create_info,
                    ptr::null(),
                    &mut handle,
                )
            },
            "vkCreatePipelineLayout",
        )?;
        Ok(PipelineLayoutHandle {
            device: self.device,
            functions: self.functions,
            handle,
        })
    }

    fn create_shader_module(&self, spirv: &[u32]) -> Result<ShaderModuleHandle, VulkanError> {
        let create_info = VkShaderModuleCreateInfo {
            s_type: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            code_size: std::mem::size_of_val(spirv),
            p_code: spirv.as_ptr(),
        };
        let mut handle = 0usize;
        vk_check(
            unsafe {
                (self.functions.create_shader_module)(
                    self.device,
                    &create_info,
                    ptr::null(),
                    &mut handle,
                )
            },
            "vkCreateShaderModule",
        )?;
        Ok(ShaderModuleHandle {
            device: self.device,
            functions: self.functions,
            handle,
        })
    }

    fn create_compute_pipeline(
        &self,
        shader_module: VkShaderModule,
        pipeline_layout: VkPipelineLayout,
    ) -> Result<PipelineHandle, VulkanError> {
        let entry = CString::new("main").expect("static string");
        let stage = VkPipelineShaderStageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            stage: VK_SHADER_STAGE_COMPUTE_BIT,
            module: shader_module,
            p_name: entry.as_ptr(),
            p_specialization_info: ptr::null(),
        };
        let create_info = VkComputePipelineCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            stage,
            layout: pipeline_layout,
            base_pipeline_handle: 0,
            base_pipeline_index: -1,
        };
        let mut handle = 0usize;
        vk_check(
            unsafe {
                (self.functions.create_compute_pipelines)(
                    self.device,
                    0,
                    1,
                    &create_info,
                    ptr::null(),
                    &mut handle,
                )
            },
            "vkCreateComputePipelines",
        )?;
        Ok(PipelineHandle {
            device: self.device,
            functions: self.functions,
            handle,
        })
    }

    fn create_descriptor_pool(
        &self,
        descriptor_count: u32,
        max_sets: u32,
    ) -> Result<DescriptorPoolHandle, VulkanError> {
        let pool_size = VkDescriptorPoolSize {
            descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count,
        };
        let create_info = VkDescriptorPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            max_sets,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };
        let mut handle = 0usize;
        vk_check(
            unsafe {
                (self.functions.create_descriptor_pool)(
                    self.device,
                    &create_info,
                    ptr::null(),
                    &mut handle,
                )
            },
            "vkCreateDescriptorPool",
        )?;
        Ok(DescriptorPoolHandle {
            device: self.device,
            functions: self.functions,
            handle,
        })
    }

    fn allocate_descriptor_set(
        &self,
        pool: VkDescriptorPool,
        layout: VkDescriptorSetLayout,
    ) -> Result<VkDescriptorSet, VulkanError> {
        let allocate_info = VkDescriptorSetAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: pool,
            descriptor_set_count: 1,
            p_set_layouts: &layout,
        };
        let mut descriptor_set = 0usize;
        vk_check(
            unsafe {
                (self.functions.allocate_descriptor_sets)(
                    self.device,
                    &allocate_info,
                    &mut descriptor_set,
                )
            },
            "vkAllocateDescriptorSets",
        )?;
        Ok(descriptor_set)
    }

    fn update_storage_descriptor_set(
        &self,
        descriptor_set: VkDescriptorSet,
        buffers: &[VkDescriptorBufferInfo],
    ) {
        let writes = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer_info)| VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: descriptor_set,
                dst_binding: binding as u32,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: buffer_info,
                p_texel_buffer_view: ptr::null(),
            })
            .collect::<Vec<_>>();
        unsafe {
            (self.functions.update_descriptor_sets)(
                self.device,
                writes.len() as u32,
                writes.as_ptr(),
                0,
                ptr::null(),
            );
        }
    }

    fn create_command_pool(
        &self,
        queue_family_index: u32,
    ) -> Result<CommandPoolHandle, VulkanError> {
        let create_info = VkCommandPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            queue_family_index,
        };
        let mut handle = 0usize;
        vk_check(
            unsafe {
                (self.functions.create_command_pool)(
                    self.device,
                    &create_info,
                    ptr::null(),
                    &mut handle,
                )
            },
            "vkCreateCommandPool",
        )?;
        Ok(CommandPoolHandle {
            device: self.device,
            functions: self.functions,
            handle,
        })
    }

    fn allocate_command_buffer(
        &self,
        command_pool: VkCommandPool,
    ) -> Result<VkCommandBuffer, VulkanError> {
        let allocate_info = VkCommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut command_buffer = 0usize;
        vk_check(
            unsafe {
                (self.functions.allocate_command_buffers)(
                    self.device,
                    &allocate_info,
                    &mut command_buffer,
                )
            },
            "vkAllocateCommandBuffers",
        )?;
        Ok(command_buffer)
    }

    fn record_storage_dispatches(
        &self,
        command_buffer: VkCommandBuffer,
        dispatches: &[StorageDispatch<'_>],
    ) -> Result<(), VulkanError> {
        self.record_storage_dispatches_with_copies(command_buffer, dispatches, &[])
    }

    fn record_storage_dispatches_with_copies(
        &self,
        command_buffer: VkCommandBuffer,
        dispatches: &[StorageDispatch<'_>],
        trailing_copies: &[BufferCopy<'_>],
    ) -> Result<(), VulkanError> {
        let begin_info = VkCommandBufferBeginInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: 0,
            p_inheritance_info: ptr::null(),
        };
        vk_check(
            unsafe { (self.functions.begin_command_buffer)(command_buffer, &begin_info) },
            "vkBeginCommandBuffer",
        )?;
        if dispatches.is_empty() && trailing_copies.is_empty() {
            return vk_check(
                unsafe { (self.functions.end_command_buffer)(command_buffer) },
                "vkEndCommandBuffer",
            );
        };
        if !dispatches.is_empty() {
            let transfer_barrier = VkMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
                dst_access_mask: VK_ACCESS_SHADER_READ_BIT,
            };
            unsafe {
                (self.functions.cmd_pipeline_barrier)(
                    command_buffer,
                    VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1,
                    &transfer_barrier,
                    0,
                    ptr::null(),
                    0,
                    ptr::null(),
                );
            }
        }
        let mut bound_pipeline = 0usize;
        for (index, dispatch) in dispatches.iter().enumerate() {
            unsafe {
                if dispatch.pipeline != bound_pipeline {
                    (self.functions.cmd_bind_pipeline)(
                        command_buffer,
                        VK_PIPELINE_BIND_POINT_COMPUTE,
                        dispatch.pipeline,
                    );
                    bound_pipeline = dispatch.pipeline;
                }
                (self.functions.cmd_bind_descriptor_sets)(
                    command_buffer,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    dispatch.pipeline_layout,
                    0,
                    1,
                    &dispatch.descriptor_set,
                    0,
                    ptr::null(),
                );
                (self.functions.cmd_push_constants)(
                    command_buffer,
                    dispatch.pipeline_layout,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    dispatch.push_constants.len() as u32,
                    dispatch.push_constants.as_ptr() as *const c_void,
                );
                (self.functions.cmd_dispatch)(
                    command_buffer,
                    dispatch.groups[0],
                    dispatch.groups[1],
                    dispatch.groups[2],
                );
            }
            if index + 1 < dispatches.len() {
                let barrier = VkMemoryBarrier {
                    s_type: VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    p_next: ptr::null(),
                    src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
                    dst_access_mask: VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                };
                unsafe {
                    (self.functions.cmd_pipeline_barrier)(
                        command_buffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0,
                        1,
                        &barrier,
                        0,
                        ptr::null(),
                        0,
                        ptr::null(),
                    );
                }
            }
        }
        if !trailing_copies.is_empty() {
            let barrier = VkMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_TRANSFER_READ_BIT,
            };
            unsafe {
                (self.functions.cmd_pipeline_barrier)(
                    command_buffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0,
                    1,
                    &barrier,
                    0,
                    ptr::null(),
                    0,
                    ptr::null(),
                );
            }
            for copy_request in trailing_copies {
                let copy = VkBufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: copy_request.byte_len,
                };
                unsafe {
                    (self.functions.cmd_copy_buffer)(
                        command_buffer,
                        copy_request.source.buffer,
                        copy_request.destination.buffer,
                        1,
                        &copy,
                    );
                }
            }
        }
        vk_check(
            unsafe { (self.functions.end_command_buffer)(command_buffer) },
            "vkEndCommandBuffer",
        )
    }

    fn record_copy_buffer(
        &self,
        command_buffer: VkCommandBuffer,
        source: &HostBuffer,
        destination: &HostBuffer,
        byte_len: u64,
    ) -> Result<(), VulkanError> {
        self.record_copy_buffers(
            command_buffer,
            &[BufferCopy {
                source,
                destination,
                byte_len,
            }],
        )
    }

    fn record_copy_buffers(
        &self,
        command_buffer: VkCommandBuffer,
        copies: &[BufferCopy<'_>],
    ) -> Result<(), VulkanError> {
        let begin_info = VkCommandBufferBeginInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: 0,
            p_inheritance_info: ptr::null(),
        };
        vk_check(
            unsafe { (self.functions.begin_command_buffer)(command_buffer, &begin_info) },
            "vkBeginCommandBuffer",
        )?;
        for copy_request in copies {
            let copy = VkBufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: copy_request.byte_len,
            };
            unsafe {
                (self.functions.cmd_copy_buffer)(
                    command_buffer,
                    copy_request.source.buffer,
                    copy_request.destination.buffer,
                    1,
                    &copy,
                );
            }
        }
        vk_check(
            unsafe { (self.functions.end_command_buffer)(command_buffer) },
            "vkEndCommandBuffer",
        )
    }

    fn copy_buffer(
        &self,
        queue_family_index: u32,
        source: &HostBuffer,
        destination: &HostBuffer,
        byte_len: u64,
    ) -> Result<(), VulkanError> {
        let command_pool = self.create_command_pool(queue_family_index)?;
        let command_buffer = self.allocate_command_buffer(command_pool.handle)?;
        self.record_copy_buffer(command_buffer, source, destination, byte_len)?;
        self.submit_and_wait(command_buffer)
    }

    fn copy_buffers(
        &self,
        queue_family_index: u32,
        copies: &[BufferCopy<'_>],
    ) -> Result<(), VulkanError> {
        if copies.is_empty() {
            return Ok(());
        }
        let command_pool = self.create_command_pool(queue_family_index)?;
        let command_buffer = self.allocate_command_buffer(command_pool.handle)?;
        self.record_copy_buffers(command_buffer, copies)?;
        self.submit_and_wait(command_buffer)
    }

    fn submit_and_wait(&self, command_buffer: VkCommandBuffer) -> Result<(), VulkanError> {
        let fence_info = VkFenceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
        };
        let mut fence = 0usize;
        vk_check(
            unsafe {
                (self.functions.create_fence)(self.device, &fence_info, ptr::null(), &mut fence)
            },
            "vkCreateFence",
        )?;
        let _fence = FenceHandle {
            device: self.device,
            functions: self.functions,
            handle: fence,
        };
        let submit_info = VkSubmitInfo {
            s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
        };
        vk_check(
            unsafe { (self.functions.queue_submit)(self.queue, 1, &submit_info, fence) },
            "vkQueueSubmit",
        )?;
        vk_check(
            unsafe { (self.functions.wait_for_fences)(self.device, 1, &fence, 1, u64::MAX) },
            "vkWaitForFences",
        )
    }

    fn wait_idle(&self) -> Result<(), VulkanError> {
        vk_check(
            unsafe { (self.functions.device_wait_idle)(self.device) },
            "vkDeviceWaitIdle",
        )
    }

    fn find_memory_type(&self, type_bits: u32, properties: VkFlags) -> Result<u32, VulkanError> {
        for index in 0..self.memory_properties.memory_type_count {
            let supported = type_bits & (1 << index) != 0;
            let memory_type = self.memory_properties.memory_types[index as usize];
            if supported && memory_type.property_flags & properties == properties {
                return Ok(index);
            }
        }
        Err(VulkanError::new(format!(
            "no Vulkan memory type supports bitmask 0x{type_bits:08x} with properties 0x{properties:08x}"
        )))
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            (self.functions.device_wait_idle)(self.device);
            (self.functions.destroy_device)(self.device, ptr::null());
        }
    }
}

struct HostBuffer {
    device: VkDevice,
    functions: VulkanDeviceFunctions,
    buffer: VkBuffer,
    memory: VkDeviceMemory,
    size: u64,
    memory_properties: VkFlags,
}

impl HostBuffer {
    fn write_slice<T: Copy>(&self, values: &[T]) -> Result<(), VulkanError> {
        let byte_len = bytes_len::<T>(values.len())?;
        if byte_len > self.size {
            return Err(VulkanError::new(format!(
                "host buffer write of {byte_len} bytes exceeds {} bytes",
                self.size
            )));
        }
        let mut mapped = ptr::null_mut();
        vk_check(
            unsafe {
                (self.functions.map_memory)(self.device, self.memory, 0, byte_len, 0, &mut mapped)
            },
            "vkMapMemory",
        )?;
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr() as *const u8,
                mapped as *mut u8,
                byte_len as usize,
            );
        }
        if self.memory_properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT == 0 {
            self.flush_range(byte_len)?;
        }
        unsafe {
            (self.functions.unmap_memory)(self.device, self.memory);
        }
        Ok(())
    }

    fn read_vec<T: Copy>(&self, count: usize) -> Result<Vec<T>, VulkanError> {
        let byte_len = bytes_len::<T>(count)?;
        if byte_len > self.size {
            return Err(VulkanError::new(format!(
                "host buffer read of {byte_len} bytes exceeds {} bytes",
                self.size
            )));
        }
        self.invalidate_range(byte_len)?;
        let mut values = Vec::<T>::with_capacity(count);
        let mut mapped = ptr::null_mut();
        vk_check(
            unsafe {
                (self.functions.map_memory)(self.device, self.memory, 0, byte_len, 0, &mut mapped)
            },
            "vkMapMemory",
        )?;
        unsafe {
            ptr::copy_nonoverlapping(
                mapped as *const u8,
                values.as_mut_ptr() as *mut u8,
                byte_len as usize,
            );
            values.set_len(count);
            (self.functions.unmap_memory)(self.device, self.memory);
        }
        Ok(values)
    }

    fn flush_range(&self, _byte_len: u64) -> Result<(), VulkanError> {
        let range = VkMappedMemoryRange {
            s_type: VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            p_next: ptr::null(),
            memory: self.memory,
            offset: 0,
            size: VK_WHOLE_SIZE,
        };
        vk_check(
            unsafe { (self.functions.flush_mapped_memory_ranges)(self.device, 1, &range) },
            "vkFlushMappedMemoryRanges",
        )
    }

    fn invalidate_range(&self, _byte_len: u64) -> Result<(), VulkanError> {
        let range = VkMappedMemoryRange {
            s_type: VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            p_next: ptr::null(),
            memory: self.memory,
            offset: 0,
            size: VK_WHOLE_SIZE,
        };
        vk_check(
            unsafe { (self.functions.invalidate_mapped_memory_ranges)(self.device, 1, &range) },
            "vkInvalidateMappedMemoryRanges",
        )
    }

    fn descriptor_info(&self) -> VkDescriptorBufferInfo {
        VkDescriptorBufferInfo {
            buffer: self.buffer,
            offset: 0,
            range: VK_WHOLE_SIZE,
        }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        unsafe {
            (self.functions.destroy_buffer)(self.device, self.buffer, ptr::null());
            (self.functions.free_memory)(self.device, self.memory, ptr::null());
        }
    }
}

struct ShaderModuleHandle {
    device: VkDevice,
    functions: VulkanDeviceFunctions,
    handle: VkShaderModule,
}

impl Drop for ShaderModuleHandle {
    fn drop(&mut self) {
        unsafe {
            (self.functions.destroy_shader_module)(self.device, self.handle, ptr::null());
        }
    }
}

struct DescriptorSetLayoutHandle {
    device: VkDevice,
    functions: VulkanDeviceFunctions,
    handle: VkDescriptorSetLayout,
}

impl Drop for DescriptorSetLayoutHandle {
    fn drop(&mut self) {
        unsafe {
            (self.functions.destroy_descriptor_set_layout)(self.device, self.handle, ptr::null());
        }
    }
}

struct PipelineLayoutHandle {
    device: VkDevice,
    functions: VulkanDeviceFunctions,
    handle: VkPipelineLayout,
}

impl Drop for PipelineLayoutHandle {
    fn drop(&mut self) {
        unsafe {
            (self.functions.destroy_pipeline_layout)(self.device, self.handle, ptr::null());
        }
    }
}

struct PipelineHandle {
    device: VkDevice,
    functions: VulkanDeviceFunctions,
    handle: VkPipeline,
}

impl Drop for PipelineHandle {
    fn drop(&mut self) {
        unsafe {
            (self.functions.destroy_pipeline)(self.device, self.handle, ptr::null());
        }
    }
}

struct DescriptorPoolHandle {
    device: VkDevice,
    functions: VulkanDeviceFunctions,
    handle: VkDescriptorPool,
}

impl Drop for DescriptorPoolHandle {
    fn drop(&mut self) {
        unsafe {
            (self.functions.destroy_descriptor_pool)(self.device, self.handle, ptr::null());
        }
    }
}

struct CommandPoolHandle {
    device: VkDevice,
    functions: VulkanDeviceFunctions,
    handle: VkCommandPool,
}

impl Drop for CommandPoolHandle {
    fn drop(&mut self) {
        unsafe {
            (self.functions.destroy_command_pool)(self.device, self.handle, ptr::null());
        }
    }
}

struct FenceHandle {
    device: VkDevice,
    functions: VulkanDeviceFunctions,
    handle: VkFence,
}

impl Drop for FenceHandle {
    fn drop(&mut self) {
        unsafe {
            (self.functions.destroy_fence)(self.device, self.handle, ptr::null());
        }
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            (self.destroy_instance)(self.instance, ptr::null());
        }
    }
}
