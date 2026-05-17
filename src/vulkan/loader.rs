struct VulkanLoader {
    module: *mut c_void,
    get_instance_proc_addr: VkGetInstanceProcAddr,
}

impl VulkanLoader {
    fn open() -> Result<Self, VulkanError> {
        let dll = CString::new("vulkan-1.dll").expect("static string");
        let module = unsafe { LoadLibraryA(dll.as_ptr()) };
        if module.is_null() {
            return Err(VulkanError::new("failed to load vulkan-1.dll"));
        }
        let name = CString::new("vkGetInstanceProcAddr").expect("static string");
        let pointer = unsafe { GetProcAddress(module, name.as_ptr()) };
        if pointer.is_null() {
            unsafe {
                FreeLibrary(module);
            }
            return Err(VulkanError::new(
                "failed to load vkGetInstanceProcAddr from vulkan-1.dll",
            ));
        }
        Ok(Self {
            module,
            get_instance_proc_addr: unsafe {
                std::mem::transmute::<*mut c_void, VkGetInstanceProcAddr>(pointer)
            },
        })
    }

    fn load<T>(&self, instance: VkInstance, name: &str) -> Result<T, VulkanError>
    where
        T: Copy,
    {
        let name = CString::new(name).expect("Vulkan function name");
        let pointer = unsafe { (self.get_instance_proc_addr)(instance, name.as_ptr()) }
            .ok_or_else(|| {
                VulkanError::new(format!("failed to load {}", name.to_string_lossy()))
            })?;
        Ok(unsafe { std::mem::transmute_copy::<unsafe extern "system" fn(), T>(&pointer) })
    }

    fn create_instance(&self) -> Result<VulkanInstance, VulkanError> {
        let vk_create_instance = self.load::<VkCreateInstance>(0, "vkCreateInstance")?;
        let app_name = CString::new("vk-j2k-dec").expect("static string");
        let app_info = VkApplicationInfo {
            s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            p_next: ptr::null(),
            p_application_name: app_name.as_ptr(),
            application_version: 1,
            p_engine_name: app_name.as_ptr(),
            engine_version: 1,
            api_version: make_api_version(1, 0, 0),
        };
        let create_info = VkInstanceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            p_application_info: &app_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: ptr::null(),
        };
        let mut instance = 0usize;
        let status = unsafe { vk_create_instance(&create_info, ptr::null(), &mut instance) };
        if status != VK_SUCCESS {
            return Err(VulkanError::new(format!(
                "vkCreateInstance failed with status {status}"
            )));
        }

        Ok(VulkanInstance {
            instance,
            destroy_instance: self.load(instance, "vkDestroyInstance")?,
            get_device_proc_addr: self.load(instance, "vkGetDeviceProcAddr")?,
            enumerate_physical_devices: self.load(instance, "vkEnumeratePhysicalDevices")?,
            get_physical_device_properties: self.load(instance, "vkGetPhysicalDeviceProperties")?,
            get_physical_device_queue_family_properties: self
                .load(instance, "vkGetPhysicalDeviceQueueFamilyProperties")?,
            get_physical_device_memory_properties: self
                .load(instance, "vkGetPhysicalDeviceMemoryProperties")?,
            create_device: self.load(instance, "vkCreateDevice")?,
        })
    }
}

impl Drop for VulkanLoader {
    fn drop(&mut self) {
        unsafe {
            FreeLibrary(self.module);
        }
    }
}

struct VulkanInstance {
    instance: VkInstance,
    destroy_instance: VkDestroyInstance,
    get_device_proc_addr: VkGetDeviceProcAddr,
    enumerate_physical_devices: VkEnumeratePhysicalDevices,
    get_physical_device_properties: VkGetPhysicalDeviceProperties,
    get_physical_device_queue_family_properties: VkGetPhysicalDeviceQueueFamilyProperties,
    get_physical_device_memory_properties: VkGetPhysicalDeviceMemoryProperties,
    create_device: VkCreateDevice,
}

impl VulkanInstance {
    fn load_device<T>(&self, device: VkDevice, name: &str) -> Result<T, VulkanError>
    where
        T: Copy,
    {
        let name = CString::new(name).expect("Vulkan device function name");
        let pointer =
            unsafe { (self.get_device_proc_addr)(device, name.as_ptr()) }.ok_or_else(|| {
                VulkanError::new(format!(
                    "failed to load device function {}",
                    name.to_string_lossy()
                ))
            })?;
        Ok(unsafe { std::mem::transmute_copy::<unsafe extern "system" fn(), T>(&pointer) })
    }

    fn enumerate_devices(&self) -> Result<Vec<VulkanDeviceSummary>, VulkanError> {
        let mut count = 0u32;
        let status = unsafe {
            (self.enumerate_physical_devices)(self.instance, &mut count, ptr::null_mut())
        };
        if status != VK_SUCCESS {
            return Err(VulkanError::new(format!(
                "vkEnumeratePhysicalDevices failed with status {status}"
            )));
        }
        let mut devices = vec![0usize; count as usize];
        let status = unsafe {
            (self.enumerate_physical_devices)(self.instance, &mut count, devices.as_mut_ptr())
        };
        if status != VK_SUCCESS {
            return Err(VulkanError::new(format!(
                "vkEnumeratePhysicalDevices failed with status {status}"
            )));
        }

        devices
            .into_iter()
            .take(count as usize)
            .map(|device| self.device_summary(device))
            .collect()
    }

    fn device_summary(&self, device: VkPhysicalDevice) -> Result<VulkanDeviceSummary, VulkanError> {
        let mut properties = unsafe { std::mem::zeroed::<VkPhysicalDeviceProperties>() };
        unsafe {
            (self.get_physical_device_properties)(device, &mut properties);
        }
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let mut queue_count = 0u32;
        unsafe {
            (self.get_physical_device_queue_family_properties)(
                device,
                &mut queue_count,
                ptr::null_mut(),
            );
        }
        let mut queue_families =
            vec![unsafe { std::mem::zeroed::<VkQueueFamilyProperties>() }; queue_count as usize];
        unsafe {
            (self.get_physical_device_queue_family_properties)(
                device,
                &mut queue_count,
                queue_families.as_mut_ptr(),
            );
        }

        Ok(VulkanDeviceSummary {
            name,
            api_version: format_api_version(properties.api_version),
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            device_type: VulkanDeviceType::from(properties.device_type),
            queue_families: queue_families
                .into_iter()
                .take(queue_count as usize)
                .enumerate()
                .map(|(index, family)| VulkanQueueFamilySummary {
                    index: index as u32,
                    queue_count: family.queue_count,
                    graphics: family.queue_flags & VK_QUEUE_GRAPHICS_BIT != 0,
                    compute: family.queue_flags & VK_QUEUE_COMPUTE_BIT != 0,
                    transfer: family.queue_flags & VK_QUEUE_TRANSFER_BIT != 0,
                })
                .collect(),
        })
    }

    fn select_compute_device(
        &self,
    ) -> Result<(VkPhysicalDevice, VulkanDeviceSummary), VulkanError> {
        let mut count = 0u32;
        let status = unsafe {
            (self.enumerate_physical_devices)(self.instance, &mut count, ptr::null_mut())
        };
        if status != VK_SUCCESS {
            return Err(VulkanError::new(format!(
                "vkEnumeratePhysicalDevices failed with status {status}"
            )));
        }
        let mut devices = vec![0usize; count as usize];
        let status = unsafe {
            (self.enumerate_physical_devices)(self.instance, &mut count, devices.as_mut_ptr())
        };
        if status != VK_SUCCESS {
            return Err(VulkanError::new(format!(
                "vkEnumeratePhysicalDevices failed with status {status}"
            )));
        }

        devices
            .into_iter()
            .take(count as usize)
            .filter_map(|device| {
                let summary = self.device_summary(device).ok()?;
                if summary.first_compute_queue_family().is_some() {
                    Some((device, summary))
                } else {
                    None
                }
            })
            .min_by_key(|(_, summary)| summary.device_type != VulkanDeviceType::DiscreteGpu)
            .ok_or_else(|| VulkanError::new("no Vulkan device with compute queue found"))
    }

    fn create_compute_device(
        &self,
        physical_device: VkPhysicalDevice,
        queue_family_index: u32,
    ) -> Result<VulkanDevice, VulkanError> {
        let priority = 1.0f32;
        let queue_create_info = VkDeviceQueueCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: &priority,
        };
        let create_info = VkDeviceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: ptr::null(),
            p_enabled_features: ptr::null(),
        };
        let mut device = 0usize;
        let status = unsafe {
            (self.create_device)(physical_device, &create_info, ptr::null(), &mut device)
        };
        if status != VK_SUCCESS {
            return Err(VulkanError::new(format!(
                "vkCreateDevice failed with status {status}"
            )));
        }

        let functions = VulkanDeviceFunctions {
            destroy_device: self.load_device(device, "vkDestroyDevice")?,
            get_device_queue: self.load_device(device, "vkGetDeviceQueue")?,
            create_buffer: self.load_device(device, "vkCreateBuffer")?,
            destroy_buffer: self.load_device(device, "vkDestroyBuffer")?,
            get_buffer_memory_requirements: self
                .load_device(device, "vkGetBufferMemoryRequirements")?,
            allocate_memory: self.load_device(device, "vkAllocateMemory")?,
            free_memory: self.load_device(device, "vkFreeMemory")?,
            bind_buffer_memory: self.load_device(device, "vkBindBufferMemory")?,
            map_memory: self.load_device(device, "vkMapMemory")?,
            unmap_memory: self.load_device(device, "vkUnmapMemory")?,
            invalidate_mapped_memory_ranges: self
                .load_device(device, "vkInvalidateMappedMemoryRanges")?,
            create_shader_module: self.load_device(device, "vkCreateShaderModule")?,
            destroy_shader_module: self.load_device(device, "vkDestroyShaderModule")?,
            create_descriptor_set_layout: self
                .load_device(device, "vkCreateDescriptorSetLayout")?,
            destroy_descriptor_set_layout: self
                .load_device(device, "vkDestroyDescriptorSetLayout")?,
            create_pipeline_layout: self.load_device(device, "vkCreatePipelineLayout")?,
            destroy_pipeline_layout: self.load_device(device, "vkDestroyPipelineLayout")?,
            create_compute_pipelines: self.load_device(device, "vkCreateComputePipelines")?,
            destroy_pipeline: self.load_device(device, "vkDestroyPipeline")?,
            create_descriptor_pool: self.load_device(device, "vkCreateDescriptorPool")?,
            destroy_descriptor_pool: self.load_device(device, "vkDestroyDescriptorPool")?,
            allocate_descriptor_sets: self.load_device(device, "vkAllocateDescriptorSets")?,
            update_descriptor_sets: self.load_device(device, "vkUpdateDescriptorSets")?,
            create_command_pool: self.load_device(device, "vkCreateCommandPool")?,
            destroy_command_pool: self.load_device(device, "vkDestroyCommandPool")?,
            allocate_command_buffers: self.load_device(device, "vkAllocateCommandBuffers")?,
            begin_command_buffer: self.load_device(device, "vkBeginCommandBuffer")?,
            end_command_buffer: self.load_device(device, "vkEndCommandBuffer")?,
            cmd_bind_pipeline: self.load_device(device, "vkCmdBindPipeline")?,
            cmd_bind_descriptor_sets: self.load_device(device, "vkCmdBindDescriptorSets")?,
            cmd_push_constants: self.load_device(device, "vkCmdPushConstants")?,
            cmd_dispatch: self.load_device(device, "vkCmdDispatch")?,
            cmd_copy_buffer: self.load_device(device, "vkCmdCopyBuffer")?,
            create_fence: self.load_device(device, "vkCreateFence")?,
            destroy_fence: self.load_device(device, "vkDestroyFence")?,
            queue_submit: self.load_device(device, "vkQueueSubmit")?,
            wait_for_fences: self.load_device(device, "vkWaitForFences")?,
            device_wait_idle: self.load_device(device, "vkDeviceWaitIdle")?,
        };

        let mut queue = 0usize;
        unsafe {
            (functions.get_device_queue)(device, queue_family_index, 0, &mut queue);
        }
        let mut memory_properties =
            unsafe { std::mem::zeroed::<VkPhysicalDeviceMemoryProperties>() };
        unsafe {
            (self.get_physical_device_memory_properties)(physical_device, &mut memory_properties);
        }

        Ok(VulkanDevice {
            device,
            queue,
            functions,
            memory_properties,
        })
    }
}

