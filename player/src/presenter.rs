use std::ffi::CStr;

use ash::vk;

const PRESENT_SHADER_SPV: &[u8] = include_bytes!("../../shaders/present_rgb48.comp.spv");

pub struct VulkanPresenter {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    pub hdr_active: bool,
    image_available_sem: vk::Semaphore,
    render_finished_sem: vk::Semaphore,
    in_flight_fence: vk::Fence,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    rgb48_buffer: vk::Buffer,
    rgb48_memory: vk::DeviceMemory,
    rgb48_size: u64,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PresentPushConstants {
    pub width: u32,
    pub height: u32,
    pub transfer_func: u32,
    pub _pad: u32,
    pub color_matrix: [f32; 9],
    pub offset: [f32; 3],
    pub input_scale: f32,
}

impl VulkanPresenter {
    pub unsafe fn new(
        window: &winit::window::Window,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

        let entry = ash::Entry::linked();

        let app_info = vk::ApplicationInfo::default()
            .application_name(CStr::from_bytes_with_nul(b"vk-j2k-player\0").unwrap())
            .api_version(vk::make_api_version(0, 1, 2, 0));

        let display_handle = window.display_handle().map_err(|e| e.to_string())?;
        let window_handle = window.window_handle().map_err(|e| e.to_string())?;

        let mut instance_extensions =
            ash_window::enumerate_required_extensions(display_handle.as_raw())
                .map_err(|e| format!("enumerate extensions: {e}"))?
                .to_vec();

        // Add HDR extension if available
        instance_extensions.push(ash::khr::get_surface_capabilities2::NAME.as_ptr());

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);

        let instance = entry
            .create_instance(&instance_info, None)
            .map_err(|e| format!("create instance: {e}"))?;

        let surface = ash_window::create_surface(
            &entry,
            &instance,
            display_handle.as_raw(),
            window_handle.as_raw(),
            None,
        )
        .map_err(|e| format!("create surface: {e}"))?;

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        let (physical_device, queue_family) =
            select_device(&instance, &surface_loader, surface)?;

        let queue_priority = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priority);

        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extensions);

        let device = instance
            .create_device(physical_device, &device_info, None)
            .map_err(|e| format!("create device: {e}"))?;

        let queue = device.get_device_queue(queue_family, 0);
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

        let (swapchain, swapchain_images, swapchain_views, swapchain_format, swapchain_extent, hdr_active) =
            create_swapchain(
                &instance,
                &device,
                &surface_loader,
                &swapchain_loader,
                physical_device,
                surface,
                width,
                height,
                vk::SwapchainKHR::null(),
            )?;

        if hdr_active {
            eprintln!("vk-j2k-player: HDR10 swapchain active (format {:?})", swapchain_format);
        } else {
            eprintln!("vk-j2k-player: SDR swapchain (format {:?})", swapchain_format);
        }

        let sem_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let image_available_sem = device.create_semaphore(&sem_info, None).map_err(|e| e.to_string())?;
        let render_finished_sem = device.create_semaphore(&sem_info, None).map_err(|e| e.to_string())?;
        let in_flight_fence = device.create_fence(&fence_info, None).map_err(|e| e.to_string())?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = device.create_command_pool(&pool_info, None).map_err(|e| e.to_string())?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = device.allocate_command_buffers(&alloc_info).map_err(|e| e.to_string())?[0];

        let rgb48_size = (width as u64) * (height as u64) * 3 * 2;
        let (rgb48_buffer, rgb48_memory) = create_host_visible_buffer(
            &instance, &device, physical_device, rgb48_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        let (descriptor_set_layout, descriptor_pool, descriptor_sets, pipeline_layout, pipeline) =
            create_present_pipeline(&device, &swapchain_views, rgb48_buffer, rgb48_size)?;

        Ok(Self {
            _entry: entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            queue,
            queue_family,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_views,
            swapchain_format,
            swapchain_extent,
            hdr_active,
            image_available_sem,
            render_finished_sem,
            in_flight_fence,
            command_pool,
            command_buffer,
            rgb48_buffer,
            rgb48_memory,
            rgb48_size,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            pipeline_layout,
            pipeline,
            width,
            height,
        })
    }

    pub unsafe fn upload_rgb48(&self, samples: &[u16]) {
        let data_ptr = self.device
            .map_memory(self.rgb48_memory, 0, self.rgb48_size, vk::MemoryMapFlags::empty())
            .expect("map memory");
        let byte_count = (samples.len() * 2).min(self.rgb48_size as usize);
        std::ptr::copy_nonoverlapping(
            samples.as_ptr() as *const u8,
            data_ptr as *mut u8,
            byte_count,
        );
        self.device.unmap_memory(self.rgb48_memory);
    }

    pub unsafe fn present_frame(&self, push_constants: &PresentPushConstants) -> Result<bool, String> {
        self.device
            .wait_for_fences(&[self.in_flight_fence], true, u64::MAX)
            .map_err(|e| e.to_string())?;
        self.device.reset_fences(&[self.in_flight_fence]).map_err(|e| e.to_string())?;

        let acquire_result = self.swapchain_loader.acquire_next_image(
            self.swapchain,
            u64::MAX,
            self.image_available_sem,
            vk::Fence::null(),
        );

        let image_index = match acquire_result {
            Ok((index, _)) => index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
            Err(e) => return Err(format!("acquire image: {e}")),
        };

        self.device
            .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
            .map_err(|e| e.to_string())?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.begin_command_buffer(self.command_buffer, &begin_info).map_err(|e| e.to_string())?;

        // Transition swapchain image to GENERAL for compute write
        let barrier_to_general = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(self.swapchain_images[image_index as usize])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        self.device.cmd_pipeline_barrier(
            self.command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier_to_general],
        );

        // Bind pipeline and descriptors
        self.device.cmd_bind_pipeline(
            self.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline,
        );
        self.device.cmd_bind_descriptor_sets(
            self.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            0,
            &[self.descriptor_sets[image_index as usize]],
            &[],
        );

        let pc_bytes: &[u8] = std::slice::from_raw_parts(
            push_constants as *const PresentPushConstants as *const u8,
            std::mem::size_of::<PresentPushConstants>(),
        );
        self.device.cmd_push_constants(
            self.command_buffer,
            self.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );

        let groups_x = (self.width + 15) / 16;
        let groups_y = (self.height + 15) / 16;
        self.device.cmd_dispatch(self.command_buffer, groups_x, groups_y, 1);

        // Transition swapchain image to PRESENT_SRC
        let barrier_to_present = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(self.swapchain_images[image_index as usize])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        self.device.cmd_pipeline_barrier(
            self.command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier_to_present],
        );

        self.device.end_command_buffer(self.command_buffer).map_err(|e| e.to_string())?;

        let wait_semaphores = [self.image_available_sem];
        let wait_stages = [vk::PipelineStageFlags::COMPUTE_SHADER];
        let signal_semaphores = [self.render_finished_sem];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&self.command_buffer))
            .signal_semaphores(&signal_semaphores);

        self.device
            .queue_submit(self.queue, &[submit_info], self.in_flight_fence)
            .map_err(|e| format!("queue submit: {e}"))?;

        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        match self.swapchain_loader.queue_present(self.queue, &present_info) {
            Ok(_) => Ok(false),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => Ok(true),
            Err(e) => Err(format!("queue present: {e}")),
        }
    }
}

impl Drop for VulkanPresenter {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_buffer(self.rgb48_buffer, None);
            self.device.free_memory(self.rgb48_memory, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_fence(self.in_flight_fence, None);
            self.device.destroy_semaphore(self.render_finished_sem, None);
            self.device.destroy_semaphore(self.image_available_sem, None);
            for &view in &self.swapchain_views {
                self.device.destroy_image_view(view, None);
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe fn select_device(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32), String> {
    let devices = instance.enumerate_physical_devices().map_err(|e| e.to_string())?;

    for &device in &devices {
        let props = instance.get_physical_device_queue_family_properties(device);
        for (i, family) in props.iter().enumerate() {
            let supports_graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let supports_compute = family.queue_flags.contains(vk::QueueFlags::COMPUTE);
            let supports_surface = surface_loader
                .get_physical_device_surface_support(device, i as u32, surface)
                .unwrap_or(false);

            if supports_graphics && supports_compute && supports_surface {
                let dev_props = instance.get_physical_device_properties(device);
                let name = CStr::from_ptr(dev_props.device_name.as_ptr()).to_string_lossy();
                eprintln!("vk-j2k-player: using device {name}, queue family {i}");
                return Ok((device, i as u32));
            }
        }
    }
    Err("no suitable Vulkan device found".to_string())
}

unsafe fn create_swapchain(
    instance: &ash::Instance,
    device: &ash::Device,
    surface_loader: &ash::khr::surface::Instance,
    swapchain_loader: &ash::khr::swapchain::Device,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    width: u32,
    height: u32,
    old_swapchain: vk::SwapchainKHR,
) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, Vec<vk::ImageView>, vk::Format, vk::Extent2D, bool), String> {
    let caps = surface_loader
        .get_physical_device_surface_capabilities(physical_device, surface)
        .map_err(|e| e.to_string())?;
    let formats = surface_loader
        .get_physical_device_surface_formats(physical_device, surface)
        .map_err(|e| e.to_string())?;

    // Try HDR10 format first
    let hdr_format = formats.iter().find(|f| {
        f.format == vk::Format::A2B10G10R10_UNORM_PACK32
            && f.color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT
    });

    let (chosen_format, hdr_active) = if let Some(hdr) = hdr_format {
        (*hdr, true)
    } else {
        let sdr = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| formats.first())
            .copied()
            .ok_or("no surface format available")?;
        (sdr, false)
    };

    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };

    let image_count = (caps.min_image_count + 1).min(
        if caps.max_image_count == 0 { u32::MAX } else { caps.max_image_count }
    );

    let supported_usage = caps.supported_usage_flags;
    let use_storage = supported_usage.contains(vk::ImageUsageFlags::STORAGE);
    if !use_storage {
        eprintln!("vk-j2k-player: WARNING — swapchain does not support STORAGE usage, using TRANSFER_DST fallback");
    }

    let image_usage = if use_storage {
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT
    } else {
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT
    };

    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(chosen_format.format)
        .image_color_space(chosen_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(image_usage)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(old_swapchain);

    let swapchain = swapchain_loader
        .create_swapchain(&create_info, None)
        .map_err(|e| format!("create swapchain: {e}"))?;

    let images = swapchain_loader
        .get_swapchain_images(swapchain)
        .map_err(|e| e.to_string())?;

    let views: Vec<vk::ImageView> = images
        .iter()
        .map(|&image| {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(chosen_format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            device.create_image_view(&view_info, None).expect("create image view")
        })
        .collect();

    Ok((swapchain, images, views, chosen_format.format, extent, hdr_active))
}

unsafe fn create_host_visible_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&buffer_info, None).map_err(|e| e.to_string())?;
    let mem_reqs = device.get_buffer_memory_requirements(buffer);

    let mem_props = instance.get_physical_device_memory_properties(physical_device);
    let mem_type = find_memory_type(
        &mem_props,
        mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    ).ok_or("no suitable memory type")?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(mem_type);

    let memory = device.allocate_memory(&alloc_info, None).map_err(|e| e.to_string())?;
    device.bind_buffer_memory(buffer, memory, 0).map_err(|e| e.to_string())?;

    Ok((buffer, memory))
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..props.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && props.memory_types[i as usize].property_flags.contains(required)
        {
            return Some(i);
        }
    }
    None
}

unsafe fn create_present_pipeline(
    device: &ash::Device,
    swapchain_views: &[vk::ImageView],
    rgb48_buffer: vk::Buffer,
    rgb48_size: u64,
) -> Result<(vk::DescriptorSetLayout, vk::DescriptorPool, Vec<vk::DescriptorSet>, vk::PipelineLayout, vk::Pipeline), String> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let set_layout = device.create_descriptor_set_layout(&layout_info, None).map_err(|e| e.to_string())?;

    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<PresentPushConstants>() as u32);

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&set_layout))
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));
    let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None).map_err(|e| e.to_string())?;

    let shader_code = ash::util::read_spv(&mut std::io::Cursor::new(PRESENT_SHADER_SPV))
        .map_err(|e| format!("read spv: {e}"))?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
    let shader_module = device.create_shader_module(&shader_info, None).map_err(|e| e.to_string())?;

    let entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(entry_name);

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(pipeline_layout);

    let pipelines = device
        .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        .map_err(|(_pipelines, e)| format!("create pipeline: {e}"))?;
    let pipeline = pipelines[0];

    device.destroy_shader_module(shader_module, None);

    // Descriptor pool and sets
    let image_count = swapchain_views.len() as u32;
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(image_count),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(image_count),
    ];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(image_count)
        .pool_sizes(&pool_sizes);
    let descriptor_pool = device.create_descriptor_pool(&pool_info, None).map_err(|e| e.to_string())?;

    let layouts: Vec<vk::DescriptorSetLayout> = vec![set_layout; image_count as usize];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    let descriptor_sets = device.allocate_descriptor_sets(&alloc_info).map_err(|e| e.to_string())?;

    for (i, &set) in descriptor_sets.iter().enumerate() {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(rgb48_buffer)
            .offset(0)
            .range(rgb48_size);

        let image_info = vk::DescriptorImageInfo::default()
            .image_view(swapchain_views[i])
            .image_layout(vk::ImageLayout::GENERAL);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_info)),
        ];

        device.update_descriptor_sets(&writes, &[]);
    }

    Ok((set_layout, descriptor_pool, descriptor_sets, pipeline_layout, pipeline))
}
