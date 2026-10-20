// The bridge is optional and loaded only for explicit NVENC benchmark requests.
#[repr(C)]
struct NvencSharedBuffer {
    memory: u64,
    allocation_size: u64,
    size: u64,
}
type NvencCreate = unsafe extern "C" fn(
    usize,
    usize,
    usize,
    *const NvencSharedBuffer,
    u32,
    u32,
    u32,
    *const c_char,
    *const c_char,
    *const c_char,
    i32,
) -> *mut c_void;
type NvencSubmit = unsafe extern "C" fn(*mut c_void, u32, u32, i64, i32) -> i32;
type NvencFinish = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64, *mut u64) -> i32;
type NvencDestroy = unsafe extern "C" fn(*mut c_void);
type NvencError = unsafe extern "C" fn() -> *const c_char;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
}

struct NvencEncoder {
    session: *mut c_void,
    module: *mut c_void,
    submit: NvencSubmit,
    finish: NvencFinish,
    destroy: NvencDestroy,
    error: NvencError,
    verify: bool,
}

impl NvencEncoder {
    fn error(&self) -> VulkanError {
        VulkanError::new(
            unsafe { CStr::from_ptr((self.error)()) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

impl Drop for NvencEncoder {
    fn drop(&mut self) {
        unsafe {
            if !self.session.is_null() {
                (self.destroy)(self.session);
            }
            FreeLibrary(self.module);
        }
    }
}

pub struct NvencResult {
    pub frames: u64,
    pub packets: u64,
    pub frames_hash: Option<u64>,
}

impl VulkanTier1Rgb48Pipeline {
    /// Initializes H.264 encoding of exported GPU buffers. The bridge uses UUID matching,
    /// imports each allocation once, and owns the imports for the pipeline's lifetime.
    pub fn start_nvenc(
        &mut self,
        output: &str,
        preset: &str,
        qp: i32,
        verify: bool,
    ) -> Result<(), VulkanError> {
        if !self.export_rgb48
            || self.nvenc.is_some()
            || self.slots.iter().any(|s| s.pending_frames != 0)
        {
            return Err(VulkanError::new(
                "NVENC requires an idle export_rgb48 pipeline without an encoder",
            ));
        }
        if !(0..=51).contains(&qp) {
            return Err(VulkanError::new("NVENC QP must be 0..51"));
        }
        let base = fs::canonicalize("target/nvenc").map_err(|e| {
            VulkanError::new(format!("build scripts/build-nvenc-bridge.ps1 first: {e}"))
        })?;
        let dll = base.join("nvenc_bridge.dll");
        let path: Vec<u16> = dll
            .as_os_str()
            .to_string_lossy()
            .encode_utf16()
            .chain(Some(0))
            .collect();
        // Search dependencies in this DLL's directory and trusted default directories.
        let module = unsafe { LoadLibraryExW(path.as_ptr(), ptr::null_mut(), 0x100 | 0x1000) };
        if module.is_null() {
            return Err(VulkanError::new(format!(
                "cannot load {}: {}",
                dll.display(),
                std::io::Error::last_os_error()
            )));
        }
        let load = |name: &CStr| -> Result<*mut c_void, VulkanError> {
            let p = unsafe { GetProcAddress(module, name.as_ptr()) };
            if p.is_null() {
                Err(VulkanError::new(format!(
                    "missing bridge symbol {}",
                    name.to_string_lossy()
                )))
            } else {
                Ok(p)
            }
        };
        let symbols = (|| unsafe {
            Ok::<_, VulkanError>((
                std::mem::transmute::<*mut c_void, NvencCreate>(load(c"nvenc_create")?),
                std::mem::transmute::<*mut c_void, NvencSubmit>(load(c"nvenc_submit_batch")?),
                std::mem::transmute::<*mut c_void, NvencFinish>(load(c"nvenc_finish")?),
                std::mem::transmute::<*mut c_void, NvencDestroy>(load(c"nvenc_destroy")?),
                std::mem::transmute::<*mut c_void, NvencError>(load(c"nvenc_error")?),
            ))
        })();
        let (create, submit, finish, destroy, error) = match symbols {
            Ok(s) => s,
            Err(e) => {
                unsafe {
                    FreeLibrary(module);
                }
                return Err(e);
            }
        };
        let mut encoder = NvencEncoder {
            session: ptr::null_mut(),
            module,
            submit,
            finish,
            destroy,
            error,
            verify,
        };
        let buffers = self
            .slots
            .iter()
            .flat_map(|slot| slot.frames.iter())
            .map(|frame| {
                let b = &frame.rgb48_packer.output_buffer;
                NvencSharedBuffer {
                    memory: b.memory as u64,
                    allocation_size: b.allocation_size,
                    size: b.size,
                }
            })
            .collect::<Vec<_>>();
        let (width, height) = match &self.slots[0].frames[0].idwt {
            FrameIdwt::Fused(levels) => (levels.width, levels.height),
            FrameIdwt::Separable(components) => (components[0].width, components[0].height),
        };
        let cstring = |s: &str| {
            CString::new(s).map_err(|_| VulkanError::new("NVENC argument contains a NUL byte"))
        };
        let output = cstring(output)?;
        let preset = cstring(preset)?;
        // Strip the Windows canonicalization prefix: CUDA's module loader expects a normal path.
        let ptx_path = base
            .join("rgb48_to_nv12.ptx")
            .to_string_lossy()
            .trim_start_matches(r"\\?\")
            .to_string();
        let ptx = cstring(&ptx_path)?;
        encoder.session = unsafe {
            create(
                self.context.instance.instance,
                self.context.physical_device,
                self.context.device.device,
                buffers.as_ptr(),
                buffers.len() as u32,
                width,
                height,
                output.as_ptr(),
                ptx.as_ptr(),
                preset.as_ptr(),
                qp,
            )
        };
        if encoder.session.is_null() {
            return Err(encoder.error());
        }
        self.nvenc = Some(encoder);
        Ok(())
    }

    /// Waits for Vulkan's release fence, converts shared RGB48 to CUDA NV12 and encodes.
    /// Each conversion completes before this slot can be reused; NVENC retains separate
    /// CUDA NV12 frame references until it has finished consuming them.
    pub fn wait_and_encode_batch(
        &mut self,
        slot_index: usize,
        first_frame: usize,
    ) -> Result<f64, VulkanError> {
        let encoder = self
            .nvenc
            .as_ref()
            .ok_or_else(|| VulkanError::new("NVENC is not initialized"))?;
        let slot = self
            .slots
            .get_mut(slot_index)
            .ok_or_else(|| VulkanError::new("invalid NVENC slot"))?;
        if slot.pending_frames == 0 {
            return Err(VulkanError::new("no NVENC batch pending"));
        }
        let start = std::time::Instant::now();
        self.context.device.wait_fence(slot.fence.handle)?;
        let wait_ms = start.elapsed().as_secs_f64() * 1000.0;
        if let Some(pool) = slot.query_pool.as_ref() {
            let ticks = self
                .context
                .device
                .read_timestamps(pool, TIER1_PIPELINE_TIMESTAMP_QUERIES)?;
            slot.last_stage_times = Some(Tier1GpuStageTimes::from_ticks(
                &ticks,
                self.timestamp_mask,
                self.timestamp_period_ns,
                None,
            ));
        }
        let index = slot_index * self.max_batch_frames;
        let status = unsafe {
            (encoder.submit)(
                encoder.session,
                index as u32,
                slot.pending_frames as u32,
                first_frame as i64,
                i32::from(encoder.verify),
            )
        };
        if status != 0 {
            return Err(encoder.error());
        }
        slot.pending_frames = 0;
        Ok(wait_ms)
    }

    /// Flushes delayed encoder packets and finalizes the MP4 container.
    pub fn finish_nvenc(&mut self) -> Result<NvencResult, VulkanError> {
        if self.slots.iter().any(|s| s.pending_frames != 0) {
            return Err(VulkanError::new(
                "drain Vulkan slots before finishing NVENC",
            ));
        }
        let encoder = self
            .nvenc
            .as_ref()
            .ok_or_else(|| VulkanError::new("NVENC is not initialized"))?;
        let (mut frames, mut packets, mut hash) = (0, 0, 0);
        let status =
            unsafe { (encoder.finish)(encoder.session, &mut frames, &mut packets, &mut hash) };
        if status != 0 {
            return Err(encoder.error());
        }
        Ok(NvencResult {
            frames,
            packets,
            frames_hash: encoder.verify.then_some(hash),
        })
    }
}
