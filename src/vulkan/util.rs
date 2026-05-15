fn read_spirv(path: &str) -> Result<Vec<u32>, VulkanError> {
    let bytes = fs::read(path)
        .map_err(|err| VulkanError::new(format!("failed to read SPIR-V file {path}: {err}")))?;
    if bytes.len() % 4 != 0 {
        return Err(VulkanError::new(format!(
            "SPIR-V file {path} has non-u32 byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn bytes_len<T>(count: usize) -> Result<u64, VulkanError> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .map(|value| value as u64)
        .ok_or_else(|| VulkanError::new("buffer byte length overflow"))
}

fn bytes_of<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn vk_check(status: i32, name: &str) -> Result<(), VulkanError> {
    if status == VK_SUCCESS {
        Ok(())
    } else {
        Err(VulkanError::new(format!(
            "{name} failed with status {status}"
        )))
    }
}

fn make_api_version(major: u32, minor: u32, patch: u32) -> u32 {
    (major << 22) | (minor << 12) | patch
}

fn format_api_version(version: u32) -> String {
    let major = version >> 22;
    let minor = (version >> 12) & 0x03ff;
    let patch = version & 0x0fff;
    format!("{major}.{minor}.{patch}")
}
