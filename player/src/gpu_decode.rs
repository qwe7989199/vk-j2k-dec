use std::io::Write;
use std::path::PathBuf;

use vk_j2k_dec::decode::{DecodedRgb48, DecodeError, ProfiledDecodedRgb48};
use vk_j2k_dec::vulkan::VulkanRgb48Pipeline;

const IDWT53_SPV: &[u8] = include_bytes!("../../shaders/idwt53_1d.comp.spv");
const COLOR_SPV: &[u8] = include_bytes!("../../shaders/color_convert.comp.spv");

pub struct GpuDecoder {
    pipeline: VulkanRgb48Pipeline,
    _temp_dir: PathBuf,
}

impl GpuDecoder {
    pub fn new() -> Result<Self, String> {
        let temp_dir = std::env::temp_dir().join("vk-j2k-player-shaders");
        std::fs::create_dir_all(&temp_dir)
            .map_err(|e| format!("create temp dir: {e}"))?;

        let idwt53_path = temp_dir.join("idwt53_1d.comp.spv");
        let color_path = temp_dir.join("color_convert.comp.spv");

        std::fs::File::create(&idwt53_path)
            .and_then(|mut f| f.write_all(IDWT53_SPV))
            .map_err(|e| format!("write idwt53 shader: {e}"))?;

        std::fs::File::create(&color_path)
            .and_then(|mut f| f.write_all(COLOR_SPV))
            .map_err(|e| format!("write color shader: {e}"))?;

        let pipeline = VulkanRgb48Pipeline::new(
            idwt53_path.to_str().unwrap_or(""),
            color_path.to_str().unwrap_or(""),
        ).map_err(|e| format!("create GPU pipeline: {e}"))?;

        eprintln!("vk-j2k-player: GPU decode pipeline ready");

        Ok(Self { pipeline, _temp_dir: temp_dir })
    }

    pub fn decode(&self, data: &[u8]) -> Result<DecodedRgb48, DecodeError> {
        vk_j2k_dec::decode::decode_rgb48_with_persistent_gpu_pipeline_profiled(
            data, 0, &self.pipeline,
        ).map(|profiled| profiled.image)
    }
}
