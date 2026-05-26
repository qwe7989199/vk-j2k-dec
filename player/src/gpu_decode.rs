use std::io::Write;
use std::path::PathBuf;

use vk_j2k_dec::decode::{DecodedRgb48, DecodeError};

const CLEANUP_SPV: &[u8] = include_bytes!("../../shaders/tier1_cleanup_smoke.comp.spv");
const SCATTER_SPV: &[u8] = include_bytes!("../../shaders/tier1_scatter_smoke.comp.spv");
const IDWT53_OFFSET_SPV: &[u8] = include_bytes!("../../shaders/idwt53_offset_1d.comp.spv");
const COLOR_I32_SPV: &[u8] = include_bytes!("../../shaders/color_convert_i32.comp.spv");

pub struct GpuDecoder {
    cleanup_path: PathBuf,
    scatter_path: PathBuf,
    idwt_path: PathBuf,
    color_path: PathBuf,
    _temp_dir: PathBuf,
}

impl GpuDecoder {
    pub fn new() -> Result<Self, String> {
        let temp_dir = std::env::temp_dir().join("vk-j2k-player-shaders");
        std::fs::create_dir_all(&temp_dir)
            .map_err(|e| format!("create temp dir: {e}"))?;

        let cleanup_path = temp_dir.join("tier1_cleanup_smoke.comp.spv");
        let scatter_path = temp_dir.join("tier1_scatter_smoke.comp.spv");
        let idwt_path = temp_dir.join("idwt53_offset_1d.comp.spv");
        let color_path = temp_dir.join("color_convert_i32.comp.spv");

        write_spv(&cleanup_path, CLEANUP_SPV)?;
        write_spv(&scatter_path, SCATTER_SPV)?;
        write_spv(&idwt_path, IDWT53_OFFSET_SPV)?;
        write_spv(&color_path, COLOR_I32_SPV)?;

        eprintln!("vk-j2k-player: full GPU Tier-1 decode pipeline ready");

        Ok(Self { cleanup_path, scatter_path, idwt_path, color_path, _temp_dir: temp_dir })
    }

    pub fn decode(&self, data: &[u8]) -> Result<DecodedRgb48, DecodeError> {
        vk_j2k_dec::decode::decode_rgb48_full_gpu(
            data,
            self.cleanup_path.to_str().unwrap_or(""),
            self.scatter_path.to_str().unwrap_or(""),
            self.idwt_path.to_str().unwrap_or(""),
            self.color_path.to_str().unwrap_or(""),
        )
    }
}

fn write_spv(path: &PathBuf, data: &[u8]) -> Result<(), String> {
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(data))
        .map_err(|e| format!("write {}: {e}", path.display()))
}
