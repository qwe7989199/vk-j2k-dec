use std::io::Write;
use std::path::PathBuf;

use vk_j2k_dec::decode::{DecodeError, DecodedRgb48};

const CLEANUP_SPV: &[u8] = include_bytes!("../../shaders/tier1_cleanup_smoke.comp.spv");
const SCATTER_SPV: &[u8] = include_bytes!("../../shaders/tier1_scatter_smoke.comp.spv");
const DEQUANT_SPV: &[u8] = include_bytes!("../../shaders/dequant.comp.spv");
const IDWT53_OFFSET_SPV: &[u8] = include_bytes!("../../shaders/idwt53_offset_1d.comp.spv");
const IDWT97_OFFSET_SPV: &[u8] = include_bytes!("../../shaders/idwt97_offset_1d.comp.spv");
const COLOR_I32_SPV: &[u8] = include_bytes!("../../shaders/color_convert_i32.comp.spv");
const COLOR_FLOAT_SPV: &[u8] = include_bytes!("../../shaders/color_convert.comp.spv");

pub struct GpuDecoder {
    cleanup_path: PathBuf,
    scatter_path: PathBuf,
    dequant_path: PathBuf,
    idwt53_path: PathBuf,
    idwt97_path: PathBuf,
    color_i32_path: PathBuf,
    color_float_path: PathBuf,
    _temp_dir: PathBuf,
}

impl GpuDecoder {
    pub fn new() -> Result<Self, String> {
        let temp_dir = std::env::temp_dir().join("vk-j2k-player-shaders");
        std::fs::create_dir_all(&temp_dir).map_err(|e| format!("create temp dir: {e}"))?;

        let cleanup_path = temp_dir.join("tier1_cleanup_smoke.comp.spv");
        let scatter_path = temp_dir.join("tier1_scatter_smoke.comp.spv");
        let dequant_path = temp_dir.join("dequant.comp.spv");
        let idwt53_path = temp_dir.join("idwt53_offset_1d.comp.spv");
        let idwt97_path = temp_dir.join("idwt97_offset_1d.comp.spv");
        let color_i32_path = temp_dir.join("color_convert_i32.comp.spv");
        let color_float_path = temp_dir.join("color_convert.comp.spv");

        write_spv(&cleanup_path, CLEANUP_SPV)?;
        write_spv(&scatter_path, SCATTER_SPV)?;
        write_spv(&dequant_path, DEQUANT_SPV)?;
        write_spv(&idwt53_path, IDWT53_OFFSET_SPV)?;
        write_spv(&idwt97_path, IDWT97_OFFSET_SPV)?;
        write_spv(&color_i32_path, COLOR_I32_SPV)?;
        write_spv(&color_float_path, COLOR_FLOAT_SPV)?;

        eprintln!("vk-j2k-player: full GPU decode pipeline ready (5/3 + 9/7)");

        Ok(Self {
            cleanup_path,
            scatter_path,
            dequant_path,
            idwt53_path,
            idwt97_path,
            color_i32_path,
            color_float_path,
            _temp_dir: temp_dir,
        })
    }

    pub fn decode(&self, data: &[u8]) -> Result<DecodedRgb48, DecodeError> {
        let codestream = vk_j2k_dec::codestream::Codestream::parse(data)
            .map_err(|e| DecodeError::new(e.to_string()))?;
        let is_97 = codestream
            .cod
            .as_ref()
            .map(|c| {
                matches!(
                    c.transform,
                    vk_j2k_dec::codestream::WaveletTransform::Irreversible97
                )
            })
            .unwrap_or(false);

        if is_97 {
            vk_j2k_dec::decode::decode_rgb48_full_gpu_97(
                data,
                self.cleanup_path.to_str().unwrap_or(""),
                self.scatter_path.to_str().unwrap_or(""),
                self.dequant_path.to_str().unwrap_or(""),
                self.idwt97_path.to_str().unwrap_or(""),
                self.color_float_path.to_str().unwrap_or(""),
            )
        } else {
            vk_j2k_dec::decode::decode_rgb48_full_gpu(
                data,
                self.cleanup_path.to_str().unwrap_or(""),
                self.scatter_path.to_str().unwrap_or(""),
                self.idwt53_path.to_str().unwrap_or(""),
                self.color_i32_path.to_str().unwrap_or(""),
            )
        }
    }
}

fn write_spv(path: &PathBuf, data: &[u8]) -> Result<(), String> {
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(data))
        .map_err(|e| format!("write {}: {e}", path.display()))
}
