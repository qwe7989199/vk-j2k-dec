use std::process::ExitCode;

use vk_j2k_dec::vulkan::{
    enumerate_devices, run_color_smoke, run_dequant_smoke, run_idwt53_smoke, run_idwt97_smoke,
    run_irreversible_smoke,
};

pub fn print_vulkan_info() -> ExitCode {
    let devices = match enumerate_devices() {
        Ok(devices) => devices,
        Err(err) => {
            eprintln!("vulkan: unavailable ({err})");
            return ExitCode::FAILURE;
        }
    };

    println!("vulkan devices: {}", devices.len());
    for (device_index, device) in devices.iter().enumerate() {
        println!(
            "device {device_index}: {} ({}, Vulkan {}, vendor 0x{:04x}, device 0x{:04x})",
            device.name, device.device_type, device.api_version, device.vendor_id, device.device_id
        );
        for family in &device.queue_families {
            println!(
                "  queue family {}: count {}, graphics {}, compute {}, transfer {}",
                family.index, family.queue_count, family.graphics, family.compute, family.transfer
            );
        }
        match device.first_compute_queue_family() {
            Some(family) => println!("  selected compute queue family: {}", family.index),
            None => println!("  selected compute queue family: none"),
        }
    }

    ExitCode::SUCCESS
}

pub fn run_vulkan_smoke(shader_path: &str) -> ExitCode {
    let result = match run_dequant_smoke(shader_path) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("vulkan smoke: failed ({err})");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "vulkan smoke: device {}, queue family {}",
        result.device_name, result.queue_family_index
    );
    println!("  input: {:?}", result.input);
    println!("  output: {:?}", result.output);
    println!("  validation: passed");
    ExitCode::SUCCESS
}

pub fn run_vulkan_color_smoke(shader_path: &str) -> ExitCode {
    let result = match run_color_smoke(shader_path) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("vulkan color smoke: failed ({err})");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "vulkan color smoke: device {}, queue family {}",
        result.device_name, result.queue_family_index
    );
    println!("  none output rgb16 words: {:?}", result.none_output);
    println!("  rct output rgb16 words: {:?}", result.rct_output);
    println!("  ict output rgb16 words: {:?}", result.ict_output);
    println!(
        "  float component output rgb16 words: {:?}",
        result.float_component_output
    );
    println!("  validation: passed");
    ExitCode::SUCCESS
}

pub fn run_vulkan_idwt53_smoke(shader_path: &str) -> ExitCode {
    let result = match run_idwt53_smoke(shader_path) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("vulkan idwt53 smoke: failed ({err})");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "vulkan idwt53 smoke: device {}, queue family {}",
        result.device_name, result.queue_family_index
    );
    println!("  row output: {:?}", result.row_output);
    println!("  column output: {:?}", result.column_output);
    println!("  level output: {:?}", result.level_output);
    println!("  validation: passed");
    ExitCode::SUCCESS
}

pub fn run_vulkan_idwt97_smoke(shader_path: &str) -> ExitCode {
    let result = match run_idwt97_smoke(shader_path) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("vulkan idwt97 smoke: failed ({err})");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "vulkan idwt97 smoke: device {}, queue family {}",
        result.device_name, result.queue_family_index
    );
    println!("  row output: {:?}", result.row_output);
    println!("  column output: {:?}", result.column_output);
    println!("  level output: {:?}", result.level_output);
    println!("  component output: {:?}", result.component_output);
    println!("  validation: passed");
    ExitCode::SUCCESS
}

pub fn run_vulkan_irreversible_smoke(
    idwt97_shader_path: &str,
    color_shader_path: &str,
) -> ExitCode {
    let result = match run_irreversible_smoke(idwt97_shader_path, color_shader_path) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("vulkan irreversible smoke: failed ({err})");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "vulkan irreversible smoke: device {}, queue family {}",
        result.device_name, result.queue_family_index
    );
    println!("  cpu rgb16 words: {:?}", result.cpu_samples);
    println!("  gpu rgb16 words: {:?}", result.gpu_samples);
    println!("  validation: passed");
    ExitCode::SUCCESS
}
