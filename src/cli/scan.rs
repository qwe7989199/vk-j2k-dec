use std::fs;
use std::process::ExitCode;

use vk_j2k_dec::codestream::{Codestream, CodestreamScanHit, scan_codestreams};

pub fn scan_summary(path: &str) -> ExitCode {
    let data = match fs::read(path) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {path}: {err}");
            return ExitCode::FAILURE;
        }
    };

    let hits = scan_codestreams(&data, usize::MAX);
    println!("file: {path}");
    println!("jpeg2000 codestream hits: {}", hits.len());

    if let Some(first) = hits.first() {
        print_hit_summary("first", first);
    }
    if let Some(last) = hits.last()
        && hits.len() > 1
    {
        print_hit_summary("last", last);
    }

    ExitCode::SUCCESS
}

pub fn scan_file(path: &str, max_hits: usize) -> ExitCode {
    let data = match fs::read(path) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("failed to read {path}: {err}");
            return ExitCode::FAILURE;
        }
    };

    let hits = scan_codestreams(&data, max_hits);
    println!("file: {path}");
    println!("jpeg2000 codestream hits: {}", hits.len());

    for (index, hit) in hits.iter().enumerate() {
        println!(
            "hit {index}: offset 0x{:08x}, length {} bytes",
            hit.offset, hit.codestream.byte_len
        );
        if let Some(siz) = &hit.codestream.siz {
            println!(
                "  image: {}x{}, components {}",
                siz.width,
                siz.height,
                siz.components.len()
            );
            for (component_index, component) in siz.components.iter().enumerate() {
                println!(
                    "  component {component_index}: {}-bit, subsampling {}x{}",
                    component.precision, component.x_subsampling, component.y_subsampling
                );
            }
        }
        if let Some(cod) = &hit.codestream.cod {
            println!(
                "  cod: progression {}, layers {}, transform {}, codeblock {}x{}",
                cod.progression_order,
                cod.layers,
                cod.transform,
                cod.codeblock_width,
                cod.codeblock_height
            );
        }
        print_packet_summary(&hit.codestream, "  ");
        let profile = hit.codestream.first_profile_report();
        if profile.is_supported() {
            println!("  first profile: supported");
        } else {
            println!("  first profile: unsupported");
            for issue in profile.issues {
                println!("    - {}", issue.message);
            }
        }
    }

    ExitCode::SUCCESS
}

fn print_hit_summary(label: &str, hit: &CodestreamScanHit) {
    println!(
        "{label}: offset 0x{:08x}, length {} bytes",
        hit.offset, hit.codestream.byte_len
    );

    if let Some(siz) = &hit.codestream.siz {
        println!(
            "  image: {}x{}, components {}",
            siz.width,
            siz.height,
            siz.components.len()
        );
    }
    if let Some(cod) = &hit.codestream.cod {
        println!(
            "  cod: progression {}, layers {}, transform {}, codeblock {}x{}",
            cod.progression_order,
            cod.layers,
            cod.transform,
            cod.codeblock_width,
            cod.codeblock_height
        );
    }

    let profile = hit.codestream.first_profile_report();
    let status = if profile.is_supported() {
        "supported"
    } else {
        "unsupported"
    };
    println!("  first profile: {status}");
    print_packet_summary(&hit.codestream, "  ");
}

fn print_packet_summary(codestream: &Codestream, indent: &str) {
    match codestream.packet_sequence() {
        Ok(packets) => {
            let layers = codestream.cod.as_ref().map(|cod| cod.layers).unwrap_or(0);
            let packets_per_layer = if layers == 0 {
                0
            } else {
                packets.len() / usize::from(layers)
            };
            println!(
                "{indent}packet grid: {packets_per_layer} packets/layer, {} packets total for {layers} layer(s)",
                packets.len()
            );
        }
        Err(err) => println!("{indent}packet grid: unavailable ({err})"),
    }
}
