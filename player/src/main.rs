mod color_space;
mod frame_clock;
mod frame_reader;

use std::env;
use std::process::ExitCode;

use color_space::{ColorConfig, ColorSpace};
use frame_clock::FrameClock;
use frame_reader::FrameReader;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let program = args.first().map(|s| s.as_str()).unwrap_or("vk-j2k-player");

    let mut input = None;
    let mut colorspace = None;
    let mut ffmpeg_path = "ffmpeg".to_string();
    let mut fps = 24.0f64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--colorspace" | "-c" => {
                i += 1;
                if i < args.len() {
                    colorspace = color_space::parse_colorspace(&args[i]);
                    if colorspace.is_none() {
                        eprintln!("unknown colorspace: {}", args[i]);
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--ffmpeg" => {
                i += 1;
                if i < args.len() {
                    ffmpeg_path = args[i].clone();
                }
            }
            "--fps" => {
                i += 1;
                if i < args.len() {
                    fps = args[i].parse().unwrap_or(24.0);
                }
            }
            "--help" | "-h" => {
                print_usage(program);
                return ExitCode::SUCCESS;
            }
            other => {
                if input.is_none() {
                    input = Some(other.to_string());
                } else {
                    eprintln!("unexpected argument: {other}");
                    return ExitCode::FAILURE;
                }
            }
        }
        i += 1;
    }

    let Some(input_path) = input else {
        print_usage(program);
        return ExitCode::FAILURE;
    };

    let cs = colorspace.unwrap_or(ColorSpace::Passthrough);
    eprintln!("vk-j2k-player: input {input_path}, fps {fps}");

    if let Err(e) = run_player(&input_path, cs, fps, &ffmpeg_path) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn print_usage(program: &str) {
    eprintln!("usage: {program} <input.mxf|input.j2k> [options]");
    eprintln!("options:");
    eprintln!("  --colorspace <xyz|bt709|bt2020|rgb>  Color space (default: rgb passthrough)");
    eprintln!("  --ffmpeg <path>                      FFmpeg binary path (default: ffmpeg)");
    eprintln!("  --fps <rate>                         Playback frame rate (default: 24)");
}

fn run_player(
    input_path: &str,
    colorspace: ColorSpace,
    fps: f64,
    ffmpeg_path: &str,
) -> Result<(), String> {
    use vk_j2k_dec::codestream::Codestream;

    let is_mxf = input_path.to_lowercase().ends_with(".mxf");

    let first_frame_bytes = if is_mxf {
        let mut reader = FrameReader::open(input_path, ffmpeg_path)?;
        let frame = reader.next_frame().ok_or("no frames in input")?;
        frame.to_vec()
    } else {
        std::fs::read(input_path).map_err(|e| format!("failed to read {input_path}: {e}"))?
    };

    let codestream = Codestream::parse(&first_frame_bytes)
        .map_err(|e| format!("failed to parse codestream: {e}"))?;
    let siz = codestream.siz.as_ref().ok_or("missing SIZ marker")?;
    let width = siz.width;
    let height = siz.height;
    let precision = siz.components.first().map(|c| c.precision).unwrap_or(12);

    eprintln!(
        "vk-j2k-player: {}x{} {}-bit, {} components",
        width, height, precision, siz.components.len()
    );

    let color_config = ColorConfig::for_colorspace(colorspace, precision);
    let _clock = FrameClock::new(fps);

    // TODO: Full Vulkan surface pipeline
    // For now, print what we would do
    eprintln!(
        "vk-j2k-player: color matrix scale={}, transfer_func={}",
        color_config.input_scale,
        color_config.transfer_func_value()
    );
    eprintln!("vk-j2k-player: would create {}x{} window and decode at {fps} fps", width, height);
    eprintln!("vk-j2k-player: Vulkan surface presentation not yet wired — use decode CLI for now");

    Ok(())
}
