mod color_space;
mod frame_clock;
mod frame_reader;

use std::env;
use std::num::NonZeroU32;
use std::process::ExitCode;
use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

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

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = PlayerApp::new(input_path, cs, fps, ffmpeg_path);
    event_loop.run_app(&mut app).expect("event loop failed");

    ExitCode::SUCCESS
}

fn print_usage(program: &str) {
    eprintln!("usage: {program} <input.mxf|input.j2k> [options]");
    eprintln!("options:");
    eprintln!("  --colorspace <xyz|bt709|bt2020|rgb>  Color space (default: rgb passthrough)");
    eprintln!("  --ffmpeg <path>                      FFmpeg binary path (default: ffmpeg)");
    eprintln!("  --fps <rate>                         Playback frame rate (default: 24)");
}

struct PlayerApp {
    input_path: String,
    colorspace: ColorSpace,
    fps: f64,
    ffmpeg_path: String,
    state: Option<PlayerState>,
}

struct PlayerState {
    window: Arc<Window>,
    surface: softbuffer::Surface<Arc<Window>, Arc<Window>>,
    frame_reader: Option<FrameReader>,
    single_frame: Option<Vec<u8>>,
    clock: FrameClock,
    color_config: ColorConfig,
    width: u32,
    height: u32,
    frame_pixels: Vec<u32>,
    playing: bool,
    decoded_count: u64,
}

impl PlayerApp {
    fn new(input_path: String, colorspace: ColorSpace, fps: f64, ffmpeg_path: String) -> Self {
        Self {
            input_path,
            colorspace,
            fps,
            ffmpeg_path,
            state: None,
        }
    }
}

impl ApplicationHandler for PlayerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let is_mxf = self.input_path.to_lowercase().ends_with(".mxf");

        let (first_frame, frame_reader, single_frame) = if is_mxf {
            let mut reader = match FrameReader::open(&self.input_path, &self.ffmpeg_path) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: {e}");
                    event_loop.exit();
                    return;
                }
            };
            let frame = match reader.next_frame() {
                Some(f) => f.to_vec(),
                None => {
                    eprintln!("error: no frames in input");
                    event_loop.exit();
                    return;
                }
            };
            (frame, Some(reader), None)
        } else {
            let data = match std::fs::read(&self.input_path) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("error: failed to read {}: {e}", self.input_path);
                    event_loop.exit();
                    return;
                }
            };
            let frame = data.clone();
            (frame, None, Some(data))
        };

        let codestream = match vk_j2k_dec::codestream::Codestream::parse(&first_frame) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: failed to parse codestream: {e}");
                event_loop.exit();
                return;
            }
        };
        let siz = match &codestream.siz {
            Some(s) => s,
            None => {
                eprintln!("error: missing SIZ marker");
                event_loop.exit();
                return;
            }
        };
        let width = siz.width;
        let height = siz.height;
        let precision = siz.components.first().map(|c| c.precision).unwrap_or(12);

        eprintln!("vk-j2k-player: {}x{} {}-bit, playing at {} fps", width, height, precision, self.fps);

        let color_config = ColorConfig::for_colorspace(self.colorspace, precision);

        let window_attrs = Window::default_attributes()
            .with_title(format!("vk-j2k-player — {}", self.input_path))
            .with_inner_size(PhysicalSize::new(width, height));

        let window = Arc::new(event_loop.create_window(window_attrs).expect("failed to create window"));
        let context = softbuffer::Context::new(window.clone()).expect("failed to create softbuffer context");
        let mut surface = softbuffer::Surface::new(&context, window.clone()).expect("failed to create surface");
        surface.resize(
            NonZeroU32::new(width).unwrap(),
            NonZeroU32::new(height).unwrap(),
        ).expect("failed to resize surface");

        let pixel_count = (width * height) as usize;
        let mut frame_pixels = vec![0u32; pixel_count];
        decode_frame_to_pixels(&first_frame, &color_config, width, height, &mut frame_pixels);

        self.state = Some(PlayerState {
            window,
            surface,
            frame_reader,
            single_frame,
            clock: FrameClock::new(self.fps),
            color_config,
            width,
            height,
            frame_pixels,
            playing: true,
            decoded_count: 1,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state.is_pressed() {
                    use winit::keyboard::{Key, NamedKey};
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        Key::Named(NamedKey::Space) => {
                            state.playing = !state.playing;
                            eprintln!("vk-j2k-player: {}", if state.playing { "play" } else { "pause" });
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if state.playing {
                    state.clock.wait_for_next_frame();
                    let next_frame_bytes = if let Some(reader) = &mut state.frame_reader {
                        reader.next_frame().map(|f| f.to_vec())
                    } else {
                        state.single_frame.clone()
                    };

                    if let Some(frame_bytes) = next_frame_bytes {
                        decode_frame_to_pixels(
                            &frame_bytes,
                            &state.color_config,
                            state.width,
                            state.height,
                            &mut state.frame_pixels,
                        );
                        state.decoded_count += 1;
                    } else {
                        state.playing = false;
                        eprintln!("vk-j2k-player: end of stream ({} frames decoded)", state.decoded_count);
                    }
                }

                let mut buffer = state.surface.buffer_mut().expect("failed to get buffer");
                buffer.copy_from_slice(&state.frame_pixels);
                buffer.present().expect("failed to present");

                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn decode_frame_to_pixels(
    frame_bytes: &[u8],
    color_config: &ColorConfig,
    width: u32,
    height: u32,
    pixels: &mut [u32],
) {
    let decoded = match vk_j2k_dec::decode::decode_rgb48(frame_bytes) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("decode error: {e}");
            return;
        }
    };

    let pixel_count = (width * height) as usize;
    for i in 0..pixel_count {
        let r_raw = decoded.samples[i * 3] as f32;
        let g_raw = decoded.samples[i * 3 + 1] as f32;
        let b_raw = decoded.samples[i * 3 + 2] as f32;

        let r_in = (r_raw + color_config.offset[0]) * color_config.input_scale;
        let g_in = (g_raw + color_config.offset[1]) * color_config.input_scale;
        let b_in = (b_raw + color_config.offset[2]) * color_config.input_scale;

        let m = &color_config.matrix;
        let r_lin = m[0] * r_in + m[1] * g_in + m[2] * b_in;
        let g_lin = m[3] * r_in + m[4] * g_in + m[5] * b_in;
        let b_lin = m[6] * r_in + m[7] * g_in + m[8] * b_in;

        let r = apply_transfer(r_lin.clamp(0.0, 1.0), color_config.transfer_func_value());
        let g = apply_transfer(g_lin.clamp(0.0, 1.0), color_config.transfer_func_value());
        let b = apply_transfer(b_lin.clamp(0.0, 1.0), color_config.transfer_func_value());

        let r8 = (r * 255.0) as u32;
        let g8 = (g * 255.0) as u32;
        let b8 = (b * 255.0) as u32;
        pixels[i] = (r8 << 16) | (g8 << 8) | b8;
    }
}

fn apply_transfer(c: f32, transfer_func: u32) -> f32 {
    match transfer_func {
        1 => {
            if c <= 0.0031308 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            }
        }
        _ => c,
    }
}
