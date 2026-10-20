mod color_space;
mod frame_clock;
mod frame_reader;
mod gpu_decode;
mod presenter;

use std::env;
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
use gpu_decode::GpuDecoder;
use presenter::{PresentPushConstants, VulkanPresenter};

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
    presenter: VulkanPresenter,
    gpu_decoder: GpuDecoder,
    frame_reader: Option<FrameReader>,
    single_frame: Option<Vec<u8>>,
    clock: FrameClock,
    push_constants: PresentPushConstants,
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
                    eprintln!("error: {e}");
                    event_loop.exit();
                    return;
                }
            };
            (data.clone(), None, Some(data))
        };

        let codestream = match vk_j2k_dec::codestream::Codestream::parse(&first_frame) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: {e}");
                event_loop.exit();
                return;
            }
        };
        let siz = match &codestream.siz {
            Some(s) => s,
            None => {
                eprintln!("error: missing SIZ");
                event_loop.exit();
                return;
            }
        };
        let width = siz.width;
        let height = siz.height;
        let precision = siz.components.first().map(|c| c.precision).unwrap_or(12);

        eprintln!(
            "vk-j2k-player: {}x{} {}-bit, {} fps",
            width, height, precision, self.fps
        );

        let color_config = ColorConfig::for_colorspace(self.colorspace, precision);

        let window_attrs = Window::default_attributes()
            .with_title(format!("vk-j2k-player — {}", self.input_path))
            .with_inner_size(PhysicalSize::new(width, height));
        let window = Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("create window"),
        );

        let presenter = match unsafe { VulkanPresenter::new(&window, width, height) } {
            Ok(p) => p,
            Err(e) => {
                eprintln!("vulkan error: {e}");
                event_loop.exit();
                return;
            }
        };

        let gpu_decoder = match GpuDecoder::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("gpu decoder init error: {e}");
                event_loop.exit();
                return;
            }
        };

        let push_constants = PresentPushConstants {
            width,
            height,
            transfer_func: color_config.transfer_func_value(),
            _pad: 0,
            color_matrix: color_config.matrix,
            offset: color_config.offset,
            input_scale: color_config.input_scale,
        };

        // Decode and upload first frame
        if let Ok(decoded) = gpu_decoder.decode(&first_frame) {
            unsafe {
                presenter.upload_rgb48(&decoded.samples);
            }
        }

        self.state = Some(PlayerState {
            window,
            presenter,
            gpu_decoder,
            frame_reader,
            single_frame,
            clock: FrameClock::new(self.fps),
            push_constants,
            playing: true,
            decoded_count: 1,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state.is_pressed() {
                    use winit::keyboard::{Key, NamedKey};
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        Key::Named(NamedKey::Space) => {
                            state.playing = !state.playing;
                            eprintln!("{}", if state.playing { "play" } else { "pause" });
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if state.playing {
                    state.clock.wait_for_next_frame();

                    let next_bytes = if let Some(reader) = &mut state.frame_reader {
                        reader.next_frame().map(|f| f.to_vec())
                    } else {
                        state.single_frame.clone()
                    };

                    if let Some(frame_bytes) = next_bytes {
                        if let Ok(decoded) = state.gpu_decoder.decode(&frame_bytes) {
                            unsafe {
                                state.presenter.upload_rgb48(&decoded.samples);
                            }
                            state.decoded_count += 1;
                        }
                    } else {
                        state.playing = false;
                        eprintln!("end of stream ({} frames)", state.decoded_count);
                    }
                }

                unsafe {
                    if let Err(e) = state.presenter.present_frame(&state.push_constants) {
                        eprintln!("present error: {e}");
                    }
                }

                state.window.request_redraw();
            }
            _ => {}
        }
    }
}
