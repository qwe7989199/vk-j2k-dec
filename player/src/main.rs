use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: vk-j2k-player <input.mxf|input.j2k> [--colorspace xyz|bt709|bt2020]");
        return ExitCode::FAILURE;
    }

    let input = &args[1];
    eprintln!("vk-j2k-player: input {input}");
    eprintln!("TODO: implement playback");

    ExitCode::SUCCESS
}
