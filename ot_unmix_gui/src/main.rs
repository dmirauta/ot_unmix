mod app;
mod components;

use std::path::PathBuf;

use app::DecompGUIApp;
use eframe::NativeOptions;
use ot_unmix_core::{print_ocl_short_info, set_ocl_device};

use clap::Parser;

// TODO: Would be good to have a log to file option?
#[derive(Parser)]
struct CLIArgs {
    #[arg(long)]
    /// Print some info and quit before starting GUI
    nogui: bool,
    #[arg(short, long)]
    /// OpenCl platform/driver index
    platform: Option<usize>,
    #[arg(short, long)]
    /// Index of device from a corresponding OpenCl platform (--platform)
    device: Option<usize>,
    #[arg(long)]
    /// Preselect a dataset to load and work on
    dataset_dir: Option<PathBuf>,
    #[arg(long)]
    /// Preselect a serialised decomposition to resume work on
    decomp_path: Option<PathBuf>,
}

fn main() -> eframe::Result<()> {
    let args = CLIArgs::parse();
    let platform_idx = args.platform.unwrap_or(0);
    let device_idx = args.device.unwrap_or(0);

    print_ocl_short_info();

    set_ocl_device(platform_idx, device_idx);

    match args.nogui {
        true => Ok(()),
        false => eframe::run_native(
            "OT Unmixing GUI",
            NativeOptions::default(),
            Box::new(|_cc| {
                Ok(Box::new(DecompGUIApp::new(
                    args.dataset_dir,
                    args.decomp_path,
                )))
            }),
        ),
    }
}
