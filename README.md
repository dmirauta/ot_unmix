# OT Unmixing

OpenCL accelerated, basis variation insensitive unmixing algorithm, with an optional GUI interface.
This is an implementation of my PhD work, a preview of which is available at [unmixing via optimal transport](https://openreview.net/forum?id=QjnVVrixPW), a more extensive version is in preparation.
The older python implementation is found at [SOTU](https://github.com/dmirauta/SOTU).

Git clone recursively to include the required c/opencl submodule `git clone --recursive https://github.com/dmirauta/ot_unmix`.

## Build

Build with `cargo b -r`, or build and run the gui with the command `cargo r -r`.
The produced binary is found under `$CARGO_TARGET_DIR/release/` (cargo target dir defaults to crate root).

### Preliminaries

#### Rust

1. Install rustup from your preferred repo or download from [rust-lang.org](https://www.rust-lang.org/learn/get-started).
2. Make sure you have a toolchain installed (`rustup toolchain install stable`).

#### OpenCL

Could make optional in the future, but currently required.

Make sure to have a GPU (or CPU) driver/dev stack which provides OpenCL installed.

Driver/platform and device selection is done via command line args when launching GUI (see usage).
Some basic OpenCL info is printed with the `--nogui` option.

## Usage

If the executable is not in the search `$PATH`, can run with `cargo r -r -- [OPTIONS]`, e.g. `cargo r -r -- --help`.

```
Usage: ot_unmix_gui [OPTIONS]

Options:
      --nogui                      Print some info and quit before starting GUI
  -p, --platform <PLATFORM>        OpenCl platform/driver index
  -d, --device <DEVICE>            Index of device from a corresponding OpenCl platform (--platform)
      --dataset-dir <DATASET_DIR>  Preselect a dataset to load and work on
      --decomp-path <DECOMP_PATH>  Preselect a serialised decomposition to resume work on
  -h, --help                       Print help
```

## Docs

The documentation is built with `cargo doc --no-deps`.

Access at `$CARGO_TARGET_DIR/doc/ot_unmix_core/index.html` (or similarly for gui crate).

## Other notes

Some git dependencies (see `Cargo.toml` files) are not currently versioned, so in some cases you may have to run `cargo update` to sync changes.
