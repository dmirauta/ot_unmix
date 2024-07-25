use std::{env, path::PathBuf};

fn build_and_bindgen<AsStr: AsRef<str>>(sources: Vec<AsStr>, libname: &str) {
    let mut ccb = cc::Build::new();
    for file in sources.iter() {
        ccb.file(file.as_ref());
    }
    ccb.compile(libname);

    let mut bb = bindgen::Builder::default();
    for file in sources.into_iter() {
        bb = bb.header(file.as_ref());
    }

    let bindings = bb
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join(format!("{libname}_bindings.rs")))
        .expect("Couldn't write bindings!");
}

static C_SRC: &str = "../c_opencl_src";

fn main() {
    // TODO: Link against f32 and f64 versions of lib?
    build_and_bindgen(
        vec![format!("{C_SRC}/dists.c"), format!("{C_SRC}/lb.c")],
        "sot",
    );
}
