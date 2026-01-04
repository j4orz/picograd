use std::env;
use std::path;

use cuda_builder::CudaBuilder;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=device_kernels");

    let out_dir = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Compile the `kernels` crate to `$OUT_DIR/kernels.ptx`.
    CudaBuilder::new(manifest_dir.join("device_kernels"))
        .copy_to(out_dir.join("device_kernels.ptx"))
        .build()
        .unwrap();
}