use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let qwen_dir = manifest_dir.join("third_party/qwen-asr");

    // Compile all qwen-asr C source files into a static library
    let c_sources = [
        "qwen_asr.c",
        "qwen_asr_kernels.c",
        "qwen_asr_kernels_generic.c",
        "qwen_asr_kernels_avx.c",
        "qwen_asr_kernels_neon.c",
        "qwen_asr_audio.c",
        "qwen_asr_encoder.c",
        "qwen_asr_decoder.c",
        "qwen_asr_tokenizer.c",
        "qwen_asr_safetensors.c",
    ];

    let mut build = cc::Build::new();
    build
        .warnings(false)
        .opt_level_str("3")
        .flag("-march=native")
        .flag("-ffast-math")
        .define("USE_BLAS", None)
        .define("USE_OPENBLAS", None)
        .include("/usr/include/openblas")
        .include(qwen_dir.to_str().unwrap());

    for src in &c_sources {
        build.file(qwen_dir.join(src));
    }

    build.compile("qwen_asr");

    // Link dependencies
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");

    // Re-run build if C sources change
    for src in &c_sources {
        println!("cargo:rerun-if-changed=third_party/qwen-asr/{}", src);
    }
    println!("cargo:rerun-if-changed=third_party/qwen-asr/qwen_asr.h");
    println!("cargo:rerun-if-changed=third_party/qwen-asr/qwen_asr_audio.h");
    println!("cargo:rerun-if-changed=third_party/qwen-asr/qwen_asr_kernels.h");
}
