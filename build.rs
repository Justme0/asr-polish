use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let qwen_dir = manifest_dir.join("third_party/qwen-asr");

    // Pure C source files (kernels - no logging, stay as C)
    let c_sources = [
        "qwen_asr_kernels.c",
        "qwen_asr_kernels_generic.c",
        "qwen_asr_kernels_avx.c",
        "qwen_asr_kernels_neon.c",
    ];

    // C++ source files (use log.h which requires C++)
    let cpp_sources = [
        "qwen_asr.cpp",
        "qwen_asr_encoder.cpp",
        "qwen_asr_decoder.cpp",
        "qwen_asr_audio.cpp",
        "qwen_asr_tokenizer.cpp",
        "qwen_asr_safetensors.cpp",
    ];

    // Compile C sources
    let mut c_build = cc::Build::new();
    c_build
        .warnings(false)
        .opt_level_str("3")
        .flag("-march=native")
        .flag("-ffast-math")
        .define("USE_BLAS", None)
        .define("USE_OPENBLAS", None)
        .include("/usr/include/openblas")
        .include(qwen_dir.to_str().unwrap());

    for src in &c_sources {
        c_build.file(qwen_dir.join(src));
    }

    c_build.compile("qwen_asr_c");

    // Compile C++ sources
    let mut cxx_build = cc::Build::new();
    cxx_build
        .cpp(true)
        .warnings(false)
        .opt_level_str("3")
        .flag("-march=native")
        .flag("-ffast-math")
        .flag("-std=c++11")
        .flag("-fpermissive")
        .define("USE_BLAS", None)
        .define("USE_OPENBLAS", None)
        .include("/usr/include/openblas")
        .include(qwen_dir.to_str().unwrap());

    for src in &cpp_sources {
        cxx_build.file(qwen_dir.join(src));
    }

    cxx_build.compile("qwen_asr_cxx");

    // Link dependencies
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=stdc++");

    // Re-run build if sources change
    for src in &c_sources {
        println!("cargo:rerun-if-changed=third_party/qwen-asr/{}", src);
    }
    for src in &cpp_sources {
        println!("cargo:rerun-if-changed=third_party/qwen-asr/{}", src);
    }
    println!("cargo:rerun-if-changed=third_party/qwen-asr/qwen_asr.h");
    println!("cargo:rerun-if-changed=third_party/qwen-asr/qwen_asr_audio.h");
    println!("cargo:rerun-if-changed=third_party/qwen-asr/qwen_asr_kernels.h");
    println!("cargo:rerun-if-changed=third_party/qwen-asr/qwen_asr_tokenizer.h");
    println!("cargo:rerun-if-changed=third_party/qwen-asr/qwen_asr_safetensors.h");
    println!("cargo:rerun-if-changed=third_party/qwen-asr/log.h");
}
