//! Raw FFI bindings to the qwen-asr C library.
#![allow(dead_code)]

use std::os::raw::{c_char, c_float, c_int, c_void};

/// Opaque context type for qwen-asr model.
/// Maps to `qwen_ctx_t` in C.
#[repr(C)]
pub struct QwenCtx {
    _opaque: [u8; 0],
}

/// Token streaming callback type.
/// `piece` is the decoded token string (UTF-8), `userdata` is the user-provided pointer.
pub type QwenTokenCb =
    Option<unsafe extern "C" fn(piece: *const c_char, userdata: *mut c_void)>;

extern "C" {
    /// Load model from directory. Returns NULL on failure.
    pub fn qwen_load(model_dir: *const c_char) -> *mut QwenCtx;

    /// Free all resources.
    pub fn qwen_free(ctx: *mut QwenCtx);

    /// Set a callback to receive each decoded token as it's generated.
    pub fn qwen_set_token_callback(
        ctx: *mut QwenCtx,
        cb: QwenTokenCb,
        userdata: *mut c_void,
    );

    /// Set optional system prompt text (UTF-8). Returns 0 on success.
    pub fn qwen_set_prompt(ctx: *mut QwenCtx, prompt: *const c_char) -> c_int;

    /// Set optional forced language. Returns 0 on success, -1 if unsupported.
    pub fn qwen_set_force_language(ctx: *mut QwenCtx, language: *const c_char) -> c_int;

    /// Transcribe a WAV file, returns allocated string (caller must free).
    pub fn qwen_transcribe(ctx: *mut QwenCtx, wav_path: *const c_char) -> *mut c_char;

    /// Transcribe from raw audio samples (mono float32, 16kHz).
    /// Returns allocated string (caller must free).
    pub fn qwen_transcribe_audio(
        ctx: *mut QwenCtx,
        samples: *const c_float,
        n_samples: c_int,
    ) -> *mut c_char;

    /// Set number of threads for parallel operations.
    pub fn qwen_set_threads(n: c_int);

    /// Get number of available CPU cores.
    pub fn qwen_get_num_cpus() -> c_int;

    /// Global verbose flag.
    pub static mut qwen_verbose: c_int;

    /// Monitor mode flag.
    pub static mut qwen_monitor: c_int;
}
