#[cfg(feature = "cuda")]
pub fn has_gpu() -> bool {
    cudarc::driver::result::device::get_count()
        .map(|count| count > 0)
        .unwrap_or(false)
}
