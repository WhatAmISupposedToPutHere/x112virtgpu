use std::ffi::c_int;
use std::os::fd::IntoRawFd;

#[no_mangle]
pub extern "C" fn xshmfence_alloc_shm() -> c_int {
    match util::create_shm_file() {
        Ok((_, file)) => file.into_raw_fd(),
        Err(_) => -1,
    }
}
