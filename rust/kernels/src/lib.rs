use cuda_std::prelude::*;

pub type T = f32; // Input/output type shared with the `rustc-cuda-basic` crate.

#[allow(improper_ctypes_definitions)]
#[kernel] pub unsafe fn add(a: &[T], b: &[T], c: *mut T) {
    let i = thread::index_1d() as usize;
    if i < a.len() {
        let elem = unsafe { &mut *c.add(i) };
        *elem = a[i] + b[i];
    }
}