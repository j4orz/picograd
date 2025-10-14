//! runtime
//! Cpu uses std::simd and R5
//! Gpu uses triton and amd

pub enum Device { Cpu(CpuLang), Gpu(GpuLang) }
pub enum CpuLang { RustStd, OpenMp, R5 } pub enum GpuLang { Ocl, Tri, Amd }
pub enum Storage { Cpu(CpuBuf), Gpu(GpuBuf) }
type CpuBuf = Vec<f32>; type GpuBuf = opencl3::memory::Buffer<f32>;