#![allow(non_snake_case)]

pub mod numpy;
pub mod uop;
pub mod engine;
pub mod eagker;
pub mod graph;
pub mod runtime;

use pyo3::{FromPyObject, IntoPyObject, pyclass};

#[pyclass(eq)] #[derive(Clone, Debug, PartialEq)] pub enum Device { Cpu, Gpu, Amd, }
#[pyclass(eq)] #[derive(Clone, Debug, PartialEq)] pub enum Layout { Strided  } // Sparse, // MklDnn
#[pyclass(eq)] #[derive(Clone, Debug, PartialEq)] pub enum Dtype { Bool, Float32, Float64, Int32, Int64}

#[derive(FromPyObject)] #[derive(IntoPyObject)] #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum DtypeVal { Bool(bool), Float32(f32), Float64(f64), Int16(i16), Int32(i32), Int64(i64) } // f16 is unstable

impl From<i32> for DtypeVal {
    fn from(x: i32) -> Self {
        DtypeVal::Int32(x)
    }
}

impl From<i64> for DtypeVal {
    fn from(x: i64) -> Self {
        DtypeVal::Int64(x)
    }
}

impl From<f32> for DtypeVal {
    fn from(x: f32) -> Self {
        DtypeVal::Float32(x)
    }
}

impl From<DtypeVal> for f32 {
    fn from(value: DtypeVal) -> Self {
        match value {
            DtypeVal::Float32(x) => x,
            DtypeVal::Int32(x) => x as f32,
            DtypeVal::Int64(x) => x as f32,
            _ => todo!(), // todo: panic?
        }
    }
}

impl From<DtypeVal> for usize {
    fn from(value: DtypeVal) -> Self {
        match value {
            DtypeVal::Float32(x) => x as usize,
            DtypeVal::Float64(x) => x as usize,
            DtypeVal::Int16(x) => x as usize,
            DtypeVal::Int32(x) => x as usize,
            DtypeVal::Int64(x) => x as usize,
            _ => todo!(), // todo: panic?
        }
    }
}