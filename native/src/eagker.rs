use crate::ten::Tensor;

pub fn add(x: Tensor, y: Tensor) -> Tensor {
    let x_storage = vec![1, 2, 3];
    let y_storage = vec![4, 5, 6];

    let foo = x_storage.iter().zip(y_storage.iter()).map(|(x, y)| x+y).collect::<Vec<_>>();
    todo!()
}
pub fn add_rvv(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn add_ocl(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn add_hip(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn sub(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn sub_rvv(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn sub_ocl(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn sub_hip(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn mul(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn mul_rvv(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn mul_ocl(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn mul_hip(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn div(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn div_rvv(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn div_ocl(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn div_hip(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn matmul(x: Tensor, y: Tensor) -> Tensor { todo!() }
pub fn matmul_rvv() -> () { todo!() }
pub fn matmul_ocl() -> () { todo!() }
pub fn matmul_hip() -> () { todo!() }