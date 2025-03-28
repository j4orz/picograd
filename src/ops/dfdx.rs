use crate::{ops::Op, tpy::ones, trs::Tensor};

impl Op {
    pub fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        match self {
            Op::Add(_x, _y) => vec![
                (&ones(grad.shape.clone()) * &grad.clone()).unwrap(),
                (&ones(grad.shape.clone()) * &grad.clone()).unwrap(),
            ],
            Op::Sub(_x, _y) => {
                // d/dx(x - y) = 1, d/dy(x - y) = -1
                // let (dx, dy) = (
                //     ones(grad.shape.clone()), // clone because of python/rust memory mismatch
                //     new(vec![DtypeVal::Float32(-1.0); grad.numel()]),
                // );
                // vec![
                //     (&dx * &grad.clone()).unwrap(),
                //     (&dy * &grad.clone()).unwrap(),
                // ]
                todo!()
            }
            Op::Mul(x, y) => vec![(y * &grad.clone()).unwrap(), (x * &grad.clone()).unwrap()],
            Op::Div(x, y) => todo!(),
            Op::Neg(x) => todo!(),
            Op::Exp(x) => todo!(),
            Op::Log(x) => todo!(),
            Op::Sinh(x) => todo!(),
            Op::Cosh(x) => todo!(),
            Op::Tanh(x) => todo!(),
            // Op::Mean(x) => todo!(),
            // Op::Var(x) => todo!(),
            Op::Matmul(x, y) => todo!(),
            Op::Sum(tensor, _, _) => todo!(),
        }
    }
}
