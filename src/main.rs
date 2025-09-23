use picograd::{tensor::pyten, Device};

#[allow(non_snake_case)]
fn main() {
    println!("picotorch..under construction..");
    // TODO: -smell?: map_err: tensor/op error -> python error in lib_py.rs
    let device = Device::Gpu;
    let X = pyten::ones(vec![3, 1]);
    let Y = pyten::ones(vec![1, 3]);

    let X = X.to(&device);
    let X = Y.to(&device);
    // let Z = (&X * &Y).unwrap();
    // Z.backward();
    todo!()
}