fn main() {
    println!("picograd");
}

// use picograd::{tensor::pyten, Device};

// #[allow(non_snake_case)]
// fn main() {
//     println!("picotorch..under construction..");
//     // TODO: -smell?: map_err: tensor/op error -> python error in lib_py.rs
//     let device = Device::Gpu;
//     let X = pyten::ones(vec![3, 1]);
//     let Y = pyten::ones(vec![1, 3]);

//     let X = X.to(&device);
//     let X = Y.to(&device);
//     // let Z = (&X * &Y).unwrap();
//     // Z.backward();
//     todo!()
// }

// fn example_eager(sess: &mut Session) {
//     // Suppose we already have two materialized leaves x,y (same shape, requires_grad=true).
//     let x = make_leaf_tensor([1024,1024].to_vec(), DType::F32, sess, true);
//     let y = make_leaf_tensor([1024,1024].to_vec(), DType::F32, sess, true);

//     // z = relu(x * y + y)
//     let z = relu(&add(&mul(&x, &y, sess), &y, sess), sess);
//     // loss = reduce mean (omitted: use sum and scale)
//     let loss = z; // pretend scalar for sketch

//     backward_eager(&loss, sess);

//     // grads now live on x.grad() and y.grad()
//     assert!(x.grad().is_some() && y.grad().is_some());
// }

// fn example_lazy(sess: &mut Session) {
//     // Build delayed leaves
//     let x = delayed_leaf([1024,1024].to_vec(), DType::F32, sess, true);
//     let y = delayed_leaf([1024,1024].to_vec(), DType::F32, sess, true);

//     let z = relu(&add(&mul(&x, &y, sess), &y, sess), sess);
//     let loss = z; // pretend scalar

//     // One shot backward: compile+run grad graph
//     backward_lazy(&loss, &[x.clone(), y.clone()], sess);

//     assert!(x.grad().is_some() && y.grad().is_some());
// }

// // helpers for the demo
// fn make_leaf_tensor(shape: Vec<usize>, dtype: DType, sess: &mut Session, requires_grad: bool) -> Tensor {
//     let nbytes = numel(&shape) * dtype.size_in_bytes();
//     let storage = sess.backend.alloc(nbytes);
//     let view = View { storage: Arc::new(storage), offset_bytes: 0, shape: shape.clone(), strides: {
//         let mut st = vec![0; shape.len()]; let mut s = 1isize;
//         for (i, d) in shape.iter().enumerate().rev() { st[i]=s; s *= *d as isize; } st
//     }, dtype };
//     Tensor::from_view(view, requires_grad, sess)
// }
// fn delayed_leaf(shape: Vec<usize>, dtype: DType, sess: &mut Session, requires_grad: bool) -> Tensor {
//     let id = sess.graph.nodes.len() as NodeId;
//     // In real code, create a "Parameter" / "Input" node kind. We'll re-use Add as placeholder is wrong; for sketch we push a leaf marker.
//     sess.graph.nodes.push(UOp { id, kind: UOpKind::Add, inputs: vec![], shape: shape.clone(), dtype, attrs: Default::default() });
//     Tensor::from_node(id, shape, dtype, sess.backend.device(), requires_grad, sess)
// }
