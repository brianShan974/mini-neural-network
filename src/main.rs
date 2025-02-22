use ndarray::{Array1, Array2};

mod activation_functions;
mod utils;

mod layer;
mod loss_layer;

mod linear_layer;

mod identity_layer;

type Number = f64;
type Vector = Array1<Number>;
type Matrix = Array2<Number>;

fn main() {
    println!("Hello, world!");
}
