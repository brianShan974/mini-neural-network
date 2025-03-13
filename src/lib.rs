use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub mod activation_functions;
pub mod loss_functions;
pub mod utils;

pub mod layers;
pub mod multilayer_network;

pub mod preprocessor;
pub mod regressor;
pub mod trainer;

pub type Number = f64;
pub type Vector = Array1<Number>;
pub type Matrix = Array2<Number>;
pub type VectorView<'a> = ArrayView1<'a, Number>;
pub type MatrixView<'a> = ArrayView2<'a, Number>;
