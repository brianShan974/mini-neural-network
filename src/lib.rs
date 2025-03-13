use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use polars::prelude::Float64Type;

pub mod utils;

pub mod math_functions;

pub mod layers;
pub mod multilayer_network;

pub mod preprocessor;
pub mod regressor;
pub mod trainer;

pub type Number = f64;
pub type NumberType = Float64Type;
pub type Vector = Array1<Number>;
pub type Matrix = Array2<Number>;
pub type VectorView<'a> = ArrayView1<'a, Number>;
pub type MatrixView<'a> = ArrayView2<'a, Number>;
