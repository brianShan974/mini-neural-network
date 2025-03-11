use ndarray::{Array1, Array2};

pub mod activation_functions;
pub mod loss_functions;
pub mod utils;

pub mod layers;

pub mod multilayer_network;

type Number = f64;
type Vector = Array1<Number>;
type Matrix = Array2<Number>;
