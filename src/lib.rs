use ndarray::{Array1, Array2};

pub mod activation_functions;
pub mod loss_functions;
pub mod utils;

pub mod layers;

pub mod multilayer_network;

pub mod trainer;

pub mod preprocessor;

pub type Number = f64;
pub type Vector = Array1<Number>;
pub type Matrix = Array2<Number>;
