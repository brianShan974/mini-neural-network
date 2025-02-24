use crate::{
    Matrix,
    activation_functions::{relu, relu_derivative},
};

use super::layer::Layer;

#[derive(Default)]
pub struct ReluLayer {
    cache: Option<Matrix>,
}

impl Layer for ReluLayer {
    fn forward(&mut self, x: Matrix) -> Matrix {
        self.cache = Some(relu_derivative(x.clone()));
        relu(x)
    }

    fn backward(&mut self, grad_z: Matrix) -> Matrix {
        self.cache.clone().unwrap() * grad_z
    }
}
