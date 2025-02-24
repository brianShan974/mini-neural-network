use crate::{
    Matrix,
    activation_functions::{sigmoid, sigmoid_derivative},
};

use super::layer::Layer;

#[derive(Default)]
pub struct SigmoidLayer {
    cache: Option<Matrix>,
}

impl Layer for SigmoidLayer {
    fn forward(&mut self, x: Matrix) -> Matrix {
        self.cache = Some(sigmoid_derivative(x.clone()));
        sigmoid(x)
    }

    fn backward(&mut self, grad_z: Matrix) -> Matrix {
        self.cache.clone().unwrap() * grad_z
    }
}
