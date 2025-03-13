use crate::{
    Matrix,
    activation_functions::{sigmoid, sigmoid_derivative},
    layers::layer::Layer,
};

use super::activation_layer::ActivationLayer;

#[derive(Default)]
pub struct SigmoidLayer {
    cache: Option<Matrix>,
}

impl Layer for SigmoidLayer {
    fn forward(&mut self, x: Matrix) -> Matrix {
        self.cache = Some(sigmoid_derivative(x.clone()));
        self.eval_only(x)
    }

    fn backward(&mut self, grad_z: Matrix) -> Matrix {
        self.cache.clone().unwrap() * grad_z
    }

    fn eval_only(&self, x: Matrix) -> Matrix {
        sigmoid(x)
    }
}

impl ActivationLayer for SigmoidLayer {}
