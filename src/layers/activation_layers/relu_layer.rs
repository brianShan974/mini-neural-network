use crate::{
    Matrix,
    activation_functions::{relu, relu_derivative},
    layers::layer::Layer,
};

use super::activation_layer::ActivationLayer;

#[derive(Default)]
pub struct ReLULayer {
    cache: Option<Matrix>,
}

impl Layer for ReLULayer {
    fn forward(&mut self, x: Matrix) -> Matrix {
        self.cache = Some(relu_derivative(x.clone()));
        self.eval_only(x)
    }

    fn backward(&mut self, grad_z: Matrix) -> Matrix {
        self.cache.clone().unwrap() * grad_z
    }

    fn eval_only(&self, x: Matrix) -> Matrix {
        relu(x)
    }
}

impl ActivationLayer for ReLULayer {}
