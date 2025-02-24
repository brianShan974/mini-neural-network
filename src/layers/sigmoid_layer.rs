use crate::{
    Vector,
    activation_functions::{sigmoid, sigmoid_derivative},
};

use super::layer::Layer;

pub struct SigmoidLayer {
    cache: Vector,
}

impl Layer for SigmoidLayer {
    fn forward(&mut self, x: Vector) -> Vector {
        self.cache = sigmoid_derivative(x.clone());
        sigmoid(x)
    }

    fn backward(&self, grad_z: Vector) -> Vector {
        self.cache.clone() * grad_z
    }
}
