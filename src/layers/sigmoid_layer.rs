use crate::{
    Vector,
    activation_functions::{sigmoid, sigmoid_derivative},
};

use super::layer::Layer;

#[derive(Default)]
pub struct SigmoidLayer {
    cache: Option<Vector>,
}

impl Layer for SigmoidLayer {
    fn forward(&mut self, x: Vector) -> Vector {
        self.cache = Some(sigmoid_derivative(x.clone()));
        sigmoid(x)
    }

    fn backward(&self, grad_z: Vector) -> Vector {
        self.cache.clone().unwrap() * grad_z
    }
}
