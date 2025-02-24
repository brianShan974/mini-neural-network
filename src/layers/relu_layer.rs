use crate::{
    Vector,
    activation_functions::{relu, relu_derivative},
};

use super::layer::Layer;

#[derive(Default)]
pub struct ReluLayer {
    cache: Option<Vector>,
}

impl Layer for ReluLayer {
    fn forward(&mut self, x: Vector) -> Vector {
        self.cache = Some(relu_derivative(x.clone()));
        relu(x)
    }

    fn backward(&self, grad_z: Vector) -> Vector {
        self.cache.clone().unwrap() * grad_z
    }
}
