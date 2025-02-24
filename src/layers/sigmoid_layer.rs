use crate::{Vector, activation_functions::relu};

use super::layer::Layer;

pub struct SigmoidLayer {
    cache: Vector,
}

impl Layer for SigmoidLayer {
    fn forward(&mut self, x: Vector) -> Vector {
        self.cache = x.clone();
        relu(x)
    }

    fn backward(&self, grad_z: Vector) -> Vector {
        self.cache.clone() * grad_z
    }
}
