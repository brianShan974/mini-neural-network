use crate::{Number, Vector};

pub trait Layer {
    fn forward(&self, x: Vector) -> Vector;

    fn backward(&self, grad_z: Vector) -> Vector;

    fn update_parameters(&mut self, _learning_rate: Number) {}
}
