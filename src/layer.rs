use crate::{Number, Numbers};

pub trait Layer {
    fn forward(&self, x: Numbers) -> Numbers;

    fn backward(&self, grad_z: Numbers) -> Numbers;

    fn update_parameters(&mut self, _learning_rate: Number) {}
}
