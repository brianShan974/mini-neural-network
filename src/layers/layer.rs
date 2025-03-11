use crate::{Matrix, Number};

pub trait Layer {
    fn forward(&mut self, x: Matrix) -> Matrix;

    fn backward(&mut self, grad_z: Matrix) -> Matrix;

    fn update_parameters(&mut self, _learning_rate: Number) {}
}
