use crate::{Matrix, layers::layer::Layer};

use super::activation_layer::ActivationLayer;

#[derive(Default)]
pub struct IdentityLayer;

impl Layer for IdentityLayer {
    fn forward(&mut self, x: Matrix) -> Matrix {
        x
    }

    fn backward(&mut self, grad_z: Matrix) -> Matrix {
        grad_z
    }
}

impl ActivationLayer for IdentityLayer {}
