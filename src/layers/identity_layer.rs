use crate::Matrix;

use super::layer::Layer;

pub struct IdentityLayer;

impl Layer for IdentityLayer {
    fn forward(&mut self, x: Matrix) -> Matrix {
        x
    }

    fn backward(&self, grad_z: Matrix) -> Matrix {
        grad_z
    }
}
