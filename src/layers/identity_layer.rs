use crate::Matrix;

use super::layer::Layer;

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
