use crate::Vector;

use super::layer::Layer;

pub struct IdentityLayer;

impl Layer for IdentityLayer {
    fn forward(&self, x: Vector) -> Vector {
        x
    }

    fn backward(&self, grad_z: Vector) -> Vector {
        grad_z
    }
}
