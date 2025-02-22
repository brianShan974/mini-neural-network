use crate::{Numbers, layer::Layer};

pub struct IdentityLayer;

impl Layer for IdentityLayer {
    fn forward(&self, x: Numbers) -> Numbers {
        x
    }

    fn backward(&self, grad_z: Numbers) -> Numbers {
        grad_z
    }
}
