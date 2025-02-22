use std::cell::Cell;

use crate::Vector;

use super::layer::Layer;

pub struct SigmoidLayer {
    _cache: Cell<Vector>,
}

impl Layer for SigmoidLayer {
    fn forward(&self, _x: Vector) -> Vector {
        unimplemented!()
    }

    fn backward(&self, _grad_z: Vector) -> Vector {
        unimplemented!()
    }
}
