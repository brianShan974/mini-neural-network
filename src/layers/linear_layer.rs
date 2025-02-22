use std::cell::Cell;

use crate::{Matrix, Number, Vector};

use super::layer::Layer;

pub struct LinearLayer {
    input_size: usize,
    output_size: usize,
    weights: Matrix,
    biases: Matrix,
    w_cache: Cell<Matrix>,
    b_cache: Cell<Matrix>,
    input_cache: Cell<Vector>,
}

impl LinearLayer {
    pub fn new(input_size: usize, output_size: usize, _gain: Option<Number>) -> Self {
        unimplemented!()
    }
}

impl Layer for LinearLayer {
    fn forward(&self, _x: Vector) -> Vector {
        unimplemented!()
    }

    fn backward(&self, _grad_z: Vector) -> Vector {
        unimplemented!()
    }

    fn update_parameters(&mut self, _learning_rate: Number) {
        unimplemented!()
    }
}
