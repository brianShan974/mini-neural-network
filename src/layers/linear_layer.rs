use std::cell::Cell;

use crate::{Matrix, Number, Vector};

use super::layer::Layer;

pub struct LinearLayer {
    _input_size: usize,
    _output_size: usize,
    _weights: Matrix,
    _biases: Matrix,
    _w_cache: Cell<Matrix>,
    _b_cache: Cell<Matrix>,
    _input_cache: Cell<Vector>,
}

impl LinearLayer {
    pub fn _new(_input_size: usize, _output_size: usize, _gain: Option<Number>) -> Self {
        unimplemented!()
    }
}

impl Layer for LinearLayer {
    fn forward(&mut self, _x: Vector) -> Vector {
        unimplemented!()
    }

    fn backward(&self, _grad_z: Vector) -> Vector {
        unimplemented!()
    }

    fn update_parameters(&mut self, _learning_rate: Number) {
        unimplemented!()
    }
}
