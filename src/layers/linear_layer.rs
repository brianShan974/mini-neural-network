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
    fn forward(&mut self, x: Matrix) -> Matrix {
        unimplemented!()
    }

    fn backward(&mut self, grad_z: Matrix) -> Matrix {
        grad_z.dot(&self.weights.t())
    }

    fn update_parameters(&mut self, _learning_rate: Number) {
        unimplemented!()
    }
}
