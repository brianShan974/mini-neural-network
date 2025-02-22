use std::cell::Cell;

use crate::{Number, Vector, layer::Layer};

pub struct LinearLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<Vector>,
    biases: Vec<Vector>,
    w_cache: Cell<Vec<Vector>>,
    b_cache: Cell<Vec<Vector>>,
    input_cache: Cell<Vector>,
}

impl LinearLayer {
    pub fn new(input_size: usize, output_size: usize, _gain: Option<Number>) -> Self {
        Self {
            input_size,
            output_size,
            weights: Vec::new(),
            biases: Vec::new(),
            w_cache: Cell::new(Vec::new()),
            b_cache: Cell::new(Vec::new()),
            input_cache: Cell::new(Vec::new()),
        }
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
