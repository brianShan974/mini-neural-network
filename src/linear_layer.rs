use std::cell::Cell;

use crate::{Number, Numbers, layer::Layer};

pub struct LinearLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<Numbers>,
    biases: Vec<Numbers>,
    w_cache: Cell<Vec<Numbers>>,
    b_cache: Cell<Vec<Numbers>>,
    input_cache: Cell<Numbers>,
}

impl LinearLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
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
    fn forward(&self, x: Numbers) -> Numbers {
        unimplemented!()
    }

    fn backward(&self, grad_z: Numbers) -> Numbers {
        unimplemented!()
    }

    fn update_parameters(&mut self, _learning_rate: Number) {
        unimplemented!()
    }
}
