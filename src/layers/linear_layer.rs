use crate::{
    Matrix, Number, Vector,
    utils::{repeat, xavier_init, xavier_init_matrix},
};

use super::layer::Layer;

pub struct LinearLayer {
    input_size: usize,
    output_size: usize,
    weights: Matrix,
    bias: Vector,
    grad_w_cache: Option<Matrix>,
    grad_b_cache: Option<Matrix>,
    input_cache: Option<Matrix>,
}

impl LinearLayer {
    pub fn new(input_size: usize, output_size: usize, gain: Option<Number>) -> Self {
        let gain = if let Some(gain) = gain {
            gain
        } else {
            1.0 as Number
        };

        Self {
            input_size,
            output_size,
            weights: xavier_init_matrix((input_size, output_size), gain),
            bias: xavier_init(output_size, gain),
            grad_w_cache: None,
            grad_b_cache: None,
            input_cache: None,
        }
    }
}

impl Layer for LinearLayer {
    fn forward(&mut self, x: Matrix) -> Matrix {
        self.input_cache = Some(x.clone());

        let batch_size = x.shape()[0];
        let bias = repeat(&self.bias, batch_size);

        x.dot(&self.weights) + bias
    }

    fn backward(&mut self, grad_z: Matrix) -> Matrix {
        grad_z.dot(&self.weights.t())
    }

    fn update_parameters(&mut self, _learning_rate: Number) {
        unimplemented!()
    }
}
