use ndarray::{Array, s};

use crate::{
    Matrix, Number, Vector,
    utils::{repeat, xavier_init, xavier_init_matrix},
};

use super::layer::Layer;

pub struct LinearLayer {
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

        self.eval_only(x)
    }

    fn backward(&mut self, grad_z: Matrix) -> Matrix {
        let batch_size = grad_z.nrows();

        self.grad_w_cache = self.input_cache.clone().map(|ic: _| ic.t().dot(&grad_z));
        self.grad_b_cache = Some(Array::ones((1, batch_size)).dot(&grad_z));

        grad_z.dot(&self.weights.t())
    }

    fn update_parameters(&mut self, learning_rate: Number) {
        self.weights = &self.weights - self.grad_w_cache.clone().unwrap() * learning_rate;
        self.bias = &self.bias
            - self
                .grad_b_cache
                .clone()
                .expect("You must call forward and backward then update_parameters!")
                .slice(s![0, ..])
                .to_owned()
                * learning_rate;
    }

    fn eval_only(&self, x: Matrix) -> Matrix {
        let batch_size = x.nrows();
        let bias = repeat(self.bias.view(), batch_size);

        x.dot(&self.weights) + bias
    }
}
