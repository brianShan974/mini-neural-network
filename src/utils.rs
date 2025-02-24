use ndarray::Array;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

use crate::{Matrix, Number, Vector};

pub fn xavier_init(size: usize, gain: Number) -> Vector {
    let half_width = gain * (6.0 as Number / size as Number).sqrt();

    Array::random((size,), Uniform::new_inclusive(-half_width, half_width))
}

pub fn xavier_init_matrix(size: (usize, usize), gain: Number) -> Matrix {
    let half_width = gain * (6.0 as Number / size.0 as Number).sqrt();

    Array::random(size, Uniform::new_inclusive(-half_width, half_width))
}

pub fn sigmoid_single(x: Number) -> Number {
    1.0 as Number / (1.0 as Number + (-x).exp())
}

pub fn sigmoid_derivative_single(x: Number) -> Number {
    let s = sigmoid_single(x);
    s * (1.0 as Number - s)
}

pub fn relu_single(x: Number) -> Number {
    if x > 0.0 as Number { x } else { 0.0 as Number }
}

pub fn relu_derivative_single(x: Number) -> Number {
    if x > 0.0 as Number {
        1.0 as Number
    } else {
        0.0 as Number
    }
}
