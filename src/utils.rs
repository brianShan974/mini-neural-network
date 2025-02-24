use ndarray::{Array, Axis, stack};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

use std::iter::repeat_n;

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

pub fn repeat(x: &Vector, n: usize) -> Matrix {
    let vecs: Vec<_> = repeat_n(x.view(), n).collect();

    stack(Axis(0), &vecs).unwrap()
}
