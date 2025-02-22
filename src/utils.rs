use rand::Rng;

use crate::{Number, Numbers};

pub fn xavier_init(size: usize, gain: Number) -> Numbers {
    let rng = rand::rng();
    let half_width = gain * (6.0 as Number / size as Number).sqrt();

    rng.random_iter()
        .take(size)
        .map(|x: Number| x * 2.0 as Number * half_width - half_width)
        .collect()
}

pub fn matrix_multiply(a: Vec<Numbers>, b: Vec<Numbers>) -> Vec<Numbers> {
    assert_eq!(a[0].len(), b.len());

    let mut c = Vec::with_capacity(a.len());

    c
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
