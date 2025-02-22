use crate::{Number, Vector};

pub fn xavier_init(size: usize, gain: Number) -> Vector {
    let half_width = gain * (6.0 as Number / size as Number).sqrt();

    unimplemented!()
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
