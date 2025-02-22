use crate::{Number, Numbers};

fn sigmoid_single(x: Number) -> Number {
    1.0 as Number / (1.0 as Number + (-x).exp())
}

fn sigmoid_derivative_single(x: Number) -> Number {
    let s = sigmoid_single(x);
    s * (1.0 as Number - s)
}

fn relu_single(x: Number) -> Number {
    if x > 0.0 as Number { x } else { 0.0 as Number }
}

fn relu_derivative_single(x: Number) -> Number {
    if x > 0.0 as Number {
        1.0 as Number
    } else {
        0.0 as Number
    }
}

pub fn sigmoid(x: Numbers) -> Numbers {
    x.into_iter().map(sigmoid_single).collect()
}

pub fn sigmoid_derivative(x: Numbers) -> Numbers {
    x.into_iter().map(sigmoid_derivative_single).collect()
}

pub fn relu(x: Numbers) -> Numbers {
    x.into_iter().map(relu_single).collect()
}

pub fn relu_derivative(x: Numbers) -> Numbers {
    x.into_iter().map(relu_derivative_single).collect()
}
