use crate::{
    Numbers,
    utils::{relu_derivative_single, relu_single, sigmoid_derivative_single, sigmoid_single},
};

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
