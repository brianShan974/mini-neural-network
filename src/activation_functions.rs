use crate::{
    Vector,
    utils::{relu_derivative_single, relu_single, sigmoid_derivative_single, sigmoid_single},
};

pub fn sigmoid(x: Vector) -> Vector {
    x.into_iter().map(sigmoid_single).collect()
}

pub fn sigmoid_derivative(x: Vector) -> Vector {
    x.into_iter().map(sigmoid_derivative_single).collect()
}

pub fn relu(x: Vector) -> Vector {
    x.into_iter().map(relu_single).collect()
}

pub fn relu_derivative(x: Vector) -> Vector {
    x.into_iter().map(relu_derivative_single).collect()
}
