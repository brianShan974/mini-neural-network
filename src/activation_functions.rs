use crate::{
    Matrix, Number, Vector,
    utils::{relu_derivative_single, relu_single, sigmoid_derivative_single, sigmoid_single},
};

pub fn sigmoid(mut x: Matrix) -> Matrix {
    x.par_mapv_inplace(sigmoid_single);
    x
}

pub fn sigmoid_derivative(mut x: Matrix) -> Matrix {
    x.par_mapv_inplace(sigmoid_derivative_single);
    x
}

pub fn relu(mut x: Matrix) -> Matrix {
    x.par_mapv_inplace(relu_single);
    x
}

pub fn relu_derivative(mut x: Matrix) -> Matrix {
    x.par_mapv_inplace(relu_derivative_single);
    x
}

pub fn softmax(mut x: Vector) -> Vector {
    x.par_mapv_inplace(Number::exp);
    let sum = x.sum();

    x / sum
}
