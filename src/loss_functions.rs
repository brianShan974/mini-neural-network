use crate::{Matrix, Number};

pub fn mse(pred: &Matrix, target: &Matrix) -> Number {
    let err = pred - target;
    err.mapv(|i| i.powi(2)).mean().unwrap()
}

pub fn mse_derivative(pred: &Matrix, target: &Matrix) -> Matrix {
    2.0 as Number * (pred - target) / pred.ndim() as Number
}
