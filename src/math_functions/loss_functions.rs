use crate::{Matrix, MatrixView, Number};

pub fn mse(pred: MatrixView, target: MatrixView) -> Number {
    let err = pred.to_owned() - target;
    err.mapv(|i| i.powi(2)).mean().unwrap()
}

pub fn mse_derivative(pred: MatrixView, target: MatrixView) -> Matrix {
    2.0 as Number * (pred.to_owned() - target) / pred.ndim() as Number
}
