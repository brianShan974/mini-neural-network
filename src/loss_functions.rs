use crate::{Number, Vector};

pub fn mse(pred: &Vector, target: &Vector) -> Number {
    let err = pred - target;
    err.mapv(|i| i.powi(2)).mean().unwrap()
}

pub fn mse_derivative(pred: &Vector, target: &Vector) -> Vector {
    2.0 as Number * (pred - target) / pred.ndim() as Number
}
