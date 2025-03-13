use crate::{Matrix, Number};

pub trait LossLayer {
    fn forward(&mut self, pred: Matrix, target: Matrix) -> Number;

    fn backward(&mut self) -> Matrix;

    fn eval_only(&self, pred: Matrix, target: Matrix) -> Number;
}
