use crate::{Matrix, Number};

pub trait LossLayer {
    fn forward(&mut self, pred: Matrix, target: Matrix) -> Number;

    fn backward(&self) -> Matrix;
}
