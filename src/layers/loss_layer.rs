use crate::{Number, Vector};

pub trait LossLayer {
    fn forward(&mut self, pred: Vector, target: Vector) -> Number;

    fn backward(&self) -> Vector;
}
