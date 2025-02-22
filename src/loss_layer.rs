use crate::Vector;

pub trait LossLayer {
    fn forward(&self, pred: Vector, target: Vector);

    fn backward(&self) -> Vector;
}
