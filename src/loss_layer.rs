use crate::Numbers;

pub trait LossLayer {
    fn forward(&self, pred: Numbers, target: Numbers);

    fn backward(&self) -> Numbers;
}
