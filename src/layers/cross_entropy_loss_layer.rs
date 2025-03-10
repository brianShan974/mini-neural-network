use crate::{Number, Vector};

use super::loss_layer::LossLayer;

pub struct CrossEntropyLossLayer {
    target_prob_cache: Option<(Vector, Vector)>,
}

impl LossLayer for CrossEntropyLossLayer {
    fn forward(&mut self, pred: Vector, target: Vector) -> Number {
        unimplemented!()
    }

    fn backward(&self) -> Vector {
        unimplemented!()
    }
}
