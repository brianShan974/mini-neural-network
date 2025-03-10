use crate::Vector;

use super::loss_layer::LossLayer;

pub struct CrossEntropyLossLayer {
    y_target_cache: Option<Vector>,
    probs_cache: Option<Vector>,
}

impl LossLayer for CrossEntropyLossLayer {
    fn forward(&self, pred: Vector, target: Vector) {
        unimplemented!()
    }

    fn backward(&self) -> Vector {
        unimplemented!()
    }
}
