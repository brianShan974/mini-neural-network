use crate::{Number, Vector};

use super::loss_layer::LossLayer;

pub struct MSELossLayer {
    y_pred_cache: Option<Vector>,
    y_target_cache: Option<Vector>,
}

impl LossLayer for MSELossLayer {
    fn forward(&mut self, pred: Vector, target: Vector) -> Number {
        unimplemented!()
    }

    fn backward(&self) -> Vector {
        unimplemented!()
    }
}
