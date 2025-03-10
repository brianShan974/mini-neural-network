use crate::{Number, Vector};

use super::loss_layer::LossLayer;

pub struct MSELossLayer {
    pred_target_cache: Option<(Vector, Vector)>,
}

impl LossLayer for MSELossLayer {
    fn forward(&mut self, pred: Vector, target: Vector) -> Number {
        unimplemented!()
    }

    fn backward(&self) -> Vector {
        unimplemented!()
    }
}
