use crate::{
    Number, Vector,
    loss_functions::{mse, mse_derivative},
};

use super::loss_layer::LossLayer;

pub struct MSELossLayer {
    pred_target_cache: Option<(Vector, Vector)>,
}

impl LossLayer for MSELossLayer {
    fn forward(&mut self, pred: Vector, target: Vector) -> Number {
        let result = mse(&pred, &target);

        self.pred_target_cache = Some((pred, target));

        result
    }

    fn backward(&self) -> Vector {
        let (pred, target) = self
            .pred_target_cache
            .as_ref()
            .expect("You have to call forward before calling backward!");

        mse_derivative(pred, target)
    }
}
