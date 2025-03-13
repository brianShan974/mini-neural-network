use crate::{
    Matrix, Number,
    loss_functions::{mse, mse_derivative},
};

use super::loss_layer::LossLayer;

#[derive(Default)]
pub struct MSELossLayer {
    pred_target_cache: Option<(Matrix, Matrix)>,
}

impl LossLayer for MSELossLayer {
    fn forward(&mut self, pred: Matrix, target: Matrix) -> Number {
        let result = mse(pred.view(), target.view());

        self.pred_target_cache = Some((pred, target));

        result
    }

    fn backward(&mut self) -> Matrix {
        let (pred, target) = self
            .pred_target_cache
            .take()
            .expect("You have to call forward before calling backward!");

        mse_derivative(pred.view(), target.view())
    }

    fn eval_only(&self, pred: Matrix, target: Matrix) -> Number {
        mse(pred.view(), target.view())
    }
}
