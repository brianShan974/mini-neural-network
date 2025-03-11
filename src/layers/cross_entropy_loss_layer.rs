use crate::{Matrix, Number, activation_functions::softmax};

use super::loss_layer::LossLayer;

/// Computes the softmax followed by the negative log-likelihood loss.
#[derive(Default)]
pub struct CrossEntropyLossLayer {
    prob_target_cache: Option<(Matrix, Matrix)>,
}

impl LossLayer for CrossEntropyLossLayer {
    fn forward(&mut self, pred: Matrix, target: Matrix) -> Number {
        let n_obs = target.len();
        let prob = softmax(pred);
        let result =
            -1.0 as Number / n_obs as Number * (&target * prob.mapv(Number::ln).to_owned()).sum();

        self.prob_target_cache = Some((prob, target));

        result
    }

    fn backward(&self) -> Matrix {
        let (prob, target) = self
            .prob_target_cache
            .as_ref()
            .expect("You have to call forward before calling backward!");
        let n_obs = target.len();

        -1.0 as Number / n_obs as Number * (target - prob)
    }
}
