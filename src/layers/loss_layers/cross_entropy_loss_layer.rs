use crate::{Matrix, Number, math_functions::activation_functions::softmax};

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

    fn backward(&mut self) -> Matrix {
        let (prob, target) = self
            .prob_target_cache
            .take()
            .expect("You must call CrossEntropyLossLayer::forward before calling CrossEntropyLossLayer::backward!");
        let n_obs = target.len();

        -1.0 as Number / n_obs as Number * (target - prob)
    }

    fn eval_only(&self, pred: Matrix, target: Matrix) -> Number {
        let n_obs = target.len();
        let prob = softmax(pred);

        -1.0 as Number / n_obs as Number * (&target * prob.mapv(Number::ln).to_owned()).sum()
    }
}
