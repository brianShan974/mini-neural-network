use crate::{Number, Vector, activation_functions::softmax};

use super::loss_layer::LossLayer;

/// Computes the softmax followed by the negative log-likelihood loss.
pub struct CrossEntropyLossLayer {
    prob_target_cache: Option<(Vector, Vector)>,
}

impl LossLayer for CrossEntropyLossLayer {
    fn forward(&mut self, pred: Vector, target: Vector) -> Number {
        let n_obs = target.len();
        let prob = softmax(pred);
        let result =
            -1.0 as Number / n_obs as Number * (&target * prob.mapv(Number::ln).to_owned()).sum();

        self.prob_target_cache = Some((prob, target));

        result
    }

    fn backward(&self) -> Vector {
        let (prob, target) = self
            .prob_target_cache
            .as_ref()
            .expect("You have to call forward before calling backward!");
        let n_obs = target.len();

        -1.0 as Number / n_obs as Number * (target - prob)
    }
}
