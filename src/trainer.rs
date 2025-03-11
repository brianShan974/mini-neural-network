use ndarray::Axis;
use rand::{rng, seq::SliceRandom};

use crate::{
    Matrix, Number,
    layers::{layer::Layer, loss_layer::LossLayer},
    multilayer_network::MultiLayerNetwork,
};

pub struct Trainer {
    network: MultiLayerNetwork,
    loss_layer: Box<dyn LossLayer>,
    batch_size: usize,
    n_epoch: usize,
    learning_rate: Number,
    shuffle_flag: bool,
}

impl Trainer {
    pub fn new(
        network: MultiLayerNetwork,
        loss_layer: Box<dyn LossLayer>,
        batch_size: usize,
        n_epoch: usize,
        learning_rate: Number,
        shuffle_flag: bool,
    ) -> Self {
        Self {
            network,
            loss_layer,
            batch_size,
            n_epoch,
            learning_rate,
            shuffle_flag,
        }
    }

    pub fn train(&mut self, input_dataset: Matrix, target_dataset: Matrix, verbose: bool) {
        assert_eq!(
            input_dataset.nrows(),
            target_dataset.nrows(),
            "They must have the same amount of samples!"
        );

        let (input_dataset, target_dataset) = if self.shuffle_flag {
            Self::shuffle(input_dataset, target_dataset)
        } else {
            (input_dataset, target_dataset)
        };

        for e in 0..self.n_epoch {
            let mut input_remaining = input_dataset.view();
            let mut target_remaining = target_dataset.view();

            let mut batch = 1;

            loop {
                let (input_current, input_next) =
                    input_remaining.split_at(Axis(0), self.batch_size);
                let (target_current, target_next) =
                    target_remaining.split_at(Axis(0), self.batch_size);

                let pred_batch = self.network.forward(input_current.to_owned());
                let loss = self
                    .loss_layer
                    .forward(pred_batch, target_current.to_owned());
                let grad_loss = self.loss_layer.backward();
                self.network.backward(grad_loss);
                self.network.update_parameters(self.learning_rate);

                if verbose {
                    println!("Epoch: {e}, batch: {batch}, current loss: {loss}");
                }

                input_remaining = input_next;
                target_remaining = target_next;

                batch += 1;

                if input_remaining.nrows() > self.batch_size {
                    continue;
                } else {
                    let (input_current, _) = input_remaining.split_at(Axis(0), self.batch_size);
                    let (target_current, _) = target_remaining.split_at(Axis(0), self.batch_size);

                    let pred_batch = self.network.forward(input_current.to_owned());
                    let loss = self
                        .loss_layer
                        .forward(pred_batch, target_current.to_owned());
                    let grad_loss = self.loss_layer.backward();
                    self.network.backward(grad_loss);
                    self.network.update_parameters(self.learning_rate);

                    if verbose {
                        println!("Epoch: {e}, batch: {batch}, current loss: {loss}");
                    }

                    break;
                }
            }
        }
    }

    pub fn eval_loss(&mut self, input_dataset: Matrix, target_dataset: Matrix) -> Number {
        assert_eq!(
            input_dataset.nrows(),
            target_dataset.nrows(),
            "They must have the same amount of samples!"
        );

        let pred = self.network.forward(input_dataset);

        self.loss_layer.forward(pred, target_dataset)
    }

    fn shuffle(input_dataset: Matrix, target_dataset: Matrix) -> (Matrix, Matrix) {
        assert_eq!(
            input_dataset.nrows(),
            target_dataset.nrows(),
            "They must have the same amount of samples!"
        );

        let mut indices: Vec<usize> = (0..input_dataset.nrows()).collect();
        indices.shuffle(&mut rng());

        (
            input_dataset.select(Axis(1), &indices),
            target_dataset.select(Axis(1), &indices),
        )
    }
}
