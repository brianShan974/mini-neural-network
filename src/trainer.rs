use ndarray::Axis;
use rand::{rng, seq::SliceRandom};

use crate::{
    Matrix, Number,
    layers::{layer::Layer, loss_layers::loss_layer::LossLayer},
    multilayer_network::MultiLayerNetwork,
};

pub struct Trainer<'a> {
    network: &'a mut MultiLayerNetwork,
    loss_layer: Box<dyn LossLayer>,
    batch_size: usize,
    n_epoch: usize,
    learning_rate: Number,
}

impl<'a> Trainer<'a> {
    pub fn new(
        network: &'a mut MultiLayerNetwork,
        loss_layer: Box<dyn LossLayer>,
        batch_size: usize,
        n_epoch: usize,
        learning_rate: Number,
    ) -> Self {
        Self {
            network,
            loss_layer,
            batch_size,
            n_epoch,
            learning_rate,
        }
    }

    pub fn train(
        &mut self,
        mut input_dataset: Matrix,
        mut target_dataset: Matrix,
        shuffle: bool,
        verbose: bool,
    ) {
        assert_eq!(
            input_dataset.nrows(),
            target_dataset.nrows(),
            "They must have the same amount of samples!"
        );

        for e in 0..self.n_epoch {
            if shuffle {
                Self::shuffle(&mut input_dataset, &mut target_dataset)
            }

            let mut input_remaining = input_dataset.view();
            let mut target_remaining = target_dataset.view();

            let mut batch = 1;
            let mut next_split;

            loop {
                let remaining_rows = input_remaining.nrows();

                if remaining_rows > self.batch_size {
                    next_split = self.batch_size;
                } else {
                    next_split = remaining_rows;
                }

                let (input_current, input_next) = input_remaining.split_at(Axis(0), next_split);
                let (target_current, target_next) = target_remaining.split_at(Axis(0), next_split);

                let pred_batch = self.network.forward(input_current.to_owned());
                let loss = self
                    .loss_layer
                    .forward(pred_batch, target_current.to_owned());
                let grad_loss = self.loss_layer.backward();
                self.network.backward(grad_loss);
                self.network.update_parameters(self.learning_rate);

                if verbose {
                    println!("Epoch: {}, batch: {batch}, current loss: {loss}", e + 1);
                }

                input_remaining = input_next;
                target_remaining = target_next;

                batch += 1;

                if next_split == remaining_rows {
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

    pub fn eval_loss_only(&self, input_dataset: Matrix, target_dataset: Matrix) -> Number {
        assert_eq!(
            input_dataset.nrows(),
            target_dataset.nrows(),
            "They must have the same amount of samples!"
        );

        let pred = self.network.eval_only(input_dataset);

        self.loss_layer.eval_only(pred, target_dataset)
    }

    fn shuffle(input_dataset: &mut Matrix, target_dataset: &mut Matrix) {
        assert_eq!(
            input_dataset.nrows(),
            target_dataset.nrows(),
            "They must have the same amount of samples!"
        );

        let mut indices: Vec<usize> = (0..input_dataset.nrows()).collect();
        indices.shuffle(&mut rng());

        *input_dataset = input_dataset.select(Axis(0), &indices);
        *target_dataset = target_dataset.select(Axis(0), &indices);
    }
}
