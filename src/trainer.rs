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

    pub fn train(&mut self, input_dataset: Matrix, output_dataset: Matrix) {
        unimplemented!()
    }

    pub fn eval_loss(&mut self, input_dataset: Matrix, output_dataset: Matrix) -> Number {
        assert_eq!(
            input_dataset.shape()[0],
            output_dataset.shape()[0],
            "The shapes must match!"
        );

        let pred = self.network.forward(input_dataset);

        self.loss_layer.forward(pred, output_dataset)
    }

    fn shuffle(&mut self, input_dataset: Matrix, output_dataset: Matrix) -> (Matrix, Matrix) {
        assert_eq!(
            input_dataset.shape()[0],
            output_dataset.shape()[0],
            "They must have the same amount of samples!"
        );

        let mut indices: Vec<usize> = (0..input_dataset.shape()[0]).collect();
        indices.shuffle(&mut rng());

        (
            input_dataset.select(Axis(1), &indices),
            output_dataset.select(Axis(1), &indices),
        )
    }
}
