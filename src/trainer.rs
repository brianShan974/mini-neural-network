use crate::{Matrix, Number, layers::loss_layer::LossLayer, multilayer_network::MultiLayerNetwork};

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

    pub fn train(&mut self, input_dataset: Matrix, output_dataset: Matrix) {}

    pub fn eval_loss(&self, input_dataset: Matrix, output_dataset: Matrix) -> Matrix {
        unimplemented!()
    }

    fn shuffle(&mut self, input_dataset: Matrix, output_dataset: Matrix) -> (Matrix, Matrix) {
        unimplemented!()
    }
}
