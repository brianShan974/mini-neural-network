use polars::frame::DataFrame;

use crate::{
    Matrix, Number,
    layers::{layer::Layer, loss_layers::loss_layer::LossLayer},
    multilayer_network::MultiLayerNetwork,
    preprocessor::Preprocessor,
};

pub struct Regressor<'a> {
    network: &'a mut MultiLayerNetwork,
    loss_layer: Box<dyn LossLayer>,
    batch_size: usize,
    n_epoch: usize,
    learning_rate: Number,
    x_preprocessor: Option<Preprocessor>,
    y_preprocessor: Option<Preprocessor>,
}

impl<'a> Regressor<'a> {
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
            x_preprocessor: None,
            y_preprocessor: None,
        }
    }

    pub fn fit(&mut self, x_train: DataFrame, y_train: DataFrame, shuffle: bool) {
        unimplemented!()
    }

    pub fn predict(&mut self, x: DataFrame) -> Matrix {
        unimplemented!()
    }

    pub fn predict_only(&self, x: DataFrame) -> Matrix {
        unimplemented!()
    }

    pub fn score(&self, x: DataFrame, y: DataFrame) -> Number {
        unimplemented!()
    }

    fn preprocess_training(&mut self, x: DataFrame, y: DataFrame) -> (Matrix, Matrix) {
        unimplemented!()
    }

    fn preprocess_non_training(&self, x: DataFrame, y: DataFrame) -> (Matrix, Matrix) {
        unimplemented!()
    }

    fn forward(&mut self, x: Matrix) -> Matrix {
        unimplemented!()
    }

    fn eval_only(&self, x: Matrix) -> Matrix {
        unimplemented!()
    }
}
