use polars::{
    frame::DataFrame,
    prelude::{DataFrameOps, FillNullStrategy, IndexOrder},
};

use crate::{
    Matrix, Number, NumberType,
    layers::{layer::Layer, loss_layers::loss_layer::LossLayer},
    multilayer_network::MultiLayerNetwork,
    preprocessor::Preprocessor,
    trainer::Trainer,
};

pub struct Regressor<'a> {
    trainer: Trainer<'a>,
    loss_layer: Box<dyn LossLayer>,
    batch_size: usize,
    n_epoch: usize,
    learning_rate: Number,
    x_preprocessor: Option<Preprocessor>,
    y_preprocessor: Option<Preprocessor>,
    str_cols: Vec<&'a str>,
}

impl<'a> Regressor<'a> {
    pub fn new(
        trainer: Trainer<'a>,
        loss_layer: Box<dyn LossLayer>,
        batch_size: usize,
        n_epoch: usize,
        learning_rate: Number,
    ) -> Self {
        Self {
            trainer,
            loss_layer,
            batch_size,
            n_epoch,
            learning_rate,
            x_preprocessor: None,
            y_preprocessor: None,
            str_cols: Vec::new(),
        }
    }

    pub fn init_preprocessor(&mut self, x: DataFrame, y: DataFrame, str_cols: Vec<&'a str>) {
        self.str_cols = str_cols;
        self.preprocess_training(x, y);
    }

    pub fn fit(&mut self, x: DataFrame, y: DataFrame, shuffle: bool, verbose: bool) {
        let (x, y) = self.preprocess_training(x, y);
        self.trainer.train(x, y, shuffle, verbose);
    }

    pub fn predict(&self, x: DataFrame) -> Matrix {
        let x = self.preprocess_x_non_training(x, self.str_cols.clone());
        let y = self.trainer.predict(x);

        self.y_preprocessor
            .as_ref()
            .expect("You must initialise the preprocessor before calling predict!")
            .apply(y)
    }

    pub fn score(&self, x: DataFrame, y: DataFrame) -> Number {
        unimplemented!()
    }

    fn preprocess_training(&mut self, x: DataFrame, y: DataFrame) -> (Matrix, Matrix) {
        let str_cols = self.str_cols.clone();
        let result_x = self.preprocess_x_training(x, str_cols);
        let result_y = self.preprocess_y_training(y);

        (result_x, result_y)
    }

    fn _preprocess_non_training(
        &self,
        x: DataFrame,
        y: DataFrame,
        str_cols: Vec<&str>,
    ) -> (Matrix, Matrix) {
        (
            self.preprocess_x_non_training(x, str_cols),
            self._preprocess_y_non_training(y),
        )
    }

    fn preprocess_x_training(&mut self, x: DataFrame, str_cols: Vec<&str>) -> Matrix {
        let x = x
            .columns_to_dummies(str_cols, None, true)
            .expect("Unable to one-hot label the columns for x!")
            .fill_null(FillNullStrategy::Mean)
            .expect("Unable to fill null for x!");

        let result = x
            .to_ndarray::<NumberType>(IndexOrder::Fortran)
            .expect("Unable to convert x to ndarray!");

        self.x_preprocessor = Some(Preprocessor::new(result.view()));

        self.x_preprocessor.as_ref().unwrap().apply(result)
    }

    fn preprocess_y_training(&mut self, y: DataFrame) -> Matrix {
        let result = y
            .to_ndarray::<NumberType>(IndexOrder::Fortran)
            .expect("Unable to convert y to ndarray!");

        self.y_preprocessor = Some(Preprocessor::new(result.view()));

        self.y_preprocessor.as_ref().unwrap().apply(result)
    }

    fn preprocess_x_non_training(&self, x: DataFrame, str_cols: Vec<&str>) -> Matrix {
        let x = x
            .columns_to_dummies(str_cols, None, true)
            .expect("Unable to one-hot label the columns for x!")
            .fill_null(FillNullStrategy::Mean)
            .expect("Unable to fill null for x!");

        let result = x
            .to_ndarray::<NumberType>(IndexOrder::Fortran)
            .expect("Unable to convert x to ndarray!");

        self.x_preprocessor
            .as_ref()
            .expect(
                "You must initialise the preprocessor before calling preprocess_x_non_training!",
            )
            .apply(result)
    }

    fn _preprocess_y_non_training(&self, y: DataFrame) -> Matrix {
        let result = y
            .to_ndarray::<NumberType>(IndexOrder::Fortran)
            .expect("Unable to convert y to ndarray!");

        self.y_preprocessor
            .as_ref()
            .expect(
                "You must initialise the preprocessor before calling preprocess_y_non_training!",
            )
            .apply(result)
    }
}
