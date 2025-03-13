use polars::{
    frame::DataFrame,
    prelude::{FillNullStrategy, IndexOrder},
};

use crate::{Matrix, Number, NumberType, preprocessor::Preprocessor, trainer::Trainer};

pub struct Regressor<'a> {
    trainer: Trainer<'a>,
    x_preprocessor: Option<Preprocessor>,
    y_preprocessor: Option<Preprocessor>,
}

impl<'a> Regressor<'a> {
    pub fn new(trainer: Trainer<'a>, x: DataFrame, y: DataFrame) -> Self {
        let mut new_regressor = Self {
            trainer,
            x_preprocessor: None,
            y_preprocessor: None,
        };

        new_regressor.preprocess_training(x, y);

        new_regressor
    }

    pub fn fit(&mut self, x: DataFrame, y: DataFrame, shuffle: bool, verbose: bool) {
        let (x, y) = self.preprocess_training(x, y);
        self.trainer.train(x, y, shuffle, verbose);
    }

    pub fn predict(&self, x: DataFrame) -> Matrix {
        let x = self.preprocess_x_non_training(x);
        let y = self.trainer.predict(x);

        self.y_preprocessor
            .as_ref()
            .expect("You must initialise the preprocessor before calling Regressor::predict!")
            .apply(y)
    }

    pub fn eval_loss(&self, x: DataFrame, y: DataFrame) -> Number {
        self.trainer.eval_loss_only(
            x.to_ndarray::<NumberType>(IndexOrder::C)
                .expect("Unable to convert x to ndarray in Regressor::eval_loss!"),
            y.to_ndarray::<NumberType>(IndexOrder::C)
                .expect("Unable to convert y to ndarray in Regressor::eval_loss!"),
        )
    }

    fn preprocess_training(&mut self, x: DataFrame, y: DataFrame) -> (Matrix, Matrix) {
        let result_x = self.preprocess_x_training(x);
        let result_y = self.preprocess_y_training(y);

        (result_x, result_y)
    }

    fn _preprocess_non_training(&self, x: DataFrame, y: DataFrame) -> (Matrix, Matrix) {
        (
            self.preprocess_x_non_training(x),
            self._preprocess_y_non_training(y),
        )
    }

    fn preprocess_x_training(&mut self, x: DataFrame) -> Matrix {
        let x = x
            .fill_null(FillNullStrategy::Mean)
            .expect("Unable to fill null for x in Regressor::preprocess_x_training!");

        let result = x
            .to_ndarray::<NumberType>(IndexOrder::C)
            .expect("Unable to convert x to ndarray in Regressor::preprocess_x_training!");

        self.x_preprocessor = Some(Preprocessor::new(result.view()));

        self.x_preprocessor.as_ref().unwrap().apply(result)
    }

    fn preprocess_y_training(&mut self, y: DataFrame) -> Matrix {
        let result = y
            .to_ndarray::<NumberType>(IndexOrder::C)
            .expect("Unable to convert y to ndarray in Regressor::preprocess_y_training!");

        self.y_preprocessor = Some(Preprocessor::new(result.view()));

        self.y_preprocessor.as_ref().unwrap().apply(result)
    }

    fn preprocess_x_non_training(&self, x: DataFrame) -> Matrix {
        let x = x
            .fill_null(FillNullStrategy::Mean)
            .expect("Unable to fill null for x in Regressor::preprocess_x_non_training!");

        let result = x
            .to_ndarray::<NumberType>(IndexOrder::C)
            .expect("Unable to convert x to ndarray in Regressor::preprocess_x_non_training!");

        self.x_preprocessor
            .as_ref()
            .expect(
                "You must initialise the preprocessor before calling Regressor::preprocess_x_non_training!",
            )
            .apply(result)
    }

    fn _preprocess_y_non_training(&self, y: DataFrame) -> Matrix {
        let result = y
            .to_ndarray::<NumberType>(IndexOrder::C)
            .expect("Unable to convert y to ndarray in Regressor::_preprocess_y_non_training!");

        self.y_preprocessor
            .as_ref()
            .expect(
                "You must initialise the preprocessor before calling Regressor::_preprocess_y_non_training!",
            )
            .apply(result)
    }
}
