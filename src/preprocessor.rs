use crate::{Matrix, Number};

pub struct Preprocessor {
    min: Number,
    max: Number,
}

impl Preprocessor {
    pub fn new(dataset: Matrix) -> Self {
        Self {
            min: unimplemented!(),
            max: unimplemented!(),
        }
    }

    pub fn apply(&mut self, dataset: Matrix) -> Matrix {
        unimplemented!()
    }

    pub fn revert(&self, dataset: Matrix) -> Matrix {
        unimplemented!()
    }
}
