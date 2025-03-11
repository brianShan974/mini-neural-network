use crate::{
    Matrix, Vector,
    utils::{max_in_matrix, min_in_matrix},
};

pub struct Preprocessor {
    min: Vector,
    max: Vector,
}

impl Preprocessor {
    pub fn new(dataset: Matrix) -> Self {
        Self {
            min: min_in_matrix(&dataset),
            max: max_in_matrix(&dataset),
        }
    }

    pub fn apply(&mut self, dataset: Matrix) -> Matrix {
        unimplemented!()
    }

    pub fn revert(&self, dataset: Matrix) -> Matrix {
        unimplemented!()
    }
}
