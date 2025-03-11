use crate::{
    Matrix, Vector,
};

pub struct Preprocessor {
    min: Vector,
    max: Vector,
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
