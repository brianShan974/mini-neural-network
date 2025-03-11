use crate::{
    Matrix, Vector,
    utils::{max_in_matrix, min_in_matrix, repeat},
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
        let n = dataset.nrows();

        let min_matrix = repeat(&self.min, n);
        let max_matrix = repeat(&self.max, n);

        (dataset - &min_matrix) / (max_matrix - min_matrix)
    }

    pub fn revert(&self, dataset: Matrix) -> Matrix {
        let n = dataset.nrows();

        let min_matrix = repeat(&self.min, n);
        let max_matrix = repeat(&self.max, n);

        dataset * (max_matrix - &min_matrix) * min_matrix
    }
}
