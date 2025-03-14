use crate::{
    Matrix, MatrixView, Vector,
    utils::{max_in_matrix, min_in_matrix, repeat},
};

#[derive(Default)]
pub struct Preprocessor {
    min: Vector,
    max: Vector,
}

impl Preprocessor {
    pub fn new(dataset: MatrixView) -> Self {
        Self {
            min: min_in_matrix(dataset),
            max: max_in_matrix(dataset),
        }
    }

    pub fn apply(&self, dataset: Matrix) -> Matrix {
        let n = dataset.nrows();

        let min_matrix = repeat(self.min.view(), n);
        let max_matrix = repeat(self.max.view(), n);

        (dataset - &min_matrix) / (max_matrix - min_matrix)
    }

    pub fn revert(&self, dataset: Matrix) -> Matrix {
        let n = dataset.nrows();

        let min_matrix = repeat(self.min.view(), n);
        let max_matrix = repeat(self.max.view(), n);

        dataset * (max_matrix - &min_matrix) * min_matrix
    }
}
