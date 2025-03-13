use crate::{
    Matrix, Vector,
    utils::{max_in_matrix, min_in_matrix, repeat},
};

#[derive(Default)]
pub struct Preprocessor {
    min: Option<Vector>,
    max: Option<Vector>,
}

impl Preprocessor {
    pub fn new(dataset: &Matrix) -> Self {
        Self {
            min: Some(min_in_matrix(dataset)),
            max: Some(max_in_matrix(dataset)),
        }
    }

    pub fn apply(&self, dataset: Matrix) -> Matrix {
        let n = dataset.nrows();

        let min_matrix = repeat(
            self.min
                .as_ref()
                .expect("You must initialise the preprocessor first!"),
            n,
        );
        let max_matrix = repeat(
            self.max
                .as_ref()
                .expect("You must initialise the preprocessor first!"),
            n,
        );

        (dataset - &min_matrix) / (max_matrix - min_matrix)
    }

    pub fn revert(&self, dataset: Matrix) -> Matrix {
        let n = dataset.nrows();

        let min_matrix = repeat(
            self.min
                .as_ref()
                .expect("You must initialise the preprocessor first!"),
            n,
        );
        let max_matrix = repeat(
            self.max
                .as_ref()
                .expect("You must initialise the preprocessor first!"),
            n,
        );

        dataset * (max_matrix - &min_matrix) * min_matrix
    }
}
