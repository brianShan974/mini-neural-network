use ndarray::{Array, Array1, Axis, stack};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use ndarray_stats::QuantileExt;

use std::iter::repeat_n;

use crate::{Matrix, MatrixView, Number, Vector, VectorView};

pub fn xavier_init(size: usize, gain: Number) -> Vector {
    let half_width = gain * (6.0 as Number / size as Number).sqrt();

    Array::random((size,), Uniform::new_inclusive(-half_width, half_width))
}

pub fn xavier_init_matrix(size: (usize, usize), gain: Number) -> Matrix {
    let half_width = gain * (6.0 as Number / size.0 as Number).sqrt();

    Array::random(size, Uniform::new_inclusive(-half_width, half_width))
}

pub fn repeat(x: VectorView, n: usize) -> Matrix {
    let vecs: Vec<_> = repeat_n(x.view(), n).collect();

    stack(Axis(0), &vecs).unwrap()
}

pub fn min_in_matrix(dataset: MatrixView) -> Vector {
    dataset.map_axis(Axis(0), |view| *view.min().unwrap())
}

pub fn max_in_matrix(dataset: MatrixView) -> Vector {
    dataset.map_axis(Axis(0), |view| *view.max().unwrap())
}

pub fn argmin_in_matrix(dataset: MatrixView) -> Array1<usize> {
    dataset.map_axis(Axis(1), |view| view.argmin().unwrap())
}

pub fn argmax_in_matrix(dataset: MatrixView) -> Array1<usize> {
    dataset.map_axis(Axis(1), |view| view.argmax().unwrap())
}
