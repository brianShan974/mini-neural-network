use ndarray::Axis;
use rand::{rng, seq::SliceRandom};

use crate::{
    Matrix, Number,
    layers::{layer::Layer, loss_layers::loss_layer::LossLayer},
    multilayer_network::MultiLayerNetwork,
};

pub struct Trainer<'a> {
    network: &'a mut MultiLayerNetwork,
    loss_layer: Box<dyn LossLayer>,
    batch_size: usize,
    n_epoch: usize,
    learning_rate: Number,
}

impl<'a> Trainer<'a> {
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
        }
    }

    pub fn train(&mut self, mut x: Matrix, mut y: Matrix, shuffle: bool, verbose: bool) {
        assert_eq!(
            x.nrows(),
            y.nrows(),
            "They must have the same amount of samples!"
        );

        for e in 0..self.n_epoch {
            if shuffle {
                Self::shuffle(&mut x, &mut y)
            }

            let mut x_remaining = x.view();
            let mut y_remaining = y.view();

            let mut batch = 1;
            let mut next_split;

            loop {
                let remaining_rows = x_remaining.nrows();

                if remaining_rows > self.batch_size {
                    next_split = self.batch_size;
                } else {
                    next_split = remaining_rows;
                }

                let (x_current, x_next) = x_remaining.split_at(Axis(0), next_split);
                let (y_current, y_next) = y_remaining.split_at(Axis(0), next_split);

                let pred_batch = self.network.forward(x_current.to_owned());
                let loss = self.loss_layer.forward(pred_batch, y_current.to_owned());
                let grad_loss = self.loss_layer.backward();
                self.network.backward(grad_loss);
                self.network.update_parameters(self.learning_rate);

                if verbose {
                    println!("Epoch: {}, batch: {batch}, current loss: {loss}", e + 1);
                }

                x_remaining = x_next;
                y_remaining = y_next;

                batch += 1;

                if next_split == remaining_rows {
                    break;
                }
            }
        }
    }

    pub fn eval_loss(&mut self, x: Matrix, y: Matrix) -> Number {
        assert_eq!(
            x.nrows(),
            y.nrows(),
            "They must have the same amount of samples!"
        );

        let pred = self.network.forward(x);

        self.loss_layer.forward(pred, y)
    }

    pub fn eval_loss_only(&self, x: Matrix, y: Matrix) -> Number {
        assert_eq!(
            x.nrows(),
            y.nrows(),
            "They must have the same amount of samples!"
        );

        let pred = self.network.eval_only(x);

        self.loss_layer.eval_only(pred, y)
    }

    pub fn eval_loss_only_with_pred(&self, pred: Matrix, target: Matrix) -> Number {
        self.loss_layer.eval_only(pred, target)
    }

    pub fn predict(&self, x: Matrix) -> Matrix {
        self.network.eval_only(x)
    }

    fn shuffle(x: &mut Matrix, y: &mut Matrix) {
        assert_eq!(
            x.nrows(),
            y.nrows(),
            "They must have the same amount of samples!"
        );

        let mut indices: Vec<usize> = (0..x.nrows()).collect();
        indices.shuffle(&mut rng());

        *x = x.select(Axis(0), &indices);
        *y = y.select(Axis(0), &indices);
    }
}
