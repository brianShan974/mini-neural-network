use std::error::Error;

use mini_neural_network::{
    Number,
    layers::{
        activation_layers::{
            activation_layer::ActivationLayer, identity_layer::IdentityLayer,
            relu_layer::ReLULayer, sigmoid_layer::SigmoidLayer,
        },
        loss_layers::mse_loss_layer::MSELossLayer,
    },
    multilayer_network::MultiLayerNetwork,
    regressor::Regressor,
    trainer::Trainer,
};
use polars::prelude::{CsvParseOptions, CsvReadOptions, DataFrameOps, FillNullStrategy, SerReader};

const LEARNING_RATE: Number = 0.005;
const EPOCHS: usize = 1000;
const SHUFFLE: bool = true;
const VERBOSE: bool = true;

const STR_COL: &str = "ocean_proximity";

const N_SAMPLES: usize = 16512;
const N_OUTPUT_FEATURES: usize = 1;
const TRAIN_SIZE: usize = N_SAMPLES * 4 / 5;
const _TEST_SIZE: usize = N_SAMPLES - TRAIN_SIZE;

fn main() -> Result<(), Box<dyn Error>> {
    let path = format!("{}/example_data/housing.csv", env!("CARGO_MANIFEST_DIR"));

    let df_csv = CsvReadOptions::default()
        .with_infer_schema_length(None)
        .with_has_header(true)
        .with_parse_options(CsvParseOptions::default())
        .try_into_reader_with_file_path(Some(path.into()))?
        .finish()?
        .fill_null(FillNullStrategy::Mean)?
        .columns_to_dummies(vec![STR_COL], None, true)?;

    let df_csv = df_csv
        .sample_n_literal(N_SAMPLES, false, true, None)
        .unwrap();

    let n_cols = df_csv.width();

    let (train, test) = df_csv.split_at(TRAIN_SIZE as i64);

    let x_train = train.select_by_range(..n_cols - N_OUTPUT_FEATURES).unwrap();
    let y_train = train
        .select_by_range(n_cols - N_OUTPUT_FEATURES..n_cols)
        .unwrap();
    let x_test = test.select_by_range(..n_cols - N_OUTPUT_FEATURES).unwrap();
    let y_test = test
        .select_by_range(n_cols - N_OUTPUT_FEATURES..n_cols)
        .unwrap();

    let input_dim = n_cols - N_OUTPUT_FEATURES;
    let neurons = [64, 32, 16, N_OUTPUT_FEATURES];
    let activations: Vec<Box<dyn ActivationLayer>> = vec![
        Box::new(IdentityLayer),
        Box::new(ReLULayer::default()),
        Box::new(ReLULayer::default()),
        Box::new(SigmoidLayer::default()),
    ];
    let mut network = MultiLayerNetwork::new(input_dim, &neurons, activations);

    let trainer = Trainer::new(
        &mut network,
        Box::new(MSELossLayer::default()),
        512,
        EPOCHS,
        LEARNING_RATE,
    );

    let mut regressor = Regressor::new(trainer, x_train.clone(), y_train.clone());

    regressor.fit(x_train.clone(), y_train.clone(), SHUFFLE, VERBOSE);

    println!(
        "Sqrt train loss: {}",
        regressor.eval_loss(x_train, y_train).sqrt()
    );
    println!(
        "Sqrt validation loss: {}",
        regressor.eval_loss(x_test, y_test).sqrt()
    );

    Ok(())
}
