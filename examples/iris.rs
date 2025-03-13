use csv::ReaderBuilder;
use mini_neural_network::{
    Matrix, Number,
    layers::{
        activation_layer::ActivationLayer, identity_layer::IdentityLayer, layer::Layer,
        mse_loss_layer::MSELossLayer, relu_layer::ReluLayer,
    },
    multilayer_network::MultiLayerNetwork,
    preprocessor::Preprocessor,
    trainer::Trainer,
    utils::argmax_in_matrix,
};
use ndarray::Axis;
use ndarray_csv::Array2Reader;
use ndarray_rand::{RandomExt, SamplingStrategy};

const VERBOSE: bool = true;

const LEARNING_RATE: Number = 0.01;
const EPOCHS: usize = 1000;

const N_SAMPLES: usize = 150;
const N_FEATURES: usize = N_INPUT_FEATURES + N_OUTPUT_FEATURES;
const N_INPUT_FEATURES: usize = 4;
const N_OUTPUT_FEATURES: usize = 3;
const TRAIN_SIZE: usize = N_SAMPLES * 4 / 5;

fn main() {
    let input_dim = N_INPUT_FEATURES;
    let neurons = [16, N_OUTPUT_FEATURES];
    let activations = vec![
        ActivationLayer::ReLU(ReluLayer::default()),
        ActivationLayer::Identity(IdentityLayer),
    ];
    let mut network = MultiLayerNetwork::new(input_dim, &neurons, activations);

    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b' ')
        .from_path(format!(
            "{}/example_data/iris.dat",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
    let dataset: Matrix = reader.deserialize_array2((N_SAMPLES, N_FEATURES)).unwrap();

    let dataset = dataset.sample_axis(Axis(0), N_SAMPLES, SamplingStrategy::WithoutReplacement);

    let (train, test) = dataset.view().split_at(Axis(0), TRAIN_SIZE);

    let (x_train, y_train) = train.split_at(Axis(1), N_INPUT_FEATURES);
    let (x_test, y_test) = test.split_at(Axis(1), N_INPUT_FEATURES);

    let prep = Preprocessor::new(x_train);

    let x_train_preped = prep.apply(x_train.to_owned());
    let x_test_preped = prep.apply(x_test.to_owned());

    let mut trainer = Trainer::new(
        &mut network,
        Box::new(MSELossLayer::default()),
        8,
        EPOCHS,
        LEARNING_RATE,
        true,
    );

    trainer.train(x_train_preped.clone(), y_train.to_owned(), VERBOSE);
    println!(
        "Train loss: {}",
        trainer.eval_loss(x_train_preped, y_train.to_owned())
    );
    println!(
        "Validation loss: {}",
        trainer.eval_loss(x_test_preped.clone(), y_test.to_owned())
    );

    let preds = argmax_in_matrix(network.forward(x_test_preped).view())
        .into_dyn()
        .squeeze();
    let targets = argmax_in_matrix(y_test).into_dyn().squeeze();

    let total = preds.len();
    let accuracy = (0..total)
        .map(|i| if preds[i] == targets[i] { 1.0 } else { 0.0 })
        .sum::<Number>()
        / total as Number;

    println!("Predictions: {preds}");
    println!("Targets: {targets}");
    println!("Accuracy: {:.5}%", accuracy * 100.0);
}
