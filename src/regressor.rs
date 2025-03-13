use crate::{layers::loss_layers::loss_layer::LossLayer, multilayer_network::MultiLayerNetwork};

pub struct Regressor<'a> {
    network: &'a mut MultiLayerNetwork,
    loss_layer: Box<dyn LossLayer>,
}
