use crate::Matrix;

use super::{
    identity_layer::IdentityLayer, layer::Layer, relu_layer::ReluLayer, sigmoid_layer::SigmoidLayer,
};

pub enum ActivationLayer {
    Identity(IdentityLayer),
    ReLU(ReluLayer),
    Sigmoid(SigmoidLayer),
}

impl Layer for ActivationLayer {
    fn forward(&mut self, x: Matrix) -> Matrix {
        match self {
            Self::Identity(layer) => layer.forward(x),
            Self::ReLU(layer) => layer.forward(x),
            Self::Sigmoid(layer) => layer.forward(x),
        }
    }

    fn backward(&mut self, grad_z: Matrix) -> Matrix {
        match self {
            Self::Identity(layer) => layer.backward(grad_z),
            Self::ReLU(layer) => layer.backward(grad_z),
            Self::Sigmoid(layer) => layer.backward(grad_z),
        }
    }
}
