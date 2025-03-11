use crate::{
    Matrix, Number,
    layers::{activation_layer::ActivationLayer, layer::Layer, linear_layer::LinearLayer},
};

pub struct MultiLayerNetwork {
    layers: Vec<Box<dyn Layer>>,
}

impl MultiLayerNetwork {
    pub fn new(input_dim: usize, neurons: &[usize], mut activations: Vec<ActivationLayer>) -> Self {
        assert_eq!(
            neurons.len(),
            activations.len(),
            "There must be as many layers as activation layers!"
        );

        let mut current_input_dim = input_dim;

        let mut layers: Vec<Box<dyn Layer>> = Vec::new();

        for &neuron in neurons {
            let next_layer = LinearLayer::new(current_input_dim, neuron, None);
            layers.push(Box::new(next_layer));
            let next_activation_layer = activations.pop().unwrap();
            layers.push(Box::new(next_activation_layer));
            current_input_dim = neuron;
        }

        Self { layers }
    }
}

impl Layer for MultiLayerNetwork {
    fn forward(&mut self, x: Matrix) -> Matrix {
        let mut temp = x;

        for layer in self.layers.iter_mut() {
            temp = layer.forward(temp);
        }

        temp
    }

    fn backward(&mut self, grad_z: Matrix) -> Matrix {
        let mut temp = grad_z;

        for layer in self.layers.iter_mut().rev() {
            temp = layer.backward(temp);
        }

        temp
    }

    fn update_parameters(&mut self, learning_rate: Number) {
        for layer in self.layers.iter_mut() {
            layer.update_parameters(learning_rate);
        }
    }
}
