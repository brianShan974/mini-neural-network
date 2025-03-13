use crate::{
    Matrix, Number,
    layers::{
        activation_layers::activation_layer::ActivationLayer, layer::Layer,
        linear_layer::LinearLayer,
    },
};

pub struct MultiLayerNetwork {
    layers: Vec<Box<dyn Layer>>,
    activation_layers: Vec<Box<dyn ActivationLayer>>,
}

impl MultiLayerNetwork {
    pub fn new(
        input_dim: usize,
        neurons: &[usize],
        mut activations: Vec<Box<dyn ActivationLayer>>,
    ) -> Self {
        assert_eq!(
            neurons.len(),
            activations.len(),
            "There must be as many layers as activation layers!"
        );

        let mut current_input_dim = input_dim;

        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        let mut activation_layers: Vec<Box<dyn ActivationLayer>> = Vec::new();

        activations.reverse();

        for &neuron in neurons {
            let next_layer = Box::new(LinearLayer::new(current_input_dim, neuron, None));
            layers.push(next_layer);
            let next_activation_layer = activations.pop().unwrap();
            activation_layers.push(next_activation_layer);
            current_input_dim = neuron;
        }

        Self {
            layers,
            activation_layers,
        }
    }
}

impl Layer for MultiLayerNetwork {
    fn forward(&mut self, x: Matrix) -> Matrix {
        let num_layers = self.layers.len();

        let mut temp = x;
        for i in 0..num_layers {
            temp = self.layers[i].forward(temp);
            temp = self.activation_layers[i].forward(temp);
        }

        temp
    }

    fn backward(&mut self, grad_z: Matrix) -> Matrix {
        let num_layers = self.layers.len();

        let mut temp = grad_z;
        for i in (0..num_layers).rev() {
            temp = self.activation_layers[i].backward(temp);
            temp = self.layers[i].backward(temp);
        }

        temp
    }

    fn update_parameters(&mut self, learning_rate: Number) {
        for layer in self.layers.iter_mut() {
            layer.update_parameters(learning_rate);
        }
    }

    fn eval_only(&self, x: Matrix) -> Matrix {
        let num_layers = self.layers.len();

        let mut temp = x;
        for i in 0..num_layers {
            temp = self.layers[i].eval_only(temp);
            temp = self.activation_layers[i].eval_only(temp);
        }

        temp
    }
}
