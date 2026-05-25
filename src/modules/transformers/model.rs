use crate::modules::transformers::HIDDEN_SIZE;
use crate::modules::transformers::block::TransformerBlock;

pub struct NeuralNetwork {
    layers: Vec<TransformerBlock>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<TransformerBlock>) -> Self {
        Self { layers }
    }

    pub fn forward(&self, x: &[[f32; HIDDEN_SIZE]]) -> Vec<[f32; HIDDEN_SIZE]> {
        let mut output = x.to_vec();
        for layer in self.layers.iter() {
            output = layer.forward(&output);
        }
        output
    }
}
