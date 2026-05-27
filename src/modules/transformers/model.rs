use crate::modules::transformers::HIDDEN_SIZE;
use crate::modules::transformers::block::TransformerBlock;
use crate::modules::transformers::rms_norm::RmsNorm;

pub struct NeuralNetwork {
    layers: Vec<TransformerBlock>,
    final_norm: RmsNorm,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<TransformerBlock>, final_norm: RmsNorm) -> Self {
        Self { layers, final_norm }
    }

    pub fn forward(&mut self, x: &[[f32; HIDDEN_SIZE]]) -> Vec<[f32; HIDDEN_SIZE]> {
        let mut output = x.to_vec();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
        }
        self.final_norm.forward(&output)
    }
}
