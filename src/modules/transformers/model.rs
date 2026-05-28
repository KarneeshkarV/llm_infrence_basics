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

    pub fn forward(&mut self, x: &[f32], seq_len: usize) -> std::io::Result<Vec<f32>> {
        let mut output = x.to_vec();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output, seq_len)?;
        }
        self.final_norm.forward(&output, seq_len)
    }
}
