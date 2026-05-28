use crate::modules::loading_weights::helper::invalid_data;
use crate::modules::loading_weights::read::WeightTensor;
use crate::modules::transformers::config::ModelConfig;

pub struct RmsNorm {
    weight: Vec<f32>,
    hidden_size: usize,
    eps: f32,
}

impl RmsNorm {
    pub fn new(weight: WeightTensor, config: &ModelConfig) -> std::io::Result<Self> {
        if weight.shape != vec![config.hidden_size] {
            return Err(invalid_data(format!(
                "RMSNorm weight shape {:?}, expected [{:?}]",
                weight.shape, config.hidden_size
            )));
        }

        Ok(Self {
            weight: weight.data,
            hidden_size: config.hidden_size,
            eps: config.rms_norm_eps,
        })
    }

    pub fn forward(&self, x: &[f32], seq_len: usize) -> std::io::Result<Vec<f32>> {
        if x.len() != seq_len * self.hidden_size {
            return Err(invalid_data(format!(
                "RMSNorm input has {} values, expected {} for seq_len {seq_len} and hidden_size {}",
                x.len(),
                seq_len * self.hidden_size,
                self.hidden_size
            )));
        }

        let mut output = Vec::with_capacity(x.len());
        for token in x.chunks_exact(self.hidden_size) {
            output.extend(self.forward_token(token));
        }

        Ok(output)
    }

    fn forward_token(&self, x: &[f32]) -> Vec<f32> {
        let mean_square =
            x.iter().map(|value| value * value).sum::<f32>() / self.hidden_size as f32;
        let scale = (mean_square + self.eps).sqrt();

        x.iter()
            .zip(&self.weight)
            .map(|(value, weight)| (value / scale) * weight)
            .collect()
    }
}
