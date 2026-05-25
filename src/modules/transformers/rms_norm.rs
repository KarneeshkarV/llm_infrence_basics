use crate::modules::loading_weights::read::WeightTensor;
use crate::modules::transformers::{HIDDEN_SIZE, RMS_NORM_EPS};

pub struct RmsNorm {
    weight: [f32; HIDDEN_SIZE],
}

impl RmsNorm {
    pub fn new(weight: WeightTensor) -> Self {
        debug_assert_eq!(weight.shape, vec![HIDDEN_SIZE]);

        Self {
            weight: weight.data.try_into().unwrap(),
        }
    }

    pub fn forward(&self, x: &[[f32; HIDDEN_SIZE]]) -> Vec<[f32; HIDDEN_SIZE]> {
        x.iter().map(|token| self.forward_token(token)).collect()
    }

    fn forward_token(&self, x: &[f32; HIDDEN_SIZE]) -> [f32; HIDDEN_SIZE] {
        let mean_square = x.iter().map(|value| value * value).sum::<f32>() / HIDDEN_SIZE as f32;
        let scale = (mean_square + RMS_NORM_EPS).sqrt();

        std::array::from_fn(|index| (x[index] / scale) * self.weight[index])
    }
}
