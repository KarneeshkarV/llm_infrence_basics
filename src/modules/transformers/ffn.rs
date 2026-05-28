use crate::modules::loading_weights::helper::invalid_data;
use crate::modules::loading_weights::read::WeightTensor;
use crate::modules::transformers::config::ModelConfig;
use ndarray::Array2;

pub struct Ffn {
    gate_proj: Array2<f32>,
    up_proj: Array2<f32>,
    down_proj: Array2<f32>,
    hidden_size: usize,
}

impl Ffn {
    pub fn new(
        gate: WeightTensor,
        up: WeightTensor,
        down: WeightTensor,
        config: &ModelConfig,
    ) -> std::io::Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        Ok(Self {
            gate_proj: convert(gate, intermediate_size, hidden_size, "mlp.gate_proj.weight")?,
            up_proj: convert(up, intermediate_size, hidden_size, "mlp.up_proj.weight")?,
            down_proj: convert(down, hidden_size, intermediate_size, "mlp.down_proj.weight")?,
            hidden_size,
        })
    }

    pub fn forward(&self, x: &[f32], seq_len: usize) -> std::io::Result<Vec<f32>> {
        if x.len() != seq_len * self.hidden_size {
            return Err(invalid_data(format!(
                "FFN input has {} values, expected {} for seq_len {seq_len} and hidden_size {}",
                x.len(),
                seq_len * self.hidden_size,
                self.hidden_size
            )));
        }

        let input = Array2::from_shape_vec((seq_len, self.hidden_size), x.to_vec())
            .map_err(|err| invalid_data(format!("invalid FFN input shape: {err}")))?;

        let gate = self.gate_proj.dot(&input.t()).t().mapv(silu);
        let up = self.up_proj.dot(&input.t()).t().to_owned();
        let hidden = gate * up;
        let output = self.down_proj.dot(&hidden.t()).t().to_owned();

        Ok(output.iter().copied().collect())
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn convert(
    input: WeightTensor,
    rows: usize,
    cols: usize,
    name: &str,
) -> std::io::Result<Array2<f32>> {
    if input.shape != vec![rows, cols] {
        return Err(invalid_data(format!(
            "{name} shape {:?}, expected [{rows}, {cols}]",
            input.shape
        )));
    }

    Array2::from_shape_vec((rows, cols), input.data)
        .map_err(|err| invalid_data(format!("invalid {name} shape: {err}")))
}
