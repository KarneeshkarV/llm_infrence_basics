use crate::modules::loading_weights::read::WeightTensor;
use crate::modules::transformers::{HIDDEN_SIZE, INTERMEDIATE_SIZE};
use ndarray::Array2;

pub struct Ffn {
    gate_proj: Array2<f32>, // [1536, 576]
    up_proj: Array2<f32>,   // [1536, 576]
    down_proj: Array2<f32>, // [576, 1536]
}

impl Ffn {
    pub fn new(gate: WeightTensor, up: WeightTensor, down: WeightTensor) -> Self {
        debug_assert_eq!(gate.shape, vec![INTERMEDIATE_SIZE, HIDDEN_SIZE]);
        debug_assert_eq!(up.shape, vec![INTERMEDIATE_SIZE, HIDDEN_SIZE]);
        debug_assert_eq!(down.shape, vec![HIDDEN_SIZE, INTERMEDIATE_SIZE]);

        Self {
            gate_proj: convert(gate),
            up_proj: convert(up),
            down_proj: convert(down),
        }
    }

    pub fn forward(&self, x: &[[f32; HIDDEN_SIZE]]) -> Vec<[f32; HIDDEN_SIZE]> {
        let seq_len = x.len();
        let flat: Vec<f32> = x.iter().flatten().copied().collect();
        let input = Array2::from_shape_vec((seq_len, HIDDEN_SIZE), flat).unwrap();

        let gate = self.gate_proj.dot(&input.t()).t().mapv(silu);
        let up = self.up_proj.dot(&input.t()).t().to_owned();
        let hidden = gate * up;
        let output = self.down_proj.dot(&hidden.t()).t().to_owned();

        output
            .rows()
            .into_iter()
            .map(|row| std::array::from_fn(|index| row[index]))
            .collect()
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn convert(input: WeightTensor) -> Array2<f32> {
    debug_assert_eq!(input.shape.iter().product::<usize>(), input.data.len());
    Array2::from_shape_vec((input.shape[0], input.shape[1]), input.data).unwrap()
}
