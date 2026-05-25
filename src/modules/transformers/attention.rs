use crate::modules::loading_weights::read::WeightTensor;
use ndarray::{Array2, s};

const HIDDEN_SIZE: usize = 576;
const HEAD_DIM: usize = 64;
const NUM_Q_HEADS: usize = 9;
const NUM_KV_HEADS: usize = 3;

pub struct Attention {
    w_q: Array2<f32>, // [576, 576]
    w_k: Array2<f32>, // [192, 576]
    w_v: Array2<f32>, // [192, 576]
    w_o: Array2<f32>, // [576, 576]
}

impl Attention {
    pub fn new(q: WeightTensor, k: WeightTensor, v: WeightTensor, o: WeightTensor) -> Self {
        Self {
            w_q: convert(q),
            w_k: convert(k),
            w_v: convert(v),
            w_o: convert(o),
        }
    }
    pub fn forward(&self, x: &[[f32; HIDDEN_SIZE]]) -> Vec<Vec<f32>> {
        let seq_len = x.len();
        let flat: Vec<f32> = x.iter().flatten().copied().collect();
        let input = Array2::from_shape_vec((seq_len, HIDDEN_SIZE), flat).unwrap();
        let q = self.w_q.dot(&input.t()).t().to_owned();
        let k = self.w_k.dot(&input.t()).t().to_owned();
        let v = self.w_v.dot(&input.t()).t().to_owned();

        let kv_group_size = NUM_Q_HEADS / NUM_KV_HEADS;
        let mut concatenated = Array2::<f32>::zeros((seq_len, HIDDEN_SIZE));

        for q_head in 0..NUM_Q_HEADS {
            let kv_head = q_head / kv_group_size;

            let q_start = q_head * HEAD_DIM;
            let q_end = q_start + HEAD_DIM;
            let kv_start = kv_head * HEAD_DIM;
            let kv_end = kv_start + HEAD_DIM;

            let q_head = q.slice(s![.., q_start..q_end]);
            let k_head = k.slice(s![.., kv_start..kv_end]);
            let v_head = v.slice(s![.., kv_start..kv_end]);

            let mut scores = q_head.dot(&k_head.t());
            scores /= (HEAD_DIM as f32).sqrt();

            for mut row in scores.rows_mut() {
                let row_max = row.iter().copied().max_by(|a, b| a.total_cmp(b)).unwrap();
                row.mapv_inplace(|x| (x - row_max).exp());
                let row_sum = row.sum();
                row.mapv_inplace(|x| x / row_sum);
            }

            let head_output = scores.dot(&v_head);
            concatenated
                .slice_mut(s![.., q_start..q_end])
                .assign(&head_output);
        }

        let output = self.w_o.dot(&concatenated.t()).t().to_owned();
        output.rows().into_iter().map(|row| row.to_vec()).collect()
    }
}

fn convert(input: WeightTensor) -> Array2<f32> {
    Array2::from_shape_vec((input.shape[0], input.shape[1]), input.data).unwrap()
}
