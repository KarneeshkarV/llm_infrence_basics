use crate::modules::loading_weights::read::WeightTensor;
use ndarray::Array2;

pub struct Attention {
    w_q: Array2<f32>, // [192, 576]
    w_k: Array2<f32>, // [192, 576]
    w_v: Array2<f32>, // [192, 576]
    w_o: Array2<f32>, // [192, 576]
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
    pub fn forward(&self, x: &[[f32; 576]]) -> Vec<Vec<f32>> {
        let mut output = Vec::new();
        let seq_len = x.len();
        let flat: Vec<f32> = x.iter().flatten().copied().collect();
        let input = Array2::from_shape_vec((seq_len, 576), flat).unwrap();
        let q = self.w_q.dot(&input.t()).t().to_owned();
        let k = self.w_k.dot(&input.t()).t().to_owned();
        let v = self.w_v.dot(&input.t()).t().to_owned();
        let mut attn = q.dot(&k.t());
        attn /= 64.0_f32.sqrt();
        for mut row in attn.rows_mut() {
            let row_max = row.iter().copied().max_by(|a, b| a.total_cmp(b)).unwrap();
            row.mapv_inplace(|x| (x - row_max).exp());
            let row_sum = row.sum();
            row.mapv_inplace(|x| x / row_sum);
        }
        let attn = attn.dot(&v);
        let (attn, offset) = attn.into_raw_vec_and_offset();
        debug_assert_eq!(offset, Some(0));
        output.push(attn);
        let (q, offset) = q.into_raw_vec_and_offset();
        debug_assert_eq!(offset, Some(0));
        output.push(q);
        output
    }
}

fn convert(input: WeightTensor) -> Array2<f32> {
    Array2::from_shape_vec((input.shape[0], input.shape[1]), input.data).unwrap()
}
