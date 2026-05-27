use crate::modules::loading_weights::read::WeightTensor;
use crate::modules::transformers::{HEAD_DIM, HIDDEN_SIZE, NUM_KV_HEADS, NUM_Q_HEADS};
use ndarray::{Array2, ArrayView2, s};

pub struct Attention {
    w_q: Array2<f32>, // [576, 576]
    w_k: Array2<f32>, // [192, 576]
    w_v: Array2<f32>, // [192, 576]
    w_o: Array2<f32>, // [576, 576]
    cache: Option<Cache>,
}
pub struct Cache {
    k: Vec<f32>,
    v: Vec<f32>,
}

impl Attention {
    pub fn new(q: WeightTensor, k: WeightTensor, v: WeightTensor, o: WeightTensor) -> Self {
        Self {
            w_q: convert(q),
            w_k: convert(k),
            w_v: convert(v),
            w_o: convert(o),
            cache: None,
        }
    }
    pub fn forward(&mut self, x: &[[f32; HIDDEN_SIZE]]) -> Vec<[f32; HIDDEN_SIZE]> {
        let seq_len = x.len();
        let flat: Vec<f32> = x.iter().flatten().copied().collect();
        let input = Array2::from_shape_vec((seq_len, HIDDEN_SIZE), flat).unwrap();
        let kv_width = NUM_KV_HEADS * HEAD_DIM;
        let past_len = self.cache.as_ref().map_or(0, |c| c.k.len() / kv_width);

        let q = self.w_q.dot(&input.t()).t().to_owned();
        let mut k_new = self.w_k.dot(&input.t()).t().to_owned();
        let v_new = self.w_v.dot(&input.t()).t().to_owned();
        let q = apply_rope(&q, HEAD_DIM, past_len);
        k_new = apply_rope(&k_new, HEAD_DIM, past_len);
        let cache = self.cache.get_or_insert_with(|| Cache {
            k: Vec::new(),
            v: Vec::new(),
        });
        cache.k.extend(k_new.iter().copied());
        cache.v.extend(v_new.iter().copied());

        let kv_len = cache.k.len() / kv_width;
        let k = ArrayView2::from_shape((kv_len, kv_width), cache.k.as_slice()).unwrap();
        let v = ArrayView2::from_shape((kv_len, kv_width), cache.v.as_slice()).unwrap();

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
            mask(&mut scores, past_len);

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
        output
            .rows()
            .into_iter()
            .map(|row| std::array::from_fn(|index| row[index]))
            .collect()
    }
}

fn convert(input: WeightTensor) -> Array2<f32> {
    debug_assert_eq!(input.shape.len(), 2);
    debug_assert_eq!(input.shape.iter().product::<usize>(), input.data.len());
    Array2::from_shape_vec((input.shape[0], input.shape[1]), input.data).unwrap()
}

fn calculate_theta(position: usize, dim_pair: usize, d_model: usize) -> f32 {
    let exponent = -2.0 * dim_pair as f32 / d_model as f32;
    let base = 100000_f32.powf(exponent);

    position as f32 * base
}
fn rotate(x: f32, y: f32, theta: f32) -> (f32, f32) {
    let new_x = x * theta.cos() - y * theta.sin();
    let new_y = x * theta.sin() + y * theta.cos();

    (new_x, new_y)
}
fn apply_rope(x: &Array2<f32>, rotary_dim: usize, position_offset: usize) -> Array2<f32> {
    let mut output = x.to_owned();
    let num_heads = x.shape()[1] / rotary_dim;
    let half = rotary_dim / 2;

    for (pos, mut row) in output.rows_mut().into_iter().enumerate() {
        let abs_pos = pos + position_offset;
        for head in 0..num_heads {
            let head_start = head * rotary_dim;
            for j in 0..half {
                let theta = calculate_theta(abs_pos, j, rotary_dim);
                let x = row[head_start + j];
                let y = row[head_start + j + half];
                let (new_x, new_y) = rotate(x, y, theta);
                row[head_start + j] = new_x;
                row[head_start + j + half] = new_y;
            }
        }
    }

    output
}

fn mask(scores: &mut Array2<f32>, offset: usize) {
    for i in 0..scores.shape()[0] {
        for j in 0..scores.shape()[1] {
            if j > i + offset {
                scores[[i, j]] = -f32::INFINITY;
            }
        }
    }
}
