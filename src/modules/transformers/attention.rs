use crate::modules::loading_weights::helper::invalid_data;
use crate::modules::loading_weights::read::WeightTensor;
use crate::modules::transformers::config::ModelConfig;
use ndarray::{Array2, ArrayView2, s};

pub struct Attention {
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>,
    cache: Option<Cache>,
    hidden_size: usize,
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    kv_width: usize,
    rope_theta: f32,
}
pub struct Cache {
    k: Vec<f32>,
    v: Vec<f32>,
}

impl Attention {
    pub fn new(
        q: WeightTensor,
        k: WeightTensor,
        v: WeightTensor,
        o: WeightTensor,
        config: &ModelConfig,
    ) -> std::io::Result<Self> {
        let hidden_size = config.hidden_size;
        let head_dim = config.head_dim();
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let kv_width = config.kv_width();

        Ok(Self {
            w_q: convert(q, hidden_size, hidden_size, "self_attn.q_proj.weight")?,
            w_k: convert(k, kv_width, hidden_size, "self_attn.k_proj.weight")?,
            w_v: convert(v, kv_width, hidden_size, "self_attn.v_proj.weight")?,
            w_o: convert(o, hidden_size, hidden_size, "self_attn.o_proj.weight")?,
            cache: None,
            hidden_size,
            head_dim,
            num_q_heads,
            num_kv_heads,
            kv_width,
            rope_theta: config.rope_theta,
        })
    }
    pub fn forward(&mut self, x: &[f32], seq_len: usize) -> std::io::Result<Vec<f32>> {
        if x.len() != seq_len * self.hidden_size {
            return Err(invalid_data(format!(
                "attention input has {} values, expected {} for seq_len {seq_len} and hidden_size {}",
                x.len(),
                seq_len * self.hidden_size,
                self.hidden_size
            )));
        }

        let input = Array2::from_shape_vec((seq_len, self.hidden_size), x.to_vec())
            .map_err(|err| invalid_data(format!("invalid attention input shape: {err}")))?;
        let past_len = self
            .cache
            .as_ref()
            .map_or(0, |cache| cache.k.len() / self.kv_width);

        let q = self.w_q.dot(&input.t()).t().to_owned();
        let mut k_new = self.w_k.dot(&input.t()).t().to_owned();
        let v_new = self.w_v.dot(&input.t()).t().to_owned();
        let q = apply_rope(&q, self.head_dim, past_len, self.rope_theta);
        k_new = apply_rope(&k_new, self.head_dim, past_len, self.rope_theta);
        let cache = self.cache.get_or_insert_with(|| Cache {
            k: Vec::new(),
            v: Vec::new(),
        });
        cache.k.extend(k_new.iter().copied());
        cache.v.extend(v_new.iter().copied());

        let kv_len = cache.k.len() / self.kv_width;
        let k = ArrayView2::from_shape((kv_len, self.kv_width), cache.k.as_slice())
            .map_err(|err| invalid_data(format!("invalid cached K shape: {err}")))?;
        let v = ArrayView2::from_shape((kv_len, self.kv_width), cache.v.as_slice())
            .map_err(|err| invalid_data(format!("invalid cached V shape: {err}")))?;

        let kv_group_size = self.num_q_heads / self.num_kv_heads;
        let mut concatenated = Array2::<f32>::zeros((seq_len, self.hidden_size));

        for q_head in 0..self.num_q_heads {
            let kv_head = q_head / kv_group_size;

            let q_start = q_head * self.head_dim;
            let q_end = q_start + self.head_dim;
            let kv_start = kv_head * self.head_dim;
            let kv_end = kv_start + self.head_dim;

            let q_head = q.slice(s![.., q_start..q_end]);
            let k_head = k.slice(s![.., kv_start..kv_end]);
            let v_head = v.slice(s![.., kv_start..kv_end]);

            let mut scores = q_head.dot(&k_head.t());
            scores /= (self.head_dim as f32).sqrt();
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
        Ok(output.iter().copied().collect())
    }
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

fn calculate_theta(position: usize, dim_pair: usize, d_model: usize, rope_theta: f32) -> f32 {
    let exponent = -2.0 * dim_pair as f32 / d_model as f32;
    let base = rope_theta.powf(exponent);

    position as f32 * base
}
fn rotate(x: f32, y: f32, theta: f32) -> (f32, f32) {
    let new_x = x * theta.cos() - y * theta.sin();
    let new_y = x * theta.sin() + y * theta.cos();

    (new_x, new_y)
}
fn apply_rope(
    x: &Array2<f32>,
    rotary_dim: usize,
    position_offset: usize,
    rope_theta: f32,
) -> Array2<f32> {
    let mut output = x.to_owned();
    let num_heads = x.shape()[1] / rotary_dim;
    let half = rotary_dim / 2;

    for (pos, mut row) in output.rows_mut().into_iter().enumerate() {
        let abs_pos = pos + position_offset;
        for head in 0..num_heads {
            let head_start = head * rotary_dim;
            for j in 0..half {
                let theta = calculate_theta(abs_pos, j, rotary_dim, rope_theta);
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
