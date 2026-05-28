use crate::modules::loading_weights::helper::invalid_data;
use ndarray::Array2;
use rand::RngExt;

pub fn lm_head(
    x: &[f32],
    seq_len: usize,
    hidden_size: usize,
    trans: &Array2<f32>,
) -> std::io::Result<Vec<f32>> {
    if seq_len == 0 {
        return Err(invalid_data("lm_head seq_len must be greater than zero"));
    }
    if x.len() != seq_len * hidden_size {
        return Err(invalid_data(format!(
            "lm_head input has {} values, expected {} for seq_len {seq_len} and hidden_size {hidden_size}",
            x.len(),
            seq_len * hidden_size
        )));
    }
    if trans.ncols() != hidden_size {
        return Err(invalid_data(format!(
            "lm_head weight has {} columns, expected hidden_size {hidden_size}",
            trans.ncols()
        )));
    }

    let input = Array2::from_shape_vec((seq_len, hidden_size), x.to_vec())
        .map_err(|err| invalid_data(format!("invalid lm_head input shape: {err}")))?;
    let output = input.dot(&trans.t());
    Ok(output.iter().copied().collect())
}

#[allow(dead_code)]
pub fn greedy(logits: &[f32], seq_len: usize, vocab_size: usize) -> usize {
    let start = (seq_len - 1) * vocab_size;
    let last = &logits[start..start + vocab_size];
    last.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap()
}

pub fn top_n(
    logits: &[f32],
    seq_len: usize,
    vocab_size: usize,
    temperature: f32,
    n: usize,
) -> std::io::Result<usize> {
    if seq_len == 0 {
        return Err(invalid_data("top_n seq_len must be greater than zero"));
    }
    if logits.len() != seq_len * vocab_size {
        return Err(invalid_data(format!(
            "logits have {} values, expected {} for seq_len {seq_len} and vocab_size {vocab_size}",
            logits.len(),
            seq_len * vocab_size
        )));
    }
    if temperature <= 0.0 {
        return Err(invalid_data("temperature must be greater than zero"));
    }
    if n == 0 {
        return Err(invalid_data("top_n sample size must be greater than zero"));
    }

    let start = (seq_len - 1) * vocab_size;
    let mut values: Vec<(usize, f32)> = logits[start..start + vocab_size]
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x / temperature))
        .collect();

    values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_n = &values[..n.min(values.len())];

    let max_logit = top_n
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);

    let exp_values: Vec<f32> = top_n.iter().map(|(_, v)| (*v - max_logit).exp()).collect();

    let sum: f32 = exp_values.iter().sum();

    let probs: Vec<f32> = exp_values.iter().map(|v| v / sum).collect();

    let mut rng = rand::rng();

    let sample: f32 = rng.random();

    let mut cumulative = 0.0;

    for ((idx, _), prob) in top_n.iter().zip(probs.iter()) {
        cumulative += prob;

        if sample < cumulative {
            return Ok(*idx);
        }
    }

    Ok(top_n[0].0)
}
