use ndarray::Array2;
use rand::RngExt;

pub fn lm_head(x: Array2<f32>, trans: &Array2<f32>) -> Array2<f32> {
    x.dot(&trans.t())
}

#[allow(dead_code)]
pub fn greedy(logits: &Array2<f32>) -> usize {
    let last = logits.row(logits.nrows() - 1);
    last.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap()
}

pub fn top_n(logits: &Array2<f32>, temperature: f32, n: usize) -> usize {
    let mut values: Vec<(usize, f32)> = logits
        .row(logits.nrows() - 1)
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
            return *idx;
        }
    }

    top_n[0].0
}
