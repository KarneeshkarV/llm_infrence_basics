use ndarray::Array2;

pub fn lm_head(x: Array2<f32>, trans: &Array2<f32>) -> Array2<f32> {
    x.dot(&trans.t())
}

pub fn greedy(logits: &Array2<f32>) -> usize {
    let last = logits.row(logits.nrows() - 1);
    last.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap()
}
