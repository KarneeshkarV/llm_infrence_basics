use crate::modules::loading_weights::read::WeightTensor;

pub fn get_embeddings(token: Vec<usize>, embedd_table: &WeightTensor) -> Vec<Vec<f32>> {
    let mut embeddings = Vec::new();
    for token in token {
        let embedding = embedd_table.row(token);
        embeddings.push(embedding);
    }
    embeddings
}
