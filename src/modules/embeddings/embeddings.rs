use crate::modules::loading_weights::helper::invalid_data;
use crate::modules::loading_weights::read::WeightTensor;

pub fn get_embeddings(
    tokens: &[usize],
    embedd_table: &WeightTensor,
    hidden_size: usize,
) -> std::io::Result<Vec<f32>> {
    if embedd_table.shape.len() != 2 || embedd_table.shape[1] != hidden_size {
        return Err(invalid_data(format!(
            "embedding table shape {:?}, expected [vocab_size, {hidden_size}]",
            embedd_table.shape
        )));
    }

    let vocab_size = embedd_table.shape[0];
    let mut embeddings = Vec::with_capacity(tokens.len() * hidden_size);
    for token in tokens {
        if *token >= vocab_size {
            return Err(invalid_data(format!(
                "token id {token} is outside embedding vocab size {vocab_size}"
            )));
        }
        let embedding = embedd_table.row(*token);
        embeddings.extend(embedding);
    }

    Ok(embeddings)
}
