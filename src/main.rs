mod modules;

use modules::embeddings::embeddings;
use modules::loading_weights::read;
use modules::tokenizer::tokenizer;
use modules::transformers::block::TransformerBlock;
use modules::transformers::lm_head;
use modules::transformers::model::NeuralNetwork;
use modules::transformers::rms_norm::RmsNorm;
use modules::transformers::{HIDDEN_SIZE, NUM_LAYERS};
use ndarray::Array2;
use std::io::Write;

fn main() -> std::io::Result<()> {
    let mut model = read::load("models/SmolLM2-135M/model.safetensors")?;
    //let block = TransformerBlock::from_weights(&mut model, 0)?;
    let mut temp_vec = vec![];
    for i in 0..NUM_LAYERS {
        temp_vec.push(TransformerBlock::from_weights(&mut model, i)?);
    }
    let final_norm = RmsNorm::new(model.remove("model.norm.weight").ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "missing tensor: model.norm.weight",
        )
    })?);
    let block = NeuralNetwork::new(temp_vec, final_norm);

    let embed = read::load_tensor_from_path(
        "models/SmolLM2-135M/model.safetensors",
        "model.embed_tokens.weight",
    )?;
    let tokenizer_path = "models/SmolLM2-135M/tokenizer.json";
    let input = "Hello, world! Karneeshkar";
    let mut token_ids = tokenizer::tokenize_text(input, tokenizer_path).unwrap();

    // lm_head is tied to the embedding matrix; clone it once so `embed` stays
    // available for the per-step embedding lookup inside the loop.
    let lm_head_weights =
        Array2::from_shape_vec((embed.shape[0], embed.shape[1]), embed.data.clone()).unwrap();

    let max_new_tokens = 30;
    let eos_id = 0;

    print!("{input}");
    std::io::stdout().flush().unwrap();

    for _ in 0..max_new_tokens {
        let embeddings = embeddings::get_embeddings(token_ids.clone(), &embed);
        let hidden_states: Vec<[f32; HIDDEN_SIZE]> = embeddings
            .into_iter()
            .map(|embedding| embedding.try_into().unwrap())
            .collect();

        let output = block.forward(&hidden_states);
        let output = Array2::from_shape_vec(
            (output.len(), HIDDEN_SIZE),
            output.iter().flatten().copied().collect(),
        )
        .unwrap();

        let logits = lm_head::lm_head(output, &lm_head_weights);
        let next_token = lm_head::greedy(&logits);

        if next_token == eos_id {
            break;
        }
        token_ids.push(next_token);

        let piece = tokenizer::decode_tokens(&[next_token], tokenizer_path).unwrap();
        print!("{piece}");
        std::io::stdout().flush().unwrap();
    }
    println!();

    Ok(())
}
