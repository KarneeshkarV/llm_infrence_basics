mod modules;

use modules::embeddings::embeddings;
use modules::loading_weights::read;
use modules::tokenizer::tokenizer;
use modules::transformers::{HIDDEN_SIZE , NUM_LAYERS};
use modules::transformers::block::TransformerBlock;
use modules::transformers::model::NeuralNetwork;

fn main() -> std::io::Result<()> {
    let mut model = read::load("models/SmolLM2-135M/model.safetensors")?;
    //let block = TransformerBlock::from_weights(&mut model, 0)?;
    let mut temp_vec = vec![];
    for i in 0..NUM_LAYERS{
        temp_vec.push(TransformerBlock::from_weights(&mut model, i)?);
    }
    let block = NeuralNetwork::new(temp_vec);

    let embed = read::load_tensor_from_path(
        "models/SmolLM2-135M/model.safetensors",
        "model.embed_tokens.weight",
    )?;
    let input = "Hello, world! Karneeshkar";
    let tokens = tokenizer::tokenize_text(input, "models/SmolLM2-135M/tokenizer.json").unwrap();
    let embeddings = embeddings::get_embeddings(tokens, &embed);

    let hidden_states: Vec<[f32; HIDDEN_SIZE]> = embeddings
        .into_iter()
        .map(|embedding| embedding.try_into().unwrap())
        .collect();

    let output = block.forward(&hidden_states);
    println!(
        "output shape: [{}, {}]",
        output.len(),
        output.first().map(|token| token.len()).unwrap_or(0)
    );

    Ok(())
}
