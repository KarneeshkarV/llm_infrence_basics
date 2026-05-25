mod modules;

use modules::embeddings::embeddings;
use modules::loading_weights::read;
use modules::tokenizer::tokenizer;
use modules::transformers::block::TransformerBlock;
use modules::transformers::lm_head;
use modules::transformers::model::NeuralNetwork;
use modules::transformers::{HIDDEN_SIZE, NUM_LAYERS};
use ndarray::Array2;

fn main() -> std::io::Result<()> {
    let mut model = read::load("models/SmolLM2-135M/model.safetensors")?;
    //let block = TransformerBlock::from_weights(&mut model, 0)?;
    let mut temp_vec = vec![];
    for i in 0..NUM_LAYERS {
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
    let output = Array2::from_shape_vec(
        (output.len(), HIDDEN_SIZE),
        output.iter().flatten().copied().collect(),
    )
    .unwrap();
    let lm_head_weights =
        Array2::from_shape_vec((embed.shape[0], embed.shape[1]), embed.data).unwrap();
    let final_logits = lm_head::lm_head(output, lm_head_weights);
    println!("logits shape: {:?}", final_logits.shape());

    let next_token = lm_head::greedy(&final_logits);
    println!("next token id: {}", next_token);

    Ok(())
}
