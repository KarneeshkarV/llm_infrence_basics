mod modules;

use modules::embeddings::embeddings;
use modules::loading_weights::read;
use modules::tokenizer::tokenizer;

fn main() -> std::io::Result<()> {
    let model = read::load("models/SmolLM2-135M/model.safetensors")?;
    let input = "Hello, world! Karneeshkar";
    let tokens = tokenizer::tokenize_text(input, "models/SmolLM2-135M/tokenizer.json");
    let embed = read::load_tensor_from_path(
        "models/SmolLM2-135M/model.safetensors",
        "model.embed_tokens.weight",
    )?;
    println!("shape: {:?}", embed.shape);
    println!("data len: {}", embed.data.len());
    println!("model tensors: {}", model.len());
    let mut tokenized_text = Vec::new();
    match tokens {
        Ok(tokens) => {
            tokenized_text = tokens;
        }
        Err(e) => println!("error: {:?}", e),
    }
    println!("input: {:?}", tokenized_text);
    let _embeddings = embeddings::get_embeddings(tokenized_text, &embed);
    //println!("embeddings: {:?}", embeddings);
    let q = &model["model.layers.0.self_attn.q_proj.weight"];
    println!("q shape: {:?}", q.shape);
    let k = &model["model.layers.0.self_attn.k_proj.weight"];
    println!("k shape: {:?}", k.shape);

    Ok(())
}
