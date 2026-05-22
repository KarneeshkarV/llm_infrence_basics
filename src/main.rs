mod modules;

use modules::loading_weights::read;
use modules::tokenizer::tokenizer;

fn main() -> std::io::Result<()> {
    let model = read::load("models/SmolLM2-135M/model.safetensors")?;
    let input = stringify!("Hello, world! Karneeshkar");
    let tokens = tokenizer::tokenize_text(input, "models/SmolLM2-135M/tokenizer.json");

    let mut tokenized_text = Vec::new();
    match tokens {
        Ok(tokens) => {
            tokenized_text = tokens;
        }
        Err(e) => println!("error: {:?}", e),
    }
    println!("input: {:?}", tokenized_text);

    Ok(())
}
