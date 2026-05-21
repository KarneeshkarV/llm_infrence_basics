mod modules;

use modules::loading_weights::read;

fn main() -> std::io::Result<()> {
    let model = read::load("models/SmolLM2-135M/model.safetensors")?;
    let input = stringify!("Hello, world!");

    Ok(())
}
