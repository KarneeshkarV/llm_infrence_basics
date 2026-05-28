mod modules;

use modules::embeddings::embeddings;
use modules::loading_weights::helper::invalid_data;
use modules::loading_weights::read;
use modules::tokenizer::tokenizer;
use modules::transformers::block::TransformerBlock;
use modules::transformers::config::ModelConfig;
use modules::transformers::lm_head;
use modules::transformers::model::NeuralNetwork;
use modules::transformers::rms_norm::RmsNorm;
use ndarray::Array2;
use std::io::Write;

fn main() -> std::io::Result<()> {
    #[cfg(feature = "cuda")]
    let use_gpu = modules::runtime::config::has_gpu();

    #[cfg(not(feature = "cuda"))]
    let use_gpu = false;

    let model_dir = "models/SmolLM2-135M";
    let config_path = format!("{model_dir}/config.json");
    let weights_path = format!("{model_dir}/model.safetensors");
    let tokenizer_path = format!("{model_dir}/tokenizer.json");

    let mut model = read::load(&weights_path)?;
    println!("{:?}", use_gpu);

    let config = ModelConfig::load(config_path)?;

    let mut temp_vec = vec![];
    for i in 0..config.num_hidden_layers {
        temp_vec.push(TransformerBlock::from_weights(&mut model, i, &config)?);
    }
    let final_norm = RmsNorm::new(
        model.remove("model.norm.weight").ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "missing tensor: model.norm.weight",
            )
        })?,
        &config,
    )?;
    let mut block = NeuralNetwork::new(temp_vec, final_norm);

    let embed = read::load_tensor_from_path(&weights_path, "model.embed_tokens.weight")?;
    if embed.shape != vec![config.vocab_size, config.hidden_size] {
        return Err(invalid_data(format!(
            "embedding table shape {:?}, expected [{}, {}]",
            embed.shape, config.vocab_size, config.hidden_size
        )));
    }

    let input = "the code to print hellow world in python is";
    let token_ids = tokenizer::tokenize_text(input, &tokenizer_path)
        .map_err(|err| invalid_data(format!("tokenization failed: {err}")))?;

    let lm_head_weights =
        Array2::from_shape_vec((embed.shape[0], embed.shape[1]), embed.data.clone())
            .map_err(|err| invalid_data(format!("invalid lm_head weight shape: {err}")))?;

    let max_new_tokens = 100;
    let eos_id = config.eos_token_id;

    print!("{input}");
    std::io::stdout().flush().unwrap();

    let mut step_input = token_ids;

    for _ in 0..max_new_tokens {
        let seq_len = step_input.len();
        let embeddings = embeddings::get_embeddings(&step_input, &embed, config.hidden_size)?;
        let output = block.forward(&embeddings, seq_len)?;
        let logits = lm_head::lm_head(&output, seq_len, config.hidden_size, &lm_head_weights)?;
        let next_token = lm_head::top_n(&logits, seq_len, config.vocab_size, 0.5, 5)?;

        if next_token == eos_id {
            break;
        }

        let piece = tokenizer::decode_tokens(&[next_token], &tokenizer_path)
            .map_err(|err| invalid_data(format!("decode failed: {err}")))?;
        print!("{piece}");
        std::io::stdout().flush().unwrap();

        step_input = vec![next_token];
    }
    println!();

    Ok(())
}
