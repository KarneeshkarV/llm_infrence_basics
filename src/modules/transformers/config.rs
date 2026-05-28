use crate::modules::loading_weights::helper::invalid_data;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub eos_token_id: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

impl ModelConfig {
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let json = fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&json)
            .map_err(|err| invalid_data(format!("invalid model config JSON: {err}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    pub fn validate(&self) -> std::io::Result<()> {
        if self.hidden_size == 0 {
            return Err(invalid_data("hidden_size must be greater than zero"));
        }
        if self.num_attention_heads == 0 {
            return Err(invalid_data(
                "num_attention_heads must be greater than zero",
            ));
        }
        if self.num_key_value_heads == 0 {
            return Err(invalid_data(
                "num_key_value_heads must be greater than zero",
            ));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(invalid_data(format!(
                "hidden_size {} must be divisible by num_attention_heads {}",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(invalid_data(format!(
                "num_attention_heads {} must be divisible by num_key_value_heads {}",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.intermediate_size == 0 {
            return Err(invalid_data("intermediate_size must be greater than zero"));
        }
        if self.vocab_size == 0 {
            return Err(invalid_data("vocab_size must be greater than zero"));
        }
        Ok(())
    }
}

fn default_rope_theta() -> f32 {
    100000.0
}



