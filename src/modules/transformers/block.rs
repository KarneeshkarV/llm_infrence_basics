use crate::modules::loading_weights::helper::invalid_data;
use crate::modules::loading_weights::read::WeightTensor;
use crate::modules::transformers::attention::Attention;
use crate::modules::transformers::config::ModelConfig;
use crate::modules::transformers::ffn::Ffn;
use crate::modules::transformers::rms_norm::RmsNorm;
use std::collections::BTreeMap;
use std::io::{Error, ErrorKind, Result};

pub struct TransformerBlock {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Ffn,
}

impl TransformerBlock {
    pub fn new(
        input_layernorm: WeightTensor,
        q_proj: WeightTensor,
        k_proj: WeightTensor,
        v_proj: WeightTensor,
        o_proj: WeightTensor,
        post_attention_layernorm: WeightTensor,
        gate_proj: WeightTensor,
        up_proj: WeightTensor,
        down_proj: WeightTensor,
        config: &ModelConfig,
    ) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::new(input_layernorm, config)?,
            self_attn: Attention::new(q_proj, k_proj, v_proj, o_proj, config)?,
            post_attention_layernorm: RmsNorm::new(post_attention_layernorm, config)?,
            mlp: Ffn::new(gate_proj, up_proj, down_proj, config)?,
        })
    }

    pub fn from_weights(
        weights: &mut BTreeMap<String, WeightTensor>,
        layer: usize,
        config: &ModelConfig,
    ) -> Result<Self> {
        Self::new(
            take(weights, layer, "input_layernorm.weight")?,
            take(weights, layer, "self_attn.q_proj.weight")?,
            take(weights, layer, "self_attn.k_proj.weight")?,
            take(weights, layer, "self_attn.v_proj.weight")?,
            take(weights, layer, "self_attn.o_proj.weight")?,
            take(weights, layer, "post_attention_layernorm.weight")?,
            take(weights, layer, "mlp.gate_proj.weight")?,
            take(weights, layer, "mlp.up_proj.weight")?,
            take(weights, layer, "mlp.down_proj.weight")?,
            config,
        )
    }

    pub fn forward(&mut self, x: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let normed_input = self.input_layernorm.forward(x, seq_len)?;
        let attn_output = self.self_attn.forward(&normed_input, seq_len)?;
        let attn_residual = add_residual(x, &attn_output)?;

        let normed_attn = self
            .post_attention_layernorm
            .forward(&attn_residual, seq_len)?;
        let ffn_output = self.mlp.forward(&normed_attn, seq_len)?;

        add_residual(&attn_residual, &ffn_output)
    }
}

fn add_residual(left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
    if left.len() != right.len() {
        return Err(invalid_data(format!(
            "residual inputs have different lengths: {} and {}",
            left.len(),
            right.len()
        )));
    }

    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| *left + *right)
        .collect())
}

fn take(
    weights: &mut BTreeMap<String, WeightTensor>,
    layer: usize,
    suffix: &str,
) -> Result<WeightTensor> {
    let name = format!("model.layers.{layer}.{suffix}");
    weights
        .remove(&name)
        .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("missing tensor: {name}")))
}
