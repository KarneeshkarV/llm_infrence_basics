use crate::modules::loading_weights::read::WeightTensor;
use crate::modules::transformers::HIDDEN_SIZE;
use crate::modules::transformers::attention::Attention;
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
    ) -> Self {
        Self {
            input_layernorm: RmsNorm::new(input_layernorm),
            self_attn: Attention::new(q_proj, k_proj, v_proj, o_proj),
            post_attention_layernorm: RmsNorm::new(post_attention_layernorm),
            mlp: Ffn::new(gate_proj, up_proj, down_proj),
        }
    }

    pub fn from_weights(
        weights: &mut BTreeMap<String, WeightTensor>,
        layer: usize,
    ) -> Result<Self> {
        Ok(Self::new(
            take(weights, layer, "input_layernorm.weight")?,
            take(weights, layer, "self_attn.q_proj.weight")?,
            take(weights, layer, "self_attn.k_proj.weight")?,
            take(weights, layer, "self_attn.v_proj.weight")?,
            take(weights, layer, "self_attn.o_proj.weight")?,
            take(weights, layer, "post_attention_layernorm.weight")?,
            take(weights, layer, "mlp.gate_proj.weight")?,
            take(weights, layer, "mlp.up_proj.weight")?,
            take(weights, layer, "mlp.down_proj.weight")?,
        ))
    }

    pub fn forward(&self, x: &[[f32; HIDDEN_SIZE]]) -> Vec<[f32; HIDDEN_SIZE]> {
        let normed_input = self.input_layernorm.forward(x);
        let attn_output = self.self_attn.forward(&normed_input);
        let attn_residual = add_residual(x, &attn_output);

        let normed_attn = self.post_attention_layernorm.forward(&attn_residual);
        let ffn_output = self.mlp.forward(&normed_attn);

        add_residual(&attn_residual, &ffn_output)
    }
}

fn add_residual(
    left: &[[f32; HIDDEN_SIZE]],
    right: &[[f32; HIDDEN_SIZE]],
) -> Vec<[f32; HIDDEN_SIZE]> {
    debug_assert_eq!(left.len(), right.len());

    left.iter()
        .zip(right)
        .map(|(left, right)| std::array::from_fn(|index| left[index] + right[index]))
        .collect()
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
