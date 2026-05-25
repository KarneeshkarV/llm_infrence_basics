# Shortcuts taken (things to fix later)

## Weight Loading
- [ ] `.bin` support — currently only handles `.safetensors` format
- [ ] validate tensor dtype at load time (we trust whatever the file says)

## Model Config
- [ ] `.config` for attention heads — `HIDDEN_SIZE`, `HEAD_DIM`, `NUM_Q_HEADS`, `NUM_KV_HEADS` are hardcoded constants in `attention.rs` for SmolLM2-135M only. Should read from `config.json`.

## Attention
- [ ] RoPE (Rotary Position Encoding) — not implemented at all. SmolLM2 uses RoPE applied to Q and K inside each head before the dot product.
- [ ] Causal mask — no masking applied. Token `i` currently attends to future tokens, which is wrong for autoregressive generation. Need to mask upper triangle of scores to `-inf` before softmax.
- [ ] KV cache — recomputing K and V for all tokens on every forward pass. Real inference caches past K/V and only computes new token.

## Transformer Block
- [ ] RMSNorm — no layer norm before attention or before FFN. SmolLM2 uses RMSNorm.
- [ ] FFN (Feed-Forward Network) — not implemented. Each transformer block has attention + FFN + residuals.
- [ ] Residual connections — attention output should be added back to input (`x = x + attn_out`), not returned raw.
- [ ] 30 layers — only one attention block exists. Need a loop over all 30 transformer layers loading the right weights.

## Inference
- [ ] `lm_head` — final linear projection from hidden dim (576) to vocab size (49152) for next-token prediction. Not implemented.
- [ ] Sampling — no temperature, top-k, top-p, greedy decode. Nothing after logits.
- [ ] Proper return type — `forward()` returns `Vec<Vec<f32>>` instead of `Array2<f32>`. Makes chaining layers awkward.

## Error Handling
- [ ] `unwrap()` calls in `attention.rs` forward pass — shape mismatches will panic with no useful message.
