# Todo

## Necessary (needed for correct inference)

- [ ] RoPE (Rotary Position Encoding) — not implemented. SmolLM2 uses RoPE applied to Q and K inside each head before the dot product.
- [ ] Causal mask — token `i` currently attends to future tokens. Need to mask upper triangle of scores to `-inf` before softmax.
- [ ] `lm_head` — final linear projection from hidden dim (576) to vocab size (49152) for next-token prediction.
- [ ] Sampling — no greedy decode or token selection after logits. Nothing produces actual text yet.

## Improvements (correctness ok, but rough)

- [ ] Read model config from `config.json` — `HIDDEN_SIZE`, `HEAD_DIM`, `NUM_Q_HEADS`, `NUM_KV_HEADS`, `NUM_LAYERS` are all hardcoded constants. Should be loaded at startup.
- [ ] Load all weights in one pass — currently `load()` builds a `BTreeMap<String, WeightTensor>` holding all tensors in memory, then `from_weights` removes them one by one. Could stream weights directly into blocks without the intermediate map.
- [ ] KV cache — recomputing K and V for all tokens on every forward pass. Real inference caches past K/V and only appends the new token.
- [ ] `.bin` support — currently only handles `.safetensors` format.
- [ ] `unwrap()` calls in `attention.rs` forward pass — shape mismatches will panic with no useful message.

## Done

- [x] RMSNorm
- [x] FFN with SwiGLU activation
- [x] Residual connections
- [x] GQA head splitting in attention
- [x] Numerically stable softmax
- [x] 30-layer loop in `NeuralNetwork`
- [x] `TransformerBlock::from_weights` loads weights by name and removes from map
