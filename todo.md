# Todo

## Necessary (needed for correct inference)

All done — the engine produces coherent text end to end.

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
- [x] RoPE applied to Q and K per head before the score dot product
- [x] Causal mask — upper triangle set to `-inf` before softmax
- [x] Final RMSNorm (`model.norm`) before `lm_head`
- [x] `lm_head` — projection to vocab logits (reuses tied embedding weights)
- [x] Greedy sampling — argmax over last-position logits
- [x] `decode_tokens` — inverts vocab + byte-level encoding back to text
- [x] Autoregressive generation loop with EOS stop
