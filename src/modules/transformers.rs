pub mod attention;
pub mod block;
pub mod ffn;
pub mod rms_norm;
pub mod model;

pub const HIDDEN_SIZE: usize = 576;
pub const HEAD_DIM: usize = 64;
pub const NUM_Q_HEADS: usize = 9;
pub const NUM_KV_HEADS: usize = 3;
pub const INTERMEDIATE_SIZE: usize = 1536;
pub const RMS_NORM_EPS: f32 = 1e-5;
pub const NUM_LAYERS: usize = 30;
