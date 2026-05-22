use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}

#[derive(Debug, Deserialize)]
pub struct Header {
    #[serde(rename = "__metadata__", default)]
    pub metadata: Option<BTreeMap<String, String>>,
    #[serde(flatten)]
    pub tensors: BTreeMap<String, TensorInfo>,
}

pub fn load(path: &str) -> std::io::Result<Header> {
    let path = Path::new(path);
    match path.extension() {
        Some(ext) if ext == "safetensors" => (),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a safetensors file",
            ));
        }
    }
    let mut file = File::open(path)?;

    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf);

    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)?;

    let header: Header =
        serde_json::from_slice(&header_bytes).expect("invalid safetensors JSON header");

    let data_start = 8u64 + header_len;

    let (name, info) = header.tensors.iter().next().expect("no tensors");
    let abs_start = data_start + info.data_offsets[0];
    let n_bytes = (info.data_offsets[1] - info.data_offsets[0]) as usize;

    file.seek(SeekFrom::Start(abs_start))?;
    let mut buf = vec![0u8; n_bytes];
    file.read_exact(&mut buf)?;

    println!("tensors: {}", header.tensors.len());
    println!(
        "first: {name}  dtype={}  shape={:?}  bytes={}",
        info.dtype, info.shape, n_bytes
    );
    match header.metadata.as_ref().and_then(|m| m.keys().next()) {
        Some(k) => println!("metadata first key: {k}"),
        None => println!("metadata: <none>"),
    }

    Ok(header)
}
