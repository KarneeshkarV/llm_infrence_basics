use super::helper::{bytes_to_f32, invalid_data};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde::Deserialize;

#[derive(Debug)]
pub struct WeightTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}
#[derive(Debug, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}

#[derive(Debug, Deserialize)]
pub struct Header {
    #[serde(rename = "__metadata__", default)]
    pub _metadata: Option<BTreeMap<String, String>>,
    #[serde(flatten)]
    pub tensors: BTreeMap<String, TensorInfo>,
}

impl WeightTensor {
    pub fn row(&self, index: usize) -> Vec<f32> {
        let row_size = self.shape.last().copied().unwrap_or(0);
        let start = index * row_size;
        let end = start + row_size;

        self.data[start..end].to_vec()
    }
}

fn read_header(file: &mut File) -> std::io::Result<(Header, u64)> {
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf);

    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)?;

    let header: Header = serde_json::from_slice(&header_bytes)
        .map_err(|err| invalid_data(format!("invalid safetensors JSON header: {err}")))?;

    Ok((header, 8u64 + header_len))
}

pub fn load_tensor(
    file: &mut File,
    data_start: u64,
    header: &Header,
    name: &str,
) -> std::io::Result<WeightTensor> {
    let info = header
        .tensors
        .get(name)
        .ok_or_else(|| invalid_data(format!("tensor not found: {name}")))?;

    let start = info.data_offsets[0];
    let end = info.data_offsets[1];
    if end < start {
        return Err(invalid_data(format!(
            "invalid data offsets for tensor: {name}"
        )));
    }

    let n_bytes = (end - start) as usize;
    file.seek(SeekFrom::Start(data_start + start))?;
    let mut buf = vec![0u8; n_bytes];
    file.read_exact(&mut buf)?;

    let data = bytes_to_f32(&info.dtype, &buf)?;
    let expected_items = info.shape.iter().product::<usize>();
    if data.len() != expected_items {
        return Err(invalid_data(format!(
            "tensor {name} has {} values, but shape {:?} expects {expected_items}",
            data.len(),
            info.shape
        )));
    }

    Ok(WeightTensor {
        shape: info.shape.clone(),
        data,
    })
}

pub fn load_tensor_from_path(path: &str, name: &str) -> std::io::Result<WeightTensor> {
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
    let (header, data_start) = read_header(&mut file)?;
    load_tensor(&mut file, data_start, &header, name)
}

pub fn load(path: &str) -> std::io::Result<BTreeMap<String, WeightTensor>> {
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
    let (header, data_start) = read_header(&mut file)?;

    let mut weights = BTreeMap::new();
    for name in header.tensors.keys() {
        if name == "model.embed_tokens.weight" {
            continue;
        }

        let tensor = load_tensor(&mut file, data_start, &header, name)?;
        weights.insert(name.clone(), tensor);
    }

    Ok(weights)
}
