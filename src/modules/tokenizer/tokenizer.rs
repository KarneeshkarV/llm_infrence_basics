use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Deserialize)]
pub struct BpeModel {
    pub vocab: HashMap<String, usize>,
    pub merges: Vec<String>,
    //pub special_tokens: HashMap<String, usize>,
}

#[derive(Deserialize)]
pub struct TokenizerFile {
    pub model: Option<BpeModel>,
}

fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bytes: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut chars: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
    let mut next = 0u32;

    for b in 0u8..=u8::MAX {
        if !bytes.contains(&b) {
            bytes.push(b);
            chars.push(256 + next);
            next += 1;
        }
    }

    bytes
        .into_iter()
        .zip(chars)
        .filter_map(|(byte, code_point)| char::from_u32(code_point).map(|ch| (byte, ch)))
        .collect()
}

fn apply_bpe(mut tokens: Vec<String>, merges: &HashMap<(String, String), usize>) -> Vec<String> {
    loop {
        let mut best_pair: Option<(usize, usize)> = None;

        for i in 0..tokens.len().saturating_sub(1) {
            let pair = (tokens[i].clone(), tokens[i + 1].clone());

            if let Some(&rank) = merges.get(&pair) {
                match best_pair {
                    Some((_, current_best)) if rank >= current_best => {}
                    _ => best_pair = Some((i, rank)),
                }
            }
        }

        let Some((i, _)) = best_pair else {
            break;
        };

        let merged = format!("{}{}", tokens[i], tokens[i + 1]);
        tokens.splice(i..=i + 1, [merged]);
    }

    tokens
}

pub fn tokenize_text(
    text: &str,
    tokenizer_file_path: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let path = Path::new(tokenizer_file_path);
    match path.extension() {
        Some(ext) if ext == "json" => (),
        _ => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a json file",
            )));
        }
    }

    let tokenizer_json = std::fs::read_to_string(tokenizer_file_path)?;
    let tokenizer_file: TokenizerFile = serde_json::from_str(&tokenizer_json)?;
    let model = tokenizer_file.model.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "tokenizer file does not contain a BPE model",
        )
    })?;

    let merges: HashMap<(String, String), usize> = model
        .merges
        .iter()
        .enumerate()
        .map(|(rank, merge_str)| {
            let mut parts = merge_str.split(' ');
            let a = parts.next().unwrap_or_default().to_string();
            let b = parts.next().unwrap_or_default().to_string();
            ((a, b), rank)
        })
        .collect();
    let byte_encoder = bytes_to_unicode();
    let initial_tokens = text
        .as_bytes()
        .iter()
        .map(|byte| byte_encoder[byte].to_string())
        .collect();

    let tokens = apply_bpe(initial_tokens, &merges);
    let unknown_tokens: Vec<&String> = tokens
        .iter()
        .filter(|token| !model.vocab.contains_key(*token))
        .collect();

    if !unknown_tokens.is_empty() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("tokenizer emitted tokens missing from vocab: {unknown_tokens:?}"),
        )));
    }

    Ok(tokens)
}
