use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Deserialize)]
pub struct BpeModel {
    pub vocab: HashMap<String, usize>,
    pub merges: Vec<String>,
}

#[derive(Deserialize)]
pub struct AddedToken {
    pub id: usize,
    pub content: String,
    #[serde(default)]
    pub special: bool,
}

#[derive(Deserialize)]
pub struct TokenizerFile {
    pub model: Option<BpeModel>,
    #[serde(default)]
    pub added_tokens: Vec<AddedToken>,
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

fn is_letter(ch: char) -> bool {
    ch.is_alphabetic()
}

fn is_number(ch: char) -> bool {
    ch.is_numeric()
}

fn byte_level_encode(text: &str, byte_encoder: &HashMap<u8, char>) -> String {
    text.as_bytes()
        .iter()
        .map(|byte| byte_encoder[byte])
        .collect()
}

fn split_byte_level_words(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut i = 0;

    while i < chars.len() {
        let start = chars[i].0;
        let mut end = start + chars[i].1.len_utf8();
        let mut j = i;
        let has_leading_space = chars[j].1 == ' ';

        if has_leading_space && j + 1 < chars.len() {
            let next = chars[j + 1].1;
            if is_letter(next) {
                j += 1;
                while j < chars.len() && is_letter(chars[j].1) {
                    end = chars[j].0 + chars[j].1.len_utf8();
                    j += 1;
                }
                pieces.push(&text[start..end]);
                i = j;
                continue;
            }

            if is_number(next) {
                pieces.push(&text[start..chars[j + 1].0]);
                i += 1;
                continue;
            }

            if !next.is_whitespace() {
                j += 1;
                while j < chars.len()
                    && !chars[j].1.is_whitespace()
                    && !is_letter(chars[j].1)
                    && !is_number(chars[j].1)
                {
                    end = chars[j].0 + chars[j].1.len_utf8();
                    j += 1;
                }
                pieces.push(&text[start..end]);
                i = j;
                continue;
            }
        }

        let ch = chars[i].1;
        if is_letter(ch) {
            j += 1;
            while j < chars.len() && is_letter(chars[j].1) {
                end = chars[j].0 + chars[j].1.len_utf8();
                j += 1;
            }
            pieces.push(&text[start..end]);
            i = j;
        } else if is_number(ch) {
            pieces.push(&text[start..end]);
            i += 1;
        } else if ch.is_whitespace() {
            j += 1;
            while j < chars.len() && chars[j].1.is_whitespace() {
                end = chars[j].0 + chars[j].1.len_utf8();
                j += 1;
            }
            pieces.push(&text[start..end]);
            i = j;
        } else {
            j += 1;
            while j < chars.len()
                && !chars[j].1.is_whitespace()
                && !is_letter(chars[j].1)
                && !is_number(chars[j].1)
            {
                end = chars[j].0 + chars[j].1.len_utf8();
                j += 1;
            }
            pieces.push(&text[start..end]);
            i = j;
        }
    }

    pieces
}

fn split_special_tokens<'a>(text: &'a str, special_tokens: &HashSet<&str>) -> Vec<&'a str> {
    if special_tokens.is_empty() {
        return vec![text];
    }

    let mut pieces = Vec::new();
    let mut cursor = 0;

    while cursor < text.len() {
        let next_special = special_tokens
            .iter()
            .filter_map(|token| {
                text[cursor..]
                    .find(token)
                    .map(|offset| (cursor + offset, *token))
            })
            .min_by_key(|(index, token)| (*index, std::cmp::Reverse(token.len())));

        let Some((index, token)) = next_special else {
            pieces.push(&text[cursor..]);
            break;
        };

        if index > cursor {
            pieces.push(&text[cursor..index]);
        }

        pieces.push(&text[index..index + token.len()]);
        cursor = index + token.len();
    }

    pieces
}

pub fn tokenize_text(
    text: &str,
    tokenizer_file_path: &str,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
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
    let special_tokens: HashSet<&str> = tokenizer_file
        .added_tokens
        .iter()
        .filter(|token| token.special)
        .map(|token| token.content.as_str())
        .collect();
    let added_token_ids: HashMap<&str, usize> = tokenizer_file
        .added_tokens
        .iter()
        .map(|token| (token.content.as_str(), token.id))
        .collect();
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

    let mut token_ids = Vec::new();
    for section in split_special_tokens(text, &special_tokens) {
        if section.is_empty() {
            continue;
        }

        if let Some(id) = added_token_ids
            .get(section)
            .or_else(|| model.vocab.get(section))
        {
            token_ids.push(*id);
            continue;
        }

        for word in split_byte_level_words(section) {
            let encoded = byte_level_encode(word, &byte_encoder);
            let initial_tokens = encoded.chars().map(|ch| ch.to_string()).collect();
            let tokens = apply_bpe(initial_tokens, &merges);

            for token in tokens {
                let id = model.vocab.get(&token).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("tokenizer emitted token missing from vocab: {token}"),
                    )
                })?;
                token_ids.push(*id);
            }
        }
    }

    Ok(token_ids)
}

pub fn decode_tokens(
    token_ids: &[usize],
    tokenizer_file_path: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let tokenizer_json = std::fs::read_to_string(tokenizer_file_path)?;
    let tokenizer_file: TokenizerFile = serde_json::from_str(&tokenizer_json)?;
    let model = tokenizer_file.model.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "tokenizer file does not contain a BPE model",
        )
    })?;

    // inverse of step 4: id -> token string (added tokens override vocab)
    let mut id_to_token: HashMap<usize, String> = model
        .vocab
        .into_iter()
        .map(|(token, id)| (id, token))
        .collect();
    for token in tokenizer_file.added_tokens {
        id_to_token.insert(token.id, token.content);
    }

    // inverse of step 2: byte-level unicode char -> original byte
    let byte_decoder: HashMap<char, u8> = bytes_to_unicode()
        .into_iter()
        .map(|(byte, ch)| (ch, byte))
        .collect();

    // ids -> token strings -> one byte-level-encoded string
    let mut encoded = String::new();
    for id in token_ids {
        let token = id_to_token.get(id).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("token id missing from vocab: {id}"),
            )
        })?;
        encoded.push_str(token);
    }

    // map each char back to its raw byte, then interpret the bytes as UTF-8
    let bytes: Vec<u8> = encoded
        .chars()
        .filter_map(|ch| byte_decoder.get(&ch).copied())
        .collect();

    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

#[cfg(test)]
mod tests {
    use super::tokenize_text;

    const TOKENIZER_PATH: &str = "models/SmolLM2-135M/tokenizer.json";

    #[test]
    fn tokenizes_demo_text_to_ids() {
        let tokens = tokenize_text("Hello, world! Karneeshkar", TOKENIZER_PATH).unwrap();
        assert_eq!(tokens, vec![19556, 28, 905, 17, 13463, 662, 6509, 25616]);
    }

    #[test]
    fn preserves_special_tokens() {
        let tokens = tokenize_text("<|im_start|>user", TOKENIZER_PATH).unwrap();
        assert_eq!(tokens[0], 1);
    }
}
