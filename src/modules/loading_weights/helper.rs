pub fn invalid_data(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into())
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits & 0x7c00) >> 10;
    let frac = bits & 0x03ff;

    let f32_bits = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut frac = frac as u32;
                let mut exp = -14i32;
                while (frac & 0x0400) == 0 {
                    frac <<= 1;
                    exp -= 1;
                }
                frac &= 0x03ff;
                sign | (((exp + 127) as u32) << 23) | (frac << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | ((frac as u32) << 13),
        _ => sign | (((exp as u32) + 112) << 23) | ((frac as u32) << 13),
    };

    f32::from_bits(f32_bits)
}

pub fn bytes_to_f32(dtype: &str, bytes: &[u8]) -> std::io::Result<Vec<f32>> {
    match dtype {
        "F32" => {
            if bytes.len() % 4 != 0 {
                return Err(invalid_data("F32 tensor byte length is not divisible by 4"));
            }

            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect())
        }
        "F16" => {
            if bytes.len() % 2 != 0 {
                return Err(invalid_data("F16 tensor byte length is not divisible by 2"));
            }

            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| f16_to_f32(u16::from_le_bytes(chunk.try_into().unwrap())))
                .collect())
        }
        "BF16" => {
            if bytes.len() % 2 != 0 {
                return Err(invalid_data(
                    "BF16 tensor byte length is not divisible by 2",
                ));
            }

            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bf16 = u16::from_le_bytes(chunk.try_into().unwrap());
                    f32::from_bits((bf16 as u32) << 16)
                })
                .collect())
        }
        _ => Err(invalid_data(format!("unsupported tensor dtype: {dtype}"))),
    }
}
