use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

fn permute_pixels(
    src_data: &[u8],
    src_texel_size: u8,
    src_to_dst_map: &HashMap<u8, u8>,
    dst_texel_size: u8,
) -> Vec<u8> {
    // The conversion from hashmap to vec is done because it grants a 4x speedup
    // while looking up the keys
    let mut src_to_dst_map_vec: Vec<Option<u8>> =
        vec![None; (*src_to_dst_map.keys().max().unwrap() + 1) as usize];
    src_to_dst_map.iter().for_each(|(k, v)| {
        src_to_dst_map_vec[*k as usize] = Some(*v as u8);
    });

    let mut out_data = Vec::<u8>::new();
    out_data.resize(
        (src_data.len() / src_texel_size as usize) * dst_texel_size as usize,
        0,
    );
    let mut written_out_data: usize = 0;

    for src_texel_idx in 0..src_data.len() / src_texel_size as usize {
        for src_byte_idx in 0..std::cmp::min(src_texel_size, src_to_dst_map_vec.len() as u8) {
            if let Some(dst_byte_idx) = src_to_dst_map_vec[src_byte_idx as usize] {
                out_data[written_out_data + dst_byte_idx as usize] =
                    src_data[src_texel_idx * src_texel_size as usize + src_byte_idx as usize];
            }
        }
        written_out_data += dst_texel_size as usize;
    }
    out_data
}

fn permute_pixels_same_size_x86_ssse(
    src_data: &mut [u8],
    texel_size: u8,
    src_to_dst_map: &HashMap<u8, u8>,
) {
    use core::arch::x86_64::*;
    unsafe {
        // Converting the hashmap to a mask
        // HashMap::from([(0, 2), (1, 0), (2, 1)]);

        let mut mask_arr: [i8; core::mem::size_of::<__m128i>()] = [0; 16];

        let simd_block_iter = (0..core::mem::size_of::<__m128i>()).step_by(texel_size as usize);
        for texel_starting_byte in simd_block_iter {
            for texel_component_idx in 0..texel_size {
                if let Some(dst_idx) = src_to_dst_map.get(&texel_component_idx) {
                    mask_arr[texel_starting_byte + *dst_idx as usize] =
                        (texel_starting_byte as u8 + texel_component_idx) as i8;
                }
            }
        }
        let mask: __m128i = _mm_loadu_si128(mask_arr.as_ptr() as *const __m128i);

        for i in (0..src_data.len()).step_by(core::mem::size_of::<__m128i>()) {
            let mem_address = src_data.as_ptr().add(i) as *mut __m128i;

            let data: __m128i = _mm_loadu_si128(mem_address);
            let out_data = _mm_shuffle_epi8(data, mask);
            _mm_storeu_si128(mem_address, out_data);
        }
    }
}

// Permutes pixels with avx2 instructions (slower than ssse version on my processor)
fn permute_pixels_same_size_x86_avx2(
    src_data: &mut [u8],
    texel_size: u8,
    src_to_dst_map: &HashMap<u8, u8>,
) {
    use core::arch::x86_64::*;
    unsafe {
        // Converting the hashmap to a mask
        // HashMap::from([(0, 2), (1, 0), (2, 1)]);
        _mm256_zeroall();

        let mut mask_arr: [i8; core::mem::size_of::<__m256i>()] = [0; 32];
        let simd_block_iter = (0..core::mem::size_of::<__m256i>()).step_by(texel_size as usize);
        for texel_starting_byte in simd_block_iter {
            for texel_component_idx in 0..texel_size {
                if let Some(dst_idx) = src_to_dst_map.get(&texel_component_idx) {
                    mask_arr[texel_starting_byte + *dst_idx as usize] =
                        (texel_starting_byte as u8 + texel_component_idx) as i8;
                }
            }
        }

        let mask: __m256i = _mm256_loadu_si256(mask_arr.as_ptr() as *const __m256i);
        for i in (0..src_data.len()).step_by(core::mem::size_of::<__m256i>()) {
            let mem_address = src_data.as_ptr().add(i) as *mut __m256i;

            // Unaligned loads because if we want to do aligned then we need to
            // make sure our vec is allocated within 32 byte boundaries
            let data: __m256i = _mm256_loadu_si256(mem_address);
            let out_data = _mm256_shuffle_epi8(data, mask);
            _mm256_storeu_si256(mem_address, out_data);
        }
    }
}

fn normal_permute(c: &mut Criterion) {
    let mut input_data = (0..268_435_456)
        .map(|i| (i % 256) as u8)
        .collect::<Vec<u8>>();
    let conversion_map = HashMap::from([(0, 2), (1, 0), (2, 1), (3, 3)]);

    c.bench_function("Normal Permute", |b| {
        b.iter(|| input_data = permute_pixels(&input_data, 4, &conversion_map, 4))
    });
}

fn simd_permute(c: &mut Criterion) {
    let mut input_data = (0..268_435_456)
        .map(|i| (i % 256) as u8)
        .collect::<Vec<u8>>();
    let conversion_map = HashMap::from([(0, 2), (1, 0), (2, 1), (3, 3)]);

    c.bench_function("Simd Permute", |b| {
        b.iter(|| {
            permute_pixels_same_size_x86_ssse(&mut input_data, 4, &conversion_map);
        })
    });
}

fn simd_avx2_permute(c: &mut Criterion) {
    let mut input_data = (0..268_435_456)
        .map(|i| (i % 256) as u8)
        .collect::<Vec<u8>>();
    let conversion_map = HashMap::from([(0, 2), (1, 0), (2, 1), (3, 3)]);

    c.bench_function("Simd AVX2 Permute", |b| {
        b.iter(|| {
            permute_pixels_same_size_x86_avx2(&mut input_data, 4, &conversion_map);
        })
    });
}

criterion_group!(name = benches; config = Criterion::default().sample_size(10).measurement_time(std::time::Duration::from_secs(10)); targets = normal_permute, simd_permute, simd_avx2_permute);
criterion_main!(benches);
