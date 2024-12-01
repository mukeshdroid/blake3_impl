//! This is the reference implementation of BLAKE3. It is used for testing and
//! as a readable example of the algorithms involved. Section 5.1 of [the BLAKE3
//! spec](https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf)
//! discusses this implementation. You can render docs for this implementation
//! by running `cargo doc --open` in this directory.
//!
//! # Example
//!
//! ```
//! let mut hasher = blake3_impl::utils::ref_impl::Hasher::new();
//! hasher.update(b"abc");
//! hasher.update(b"def");
//! let mut hash = [0; 32];
//! hasher.finalize(&mut hash);
//! let mut extended_hash = [0; 500];
//! hasher.finalize(&mut extended_hash);
//! assert_eq!(hash, extended_hash[..32]);
//! ```

use core::cmp::min;

const OUT_LEN: usize = 32;
const KEY_LEN: usize = 32;
const BLOCK_LEN: usize = 64;
const CHUNK_LEN: usize = 1024;

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;
const KEYED_HASH: u32 = 1 << 4;
const DERIVE_KEY_CONTEXT: u32 = 1 << 5;
const DERIVE_KEY_MATERIAL: u32 = 1 << 6;

const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

// The mixing function, G, which mixes either a column or a diagonal.
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

fn round(state: &mut [u32; 16], m: &[u32; 16]) {
    // Mix the columns.
    g(state, 0, 4, 8, 12, m[0], m[1]);
    g(state, 1, 5, 9, 13, m[2], m[3]);
    g(state, 2, 6, 10, 14, m[4], m[5]);
    g(state, 3, 7, 11, 15, m[6], m[7]);
    // Mix the diagonals.
    g(state, 0, 5, 10, 15, m[8], m[9]);
    g(state, 1, 6, 11, 12, m[10], m[11]);
    g(state, 2, 7, 8, 13, m[12], m[13]);
    g(state, 3, 4, 9, 14, m[14], m[15]);
}

fn permute(m: &mut [u32; 16]) {
    let mut permuted = [0; 16];
    for i in 0..16 {
        permuted[i] = m[MSG_PERMUTATION[i]];
    }
    *m = permuted;
}

fn compress(
    chaining_value: &[u32; 8],
    block_words: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    let counter_low = counter as u32;
    let counter_high = (counter >> 32) as u32;
    #[rustfmt::skip]
    let mut state = [
        chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3],
        chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7],
        IV[0],             IV[1],             IV[2],             IV[3],
        counter_low,       counter_high,      block_len,         flags,
    ];
    let mut block = *block_words;

    round(&mut state, &block); // round 1
    permute(&mut block);
    round(&mut state, &block); // round 2
    permute(&mut block);
    round(&mut state, &block); // round 3
    permute(&mut block);
    round(&mut state, &block); // round 4
    permute(&mut block);
    round(&mut state, &block); // round 5
    permute(&mut block);
    round(&mut state, &block); // round 6
    permute(&mut block);
    round(&mut state, &block); // round 7

    for i in 0..8 {
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }
    state
}

fn first_8_words(compression_output: [u32; 16]) -> [u32; 8] {
    compression_output[0..8].try_into().unwrap()
}

fn words_from_little_endian_bytes(bytes: &[u8], words: &mut [u32]) {
    debug_assert_eq!(bytes.len(), 4 * words.len());
    for (four_bytes, word) in bytes.chunks_exact(4).zip(words) {
        *word = u32::from_le_bytes(four_bytes.try_into().unwrap());
    }
}

// Each chunk or parent node can produce either an 8-word chaining value or, by
// setting the ROOT flag, any number of final output bytes. The Output struct
// captures the state just prior to choosing between those two possibilities.
struct Output {
    input_chaining_value: [u32; 8],
    block_words: [u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
}

impl Output {
    fn chaining_value(&self) -> [u32; 8] {
        first_8_words(compress(
            &self.input_chaining_value,
            &self.block_words,
            self.counter,
            self.block_len,
            self.flags,
        ))
    }

    fn root_output_bytes(&self, out_slice: &mut [u8]) {
        let mut output_block_counter = 0;
        for out_block in out_slice.chunks_mut(2 * OUT_LEN) {
            let words = compress(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                self.flags | ROOT,
            );
            // The output length might not be a multiple of 4.
            for (word, out_word) in words.iter().zip(out_block.chunks_mut(4)) {
                out_word.copy_from_slice(&word.to_le_bytes()[..out_word.len()]);
            }
            output_block_counter += 1;
        }
    }
}

struct ChunkState {
    chaining_value: [u32; 8],
    chunk_counter: u64,
    block: [u8; BLOCK_LEN],
    block_len: u8,
    blocks_compressed: u8,
    flags: u32,
}

impl ChunkState {
    fn new(key_words: [u32; 8], chunk_counter: u64, flags: u32) -> Self {
        Self {
            chaining_value: key_words,
            chunk_counter,
            block: [0; BLOCK_LEN],
            block_len: 0,
            blocks_compressed: 0,
            flags,
        }
    }

    fn len(&self) -> usize {
        BLOCK_LEN * self.blocks_compressed as usize + self.block_len as usize
    }

    fn start_flag(&self) -> u32 {
        if self.blocks_compressed == 0 {
            CHUNK_START
        } else {
            0
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            // If the block buffer is full, compress it and clear it. More
            // input is coming, so this compression is not CHUNK_END.
            if self.block_len as usize == BLOCK_LEN {
                let mut block_words = [0; 16];
                words_from_little_endian_bytes(&self.block, &mut block_words);
                self.chaining_value = first_8_words(compress(
                    &self.chaining_value,
                    &block_words,
                    self.chunk_counter,
                    BLOCK_LEN as u32,
                    self.flags | self.start_flag(),
                ));
                self.blocks_compressed += 1;
                self.block = [0; BLOCK_LEN];
                self.block_len = 0;
            }

            // Copy input bytes into the block buffer.
            let want = BLOCK_LEN - self.block_len as usize;
            let take = min(want, input.len());
            self.block[self.block_len as usize..][..take].copy_from_slice(&input[..take]);
            self.block_len += take as u8;
            input = &input[take..];
        }
    }

    fn output(&self) -> Output {
        let mut block_words = [0; 16];
        words_from_little_endian_bytes(&self.block, &mut block_words);
        Output {
            input_chaining_value: self.chaining_value,
            block_words,
            counter: self.chunk_counter,
            block_len: self.block_len as u32,
            flags: self.flags | self.start_flag() | CHUNK_END,
        }
    }
}

fn parent_output(
    left_child_cv: [u32; 8],
    right_child_cv: [u32; 8],
    key_words: [u32; 8],
    flags: u32,
) -> Output {
    let mut block_words = [0; 16];
    block_words[..8].copy_from_slice(&left_child_cv);
    block_words[8..].copy_from_slice(&right_child_cv);
    Output {
        input_chaining_value: key_words,
        block_words,
        counter: 0,                  // Always 0 for parent nodes.
        block_len: BLOCK_LEN as u32, // Always BLOCK_LEN (64) for parent nodes.
        flags: PARENT | flags,
    }
}

fn parent_cv(
    left_child_cv: [u32; 8],
    right_child_cv: [u32; 8],
    key_words: [u32; 8],
    flags: u32,
) -> [u32; 8] {
    parent_output(left_child_cv, right_child_cv, key_words, flags).chaining_value()
}

/// An incremental hasher that can accept any number of writes.
pub struct Hasher {
    chunk_state: ChunkState,
    key_words: [u32; 8],
    cv_stack: [[u32; 8]; 54], // Space for 54 subtree chaining values:
    cv_stack_len: u8,         // 2^54 * CHUNK_LEN = 2^64
    flags: u32,
}

impl Hasher {
    fn new_internal(key_words: [u32; 8], flags: u32) -> Self {
        Self {
            chunk_state: ChunkState::new(key_words, 0, flags),
            key_words,
            cv_stack: [[0; 8]; 54],
            cv_stack_len: 0,
            flags,
        }
    }

    /// Construct a new `Hasher` for the regular hash function.
    pub fn new() -> Self {
        Self::new_internal(IV, 0)
    }

    /// Construct a new `Hasher` for the keyed hash function.
    pub fn new_keyed(key: &[u8; KEY_LEN]) -> Self {
        let mut key_words = [0; 8];
        words_from_little_endian_bytes(key, &mut key_words);
        Self::new_internal(key_words, KEYED_HASH)
    }

    /// Construct a new `Hasher` for the key derivation function. The context
    /// string should be hardcoded, globally unique, and application-specific.
    pub fn new_derive_key(context: &str) -> Self {
        let mut context_hasher = Self::new_internal(IV, DERIVE_KEY_CONTEXT);
        context_hasher.update(context.as_bytes());
        let mut context_key = [0; KEY_LEN];
        context_hasher.finalize(&mut context_key);
        let mut context_key_words = [0; 8];
        words_from_little_endian_bytes(&context_key, &mut context_key_words);
        Self::new_internal(context_key_words, DERIVE_KEY_MATERIAL)
    }

    fn push_stack(&mut self, cv: [u32; 8]) {
        self.cv_stack[self.cv_stack_len as usize] = cv;
        self.cv_stack_len += 1;
    }

    fn pop_stack(&mut self) -> [u32; 8] {
        self.cv_stack_len -= 1;
        self.cv_stack[self.cv_stack_len as usize]
    }

    // Section 5.1.2 of the BLAKE3 spec explains this algorithm in more detail.
    fn add_chunk_chaining_value(&mut self, mut new_cv: [u32; 8], mut total_chunks: u64) {
        // This chunk might complete some subtrees. For each completed subtree,
        // its left child will be the current top entry in the CV stack, and
        // its right child will be the current value of `new_cv`. Pop each left
        // child off the stack, merge it with `new_cv`, and overwrite `new_cv`
        // with the result. After all these merges, push the final value of
        // `new_cv` onto the stack. The number of completed subtrees is given
        // by the number of trailing 0-bits in the new total number of chunks.
        while total_chunks & 1 == 0 {
            new_cv = parent_cv(self.pop_stack(), new_cv, self.key_words, self.flags);
            total_chunks >>= 1;
        }
        self.push_stack(new_cv);
    }

    /// Add input to the hash state. This can be called any number of times.
    pub fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            // If the current chunk is complete, finalize it and reset the
            // chunk state. More input is coming, so this chunk is not ROOT.
            if self.chunk_state.len() == CHUNK_LEN {
                let chunk_cv = self.chunk_state.output().chaining_value();
                let total_chunks = self.chunk_state.chunk_counter + 1;
                self.add_chunk_chaining_value(chunk_cv, total_chunks);
                self.chunk_state = ChunkState::new(self.key_words, total_chunks, self.flags);
            }

            // Compress input bytes into the current chunk state.
            let want = CHUNK_LEN - self.chunk_state.len();
            let take = min(want, input.len());
            self.chunk_state.update(&input[..take]);
            input = &input[take..];
        }
    }

    /// Finalize the hash and write any number of output bytes.
    pub fn finalize(&self, out_slice: &mut [u8]) {
        // Starting with the Output from the current chunk, compute all the
        // parent chaining values along the right edge of the tree, until we
        // have the root Output.
        let mut output = self.chunk_state.output();
        let mut parent_nodes_remaining = self.cv_stack_len as usize;
        while parent_nodes_remaining > 0 {
            parent_nodes_remaining -= 1;
            output = parent_output(
                self.cv_stack[parent_nodes_remaining],
                output.chaining_value(),
                self.key_words,
                self.flags,
            );
        }
        output.root_output_bytes(out_slice);
    }
}

// remember that the blocklen in compress function expects number of bytes not number of words.
#[cfg(test)]
mod tests {
    use crate::utils::input_msg;

    use super::*;

    fn u32_array_to_hex_le(values: &[u32]) -> String {
        // Convert each `u32` to little-endian bytes, flatten the bytes, and map to a hex string
        values
            .iter()
            .flat_map(|&value| value.to_le_bytes()) // Convert to little-endian bytes
            .map(|byte| format!("{:02x}", byte)) // Format each byte as a 2-digit hex
            .collect::<String>() // Collect into a single string
    }

    #[test]
    fn test_empty_string() {
        let input: [u32; 16] = [0x00000000; 16];
        let hash_out = compress(&IV, &input, 0, 4, ROOT | CHUNK_END | CHUNK_START);
        println!("Hash Computed :: {}", u32_array_to_hex_le(&hash_out));


        //// Currently verifying manually from https://connor4312.github.io/blake3/index.html.
        //// Make sure to verify using blake3 rust implementaton and import it from official crate
        // let mut hasher = Hasher::new();
        // hasher.update(b"00000000000000000000000000000000");

        // let mut hash_official = [0; 32];
        // hasher.finalize(&mut hash_official);

        // println!("Hash Official :: {:?}", hash_official);
    }

    fn hex_to_u32_array(hex: &str) -> Result<[u32; 16], String> {
        // Ensure the hex string is exactly 128 characters long
        if hex.len() != 128 {
            return Err("Hex string must be exactly 128 characters long".to_string());
        }

        // Create an array to hold the 16 `u32` values
        let mut result = [0u32; 16];

        // Process the string in chunks of 8 characters
        for (i, chunk) in hex.as_bytes().chunks(8).enumerate() {
            // Convert the chunk to a string and reverse it for LE interpretation
            let chunk_str = std::str::from_utf8(chunk).expect("Invalid UTF-8");

            // Reverse the bytes for Little Endian interpretation
            let reversed_hex: String = chunk_str
                .as_bytes()
                .chunks(2) // Break into 2-character chunks (representing bytes)
                .rev() // Reverse the byte order
                .flat_map(std::str::from_utf8)
                .collect();

            // Parse the reversed string as a `u32`
            result[i] = u32::from_str_radix(&reversed_hex, 16)
                .map_err(|_| format!("Invalid hex chunk: {}", reversed_hex))?;
        }

        Ok(result)
    }

    #[test]
    fn test_all_ones_string() {
        let input1_str =  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

        let input1: [u32; 16] = hex_to_u32_array(&input1_str).unwrap();
        println!("{:?}", input1);
        let hash1 = compress(&IV, &input1, 0, 64, 11);

        println!("Hash1: {}", u32_array_to_hex_le(&hash1));
    }

    #[test]
    fn test_random() {
        let input1_str =  "12fade44ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

        let input1: [u32; 16] = hex_to_u32_array(&input1_str).unwrap();
        println!("{:?}", input1);
        let hash1 = compress(&IV, &input1, 0, 64, 11);

        println!("Hash1: {}", u32_array_to_hex_le(&hash1));
    }

    fn first_8_words(compression_output: [u32; 16]) -> [u32; 8] {
        compression_output[0..8].try_into().unwrap()
    }

    #[test]
    fn test_chaining_for_msgblocks() {
        let input1_str =  "48656c6c6f204d79206e616d65206973204d756b6573682054697761726920616e64204920616d20747279696e6720746f20666967757265206f757420575446";

        let input1: [u32; 16] = hex_to_u32_array(&input1_str).unwrap();

        let hash1 = compress(&IV, &input1, 0, 64, CHUNK_START);

        println!("\nHash1: {}\n", u32_array_to_hex_le(&hash1));

        let input2_str= "207468697320636f6d7072657373696f6e2066756e6374696f6e206973206e6f7420636861696e696e672077656c6c2e2055474848484848484848484848482e";
        let input2: [u32; 16] = hex_to_u32_array(input2_str).unwrap();


        
        let chaining_val: [u32; 8] = first_8_words(hash1);
        for v in chaining_val{
            print!("{:x}",v);
        }

        println!("\nChaining Val{:?}\n", chaining_val);


        let hash2 = compress(&chaining_val, &input2, 0, 64, ROOT | CHUNK_END);

        println!("Hash2: {}\n", u32_array_to_hex_le(&hash2));

        // let mut hasher = Hasher::new();
        // let mut hash_official = [0;32];
        // hasher.update(&hash_official);

        // println!("Official Hash: {}",u32_array_to_hex_le(hash_official));
    }
    
    #[test]
    fn test_structOutput(){
        let input1_str =  "48656c6c6f204d79206e616d65206973204d756b6573682054697761726920616e64204920616d20747279696e6720746f20666967757265206f757420575446";

        let input1: [u32; 16] = hex_to_u32_array(&input1_str).unwrap();


        let out = Output{
            input_chaining_value : IV,
            block_words : input1,
            counter:0,
            block_len:64,
            flags: ROOT | CHUNK_END | CHUNK_START
        };

        println!("{:?}",out.chaining_value());

        for v in out.chaining_value(){
            print!("{:x} ",v);
        }

        let input2: [u32; 16] = [0x00000000;16];

        let out = compress(&out.chaining_value(), &input2, 0, 1, ROOT | CHUNK_END | CHUNK_START);

        println!{"\n{}",u32_array_to_hex_le(&out)};
    }
}
