pub mod utils;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

///define initial values
const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// define permutation map
const PERMUTE: [u8; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// the main blake3 function that outouts the hash of the input as a hex string
pub fn blake3(input: String) -> String {
    // // convert input to chunks
    // let chunks = input_to_chunks(input);

    //for each chunk, convert them to msg blocks

    //for each msg blocks, convert to msg words using LE

    //

    todo!("This function needs to be implemented");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
