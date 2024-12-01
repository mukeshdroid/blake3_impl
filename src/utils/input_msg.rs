/// takes an hex string of arbitary length as input
/// adds padding if needed and outputs Vectors of chunks which are each 1024 bytes
/// 1024 byte = 2048 hex chars
fn input_to_chunks(input: &str) -> Vec<String> {
    //since our msg will always be less than 1024 bytes, lets not worry about this for now.
    //if less than chunk size simply pad and return
    // if input.len() <= 2048{
    // input.to_owned() + "0".repeat(2048 - input.len())
    // }

    todo!();
}

/// takes a chunk of 1024 bytes and outputs Vec of 16 msg blocks each of 64 bytes
/// containing 16 message words.
fn chunk_to_msgblocks(chunk: &str) -> Vec<String> {
    todo!("This function needs to be implemented");
}

/// takes a msgblock of 16 words and converts them to a
/// vector of 16 u32 words in Little Endian notation
fn msgblock_to_words(msgblock: &str) -> Vec<u32> {
    let msg_words: Vec<u32> = Vec::new();

    msg_words
}
