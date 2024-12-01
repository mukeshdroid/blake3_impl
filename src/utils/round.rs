use crate::PERMUTE;

///create the initial state
fn init_state() -> [u32; 16] {
    todo!()
}

///apply the quarter-round state update function G
fn G(a: &mut u32, b: &mut u32, c: &mut u32, d: &mut u32, m0: &mut u32, m1: &mut u32) {
    *a = *a + *b + *m0;
}

///apply permutation
fn permute(state: [u32; 16]) -> [u32; 16] {
    todo!();
}

///compression function
fn compression(state: Vec<u32>) -> Vec<u32> {
    todo!()
}
