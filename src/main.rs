extern crate ocl;

use std::str::FromStr;

mod network;
mod gpu_math;
mod cl_utils;
mod tests;


fn main() {
    tests::ac::test_ac::run();
}

pub fn get_network_args() -> (u32, f32, Vec<usize>) {
    let args: Vec<String> = std::env::args().collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();

    let epochs = u32::from_str(args[1]).unwrap();
    let learn_rate = f32::from_str(args[2]).unwrap();
    let mut layers = vec![];
    for i in 3..args.len() {
        let arg_val = args[i];
        layers.push(usize::from_str(arg_val).unwrap());
    }
    (epochs, learn_rate, layers)
}