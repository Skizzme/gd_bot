extern crate ocl;

use std::str::FromStr;

mod network;
mod gpu_math;
mod cl_utils;
mod tests;


fn main() {
    // tests::test_2d::run();
    tests::ac::test_ac::run();
}

pub fn get_network_args(args: Option<Vec<String>>) -> (u32, f32, Vec<usize>) {
    let args: Vec<String> = match args {
        None => {
            let mut env_args: Vec<String> = std::env::args().collect();
            env_args.remove(0);
            let collected = env_args.iter().map(String::clone).collect::<Vec<String>>();
            collected
        }
        Some(args) => args,
    };
    println!("args {:?}", args);

    let epochs = u32::from_str(&args[0]).unwrap();
    let learn_rate = f32::from_str(&args[1]).unwrap();
    let mut layers = vec![];
    for i in 2..args.len() {
        let arg_val = &args[i];
        layers.push(usize::from_str(&arg_val).unwrap());
    }
    (epochs, learn_rate, layers)
}

struct BytesReader {
    bytes: Vec<u8>,
    index: usize,
    buf_4: [u8; 4],
    buf_8: [u8; 8],
}

impl BytesReader {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            index: 0,
            buf_4: [0u8; 4],
            buf_8: [0u8; 8],
        }
    }

    pub fn next_f32(&mut self) -> f32 {
        for i in 0..4 {
            self.buf_4[i] = self.bytes[self.index];
            self.index += 1;
        }
        f32::from_be_bytes(self.buf_4)
    }

    pub fn ended(&self) -> bool {
        self.index >= self.bytes.len()
    }

    pub fn next_usize(&mut self) -> usize {
        for i in 0..8 {
            self.buf_8[i] = self.bytes[self.index];
            self.index += 1;
        }
        usize::from_be_bytes(self.buf_8)
    }

    pub fn next_i32(&mut self) -> i32 {
        for i in 0..4 {
            self.buf_4[i] = self.bytes[self.index];
            self.index += 1;
        }
        i32::from_be_bytes(self.buf_4)
    }
}