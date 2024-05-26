mod network;

extern crate ocl;

use std::ffi::c_uint;
use std::fs::read_to_string;
use std::str::FromStr;
use std::string::ToString;
use std::thread::sleep;
use std::time::{Duration, Instant};
use num_format::{Locale, ToFormattedString};
use ocl::{Buffer, Device, Platform, ProQue, SpatialDims};
use ocl::ffi::c_int;
use ocl::ffi::libc::{rand, srand};
use rand::thread_rng;
use crate::network::Network;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();

    let mut layers = vec![];
    for i in 1..args.len() {
        let arg_val = args[i];
        layers.push(i32::from_str(arg_val).unwrap());
    }
    let network = Network::new(layers.clone(), None, None).unwrap();
    println!("Total weights: {}, Total biases: {}", network.weights().to_formatted_string(&Locale::en), network.biases().to_formatted_string(&Locale::en));
    let st = Instant::now();
    let out = network.calculate(vec![1.0; layers[0] as usize]);
    println!("{:?} {:?}", st.elapsed(), out);
}