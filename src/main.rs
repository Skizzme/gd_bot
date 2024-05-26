mod network;

extern crate ocl;

use std::fs::read_to_string;
use std::str::FromStr;
use std::string::ToString;
use std::thread::sleep;
use std::time::{Duration, Instant};
use ocl::{Buffer, Device, Platform, ProQue, SpatialDims};
use ocl::ffi::libc::rand;
use crate::network::Network;

fn main() {
    println!("creating network");
    let network = Network::new(vec![1024, 1024*16, 1024*32, 32], None, None).unwrap();
    println!("{} {}", network.weights().len(), network.biases().len());
    let st = Instant::now();
    let out = network.forward(vec![1.0; 1024], 16);
    println!("{:?} {:?}", st.elapsed(), out);
    loop {

    }
}