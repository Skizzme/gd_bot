extern crate ocl;

use std::str::FromStr;
use std::string::ToString;
use std::time::Instant;

use num_format::{Locale, ToFormattedString};

use crate::network::Network;

mod network;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();

    let mut layers = vec![];
    for i in 1..args.len() {
        let arg_val = args[i];
        layers.push(usize::from_str(arg_val).unwrap());
    }
    let network = Network::new(layers.clone(), None, None).unwrap();
    println!("Total weights: {}, Total biases: {}, Total mem: {}", network.weights().to_formatted_string(&Locale::en), network.biases().to_formatted_string(&Locale::en), ((network.weights()+network.biases())*i32::BITS as usize/8).to_formatted_string(&Locale::en));
    let st = Instant::now();
    let out = network.calculate(vec![1.0; layers[0] as usize]);
    println!("{:?} {:?}", st.elapsed(), out);
}