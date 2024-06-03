extern crate ocl;

use std::str::FromStr;
use std::time::Instant;

use num_format::{Locale, ToFormattedString};
use rand::random;

use crate::network::Network;

mod network;
mod gpu_math;

fn main() {
    // backprop();
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
    let out = network.forward(vec![1.0; layers[0]]);
    println!("{:?} {:?}", st.elapsed(), out);
    for i in 0..500 {
        let out = network.forward(vec![1.0; layers[0]]);
        println!("{:?}", out);
        network.backward(vec![-0.5, 0.7, 0.2], 0.1);
        // let out = network.forward(vec![0.2; layers[0]]);
        // println!("{:?}", out);
        // network.backward(vec![0.1, -0.2, 0.8], 0.1);
    }
    let out = network.forward(vec![1.0; layers[0]]);
    println!("{:?} {:?}", st.elapsed(), out);
    let out = network.forward(vec![0.2; layers[0]]);
    println!("{:?} {:?}", st.elapsed(), out);
}

pub fn error_derivative(actual: f64, desired: f64) -> f64 {
    2.0 * (actual - desired)
}

pub fn activation(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp()) * 2.0 - 1.0
    // value
}

pub fn activation_derivative(activated: f64) -> f64 {
    (2.0*activated.exp()) / (activated.exp() + 1.0).powf(2.0)
    // 1.0
}

// gradient = input * error_derivative(actual_output - desired_output)
// new_weight = original_weight - learn_rate * gradient

pub fn backprop() {
    let learn_rate = 0.1;
    let input = 1.5;
    let mut weight_1 = random::<f64>();
    let mut bias_1 = random::<f64>();
    let mut weight_2 = random::<f64>();
    let mut bias_2 = random::<f64>();
    let desired_output = -0.2;

    for i in 0..10000 {
        let layer_1_output = input * weight_1 + bias_1;
        let layer_1_activated = activation(layer_1_output);
        let layer_2_output = layer_1_activated * weight_2 + bias_2;
        let layer_2_activated = activation(layer_2_output);

        let layer_2_desired = desired_output;

        let gradient =  activation_derivative(layer_2_output) * error_derivative(layer_2_activated, layer_2_desired);
        println!("L2T {} {} {} {}", layer_2_activated, layer_1_activated, layer_2_desired, gradient);

        weight_2 = weight_2 - learn_rate * layer_1_activated * gradient;
        bias_2 = bias_2 - learn_rate * gradient;

        let layer_1_desired = gradient * weight_1;

        let gradient =  activation_derivative(layer_1_output) * error_derivative(layer_1_activated, layer_1_desired);
        println!("L1T {} {} {}", layer_1_activated, input, gradient);

        weight_1 = weight_1 - learn_rate * input * gradient;
        bias_1 = bias_1 - learn_rate * gradient;
    }

    let layer_1_output = activation(input * weight_1 + bias_1);
    let layer_2_output = activation(layer_1_output * weight_2 + bias_2);
    println!("AF {} {} {} {} {}", layer_2_output, weight_1, bias_1, weight_2, bias_2);
}