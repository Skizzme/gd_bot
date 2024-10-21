use std::fs;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener};
use std::num::ParseFloatError;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::Mutex;
use crate::{get_network_args, network};
use crate::network::cpu::CPUNetwork;
use crate::network::gpu::GPUNetwork;
use crate::network::Network;

type TNetwork = GPUNetwork;
pub fn run() {
    let mut args: Vec<String> = std::env::args().collect();
    args.remove(0);

    if args[0] == "--new" {
        args.remove(0);
        let train_data = args.remove(0);

        let (epochs, learn_rate, layers) = get_network_args(Some(args));
        let mut network = TNetwork::new(layers, None, None).unwrap();

        let (mut inputs, mut outputs) = get_training_data(train_data);
        network.train(epochs, 0.001, &mut inputs, &mut outputs, learn_rate, |_| {});
    } else if args[0] == "--train" {
        args.remove(0);
        let train_data = args.remove(0);

        let st = Instant::now();
        let (layers, biases, weights) = network::read_network(&args[0]);
        args.remove(0);
        println!("Loaded network in {:?}", st.elapsed());

        let (epochs, learn_rate, _) = get_network_args(Some(args));

        let (mut inputs, mut outputs) = get_training_data(train_data);
        let mut network = TNetwork::new(layers, Some(biases), Some(weights)).unwrap();

        network.train(epochs, 0.001, &mut inputs, &mut outputs, learn_rate, |_| {});
    } else {
        let st = Instant::now();
        let (layers, biases, weights) = network::read_network(&args[0]);
        println!("Loaded network in {:?}", st.elapsed());
        let mut network = TNetwork::new(layers, Some(biases), Some(weights)).unwrap();
        server(&mut network);
    }

    // let st = Instant::now();
    // let fwd = network.forward(&inputs[0]);
    // println!("fwd took {:?}", st.elapsed());
    // println!("{:?} {:?}", inputs[0], fwd);
    // network.save();
}

fn get_training_data(path: impl ToString) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let read = String::from_utf8(fs::read(path.to_string()).unwrap()).unwrap();
    let mut inputs = vec![];
    let mut outputs = vec![];
    for line in read.split("\n") {
        let mut values = Vec::new();
        for value in line.split(",") {
            match f32::from_str(value) {
                Ok(v) => {
                    values.push(v);
                }
                Err(_) => {}
            }
        }
        if !values.is_empty() {
            let input = values.drain(0..values.len() - 1).as_slice().to_vec();
            let output = if values[0] == 1.0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] };
            if input.len() != 130 {
                println!("{:?} {:?}", input, output);
            }
            inputs.push(input);
            outputs.push(output);
        }
    }
    (inputs, outputs)
}

pub fn server(network: &mut TNetwork) {
    let server: TcpListener = TcpListener::bind("127.0.0.1:16620").unwrap();

    let in_size = network.layers()[0].0 as i32;
    let out_size = network.layers().last().unwrap().0 as i32;
    println!("{} {} {:?}", in_size, out_size, network.layers());
    loop {
        for stream in server.incoming() {
            if let Ok(mut stream) = stream {
                let ip = match stream.peer_addr() {
                    Ok(v) => v.ip(),
                    Err(err) => {
                        println!("{}", err);
                        stream.shutdown(Shutdown::Both);
                        continue
                    }
                };

                stream.set_read_timeout(None);
                stream.set_write_timeout(None);
                // println!("Connected to: {:?}", stream.peer_addr());
                if let Ok(addr) = stream.peer_addr() {
                    // logs.log(format!("Connected to '{:?}'", addr));
                    println!("Connected to '{:?}'", addr);
                }
                stream.write(&in_size.to_be_bytes());
                stream.write(&out_size.to_be_bytes());

                'con : loop {
                    let mut input = vec![];
                    let mut buf = [0u8; 4];
                    for i in 0..in_size {
                        let res = stream.read_exact(&mut buf);
                        if res.is_err() {
                            println!("{}", res.err().unwrap());
                            break 'con;
                        }
                        input.push(f32::from_be_bytes(buf));
                    }
                    let res = stream.read_exact(&mut buf); // read the id buffer
                    if res.is_err() {
                        println!("{}", res.err().unwrap());
                        break 'con;
                    }

                    // println!("fowarding {:?}", input);
                    let st = Instant::now();
                    let result = network.forward(&input).unwrap();
                    println!("OUT {:?} {:?}", result, st.elapsed());
                    stream.write(&mut buf).expect("faield"); // send the id buffer
                    for v in &result {
                        stream.write(&(*v).to_be_bytes());
                    }
                }
            }
        }
    }
}