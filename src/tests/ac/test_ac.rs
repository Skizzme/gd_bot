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

type TNetwork = CPUNetwork;
pub fn run() {
    let (epochs, learn_rate, layers) = get_network_args();
    let read = String::from_utf8(fs::read("accels.csv").unwrap()).unwrap();
    let mut inputs = vec![];
    let mut outputs = vec![];
    for line in read.split("\n") {
        let mut values = Vec::new();
        for value in line.split(",") {
            match f32::from_str(value) {
                Ok(v) => {
                    values.push(v);
                }
                Err(_) => {
                }
            }
        }
        if !values.is_empty() {
            let input = values.drain(0..20).as_slice().to_vec();
            let output = if values[0] == 1.0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] };
            inputs.push(input);
            outputs.push(output);
        }
    }
    let (layers, biases, weights) = network::read_network("1729136452239.net");
    let mut network = TNetwork::new(layers, Some(biases), Some(weights)).unwrap();
    // let mut network = TNetwork::new(layers, None, None).unwrap();
    // let st = Instant::now();
    // let fwd = network.forward(&inputs[0]);
    // println!("fwd took {:?}", st.elapsed());
    // println!("{:?} {:?}", inputs[0], fwd);
    // network.save();
    // network.train(epochs, learn_rate, &mut inputs, &mut outputs, 0.01, |_| {});
    server(&mut network);
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
                    for i in 0..in_size {
                        let mut buf = [0u8; 4];
                        let res = stream.read_exact(&mut buf);
                        if res.is_err() {
                            println!("{}", res.err().unwrap());
                            break 'con;
                        }
                        input.push(f32::from_be_bytes(buf));
                    }

                    println!("fowarding {:?}", input);
                    let result = network.forward(&input).unwrap();
                    for v in &result {
                        stream.write(&(*v).to_be_bytes());
                    }
                    println!("sent {:?}", result);
                }
            }
        }
    }
}