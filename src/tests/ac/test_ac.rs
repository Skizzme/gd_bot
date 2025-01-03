use std::collections::LinkedList;
use std::fs;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener};
use std::num::ParseFloatError;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::Mutex;
use crate::{BytesReader, get_network_args, network};
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

#[derive(Debug)]
pub struct MoveData {
    deltaX: f32,
    deltaY: f32,
    deltaZ: f32,
    yaw: f32,
    slipperiness: f32,
    onGround: i32,
    tick: i64,
    amplifiers: [i32; 4],
    classification: i32,
}

impl MoveData {
    pub fn from_vec(values: Vec<f32>) -> MoveData {
        MoveData {
            tick:  values[0] as i64,
            deltaX: values[1],
            deltaY: values[2],
            deltaZ: values[3],
            yaw: values[4],
            slipperiness: values[5],
            onGround: values[6] as i32,
            classification: values[7] as i32,
            amplifiers: [values[8] as i32, values[9] as i32, values[10] as i32, values[11] as i32],
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.tick as f32,
            self.deltaX,
            self.deltaY,
            self.deltaZ,
            self.yaw,
            self.slipperiness,
            self.onGround as f32,
            self.classification as f32,
            self.amplifiers[0] as f32,
            self.amplifiers[1] as f32,
            self.amplifiers[2] as f32,
            self.amplifiers[3] as f32,
        ]
    }
}

fn get_training_data(path: impl ToString) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut all_data = vec![];
    match fs::read_dir(path.to_string()) {
        Ok(read) => {
            for file in read {
                if let Ok(file) = file {
                    if file.metadata().unwrap().is_file() {
                        let mut data = fs::read(file.path()).unwrap();
                        let start = if all_data.is_empty() {
                            0
                        } else {
                            4 // skip the first 4 bytes as it is an int value specifying the version as long as it is not the first file
                        };
                        all_data.append(&mut data[start..data.len()].to_vec());
                    }
                }
            }
        }
        Err(_) => {}
    }

    let mut read = BytesReader::new(all_data);
    let version = read.next_i32();
    println!("VERSION:{version}");

    let mut values = Vec::new();
    while !read.ended() {
        values.push(read.next_f32());
    }

    let mut move_datas = Vec::new();

    while !values.is_empty() {
        let data = values.drain(0..12).as_slice().to_vec();
        if data.len() == 0 {
            break;
        }

        let move_data = MoveData::from_vec(data);
        move_datas.push(move_data);
    }

    let mut inputs = vec![];
    let mut outputs = vec![];

    let mut datas = LinkedList::new();
    let (mut class0, mut class1) = (0, 0);
    for i in 0..move_datas.len() {
        let data0 = &move_datas[i];

        if data0.classification == 0 {
            class0 += 1;
        } else {
            class1 += 1;
        }

        if i < 3 {
            continue;
        }

        let data1 = &move_datas[i-1];
        let data2 = &move_datas[i-2];

        if data0.classification != data1.classification {
            datas.clear();
        }

        let mut data = vec![
            data0.deltaX, data0.deltaY, data0.deltaZ, // speeds
            data1.deltaX - data0.deltaX, data1.deltaY - data0.deltaY, data1.deltaZ - data0.deltaZ, // accelerations
            (data2.deltaX - data1.deltaX) - (data1.deltaX - data0.deltaX), // jolt X
            (data2.deltaY - data1.deltaY) - (data1.deltaY - data0.deltaY),  // jolt Y
            (data2.deltaZ - data1.deltaZ) - (data1.deltaZ - data0.deltaZ), // jolt Z
            data0.yaw, data1.yaw - data0.yaw, // yaw, yaw dif
            data0.slipperiness, data0.onGround as f32
        ];

        for v in data0.amplifiers { data.push(v as f32) };
        for v in data1.amplifiers { data.push(v as f32) };
        for v in data2.amplifiers { data.push(v as f32) };

        datas.push_back(data);

        if datas.len() >= 10 {
            let mut combined = vec![];
            for data in &datas {
                combined.append(&mut data.clone());
            }

            inputs.push(combined);
            outputs.push(if data0.classification == 1 { vec![1.0, 0.0] } else { vec![0.0, 1.0] });

            datas.pop_front();
        }

        // println!("{:?} {:?}", inputs, outputs);
        // TODO hasn't been tested yet
    }

    println!("0: {}, 1: {}", class0, class1);

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

                if let Ok(addr) = stream.peer_addr() {
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