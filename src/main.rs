mod network;

extern crate ocl;

use std::fs::read_to_string;
use std::str::FromStr;
use std::string::ToString;
use std::thread::sleep;
use std::time::{Duration, Instant};
use ocl::{Buffer, Device, Platform, ProQue, SpatialDims};
use ocl::ffi::libc::rand;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();

    let layer_nodes: i32 = i32::from_str(args[1]).unwrap();
    let in_size: i32 = i32::from_str(args[2]).unwrap();;
    let local_x: i32 = i32::from_str(args[3]).unwrap();;
    let local_y: i32 = i32::from_str(args[4]).unwrap();;

    let src = read_to_string("src\\test_comp").unwrap();
    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    println!("{:?}", device.max_wg_size());

    let pro_que = ProQue::builder().src(src).build();
    if pro_que.is_err() {
        std::fs::write("compile_error.txt", pro_que.err().unwrap().to_string()).expect("failed to write error");
        return
    }
    let mut t = Instant::now();
    let pro_que = pro_que.unwrap();
    let queue = pro_que.queue();
    // Flattens layers
    let mut layer_1_flat = vec![0f64; (layer_nodes * in_size) as usize];
    for i in 0..layer_1_flat.len() {
        layer_1_flat[i] = rand::random() as f64;
    }

    let mut inputs = vec![0f64; in_size as usize];
    for i in 0..inputs.len() {
        inputs[i] = rand::random() as f64;
    }
    println!("buf_create: {:?}", t.elapsed());
    let st = Instant::now();
    t = Instant::now();

    let weights_1: Buffer<f64> = Buffer::builder()
        .queue(queue.clone())
        .len(layer_1_flat.len())
        .build().expect("Failed to make input buffer");
    let inputs_1: Buffer<f64> = Buffer::builder()
        .queue(queue.clone())
        .len(inputs.len())
        .build().expect("Failed to make input buffer");

    weights_1.write(layer_1_flat.as_slice()).enq().expect("Failed to enq buffer 1");
    println!("layer_weight_write: {:?} {:?}", t.elapsed(), st.elapsed());
    t = Instant::now();

    inputs_1.write(inputs.as_slice()).enq().expect("Failed to enq buffer 2");
    println!("layer_inputs_write: {:?} {:?}", t.elapsed(), st.elapsed());
    t = Instant::now();

    let out_buffer: Buffer<f64> = Buffer::builder()
        .queue(queue.clone())
        .len(layer_nodes)
        .build().expect("Failed to make output buffer");

    println!("output_buf_create: {:?} {:?}", t.elapsed(), st.elapsed());
    t = Instant::now();
    let mut kernel = pro_que
        .kernel_builder("layer")
        .arg(layer_nodes)
        .arg(in_size)
        .arg(&weights_1)
        .arg(&inputs_1)
        .arg(&out_buffer)
        // .global_work_size((layer_nodes, in_size))
        // .local_work_size((local_x, local_y))
        .build()
        .unwrap();

    println!("kernel_create: {:?} {:?}", t.elapsed(), st.elapsed());
    t = Instant::now();

    unsafe {
        kernel
            .cmd()
            .global_work_size((layer_nodes, in_size))
            .local_work_size((local_x, local_y))
            .enq()
            .unwrap();
    }

    println!("gpu_exec_time: {:?} {:?}", t.elapsed(), st.elapsed());

    let mut out = vec![0f64; layer_nodes as usize];
    out_buffer.read(&mut out).enq().unwrap();
    println!("gpu_read_time: {:?} {:?} {} {} {}", t.elapsed(), st.elapsed(), out.len(), out[1], out[out.len()-1]);
    // println!("gpu_out: {:?}", out);
    t = Instant::now();

    let mut cpu_out = vec![0f64; layer_nodes as usize];
    for i in 0..layer_nodes {
        for j in 0..in_size {
            let ind: usize = (in_size * i + j) as usize;
            cpu_out[i as usize] += layer_1_flat[ind] * inputs[j as usize];
        }
    }
    println!("cpu_calc_time: {:?} {:?} {} {} {}", t.elapsed(), st.elapsed(), cpu_out.len(), cpu_out[1], cpu_out[cpu_out.len()-1]);
    // println!("cpu_out: {:?}", cpu_out);

    if args.len() > 5 && args[5] == "print" {
        for i in 0..out.len() {
            println!("CPU: {}, GPU: {} ERR: {}", cpu_out[i], out[i], (cpu_out[i] - out[i]).abs());
        }
    }
}