extern crate ocl;

use std::fs::read_to_string;
use std::string::ToString;
use std::thread::sleep;
use std::time::{Duration, Instant};
use ocl::{Buffer, Device, Platform, ProQue, SpatialDims};
use ocl::ffi::libc::rand;

const LAYER_NODES: i32 = 8192;
const IN_NODES: i32 = 2048;

const LOCAL_X: i32 = 16;
const LOCAL_Y: i32 = 16;

fn rand_float() -> f32 {
    rand::random()
}

fn main() {
    let src = read_to_string("src\\test_comp").unwrap();
    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    println!("{:?}", device.max_wg_size());

    let pro_que = ProQue::builder().src(src).dims((LAYER_NODES, IN_NODES)).build();
    if pro_que.is_err() {
        std::fs::write("compile_error.txt", pro_que.err().unwrap().to_string()).expect("failed to write error");
        return
    }
    let mut t = Instant::now();
    let pro_que = pro_que.unwrap();
    let queue = pro_que.queue();
    // Inner array is previous layer node count, outer is current layer node count
    let mut layer_1_weights = vec![vec![0f32; IN_NODES as usize]; LAYER_NODES as usize];
    // Layer weights initialization
    for i in 0..layer_1_weights.len() {
        for j in 0..layer_1_weights[i].len() {
            layer_1_weights[i][j] = rand_float();
        }
    }
    // Flattens layers
    let mut layer_1_flat = vec![0f32; (LAYER_NODES * IN_NODES) as usize];
    for i in 0..layer_1_weights.len() {
        for j in 0..layer_1_weights[i].len() {
            layer_1_flat[i*layer_1_weights[i].len()+j] = layer_1_weights[i][j];
        }
    }

    let mut inputs = vec![0f32; IN_NODES as usize];
    for i in 0..inputs.len() {
        inputs[i] = rand_float();
    }
    println!("buf_create: {:?}", t.elapsed());
    let st = Instant::now();
    t = Instant::now();

    let weights_1: Buffer<f32> = Buffer::builder()
        .queue(queue.clone())
        .len(layer_1_flat.len())
        .build().expect("Failed to make input buffer");
    let inputs_1: Buffer<f32> = Buffer::builder()
        .queue(queue.clone())
        .len(inputs.len())
        .build().expect("Failed to make input buffer");

    weights_1.write(layer_1_flat.as_slice()).enq().expect("Failed to enq buffer 1");
    println!("layer_weight_write: {:?} {:?}", t.elapsed(), st.elapsed());
    t = Instant::now();

    inputs_1.write(inputs.as_slice()).enq().expect("Failed to enq buffer 2");
    println!("layer_inputs_write: {:?} {:?}", t.elapsed(), st.elapsed());
    t = Instant::now();

    let out_buffer: Buffer<f32> = Buffer::builder()
        .queue(queue.clone())
        .len(layer_1_weights.len())
        .build().expect("Failed to make output buffer");

    println!("output_buf_create: {:?} {:?}", t.elapsed(), st.elapsed());
    t = Instant::now();
    let mut kernel = pro_que
        .kernel_builder("matcher")
        .arg(LAYER_NODES)
        .arg(IN_NODES)
        .arg(&weights_1)
        .arg(&inputs_1)
        .arg(&out_buffer)
        .local_work_size((LOCAL_X, LOCAL_Y))
        .build()
        .unwrap();

    println!("kernel_enq: {:?} {:?}", t.elapsed(), st.elapsed());
    t = Instant::now();
    println!("{:?}", kernel.default_local_work_size());
    unsafe {
        kernel.enq().unwrap();
    }

    println!("gpu_exec_time: {:?} {:?}", t.elapsed(), st.elapsed());

    let mut out = vec![0f32; layer_1_weights.len()];
    out_buffer.read(&mut out).enq().unwrap();
    println!("gpu_read_time: {:?} {:?} {} {} {}", t.elapsed(), st.elapsed(), out.len(), out[1], out[out.len()-1]);
    t = Instant::now();

    let mut cpu_out = vec![0f32; layer_1_weights.len()];
    for i in 0..layer_1_weights.len() {
        for j in 0..layer_1_weights[i].len() {
            cpu_out[i] += layer_1_weights[i][j] * inputs[j];
        }
    }
    println!("cpu_calc_time: {:?} {:?} {} {} {}", t.elapsed(), st.elapsed(), cpu_out.len(), cpu_out[1], cpu_out[cpu_out.len()-1]);
    // println!("cpu_out: {:?}", cpu_out);
}