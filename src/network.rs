#![allow(unused)]
use std::time::Instant;
use num_format::Locale::wo;

use ocl::{Buffer, OclPrm, ProQue, SpatialDims};
use rand::random;
use crate::gpu_math;

#[derive(Debug)]
pub struct Network {
    // Layer weights are flat, not a matrix
    // weights: Vec<f64>,
    // biases: Vec<f64>,
    weights: usize,
    biases: usize,
    // Each layer size. First item is the inputs to the network
    // nodes, weights_offset, biases_offset
    layers: Vec<(usize, usize, usize)>,

    gpu_proque: ProQue,
    // A single buffer containing all the weights/biases, so fewer calls to gpu
    // Values will be gotten based off of the layer sizes and offsets etc
    gpu_weights: Buffer<f64>,
    gpu_biases: Buffer<f64>,
    // Individual output buffers for each layer, so each layer can have different sizes
    // Inputs to next layer will be outputs of last layer
    gpu_layer_bufs: Vec<Buffer<f64>>,
}

pub fn randomize_buffer(buffer: &Buffer<f64>, max_work_size: u32, pro_que: &ProQue) {
    let rnd_kernel = pro_que
        .kernel_builder("random_buf")
        .arg(buffer)
        .arg(random::<u64>())
        .build()
        .expect("Failed to build rnd_kernel");

    unsafe {
        rnd_kernel
            .cmd()
            .global_work_size(buffer.len())
            .local_work_size(calculate_worksize(max_work_size as usize, buffer.len()))
            .enq()
            .expect("Failed to enq rnd_kernel")
    }
}

pub fn calculate_worksize(max: usize, size: usize) -> usize {
    let mut calc = 1;
    for i in (1..max+1).rev() {
        if (size as f32 / i as f32) % 1.0 == 0.0 {
            calc = i;
            break;
        }
    }
    calc
}

impl Network {
    pub fn new(layers: Vec<usize>, init_weights: Option<Vec<f64>>, init_biases: Option<Vec<f64>>) -> Result<Self, String> {
        let mut st = Instant::now();
        // println!("Creating network...");
        let src = include_str!("kernels.c");
        let pro_que = ProQue::builder().src(src).build();
        if pro_que.is_err() {
            std::fs::write("compile_error.txt", pro_que.err().unwrap().to_string()).expect("failed to write error");
            return Err("Compile error. Check 'compile_error.txt'".to_string());
        }
        let pro_que = pro_que.unwrap();
        let queue = pro_que.queue();
        // println!("Compiled kernels.c {:?}", st.elapsed());

        // println!("\nInitializing layers...");

        let mut network_layers = vec![];
        let mut layer_buffers = vec![];

        let mut layer_inputs = *layers.get(0).unwrap();
        // Add input layer, but since it isn't a real layer it doesn't have weights or biases
        layer_buffers.push(Buffer::builder()
                               .queue(queue.clone())
                               .len(layer_inputs)
                               .build().expect("Failed to make input buffer"));
        network_layers.push((layers[0], 0, 0));

        let mut weight_offset = 0usize;
        let mut bias_offset = 0usize;
        // First value would be the plain inputs, so start creating layers with sizes after
        for i in 1..layers.len() {
            let layer_size = *layers.get(i).unwrap();

            let layer: Buffer<f64> = Buffer::builder()
                .queue(queue.clone())
                .len(layer_size)
                .build().expect("Failed to make input buffer");
            layer_buffers.push(layer);

            network_layers.push((layer_size, weight_offset, bias_offset));

            weight_offset += layer_size*layer_inputs;
            bias_offset += layer_size;

            // Keep track of the input size for the next layer
            layer_inputs = layer_size;
        }

        st = Instant::now();
        // println!("\nStoring network on GPU...");
        let weight_buf: Buffer<f64> = Buffer::builder()
            .queue(queue.clone())
            .len(weight_offset)
            .build().expect("Failed to make input buffer");
        // Init network's weights randomly using GPU
        randomize_buffer(&weight_buf, 256, &pro_que);

        let biases_buf: Buffer<f64> = Buffer::builder()
            .queue(queue.clone())
            .len(bias_offset)
            .build().expect("Failed to make input buffer");
        // Init network's biases randomly using GPU
        randomize_buffer(&biases_buf, 256, &pro_que);

        // println!("Stored network on GPU {:?}\n", st.elapsed());

        Ok(Network {
            weights: weight_offset,
            biases: bias_offset,
            layers: network_layers,
            gpu_proque: pro_que,
            gpu_weights: weight_buf,
            gpu_biases: biases_buf,
            gpu_layer_bufs: layer_buffers,
        })
    }

    pub fn forward(&self, inputs: Vec<f64>) -> Result<Vec<f64>, String> { // Will return proper error later
        let mut layer_in_size = self.layers[0].0;
        let max_wg = self.gpu_proque.max_wg_size().expect("Failed to get max workgroup size");

        if inputs.len() != layer_in_size {
            return Err(format!("Input length is incorrect! Network input length is: {}", layer_in_size).to_string())
        }

        let mut st = Instant::now();
        let mut buf = &self.gpu_layer_bufs[0];

        buf.write(&inputs).enq().expect("Failed to write network inputs");

        // println!("\nInitialized network input {:?}", st.elapsed());
        st = Instant::now();

        for i in 1..self.layers.len() {
            let (layer_size, weights_offset, biases_offset) = self.layers[i];
            // println!("Forwarding layer: {:?}. size: {:?}, weight_offset: {:?}, bias_offset: {:?} {:?}", i, layer_size, weights_offset, biases_offset, match i { 1 => 0, _ => 1 });

            let layer_buf = &self.gpu_layer_bufs[i];

            layer_buf.cmd().fill(0.0, None).enq();

            // Maybe create these kernels once in network creation, and only enq them here?
            let forward_kernel = self.gpu_proque
                .kernel_builder("forward")
                .arg(match i { 1 => 0, _ => 1 })
                .arg(layer_in_size as u64)
                .arg(layer_size as u64)
                .arg(weights_offset as u64)
                .arg(biases_offset as u64)
                .arg(&self.gpu_weights)
                .arg(&self.gpu_biases)
                .arg(buf)
                .arg(layer_buf)
                .build()
                .unwrap();

            // println!("Created layer kernel {:?}", st.elapsed());
            st = Instant::now();

            unsafe {
                let wg_size = (calculate_worksize((max_wg as f32).sqrt() as usize, layer_size), calculate_worksize((max_wg as f32).sqrt() as usize, layer_in_size));

                forward_kernel
                    .cmd()
                    .global_work_size((layer_size, layer_in_size))
                    .local_work_size(wg_size)
                    .enq()
                    .expect("Failed to enqueue layer kernel");

                println!("Enqueued layer kernel {:?} {:?}", wg_size, st.elapsed());
                st = Instant::now();

            }

            layer_in_size = self.layers[i].0;
            buf = layer_buf;
        }

        let out_buf: Buffer<f64> = Buffer::builder()
            .queue(self.gpu_proque.queue().clone())
            .len(buf.len())
            .build()
            .expect("Failed to build final out buffer");
        unsafe {
            let wg_size_1d = calculate_worksize(max_wg, buf.len());
            let activation_kernel = self.gpu_proque
                .kernel_builder("activation")
                .arg(buf)
                .arg(&out_buf)
                .build().unwrap();

            // println!("Created final activation kernel {:?}", st.elapsed());
            st = Instant::now();

            activation_kernel
                .cmd()
                .global_work_size(buf.len())
                .local_work_size(wg_size_1d)
                .enq()
                .expect("Failed to enqueue activation kernel");

            // println!("Enqueued activation kernel ({}) {:?}", wg_size_1d, st.elapsed());
        }

        st = Instant::now();
        let mut final_out = vec![0f64; self.layers.get(self.layers.len() - 1).unwrap().0];
        out_buf.read(&mut final_out).enq().expect("Failed to read output of network");
        // println!("\nReading network output {:?}", st.elapsed());

        Ok(final_out)
    }

    pub fn read_buf(buffer: &Buffer<f64>) -> Vec<f64> {
        let mut target = vec![0.0; buffer.len()];
        buffer.read(&mut target).enq();
        target
    }

    pub fn backward(&self, mut target: Vec<f64>, learn_rate: f64) {
        let max_wg = self.gpu_proque.max_wg_size().expect("Failed to get max workgroup size");
        let mut target_buf = Buffer::builder()
            .queue(self.gpu_proque.queue().clone())
            .len(target.len())
            .build()
            .expect("Failed to create target buf");
        target_buf.write(&target).enq().unwrap();
        let mut prev_layer_sensitivities = Buffer::builder()
            .queue(self.gpu_proque.queue().clone())
            .len(target_buf.len())
            .build()
            .expect("Failed to create next target buffer");
        prev_layer_sensitivities.write(&mut target).enq().expect("Failed to write to buf");


        unsafe {
            for i in (1..self.layers.len()).rev() {
                let (layer_size, weight_offset, bias_offset) = self.layers[i];
                let (prev_size, prev_weight_offset, prev_bias_offset) = self.layers[i-1];
                if i != self.layers.len()-1 {
                    target_buf = Buffer::builder()
                        .queue(self.gpu_proque.queue().clone())
                        .len(layer_size)
                        .fill_val(0.0)
                        .build()
                        .expect("Failed to build ttarget buf");
                    // self.gpu_weights.cmd().offset(weight_offset).enq().expect("failed to enq offset cmd");
                    gpu_math::mult(&self.gpu_proque, weight_offset, &self.gpu_weights, &prev_layer_sensitivities, &target_buf);
                    // self.gpu_weights.cmd().offset(0).enq().expect("failed to enq offset cmd");
                }

                prev_layer_sensitivities = Buffer::builder()
                    .queue(self.gpu_proque.queue().clone())
                    .len(layer_size)
                    .fill_val(0.0)
                    .build()
                    .expect("Failed to create next target buffer");
                let back_kernel = self.gpu_proque
                    .kernel_builder("backward")
                    .arg(i as i32)
                    .arg(prev_size as u64)
                    .arg(weight_offset as u64)
                    .arg(bias_offset as u64)
                    .arg(learn_rate)
                    .arg(&self.gpu_layer_bufs[i-1])
                    .arg(&self.gpu_layer_bufs[i])
                    .arg(&target_buf)
                    .arg(&self.gpu_weights)
                    .arg(&self.gpu_biases)
                    .arg(&prev_layer_sensitivities)
                    .build()
                    .expect("failed to build backward kernel");

                let wg_size = (calculate_worksize((max_wg as f32).sqrt() as usize, layer_size), calculate_worksize((max_wg as f32).sqrt() as usize, prev_size));
                back_kernel
                    .cmd()
                    .global_work_size((layer_size, prev_size))
                    .local_work_size(wg_size)
                    .enq()
                    .expect("Failed to enqueue layer kernel");

                // gpu_math::mult(self.gpu_proque, )
                // println!("bgradients {:?}", gpu_math::load_buffer(&prev_layer_gradients));
                gpu_math::mult_single(&self.gpu_proque, 0, &prev_layer_sensitivities, 1.0/prev_size as f64, &prev_layer_sensitivities);
                // println!("agradients {:?}", gpu_math::load_buffer(&prev_layer_gradients));

                // println!("Enqueued backward layer kernel {:?}", wg_size);
            }
        }
    }

    pub fn layers(&self) -> &Vec<(usize, usize, usize)> {
        &self.layers
    }
    pub fn weights(&self) -> usize {
        self.weights
    }
    pub fn biases(&self) -> usize {
        self.biases
    }
}
