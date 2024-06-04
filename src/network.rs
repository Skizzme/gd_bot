#![allow(unused)]
use std::time::Instant;
use num_format::Locale::wo;

use ocl::{Buffer, OclPrm, ProQue, SpatialDims};
use rand::random;
use crate::{cl_utils, gpu_math};
use crate::cl_utils::{calculate_worksize, execute_kernel};

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
    // Stores a buffer on the GPU for each layer's sensitivities/gradients
    // Only used for back propagation
    gpu_layer_sensitivities: Vec<Buffer<f64>>,
    gpu_out_buf: Buffer<f64>,
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
        let mut layer_sensitivities = vec![];

        let mut layer_inputs = *layers.get(0).unwrap();
        // Add input layer, but since it isn't a real layer it doesn't have weights or biases
        layer_buffers.push(cl_utils::new_buffer(&pro_que, layer_inputs));
        network_layers.push((layers[0], 0, 0));

        let mut weight_offset = 0usize;
        let mut bias_offset = 0usize;
        // First value would be the plain inputs, so start creating layers with sizes after
        for i in 1..layers.len() {
            let layer_size = *layers.get(i).unwrap();

            let layer: Buffer<f64> = cl_utils::new_buffer(&pro_que, layer_size);
            layer_buffers.push(layer);

            network_layers.push((layer_size, weight_offset, bias_offset));
            layer_sensitivities.push(cl_utils::new_buffer(&pro_que, layer_inputs));

            weight_offset += layer_size*layer_inputs;
            bias_offset += layer_size;

            // Keep track of the input size for the next layer
            layer_inputs = layer_size;
        }

        st = Instant::now();
        // println!("\nStoring network on GPU...");
        let weight_buf: Buffer<f64> = cl_utils::new_buffer(&pro_que, weight_offset);
        // Init network's weights randomly using GPU
        cl_utils::randomize_buffer(&weight_buf, 256, &pro_que);

        let biases_buf: Buffer<f64> = cl_utils::new_buffer(&pro_que, bias_offset);
        // Init network's biases randomly using GPU
        cl_utils::randomize_buffer(&biases_buf, 256, &pro_que);

        println!("Stored network on GPU {:?}\n", st.elapsed());
        println!("{}", weight_buf.len());

        let gpu_out_buf = cl_utils::new_buffer_f(&pro_que, layer_inputs, 0.0);
        Ok(Network {
            weights: weight_offset,
            biases: bias_offset,
            layers: network_layers,
            gpu_proque: pro_que,
            gpu_weights: weight_buf,
            gpu_biases: biases_buf,
            gpu_layer_bufs: layer_buffers,
            gpu_layer_sensitivities: layer_sensitivities,
            gpu_out_buf,
        })
    }

    pub fn forward(&self, inputs: Vec<f64>) -> Result<Vec<f64>, String> { // Will return proper error later
        let mut layer_in_size = self.layers[0].0;

        if inputs.len() != layer_in_size {
            return Err(format!("Input length is incorrect! Network input length is: {}", layer_in_size).to_string())
        }

        let mut st = Instant::now();
        let buf = &self.gpu_layer_bufs[0];

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
                execute_kernel(&self.gpu_proque, &forward_kernel, (layer_size, layer_in_size));
                // println!("Enqueued layer kernel {:?} {:?}", wg_size, st.elapsed());
                st = Instant::now();

            }

            layer_in_size = self.layers[i].0;
            buf = layer_buf;
        }

        unsafe {
            let activation_kernel = self.gpu_proque
                .kernel_builder("activation")
                .arg(buf)
                .arg(&self.gpu_out_buf)
                .build().unwrap();

            // println!("Created final activation kernel {:?}", st.elapsed());
            st = Instant::now();

            cl_utils::execute_kernel(&self.gpu_proque, &activation_kernel, buf.len());

            // println!("Enqueued activation kernel ({}) {:?}", wg_size_1d, st.elapsed());
        }

        st = Instant::now();
        let mut final_out = vec![0f64; self.layers.get(self.layers.len() - 1).unwrap().0];
        self.gpu_out_buf.read(&mut final_out).enq().expect("Failed to read output of network");
        // println!("\nReading network output {:?}", st.elapsed());

        Ok(final_out)
    }

    pub fn read_buf(buffer: &Buffer<f64>) -> Vec<f64> {
        let mut target = vec![0.0; buffer.len()];
        buffer.read(&mut target).enq();
        target
    }

    pub fn error(&self, values: Vec<f64>, target: Vec<f64>) -> f64 {
        let mut error = 0.0;
        for i in 0..values.len() {
            error += (values[i]-target[i]).powf(2.0);
        }
        error
    }

    pub fn backward(&self, mut target: Vec<f64>, learn_rate: f64) {
        let max_wg = self.gpu_proque.max_wg_size().expect("Failed to get max workgroup size");
        let target_buf = cl_utils::new_buffer(&self.gpu_proque, target.len());
        target_buf.write(&target).enq().unwrap();

        let first_senses = cl_utils::new_buffer(&self.gpu_proque, target.len());
        gpu_math::activate_and_error_derivative(&self.gpu_proque, &self.gpu_layer_bufs[self.layers.len()-1], &target_buf, &first_senses);;
        let mut prev_layer_sensitivities = &first_senses;

        unsafe {
            for i in (1..self.layers.len()).rev() {
                let (layer_size, weight_offset, bias_offset) = self.layers[i];
                let (prev_size, prev_weight_offset, prev_bias_offset) = self.layers[i-1];

                let layer_sensitivities = &self.gpu_layer_sensitivities[i-1];
                layer_sensitivities.cmd().fill(0.0, None).enq();

                let back_kernel = self.gpu_proque
                    .kernel_builder("backward")
                    .arg(i as i32)
                    .arg(prev_size as u64)
                    .arg(weight_offset as u64)
                    .arg(bias_offset as u64)
                    .arg(learn_rate)
                    .arg(&self.gpu_layer_bufs[i-1])
                    .arg(&self.gpu_layer_bufs[i])
                    .arg(prev_layer_sensitivities)
                    .arg(&self.gpu_weights)
                    .arg(&self.gpu_biases)
                    .arg(layer_sensitivities)
                    .build()
                    .expect("failed to build backward kernel");

                execute_kernel(&self.gpu_proque, &back_kernel, (layer_size, prev_size));

                prev_layer_sensitivities = layer_sensitivities;
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
