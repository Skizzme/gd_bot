
use std::{fs, thread};
use std::time::{Duration, Instant, UNIX_EPOCH};
use num_format::Locale::{te, wo};

use ocl::{Buffer, Kernel, OclPrm, ProQue, SpatialDims};
use ocl::enums::WriteSrc;
use rand::{random, Rng, thread_rng};
use crate::{cl_utils, gpu_math};
use crate::cl_utils::{calc_ws, execute_kernel};
use num_format::{Locale, ToFormattedString};
use rand::seq::SliceRandom;
use crate::network::{Network, shuffle};
use crate::network::cpu::activate;

#[derive(Debug)]
pub struct GPUNetwork {
    // Layer weights are flat, not a matrix
    // weights: Vec<f32>,
    // biases: Vec<f32>,
    weights: usize,
    biases: usize,
    // Each layer size. First item is the inputs to the network
    // nodes, weights_offset, biases_offset
    layers: Vec<(usize, usize, usize)>,

    gpu_proque: ProQue,
    // A single buffer containing all the weights/biases, so fewer calls to gpu
    // Values will be gotten based off of the layer sizes and offsets etc
    gpu_weights: Buffer<f32>,
    gpu_biases: Buffer<f32>,
    // Individual output buffers for each layer, so each layer can have different sizes
    // Inputs to next layer will be outputs of last layer
    gpu_layer_bufs: Vec<Buffer<f32>>,
    gpu_activated_bufs: Vec<Buffer<f32>>,
    // Stores a buffer on the GPU for each layer's sensitivities/gradients
    // Only used for back propagation
    gpu_layer_sensitivities: Vec<Buffer<f32>>,
    gpu_out_buf: Buffer<f32>,
    gpu_target_buf: Buffer<f32>,
    gpu_senses_buf: Buffer<f32>,
    gpu_fw_kernels: Vec<Kernel>,

    weight_mods: Buffer<f32>,
    bias_mods: Buffer<f32>,
}

impl GPUNetwork {
    pub fn new(layers: Vec<usize>, init_biases: Option<Vec<f32>>, init_weights: Option<Vec<f32>>) -> Result<Self, String> {
        let mut st = Instant::now();
        // println!("Creating network...");
        let src = include_str!("../kernels.c");
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
        let mut activated_buffers = vec![];
        let mut layer_sensitivities = vec![];

        let mut layer_inputs = *layers.get(0).unwrap();
        // Add input layer, but since it isn't a real layer it doesn't have weights or biases
        layer_buffers.push(cl_utils::new_buffer(&pro_que, layer_inputs));
        activated_buffers.push(cl_utils::new_buffer(&pro_que, layer_inputs));
        network_layers.push((layers[0], 0, 0));

        let mut weight_offset = 0;
        let mut bias_offset = layers[0];
        // First value would be the plain inputs, so start creating layers with sizes after
        for i in 1..layers.len() {
            let layer_size = *layers.get(i).unwrap();

            layer_buffers.push(cl_utils::new_buffer(&pro_que, layer_size));
            activated_buffers.push(cl_utils::new_buffer(&pro_que, layer_size));

            println!("{} {} {}", i, bias_offset, weight_offset);
            network_layers.push((layer_size, weight_offset, bias_offset));
            layer_sensitivities.push(cl_utils::new_buffer(&pro_que, layer_inputs));

            bias_offset += layer_size;
            weight_offset += layer_size * layer_inputs;

            // Keep track of the input size for the next layer
            layer_inputs = layer_size;
        }

        st = Instant::now();
        // println!("\nStoring network on GPU...");
        let weight_buf: Buffer<f32> = cl_utils::new_buffer(&pro_que, weight_offset);
        match init_weights {
            None => {
                // Init network's weights randomly using GPU
                cl_utils::randomize_buffer(&weight_buf, 256, 5.0, &pro_que);
            }
            Some(weights) => {
                cl_utils::buf_write(&weight_buf, &weights);
            }
        }

        let biases_buf: Buffer<f32> = cl_utils::new_buffer(&pro_que, bias_offset);
        match init_biases {
            None => {
                // Init network's biases randomly using GPU
                cl_utils::randomize_buffer(&biases_buf, 256, 200.0, &pro_que);
            }
            Some(biases) => {
                cl_utils::buf_write(&biases_buf, &biases);
            }
        }

        // println!("{:?}", cl_utils::buf_read(&weight_buf));

        println!("Stored network on GPU {:?}\n", st.elapsed());
        println!("{}", weight_buf.len());

        let gpu_out_buf = cl_utils::new_buffer_f(&pro_que, layer_inputs, 0.0);
        let weight_mods = cl_utils::new_buffer(&pro_que, weight_offset);
        let bias_mods = cl_utils::new_buffer(&pro_que, bias_offset);
        let gpu_target_buf = cl_utils::new_buffer(&pro_que, *layers.last().unwrap());
        let gpu_senses_buf = cl_utils::new_buffer(&pro_que, *layers.last().unwrap());
        Ok(GPUNetwork {
            weights: weight_offset,
            biases: bias_offset,
            layers: network_layers,
            gpu_proque: pro_que,
            gpu_weights: weight_buf,
            gpu_biases: biases_buf,
            gpu_layer_bufs: layer_buffers,
            gpu_activated_bufs: activated_buffers,
            gpu_layer_sensitivities: layer_sensitivities,
            gpu_out_buf,
            gpu_target_buf,
            gpu_senses_buf,
            gpu_fw_kernels: vec![],
            weight_mods,
            bias_mods,
        })
    }
}

impl Network for GPUNetwork {
    fn forward(&mut self, inputs: &Vec<f32>) -> Result<Vec<f32>, String> { // Will return proper error later
        let mut layer_in_size = self.layers[0].0;

        if inputs.len() != layer_in_size {
            return Err(format!("Input length is incorrect! Network input length is: {}. Inputs length is: {}", layer_in_size, inputs.len()).to_string())
        }

        let mut st = Instant::now();
        let mut layer_in = &self.gpu_layer_bufs[0];

        layer_in.write(inputs).enq().expect("Failed to write network inputs");

        for i in 1..self.layers.len() {
            let (layer_size, weights_offset, biases_offset) = self.layers[i];

            let layer_out = &self.gpu_layer_bufs[i];
            layer_out.cmd().fill(0.0, None).enq();

            st = Instant::now();
            // Maybe create these kernels once in network creation, and only enq them here?
            if self.gpu_fw_kernels.len() <= i {
                let fw_kernel: Kernel = self.gpu_proque
                    .kernel_builder("forward")
                    .arg(match i { 1 => 0, _ => 1 }) // match i { 1 => 0, _ => 1 }
                    .arg(layer_in_size as u64)
                    .arg(layer_size as u64)
                    .arg(weights_offset as u64)
                    .arg(biases_offset as u64)
                    .arg(&self.gpu_weights)
                    .arg(&self.gpu_biases)
                    .arg(layer_in)
                    .arg(&self.gpu_activated_bufs[i-1])
                    .arg(layer_out)
                    .arg(&self.gpu_activated_bufs[i])
                    .build()
                    .unwrap();

                self.gpu_fw_kernels.push(fw_kernel);
            }
            let forward_kernel =

            // println!("Created layer kernel {:?}", st.elapsed());
            st = Instant::now();

            unsafe {
                execute_kernel(&self.gpu_proque, &self.gpu_fw_kernels[i-1], (layer_size, layer_in_size));
                // println!("Enqueued layer kernel {:?} {:?}", wg_size, st.elapsed());
                st = Instant::now();

            }

            layer_in_size = self.layers[i].0;
            // println!("{:?} {:?}", i, cl_utils::buf_read(&layer_buf));
            layer_in = layer_out;
        }
        // self.gpu_proque.finish();
        // println!("{:?}", st.elapsed());

        let mut last_out = cl_utils::buf_read(layer_in);
        let mut activated = Vec::new();
        for v in last_out {
            activated.push(activate(v));
        }
        // unsafe {
        //     let activation_kernel = self.gpu_proque
        //         .kernel_builder("activation")
        //         .arg(layer_in)
        //         .arg(&self.gpu_out_buf)
        //         .build().unwrap();
        //
        //     // println!("Created final activation kernel {:?}", st.elapsed());
        //     st = Instant::now();
        //
        //     execute_kernel(&self.gpu_proque, &activation_kernel, layer_in.len());
        //
        //     // println!("Enqueued activation kernel ({}) {:?}", wg_size_1d, st.elapsed());
        // }
        // buf.copy(&self.gpu_out_buf, None, None).enq();

        st = Instant::now();
        // println!("\nReading network output {:?}", st.elapsed());

        Ok(activated)
    }

    // TODO: implement proper training, using averaged gradients etc
    fn backward(&mut self, mut target: &Vec<f32>, learn_rate: f32) {
        let max_wg = self.gpu_proque.max_wg_size().expect("Failed to get max workgroup size");
        self.gpu_target_buf.write(target).enq().unwrap();

        // let last_outs = cl_utils::buf_read(&self.gpu_layer_bufs[self.layers.len()-1]);
        // let mut sensitivities = Vec::with_capacity(target.len());
        // for i in 0..target.len() {
        //     sensitivities.push(crate::network::cpu::activate(last_outs[i]) - target[i]);
        // }
        // cl_utils::buf_write(&self.gpu_senses_buf, &sensitivities);

        gpu_math::activate_and_error_derivative(&self.gpu_proque, &self.gpu_layer_bufs[self.layers.len()-1], &self.gpu_target_buf, &self.gpu_senses_buf);
        let mut prev_layer_sensitivities = &self.gpu_senses_buf;

        unsafe {
            for i in (1..self.layers.len()).rev() {
                let (layer_size, weight_offset, bias_offset) = self.layers[i];
                let (prev_size, prev_weight_offset, prev_bias_offset) = self.layers[i-1];

                let layer_sensitivities = &self.gpu_layer_sensitivities[i-1];
                layer_sensitivities.cmd().fill(0.0, None).enq();

                let st = Instant::now();
                let back_kernel = self.gpu_proque
                    .kernel_builder("backward")
                    .arg(prev_size as u64) // Input value length
                    .arg(weight_offset as u64)
                    .arg(bias_offset as u64)
                    .arg(learn_rate)
                    .arg(&self.gpu_layer_bufs[i-1]) // Input values
                    .arg(&self.gpu_layer_bufs[i]) // Output values
                    .arg(prev_layer_sensitivities) // Sensitivities
                    .arg(&self.gpu_weights)
                    .arg(&self.gpu_biases)
                    .arg(&self.weight_mods)
                    .arg(&self.bias_mods)
                    .arg(layer_sensitivities)
                    .build()
                    .expect("failed to build backward kernel");

                // println!("{} {} {}", layer_sensitivities.len(), layer_size, prev_size);
                execute_kernel(&self.gpu_proque, &back_kernel, (layer_size, prev_size));

                // println!("{:?}", cl_utils::buf_read(&layer_sensitivities));
                prev_layer_sensitivities = layer_sensitivities;
            }
        }
    }

    fn train<T: Into<Option<f32>>, F>(&mut self, epochs: u32, target_error: T, inputs: &mut Vec<Vec<f32>>, outputs: &mut Vec<Vec<f32>>, mut learn_rate: f32, mut epoch_call_back: F) -> Result<f32, String>
        where F: FnMut(&mut Self)
    {
        let target_error = target_error.into();
        if inputs.len() != outputs.len() {
            return Err("Different length of input sets to output sets".to_string())
        }
        let mut inputs = inputs.clone();
        let mut outputs = outputs.clone();
        let mut i = 0u32;
        let mut last_error = 1f32;
        let mut close_error_count = 0;
        // let mut learn_rate = 0.01f32;
        let mut last_print = Instant::now();
        let p_t_err = match target_error {None => "None".to_string(), Some(val) => format!("{:.4}", val).to_string()};
        println!("\nBeginning network training.\n  Samples: {}\n  Target-Error: {}\n  Max-Epochs: {}", inputs.len(), p_t_err.as_str(), epochs);

        let mut learn_rate_change = -0.000f32;
        let mut last_batch_error = f32::INFINITY;
        let mut batch_complete = 0;
        while i < epochs {
            let mut error = 0.0;
            let mut epoch_error_sum = 0.0;
            let mut batch_error_sum = 0.0;
            shuffle(&mut inputs, &mut outputs);
            // println!("{:?} {:?}", inputs, outputs);\
            self.weight_mods.cmd().fill(0.0, None).enq();
            self.bias_mods.cmd().fill(0.0, None).enq();
            for j in (0..inputs.len()).rev() {
                let input = &inputs[j];
                let expected = &outputs[j];
                let out = self.forward(input).unwrap();

                error = self.error(&out, expected);
                epoch_error_sum += error;
                batch_error_sum += error;

                self.backward(expected, learn_rate+learn_rate_change);
                batch_complete += 1;
                let batch_size = 8;
                if batch_complete >= batch_size {
                    gpu_math::div_second_and_add(&self.gpu_proque, &self.gpu_weights, &self.weight_mods, batch_size as f32);
                    gpu_math::div_second_and_add(&self.gpu_proque, &self.gpu_biases, &self.bias_mods, batch_size as f32);
                    self.weight_mods.cmd().fill(0.0, None).enq();
                    self.bias_mods.cmd().fill(0.0, None).enq();

                    batch_error_sum /= batch_size as f32;

                    if last_print.elapsed().as_secs_f32() > 0.2 {
                        println!("SAMPLE Error: {:.8}, Learn-Rate: {:.4} ({:.2}%), Epoch: {}/{} ({:.2}%) {:?} {:?} {:?} {} {}", batch_error_sum, learn_rate, ((learn_rate+learn_rate_change)/learn_rate*100.0), i, epochs, ((inputs.len() - j) as f32 / inputs.len() as f32) * 100.0, target_error, out, expected, inputs.len(), epoch_error_sum / (inputs.len()-j) as f32);//target_error, input, out, expected
                        last_print = Instant::now();
                    }

                    if last_batch_error != f32::INFINITY {
                        let dif = 5.0 / (last_batch_error - batch_error_sum) / 2e6 * learn_rate + learn_rate/5e4;
                        learn_rate_change += dif;
                        // println!("lear: {}", dif);
                        learn_rate_change = learn_rate_change.min(0.0).max(-learn_rate+0.00001);
                    }
                    last_batch_error = batch_error_sum;
                    batch_error_sum = 0.0f32;

                    batch_complete = 0;
                }
            }

            epoch_error_sum /= inputs.len() as f32;
            if i % 5 == 0 {
                epoch_call_back(self);
            }
            println!("EPOCH  Error: {:.8}, Learn-Rate: {:.8} ({:.8}), Epoch: {}/{} {:?}", epoch_error_sum, learn_rate, learn_rate_change, i, epochs, target_error);

            last_error = epoch_error_sum;
            if target_error.is_some() && epoch_error_sum <= target_error.unwrap() {
                break;
            }
            self.save(format!("{}_{}_{}.net", UNIX_EPOCH.elapsed().unwrap().as_millis(), epoch_error_sum, i));
            i += 1;
        }
        self.save(format!("{}_done.net", UNIX_EPOCH.elapsed().unwrap().as_millis()));
        println!("Network training completed.\n  Completed-Epochs: {}\n  Final-Error: {}\n", i, last_error);
        return Ok(0.0);
    }

    fn error(&mut self, values: &Vec<f32>, target: &Vec<f32>) -> f32 {
        let mut error = 0.0;
        for i in 0..values.len() {
            error += (values[i]-target[i]).powf(2.0);
        }
        error
    }

    fn binary_loss(&mut self, values: &Vec<f32>, target: &Vec<f32>) -> f32 {
        let mut loss = 0f32;
        for i in 0..values.len() {
            loss = loss - target[i] * values[i].abs().ln(); // TODO abs is temp
        }

        loss
    }

    fn layers(&self) -> &Vec<(usize, usize, usize)> {
        &self.layers
    }

    fn weights_len(&self) -> usize {
        self.weights
    }
    fn biases_len(&self) -> usize {
        self.biases
    }

    fn weights(&self) -> Vec<f32> {
        cl_utils::buf_read(&self.gpu_weights)
    }

    fn biases(&self) -> Vec<f32> {
        cl_utils::buf_read(&self.gpu_biases)
    }

    fn layer_bufs(&self) -> Vec<Vec<f32>> {
        let mut out = vec![];
        for buf in &self.gpu_layer_bufs {
            out.push(cl_utils::buf_read(buf));
        }
        out
    }

    fn layer_sensitivities(&self) -> Vec<Vec<f32>> {
        vec![]
    }

    fn out_buf(&self) -> Vec<f32> {
        vec![]
    }
}