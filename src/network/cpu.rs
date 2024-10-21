use std::time::{Instant, UNIX_EPOCH};
use rand::{random, Rng};
use crate::cl_utils;
use crate::cl_utils::execute_kernel;
use crate::network::{Network, shuffle};

pub fn activate(x: f32) -> f32 {
    // 1.0 / (1.0 + (-x).exp())
    (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
    // x.sin()
    // x
}

pub fn activate_derivative(x: f32) -> f32 {
    // x.exp() / (x.exp() + 1.0).powf(2.0)
    1.0 - (activate(x) * activate(x))
    // x.cos()
    // 1.0
}

pub fn random_with_capacity(capacity: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(capacity);
    for i in 0..capacity {
        let val = random::<f32>() * 2.0 - 1.0;
        // let val = random::<f32>();
        out.push(val / 10.0);
    }
    // println!("{:?}", out);
    out
}

#[derive(Clone)]
pub struct CPUNetwork {
    layers: Vec<(usize, usize, usize)>,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub layer_bufs: Vec<Vec<f32>>,
    pub layer_sensitivities: Vec<Vec<f32>>,
    pub out_buf: Vec<f32>,
    weight_mods: Vec<f32>,
    bias_mods: Vec<f32>,
}

impl CPUNetwork {
    pub fn new(layers: Vec<usize>, init_biases: Option<Vec<f32>>, init_weights: Option<Vec<f32>>) -> Result<Self, String> {
        let mut st = Instant::now();

        let mut network_layers = vec![];
        let mut layer_outputs = vec![];
        let mut layer_sensitivities = vec![];

        let mut layer_inputs = *layers.get(0).unwrap();
        // Add input layer, but since it isn't a real layer it doesn't have weights or biases
        network_layers.push((layers[0], 0, 0));
        layer_sensitivities.push(vec![0.0f32; layers[0]]);
        layer_outputs.push(Vec::with_capacity(layer_inputs));

        let mut weight_offset = 0;
        let mut bias_offset = layers[0];
        // First value would be the plain inputs, so start creating layers with sizes after
        for i in 1..layers.len() {
            let layer_size = *layers.get(i).unwrap();

            let mut layer = Vec::with_capacity(layer_size);
            for i in 0..layer_size {
                layer.push(0.0);
            }
            // layer.fill(0.0);
            // println!("LAYER SIZE {} {}", layer.len(), layer_size);
            layer_outputs.push(layer);

            // println!("{} {} {}", i, bias_offset, weight_offset);
            network_layers.push((layer_size, weight_offset, bias_offset));
            let mut senses = Vec::with_capacity(layer_size);
            for j in 0..senses.capacity() {
                senses.push(0.0);
            }
            layer_sensitivities.push(senses);

            bias_offset += layer_size;
            weight_offset += layer_size * layer_inputs;

            // Keep track of the input size for the next layer
            layer_inputs = layer_size;
        }

        let weights = init_weights.unwrap_or_else(|| random_with_capacity(weight_offset));
        let biases = init_biases.unwrap_or_else(|| random_with_capacity(bias_offset));
        println!("{} {}", weights.len(), weight_offset);

        Ok(CPUNetwork {
            layers: network_layers,
            weights,
            biases,
            layer_bufs: layer_outputs,
            layer_sensitivities,
            out_buf: random_with_capacity(layer_inputs),
            weight_mods: random_with_capacity(weight_offset),
            bias_mods: random_with_capacity(bias_offset),
        })
    }
}

impl Network for CPUNetwork {
    fn forward(&mut self, inputs: &Vec<f32>) -> Result<Vec<f32>, String> {
        let mut layer_in_size = self.layers[0].0;

        if inputs.len() != layer_in_size {
            return Err(format!("Input length is incorrect! Network input length is: {}. Inputs length is: {}", layer_in_size, inputs.len()).to_string())
        }

        self.layer_bufs[0] = inputs.clone();

        for i in 1..self.layers.len() {
            let (layer_size, weights_offset, biases_offset) = self.layers[i];

            for x in 0..layer_size {
                self.layer_bufs[i][x] = self.biases[biases_offset+x];
                for y in 0..layer_in_size {
                    let weight_index = (layer_in_size * x) + y + weights_offset;
                    let mut in_value = self.layer_bufs[i-1][y];
                    if i != 1 {
                        in_value = activate(in_value);
                    }
                    self.layer_bufs[i][x] += self.weights[weight_index] * in_value;
                }
            }

            layer_in_size = layer_size;
        }

        let last = self.layer_bufs.last().unwrap();
        self.out_buf = Vec::with_capacity(last.len());
        for i in 0..last.len() {
            self.out_buf.push(activate(last[i]));
        }

        Ok(self.out_buf.clone())
    }

    fn backward(&mut self, target: &Vec<f32>, learn_rate: f32) {
        let mut sensitivities = Vec::with_capacity(target.len());
        for i in 0..target.len() {
            sensitivities.push(activate(self.layer_bufs.last().unwrap()[i]) - target[i]);
        }
        let senses_len = self.layer_sensitivities.len();
        self.layer_sensitivities[senses_len-1] = sensitivities;

        unsafe {
            for i in (1..self.layers.len()).rev() {
                let (layer_size, weight_offset, bias_offset) = self.layers[i];
                let (prev_size, prev_weight_offset, prev_bias_offset) = self.layers[i-1];

                self.layer_sensitivities[i-1].fill(0.0);

                for x in 0..layer_size {
                    // println!("[{:?}] {}", self.layer_sensitivities[i], x);
                    let gradient = activate_derivative(self.layer_bufs[i][x]) * self.layer_sensitivities[i][x];

                    let bias_index = bias_offset + x;
                    let new_bias = self.biases[bias_index] - (learn_rate * gradient);
                    self.bias_mods[bias_index] += new_bias - self.biases[bias_index];
                    for y in 0..prev_size {
                        let weight_index = (prev_size * x) + y + weight_offset;

                        let new_weight = self.weights[weight_index] - (learn_rate * activate(self.layer_bufs[i-1][y]) * gradient);
                        self.weight_mods[weight_index] += new_weight - self.weights[weight_index];
                        self.layer_sensitivities[i-1][y] += gradient * self.weights[weight_index];
                    }
                }

            }
        }
    }

    fn train<T: Into<Option<f32>>, F>(&mut self, epochs: u32, target_error: T, inputs: &mut Vec<Vec<f32>>, outputs: &mut Vec<Vec<f32>>, mut learn_rate: f32, mut epoch_call_back: F) -> Result<f32, String> where F: FnMut(&mut Self) {
        let target_error = target_error.into().unwrap();
        let mut inputs = inputs.clone();
        let mut outputs = outputs.clone();

        let mut last_print = Instant::now();
        let mut i = 0;
        let mut batched = 0;
        let mut last_error = f32::INFINITY;
        while i < epochs {
            shuffle(&mut inputs, &mut outputs);
            let mut error_sum = 0.0;
            for j in 0..inputs.len() {
                let input = &inputs[j];
                let expected = &outputs[j];

                let out = self.forward(input).unwrap();
                let error = self.error(&out, expected);

                error_sum += error;
                if last_print.elapsed().as_secs_f32() > 0.2 {
                    println!("SAMPLE Error: {:.8}, Learn-Rate: {:.8}, Epoch: {}/{} ({:.2}%) {:?} {:?} {:?}", error, learn_rate, i, epochs, (j as f32 / inputs.len() as f32) * 100.0, target_error, out, expected);//target_error, input, out, expected
                    last_print = Instant::now();
                }

                // Accumulate mods
                self.backward(expected, learn_rate);

                batched += 1;
                if batched >= 4 {
                    // Apply mods
                    for w in 0..self.weights.len() {
                        self.weights[w] += self.weight_mods[w] / batched as f32;
                    }
                    for b in 0..self.biases.len() {
                        self.biases[b] += self.bias_mods[b] / batched as f32;
                    }
                    self.weight_mods.fill(0.0);
                    self.bias_mods.fill(0.0);

                    batched = 0;
                }

            }
            error_sum /= inputs.len() as f32;

            if last_error != f32::INFINITY {
                let dif = 5.0 / (last_error - error_sum) / 2e8;
                learn_rate += dif;
                // println!("lear: {}", dif);
                learn_rate = learn_rate.max(0.00001).min(0.1);
            }
            last_error = error_sum;
            // if error_sum < target_error {
            //     break
            // }
            if i % 50 == 0 {
                epoch_call_back(self);
            }
            println!("EPOCH  Error: {:.8}, Learn-Rate: {:.8}, Epoch: {}/{}", error_sum, learn_rate, i, epochs);

            i += 1;
        }
        self.save(format!("{}.net", UNIX_EPOCH.elapsed().unwrap().as_millis()));
        println!("Network training completed.\n  Completed-Epochs: {}\n  Final-Error: {}\n", i, last_error);
        Ok(0.0)
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
        self.weights.len()
    }

    fn biases_len(&self) -> usize {
        self.biases.len()
    }

    fn weights(&self) -> Vec<f32> {
        self.weights.clone()
    }

    fn biases(&self) -> Vec<f32> {
        self.biases.clone()
    }

    fn layer_bufs(&self) -> Vec<Vec<f32>> {
        self.layer_bufs.clone()
    }

    fn layer_sensitivities(&self) -> Vec<Vec<f32>> {
        self.layer_sensitivities.clone()
    }

    fn out_buf(&self) -> Vec<f32> {
        self.out_buf.clone()
    }
}