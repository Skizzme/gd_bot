use std::time::Instant;
use rand::random;
use crate::cl_utils;
use crate::cl_utils::execute_kernel;
use crate::network::Network;

pub fn random_with_capacity(capacity: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(capacity);
    for i in 0..capacity {
        out.push(random::<f32>() / 50.0)
    }
    println!("{:?}", out);
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
    pub fn new(layers: Vec<usize>, init_weights: Option<Vec<f32>>, init_biases: Option<Vec<f32>>) -> Result<Self, String> {
        let mut st = Instant::now();

        let mut network_layers = vec![];
        let mut layer_outputs = vec![];
        let mut layer_sensitivities = vec![];

        let mut layer_inputs = *layers.get(0).unwrap();
        // Add input layer, but since it isn't a real layer it doesn't have weights or biases
        network_layers.push((layers[0], 0, 0));
        layer_outputs.push(Vec::with_capacity(layer_inputs));

        let mut weight_offset = 0;
        let mut bias_offset = layers[0];
        // First value would be the plain inputs, so start creating layers with sizes after
        for i in 1..layers.len() {
            let layer_size = *layers.get(i).unwrap();

            let layer = Vec::with_capacity(layer_size);
            layer_outputs.push(layer);

            println!("{} {} {}", i, bias_offset, weight_offset);
            network_layers.push((layer_size, weight_offset, bias_offset));
            layer_sensitivities.push(Vec::with_capacity(layer_inputs));

            bias_offset += layer_size;
            weight_offset += layer_size * layer_inputs;

            // Keep track of the input size for the next layer
            layer_inputs = layer_size;
        }

        Ok(CPUNetwork {
            layers: network_layers,
            weights: random_with_capacity(weight_offset),
            biases: random_with_capacity(bias_offset),
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

        let mut st = Instant::now();
        let mut layer_in = &mut self.layer_bufs[0];

        // println!("\nInitialized network input {:?}", st.elapsed());
        st = Instant::now();

        for i in 1..self.layers.len() {
            let (layer_size, weights_offset, biases_offset) = self.layers[i];
            // println!("Forwarding layer: {:?}. size: {:?}, weight_offset: {:?}, bias_offset: {:?} {:?}", i, layer_size, weights_offset, biases_offset, match i { 1 => 0, _ => 1 });

            let layer_out = &mut self.layer_bufs[i];



            layer_in_size = self.layers[i].0;
            // println!("{:?} {:?}", i, cl_utils::buf_read(&layer_buf));
            layer_in = layer_out;
        }

        //TODO activate kernel

        st = Instant::now();
        // println!("\nReading network output {:?}", st.elapsed());

        Ok(self.out_buf.clone())
    }

    fn backward(&mut self, target: &Vec<f32>, learn_rate: f32) {

    }

    fn train<T: Into<Option<f32>>, F>(&mut self, epochs: u32, target_error: T, inputs: &mut Vec<Vec<f32>>, outputs: &mut Vec<Vec<f32>>, learn_rate: f32, epoch_call_back: F) -> Result<f32, String> where F: FnMut(&mut Self) {
        Ok(0.0)
    }

    fn error(&mut self, values: &Vec<f32>, target: &Vec<f32>) -> f32 {
        0.0
    }

    fn binary_loss(&mut self, values: &Vec<f32>, target: &Vec<f32>) -> f32 {
        0.0
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