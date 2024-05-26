use std::fs::read_to_string;
use std::time::Instant;
use ocl::{Buffer, ProQue};
use rand::random;

#[derive(Debug)]
pub struct Network {
    // Layer weights are flat, not a matrix
    weights: Vec<f64>,
    biases: Vec<f64>,
    // Each layer size. First item is the inputs to the network
    // nodes, weights_offset, biases_offset
    layers: Vec<(i32, i32, i32)>,

    gpu_proque: ProQue,
    // A single buffer containing all the weights, so fewer calls to gpu
    // Values will be gotten based off of the layer sizes and offsets etc
    gpu_weights: Buffer<f64>,
    // Individual output buffers for each layer, so each layer can have different sizes
    // Inputs to next layer will be outputs of last layer
    gpu_layer_bufs: Vec<Buffer<f64>>,
}

impl Network {
    pub fn new(layers: Vec<i32>, init_weights: Option<Vec<f64>>, init_biases: Option<Vec<f64>>) -> Result<Self, String> {
        println!("Compiling kernel");
        let src = read_to_string("src\\test_comp").unwrap();
        let pro_que = ProQue::builder().src(src).build();
        if pro_que.is_err() {
            std::fs::write("compile_error.txt", pro_que.err().unwrap().to_string()).expect("failed to write error");
            return Err("Compile error. Check 'compile_error.txt'".to_string());
        }
        let pro_que = pro_que.unwrap();
        let queue = pro_que.queue();

        println!("Initializing layers");
        let mut network_layers = vec![];
        let mut layer_buffers = vec![];
        let mut network_weights = vec![];
        let mut network_biases = vec![];
        let mut layer_inputs = *layers.get(0).unwrap();
        // Add input layer, but since it isn't a real layer it doesn't have weights or biases
        layer_buffers.push(Buffer::builder()
                               .queue(queue.clone())
                               .len(layer_inputs)
                               .build().expect("Failed to make input buffer"));
        network_layers.push((layers[0], 0, 0));

        let mut weight_offset = 0;
        let mut bias_offset = 0;
        // First value would be the plain inputs, so start creating layers with sizes after
        for i in 1..layers.len() {
            let layer_size = *layers.get(i).unwrap();

            let layer: Buffer<f64> = Buffer::builder()
                .queue(queue.clone())
                .len(layer_size)
                .build().expect("Failed to make input buffer");
            layer_buffers.push(layer);

            // Init weights/biases
            for j in 0..layer_size {
                for k in 0..layer_inputs {
                    network_weights.push(random::<f64>()-0.5f64)
                }
                network_biases.push(random::<f64>()-0.5f64);
            }

            network_layers.push((layer_size, weight_offset, bias_offset));

            weight_offset += layer_size*layer_inputs;
            bias_offset += layer_size;

            // Keep track of the input size for the next layer
            layer_inputs = layer_size;
        }

        println!("Storing network weights on GPU");
        let weight_buf: Buffer<f64> = Buffer::builder()
            .queue(queue.clone())
            .len(network_weights.len())
            .build().expect("Failed to make input buffer");
        // Write all layers' weights to a single buffer

        weight_buf.write(network_weights.as_slice()).enq().expect("Failed to write network weights");

        let mut network = Network {
            weights: network_weights,
            biases: network_biases,
            layers: network_layers,
            gpu_proque: pro_que,
            gpu_weights: weight_buf,
            gpu_layer_bufs: layer_buffers,
        };

        Ok(network)
    }

    pub fn forward(&self, mut inputs: Vec<f64>, work_size: i32) -> Result<Vec<f64>, String> { // Will return proper error later
        let mut layer_in_size = self.layers[0].0;

        if inputs.len() != layer_in_size as usize {
            return Err(format!("Input length is incorrect! Network input length is: {}", layer_in_size).to_string())
        }

        let mut st = Instant::now();
        let mut buf = &self.gpu_layer_bufs[0];
        buf.write(inputs.as_slice()).enq().expect("Failed to write network inputs");
        println!("Initialized network input {:?}", st.elapsed());
        st = Instant::now();
        for i in 1..self.layers.len() {
            let (layer_size, weights_offset, biases_offset) = self.layers[i];
            println!("Forwarding layer: {:?}. size: {:?}, weight_offset: {:?}, bias_offset: {:?}", i, layer_size, weights_offset, biases_offset);
            let layer_buf = &self.gpu_layer_bufs[i];
            let biases = self.biases[biases_offset as usize..(biases_offset+layer_size) as usize].iter().as_slice();
            layer_buf.write(biases).enq().expect("Failed to set layer input buffer");
            println!("Initialized layer output {:?}", st.elapsed());
            st = Instant::now();

            let layer_kernel = self.gpu_proque
                .kernel_builder("layer")
                .arg(i as i32)
                .arg(layer_in_size)
                .arg(weights_offset)
                .arg(&self.gpu_weights)
                .arg(buf)
                .arg(layer_buf)
                .build()
                .unwrap();
            println!("Created layer kernel {:?}", st.elapsed());
            st = Instant::now();

            unsafe {
                layer_kernel
                    .cmd()
                    .global_work_size((layer_size, layer_in_size))
                    .local_work_size((16, 16))
                    .enq()
                    .expect("Failed to enqueue kernel");
            }
            println!("Enqueued layer kernel {:?}", st.elapsed());
            st = Instant::now();

            layer_in_size = self.layers[i].0;
            buf = layer_buf;
        }

        let mut final_out = vec![0f64; self.layers.get(self.layers.len() - 1).unwrap().0 as usize];
        buf.read(&mut final_out).enq().expect("Failed to read output of network");
        println!("Reading network output {:?}", st.elapsed());
        st = Instant::now();

        Ok(final_out)
    }


    pub fn weights(&self) -> &Vec<f64> {
        &self.weights
    }
    pub fn biases(&self) -> &Vec<f64> {
        &self.biases
    }
    pub fn layers(&self) -> &Vec<(i32, i32, i32)> {
        &self.layers
    }
}