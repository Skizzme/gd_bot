use std::fs::read_to_string;
use std::time::Instant;
use num_format::Locale::{qu, si};
use ocl::{Buffer, ProQue, Queue};
use rand::random;

#[derive(Debug)]
pub struct Network {
    // Layer weights are flat, not a matrix
    // weights: Vec<f64>,
    // biases: Vec<f64>,
    weights: i32,
    biases: i32,
    // Each layer size. First item is the inputs to the network
    // nodes, weights_offset, biases_offset
    layers: Vec<(i32, i32, i32)>,

    gpu_proque: ProQue,
    // A single buffer containing all the weights/biases, so fewer calls to gpu
    // Values will be gotten based off of the layer sizes and offsets etc
    gpu_weights: Buffer<f64>,
    gpu_biases: Buffer<f64>,
    // Individual output buffers for each layer, so each layer can have different sizes
    // Inputs to next layer will be outputs of last layer
    gpu_layer_bufs: Vec<Buffer<f64>>,
}

pub fn randomize_buffer(buffer: &Buffer<f64>, max_work_size: i32, pro_que: &ProQue) {
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
            .local_work_size(calculate_worksize(max_work_size, buffer.len() as i32))
            .enq()
            .expect("Failed to enq rnd_kernel")
    }
}

pub fn calculate_worksize(max: i32, size: i32) -> i32 {
    let mut calc = 1;
    for i in 1..max+1 {
        if size as f32 / i as f32 == (size / i) as f32 {
            calc = i;
        }
    }
    calc
}

impl Network {
    pub fn new(layers: Vec<i32>, init_weights: Option<Vec<f64>>, init_biases: Option<Vec<f64>>) -> Result<Self, String> {
        let mut st = Instant::now();
        println!("Creating network...");
        let src = include_str!("kernels.c");
        let pro_que = ProQue::builder().src(src).build();
        if pro_que.is_err() {
            std::fs::write("compile_error.txt", pro_que.err().unwrap().to_string()).expect("failed to write error");
            return Err("Compile error. Check 'compile_error.txt'".to_string());
        }
        let pro_que = pro_que.unwrap();
        let queue = pro_que.queue();;
        println!("Compiled kernels.c {:?}", st.elapsed());
        st = Instant::now();

        println!("\nInitializing layers...");
        st = Instant::now();

        let mut network_layers = vec![];
        let mut layer_buffers = vec![];

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

            network_layers.push((layer_size, weight_offset, bias_offset));

            weight_offset += layer_size*layer_inputs;
            bias_offset += layer_size;

            // Keep track of the input size for the next layer
            layer_inputs = layer_size;
        }

        st = Instant::now();
        println!("\nStoring network on GPU...");
        let weight_buf: Buffer<f64> = Buffer::builder()
            .queue(queue.clone())
            .len(weight_offset)
            .build().expect("Failed to make input buffer");
        // Init network's weights randomly using GPU
        // randomize_buffer(&weight_buf, 256, &pro_que);

        let biases_buf: Buffer<f64> = Buffer::builder()
            .queue(queue.clone())
            .len(bias_offset)
            .build().expect("Failed to make input buffer");
        // Init network's biases randomly using GPU
        randomize_buffer(&biases_buf, 256, &pro_que);

        println!("Stored network on GPU {:?}\n", st.elapsed());
        st = Instant::now();

        let mut network = Network {
            // weights: network_weights,
            // biases: network_biases,
            weights: weight_offset,
            biases: bias_offset,
            layers: network_layers,
            gpu_proque: pro_que,
            gpu_weights: weight_buf,
            gpu_biases: biases_buf,
            gpu_layer_bufs: layer_buffers,
        };

        Ok(network)
    }

    pub fn calculate(&self, mut inputs: Vec<f64>) -> Result<Vec<f64>, String> { // Will return proper error later
        let mut layer_in_size = self.layers[0].0;
        let max_wg = self.gpu_proque.max_wg_size().expect("Failed to get max workgroup size");

        if inputs.len() != layer_in_size as usize {
            return Err(format!("Input length is incorrect! Network input length is: {}", layer_in_size).to_string())
        }

        let mut st = Instant::now();
        let mut buf = &self.gpu_layer_bufs[0];

        buf.write(&inputs).enq().expect("Failed to write network inputs");

        println!("\nInitialized network input {:?}", st.elapsed());
        st = Instant::now();

        for i in 1..self.layers.len() {
            let (layer_size, weights_offset, biases_offset) = self.layers[i];
            println!("\nForwarding layer: {:?}. size: {:?}, weight_offset: {:?}, bias_offset: {:?}", i, layer_size, weights_offset, biases_offset);

            let layer_buf = &self.gpu_layer_bufs[i];

            // Maybe create these kernels once in network creation, and only enq them here?
            let forward_kernel = self.gpu_proque
                .kernel_builder("forward")
                .arg(layer_in_size)
                .arg(layer_size)
                .arg(weights_offset)
                .arg(biases_offset)
                .arg(&self.gpu_weights)
                .arg(&self.gpu_biases)
                .arg(buf)
                .arg(layer_buf)
                .build()
                .unwrap();

            println!("Created layer kernel {:?}", st.elapsed());
            st = Instant::now();

            unsafe {
                let wg_size = (calculate_worksize((max_wg as f32).sqrt() as i32, layer_size), calculate_worksize((max_wg as f32).sqrt() as i32, layer_in_size));

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

        let mut final_out = vec![0f64; self.layers.get(self.layers.len() - 1).unwrap().0 as usize];
        buf.read(&mut final_out).enq().expect("Failed to read output of network");
        println!("\nReading network output {:?}", st.elapsed());
        st = Instant::now();

        Ok(final_out)
    }

    pub fn layers(&self) -> &Vec<(i32, i32, i32)> {
        &self.layers
    }


    pub fn weights(&self) -> i32 {
        self.weights
    }
    pub fn biases(&self) -> i32 {
        self.biases
    }
}
