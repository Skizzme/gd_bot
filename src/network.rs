use std::fs::read_to_string;
use ocl::{Buffer, ProQue};
use rand::random;

pub struct Network {
    // Layer weights are flat, not a matrix
    weights: Vec<f64>,
    biases: Vec<f64>,
    // Each layer size. First is the number of inputs
    layers: Vec<u32>,

    gpu_kernel: ProQue,
    // A single buffer containing all the weights, so fewer calls to gpu
    // Values will be gotten based off of the layer sizes and offsets etc
    gpu_weights: Buffer<f64>,
    // Individual output buffers for each layer, so each layer can have different sizes
    // Inputs to next layer will be outputs of last layer
    gpu_layer_out_bufs: Vec<Buffer<f64>>,
}

impl Network {
    pub fn new(layers: Vec<u32>, init_weights: Option<Vec<f64>>, init_biases: Option<Vec<f64>>) -> Result<Self, String> {

        let src = read_to_string("src\\test_comp").unwrap();
        let pro_que = ProQue::builder().src(src).build();
        if pro_que.is_err() {
            std::fs::write("compile_error.txt", pro_que.err().unwrap().to_string()).expect("failed to write error");
            return Err("Compile error. Check 'compile_error.txt'".to_string());
        }
        let pro_que = pro_que.unwrap();
        let queue = pro_que.queue();

        let mut layer_buffers = vec![];
        let mut network_weights = vec![];
        let mut network_biases = vec![];
        let mut layer_inputs = *layers.get(0).unwrap();
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
                    network_weights.push(random())
                }
                network_biases.push(random());
            }

            weight_offset += layer_size*layer_inputs;
            bias_offset += layer_size;

            // Keep track of the input size for the next layer
            layer_inputs = layer_size;
        }

        let weight_buf: Buffer<f64> = Buffer::builder()
            .queue(queue.clone())
            .len(network_weights.len())
            .build().expect("Failed to make input buffer");
        // Write all layers' weights to a single buffer

        let mut network = Network {
            weights: network_weights,
            biases: network_biases,
            layers: layers.clone(),
            gpu_kernel: pro_que,
            gpu_weights: weight_buf,
            gpu_layer_out_bufs: layer_buffers,
        };

        Ok(network)
    }

    // pub fn forward(&self) -> Vec<f64> {
    //
    //     pro_que
    //         .kernel_builder("layer")
    //         .arg(layer_nodes)
    //         .arg(in_size)
    //         .arg(&weights_1)
    //         .arg(&inputs_1)
    //         .arg(&out_buffer)
    //         // .global_work_size((layer_nodes, in_size))
    //         // .local_work_size((local_x, local_y))
    //         .build()
    //         .unwrap();
    // }
}