extern crate ocl;

use std::f32::consts::PI;
use std::fs::read;
use std::path;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::time::{Duration, Instant};
use gl30::*;

use num_format::{Locale, ToFormattedString};
use parking_lot::Mutex;
use rand::prelude::SliceRandom;
use rand::{random, thread_rng};
use RustUI::components::render::bounds::Bounds;
use RustUI::components::render::color::Color;
use RustUI::components::screen::{Element, ScreenTrait};
use RustUI::components::window::Window;
use RustUI::gl_binds::gl30;
use RustUI::glfw::{Action, Key, Modifiers, Scancode, WindowEvent};
use RustUI::WindowMode;

use crate::network::Network;

mod network;
mod gpu_math;
mod cl_utils;

fn main() {
    // backprop();
    let args: Vec<String> = std::env::args().collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();

    let epochs = u32::from_str(args[1]).unwrap();
    let learn_rate = f32::from_str(args[2]).unwrap();
    let mut layers = vec![];
    for i in 3..args.len() {
        let arg_val = args[i];
        layers.push(usize::from_str(arg_val).unwrap());
    }
    let network = Network::new(layers.clone(), None, None).unwrap();
    println!("Total weights: {}, Total biases: {}, Total mem: {}", network.weights().to_formatted_string(&Locale::en), network.biases().to_formatted_string(&Locale::en), ((network.weights()+network.biases())*i32::BITS as usize/8).to_formatted_string(&Locale::en));
    let st = Instant::now();
    // let mut t_inputs: Vec<Vec<f32>> = vec![vec![1.0, -0.87], vec![2.0, 0.2], vec![-1.0, 2.0], vec![0.0, -1.0]];
    // let mut t_outputs: Vec<Vec<f32>> = vec![vec![0.0, 0.25], vec![0.25, -0.6], vec![0.85, -0.2], vec![-0.25, 0.0]];
    let mut t_inputs: Vec<Vec<f32>> = vec![];
    let mut t_outputs: Vec<Vec<f32>> = vec![];

    let samples = 20f32;

    let samples = 20;
    for i in 0..samples {
        let i = i as f32;
        // println!("{}", ((i/samples as f32)*2.0*PI).sin());
        t_inputs.push(vec![i/samples as f32]);
        let degrees = (i/samples as f32)*2.0*PI;
        t_outputs.push(vec![degrees.sin()]); // + degrees.sin().powi(2)
    }

    // let mut cl_1_i = vec![];
    // let mut cl_1_o = vec![];
    // let mut cl_2_i = vec![];
    // let mut cl_2_o = vec![];
    // let mut x = 0f32;
    // while x < samples {
    //     let mut y = 0f32;
    //     while y < samples {
    //         let classification = if ((((x - samples / 2.0).powi(2) + (y - samples / 2.0).powi(2)) as f32).sqrt() > samples / 3.0) { 1.0 } else { 0.0 };
    //         let inp = vec![x/samples, y/samples];
    //         let out = vec![classification, 1.0-classification];
    //         if classification == 0.0 {
    //             cl_1_i.push(inp);
    //             cl_1_o.push(out); // + degrees.sin().powi(2)
    //         } else {
    //             cl_2_i.push(inp);
    //             cl_2_o.push(out); // + degrees.sin().powi(2)
    //         }
    //         y+=1.0;
    //     }
    //     x+= 1.0;
    // }
    // network::shuffle(&mut cl_1_i, &mut cl_1_o);
    // network::shuffle(&mut cl_2_i, &mut cl_2_o);
    // if cl_2_i.len() > cl_1_i.len() {
    //     for i in 0..(cl_2_i.len()-cl_1_i.len()) {
    //         cl_2_i.swap_remove(0);
    //         cl_2_o.swap_remove(0);
    //     }
    // }
    //
    // t_inputs.append(&mut cl_1_i);
    // t_inputs.append(&mut cl_2_i);
    // t_outputs.append(&mut cl_1_o);
    // t_outputs.append(&mut cl_2_o);

    // println!("CL1: {:?} {:?}", cl_1_i, cl_1_o);
    // println!("CL2: {:?} {:?}", cl_2_i, cl_2_o);

    let s_t_inputs = t_inputs.clone();
    let s_t_outputs = t_outputs.clone();
    let t_layers = network.layers().clone();
    let n_c = network.clone();
    let (send, receive) = channel();
    let j = std::thread::spawn(move || {
        unsafe {
            let mut window = unsafe {
                Window::create(
                    "Simulation",
                    1920,
                    1080,
                    "src\\assets\\fonts\\",
                    "",
                    vec![],
                    WindowMode::Windowed,
                    240
                )
            };
            let mut current_screen = MainScreen::new(&mut window, s_t_inputs, s_t_outputs, receive, t_layers, n_c);
            let mut last_frame = Instant::now();
            let mut last_len_toks = 0;
            // window.p_window.set_pos(1920 - width/2, 1080 - height/2);
            while !window.p_window.should_close() {
                window.frame(Box::new(&mut current_screen), last_frame);
                last_frame = Instant::now()
            }
        }
    });
    for i in 0..t_inputs.len() {
        let t_input = &t_inputs[i];
        let t_output = &t_outputs[i];
        println!("IN: {:?}, OUT: {:?}, EXPECTED: {:?}", t_input, network.forward(&t_input), t_output);
    }
    send.send((vec![], cl_utils::buf_read(&network.gpu_weights), cl_utils::buf_read(&network.gpu_biases), network.clone())).unwrap();
    std::thread::sleep(Duration::from_millis(2000));

    network.train(epochs, 0.0000005, &mut t_inputs.clone(), &mut t_outputs.clone(), learn_rate, |c_net| {
        let mut all = vec![];
        let mut n_in = 0.0 as f32;
        while n_in < 1.0 {
            all.push((vec![n_in], (network.forward(&vec![n_in]).unwrap())));
            n_in += 0.01;
        }
        send.send((all, cl_utils::buf_read(&c_net.gpu_weights), cl_utils::buf_read(&c_net.gpu_biases), c_net.clone())).unwrap();
        // let samples = 20f32;
        // let mut x = 0f32;
        // while x < samples {
        //     let mut y = 0f32;
        //     while y < samples {
        //         let inp = vec![x/samples, y/samples];
        //         all.push((inp.clone(), c_net.forward(&inp).unwrap()));
        //         y+=1.0;
        //     }
        //     x+= 1.0;
        // }
        // send.send((all, cl_utils::buf_read(&c_net.gpu_weights), cl_utils::buf_read(&c_net.gpu_biases), c_net.clone())).unwrap();
    }).unwrap();
    // send.send((vec![], cl_utils::buf_read(&network.gpu_weights), cl_utils::buf_read(&network.gpu_biases), network.clone())).unwrap();
    for i in 0..t_inputs.len() {
        let t_input = &t_inputs[i];
        let t_output = &t_outputs[i];
        println!("IN: {:?}, OUT: {:?}, EXPECTED: {:?}", t_input, network.forward(&t_input), t_output);
    }
    j.join().expect("TODO: panic message");
}

struct MainScreen {
    inputs: Vec<Vec<f32>>,
    outputs: Vec<Vec<f32>>,
    last_outputs: Vec<(Vec<f32>, Vec<f32>)>,
    receive: Receiver<(Vec<(Vec<f32>, Vec<f32>)>, Vec<f32>, Vec<f32>, Network)>,
    network_weights: Vec<f32>,
    network_biases: Vec<f32>,
    network_layers: Vec<(usize, usize, usize)>,
    network: Network,
}

impl MainScreen {
    pub unsafe fn new(window: &mut Window, inputs: Vec<Vec<f32>>, outputs: Vec<Vec<f32>>, receive: Receiver<(Vec<(Vec<f32>, Vec<f32>)>, Vec<f32>, Vec<f32>, Network)>, layers: Vec<(usize, usize, usize)>, network: Network) -> Self {
        window.fonts.set_font_bytes("ProductSans", read("src/assets/fonts/ProductSans.ttf".replace("/", path::MAIN_SEPARATOR_STR)).unwrap()).load_font("ProductSans", false);
        MainScreen {
            last_outputs: vec![(vec![0.0], vec![0f32]); outputs.len()],
            inputs,
            outputs,
            receive,
            network_weights: vec![],
            network_biases: vec![],
            network_layers: layers,
            network,
        }
    }

    unsafe fn draw_network(&mut self, window: &mut Window) {
        Color4f(1.0, 1.0, 1.0, 1.0);
        PointSize(20.0);
        Enable(LINE_SMOOTH);
        Hint(LINE_SMOOTH_HINT, NICEST);
        Begin(POINTS);
        let hori_mult = 150;
        let vert_mult = 50;
        for i in 0..self.network_layers.len() {
            let (size, w_offset, b_offset) = self.network_layers[i];
            for node in 0..size {
                let x = (i * hori_mult + 50) as f32;
                let y = (node * vert_mult + 160) as f32;
                Color4f(1.0-self.network_biases[node+b_offset]*10.0, 1.0, 1.0, 1.0);
                Vertex2f(x, y);
            }
        }
        End();
        for i in 0..self.network_layers.len() {
            // let out = cl_utils::buf_read(&self.network.gpu_layer_bufs[i]);
            let (size, w_offset, b_offset) = self.network_layers[i];
            for node in 0..size {
                let x = (i * hori_mult + 50) as f32;
                let y = (node * vert_mult + 160) as f32;
                let bias_index = node+b_offset;
                window.fonts.get_font("ProductSans").unwrap().draw_string(16.0, format!("{} {:.4} {}", bias_index, self.network_biases[bias_index], b_offset), x, y, Color::from_u32(0xff20ffff));
            }
        }
        // let nw_weights = cl_utils::buf_read(&self.network.gpu_weights);
        for i in 0..self.network_layers.len() {
            let (size, w_offset, b_offset) = self.network_layers[i];
            for node in 0..size {
                let x = (i * hori_mult + 50) as f32;
                let y = (node * vert_mult + 160) as f32;
                if i < self.network_layers.len()-1 {
                    let (n_size, n_w_offset, n_b_offset) = self.network_layers[i+1];
                    for next_node in 0..n_size {
                        let weight_index = (next_node * size) + node + n_w_offset;
                        let weight = self.network_weights[weight_index];
                        // println!("{} {}", weight_index, weight);
                        let n_x = ((i+1) * hori_mult + 50) as f32;
                        let n_y = (next_node * vert_mult + 160) as f32;
                        if weight < 0.0 {
                            Color4f(0.0, 1.0, 1.0, 0.6);
                        }  else {
                            Color4f(1.0, 1.0, 0.0, 0.6);
                        }
                        Enable(BLEND);
                        Disable(TEXTURE_2D);
                        LineWidth((weight*30.0).abs().max(1.0));
                        Begin(LINE_STRIP);
                        Vertex2f(x,y);
                        Vertex2f(n_x,n_y);
                        End();
                        window.fonts.get_font("ProductSans").unwrap().draw_string(12.0, format!("{:?}", weight_index), (x+n_x)/2.0, (y+n_y) / 2.0 + node as f32*14.0, Color::from_u32(0xffffffff));
                    }
                }
            }
        }
    }
}

impl ScreenTrait for MainScreen {
    unsafe fn draw(&mut self, window: &mut Window) {
        Enable(BLEND);
        match self.receive.try_recv() {
            Ok(values) => {
                (self.last_outputs, self.network_weights, self.network_biases, self.network) = values;
            }
            Err(_) => {}
        }
        Disable(TEXTURE_2D);
        Enable(BLEND);
        Enable(LINE_SMOOTH);
        LineWidth(2.0);
        Hint(LINE_SMOOTH_HINT, NICEST);
        Enable(POINT_SMOOTH);
        PointSize(4.0);
        Hint(POINT_SMOOTH_HINT, NICEST);
        PushMatrix();
        Translatef(800.0, 0.0, 0.0);
        Begin(LINE_STRIP);
        for i in 0..self.inputs.len() {
            let input = self.inputs[i][0];
            let output = self.outputs[i][0];
            Color::from_u32(0xff0000ff).apply();
            Vertex2f(10.0+input*700.0, output * 100.0 + 300.0);
        }
        End();
        Begin(LINE_STRIP);
        for (input, out) in &self.last_outputs {
            Color::from_u32(0xffff00ff).apply();
            Vertex2f(10.0+input[0]*700.0, out[0] * 100.0 + 300.0);
        }
        End();
        PopMatrix();
        // for i in 0..self.inputs.len() {
        //     let x = self.inputs[i][0];
        //     let y = self.inputs[i][1];
        //     let output = &self.outputs[i];
        //     Color4f(output[0], output[1], 0.0, 1.0);
        //     Vertex2f(x*20.0*5.0+5.0, y*20.0*5.0+5.0);
        // }
        // End();
        // Begin(POINTS);
        // for i in 0..self.last_outputs.len() {
        //     let x = self.last_outputs[i].0[0];
        //     let y = self.last_outputs[i].0[1];
        //     let output = &self.last_outputs[i].1;
        //     Color4f(0.0, output[1], output[0], 1.0);
        //     Vertex2f(400.0+x*20.0*5.0+5.0, y*20.0*5.0+5.0);
        // }
        // End();
        self.draw_network(window);
    }

    fn key_press(&mut self, key: Key, code: Scancode, action: Action, mods: Modifiers) {

    }

    fn event(&mut self, event: WindowEvent, window: &Window) {

    }

    fn elements(&self) -> Vec<Element> {
        vec![]
    }
}

pub fn error_derivative(actual: f32, desired: f32) -> f32 {
    2.0 * (actual - desired)
}

pub fn activation(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp()) * 2.0 - 1.0
    // value
}

pub fn activation_derivative(activated: f32) -> f32 {
    (2.0*activated.exp()) / (activated.exp() + 1.0).powf(2.0)
    // 1.0
}

// gradient = input * error_derivative(actual_output - desired_output)
// new_weight = original_weight - learn_rate * gradient

pub fn backprop() {
    let learn_rate = 0.1;
    let input = 1.5;
    let mut weight_1 = random::<f32>();
    let mut bias_1 = random::<f32>();
    let mut weight_2 = random::<f32>();
    let mut bias_2 = random::<f32>();
    let desired_output = -0.2;

    for i in 0..10000 {
        let layer_1_output = input * weight_1 + bias_1;
        let layer_1_activated = activation(layer_1_output);
        let layer_2_output = layer_1_activated * weight_2 + bias_2;
        let layer_2_activated = activation(layer_2_output);

        let layer_2_desired = desired_output;

        let gradient =  activation_derivative(layer_2_output) * error_derivative(layer_2_activated, layer_2_desired);
        println!("L2T {} {} {} {}", layer_2_activated, layer_1_activated, layer_2_desired, gradient);

        weight_2 = weight_2 - learn_rate * layer_1_activated * gradient;
        bias_2 = bias_2 - learn_rate * gradient;

        let layer_1_desired = gradient * weight_1;

        let gradient =  activation_derivative(layer_1_output) * error_derivative(layer_1_activated, layer_1_desired);
        println!("L1T {} {} {}", layer_1_activated, input, gradient);

        weight_1 = weight_1 - learn_rate * input * gradient;
        bias_1 = bias_1 - learn_rate * gradient;
    }

    let layer_1_output = activation(input * weight_1 + bias_1);
    let layer_2_output = activation(layer_1_output * weight_2 + bias_2);
    println!("AF {} {} {} {} {}", layer_2_output, weight_1, bias_1, weight_2, bias_2);
}