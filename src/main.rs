extern crate ocl;

use std::f32::consts::PI;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::time::{Duration, Instant};
use gl30::*;

use num_format::{Locale, ToFormattedString};
use parking_lot::Mutex;
use rand::random;
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

    let samples = 20;
    for i in 0..samples {
        let i = i as f32;
        // println!("{}", ((i/samples as f32)*2.0*PI).sin());
        t_inputs.push(vec![i/samples as f32]);
        let degrees = (i/samples as f32)*2.0*PI;
        t_outputs.push(vec![degrees.sin() + degrees.sin().powi(2)]); // + degrees.sin().powi(2)
    }

    let s_t_inputs = t_inputs.clone();
    let s_t_outputs = t_outputs.clone();
    let (send, receive) = channel();
    let j = std::thread::spawn(move || {
        unsafe {
            let mut window = unsafe {
                Window::create(
                    "Simulation",
                    1920 / 2,
                    1080 / 2,
                    "src\\assets\\fonts\\",
                    "",
                    vec![],
                    WindowMode::Windowed,
                    240
                )
            };
            let mut current_screen = MainScreen::new(&mut window, s_t_inputs, s_t_outputs, receive);
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

    network.train(epochs, 0.0000005, &mut t_inputs.clone(), &mut t_outputs.clone(), learn_rate, |network| {
        let mut all = vec![];
        let mut n_in = 0.0 as f32;
        while n_in < 1.0 {
            all.push((n_in, (network.forward(&vec![n_in]).unwrap())));
            n_in += 0.01;
        }
        send.send(all).unwrap();
    }).unwrap();
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
    last_outputs: Vec<(f32, Vec<f32>)>,
    receive: Receiver<Vec<(f32, Vec<f32>)>>,
}

impl MainScreen {
    pub fn new(window: &mut Window, inputs: Vec<Vec<f32>>, outputs: Vec<Vec<f32>>, receive: Receiver<Vec<(f32, Vec<f32>)>>) -> Self {
        MainScreen {
            last_outputs: vec![(0.0, vec![0f32]); outputs.len()],
            inputs,
            outputs,
            receive,
        }
    }
}

impl ScreenTrait for MainScreen {
    unsafe fn draw(&mut self, window: &mut Window) {
        match self.receive.try_recv() {
            Ok(values) => {
                self.last_outputs = values;
            }
            Err(_) => {}
        }
        Disable(TEXTURE_2D);
        Enable(BLEND);
        Enable(LINE_SMOOTH);
        LineWidth(2.0);
        Hint(LINE_SMOOTH_HINT, NICEST);
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
            Vertex2f(10.0+input*700.0, out[0] * 100.0 + 300.0);
        }
        End();
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