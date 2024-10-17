#![allow(unused)]

pub mod gpu;
pub mod cpu;

use std::{fs, thread};
use std::io::Write;
use std::time::{Duration, Instant, UNIX_EPOCH};
use num_format::Locale::{te, wo};

use ocl::{Buffer, OclPrm, ProQue, SpatialDims};
use ocl::enums::WriteSrc;
use rand::{random, Rng, thread_rng};
use crate::{cl_utils, gpu_math};
use crate::cl_utils::{calc_ws, execute_kernel};
use num_format::{Locale, ToFormattedString};
use ocl::core::OclNum;
use rand::seq::SliceRandom;

pub fn shuffle(inputs: &mut Vec<Vec<f32>>, outputs: &mut Vec<Vec<f32>>) {
    for i in (1..inputs.len()).rev() {
        // invariant: elements with index > i have been locked in place.
        let new_index = gen_index(&mut thread_rng(), i + 1);
        inputs.swap(i, new_index);
        outputs.swap(i, new_index);
    }
}

fn gen_index<R: Rng + ?Sized>(rng: &mut R, ubound: usize) -> usize {
    if ubound <= (core::u32::MAX as usize) {
        rng.gen_range(0..ubound as u32) as usize
    } else {
        rng.gen_range(0..ubound)
    }
}

pub fn read_network(path: impl ToString) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let mut index = 0;
    let mut bytes = fs::read(path.to_string()).unwrap();

    let layer_count = bytes.read_usize();
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        layers.push(bytes.read_usize());
    }

    let biases_len = bytes.read_usize();
    let mut biases = Vec::with_capacity(biases_len);
    for i in 0..biases_len {
        biases.push(bytes.read_f32());
    }

    let weights_len = bytes.read_usize();
    let mut weights = Vec::with_capacity(weights_len);
    for i in 0..weights_len {
        weights.push(bytes.read_f32());
    }

    (layers, biases, weights)
}

pub trait Network {
    fn forward(&mut self, inputs: &Vec<f32>) -> Result<Vec<f32>, String>;
    fn backward(&mut self, target: &Vec<f32>, learn_rate: f32);
    fn train<T: Into<Option<f32>>, F>(&mut self, epochs: u32, target_error: T, inputs: &mut Vec<Vec<f32>>, outputs: &mut Vec<Vec<f32>>, learn_rate: f32, epoch_call_back: F) -> Result<f32, String> where F: FnMut(&mut Self);
    fn error(&mut self, values: &Vec<f32>, target: &Vec<f32>) -> f32;
    fn binary_loss(&mut self, values: &Vec<f32>, target: &Vec<f32>) -> f32;
    fn layers(&self) -> &Vec<(usize, usize, usize)>;
    fn weights_len(&self) -> usize;
    fn biases_len(&self) -> usize;
    fn weights(&self) -> Vec<f32>;
    fn biases(&self) -> Vec<f32>;
    fn layer_bufs(&self) -> Vec<Vec<f32>>;
    fn layer_sensitivities(&self) -> Vec<Vec<f32>>;
    fn out_buf(&self) -> Vec<f32>;
    fn save(&mut self) {
        // let biases = cl_utils::buf_read(&self.gpu_biases);
        // let weights = cl_utils::buf_read(&self.gpu_weights);
        let biases = self.biases();
        let weights = self.weights();
        let mut bytes = Vec::with_capacity(biases.len() * 4 + weights.len() * 4);

        bytes.write_usize(self.layers().len());
        for (layer_size, _, _) in self.layers() {
            bytes.write_usize(*layer_size);
        }

        bytes.write_usize(biases.len());
        for v in biases {
            bytes.write_f32(v);
        }
        bytes.write_usize(weights.len());
        for v in weights {
            bytes.write_f32(v);
        }
        fs::write(format!("{}.net", UNIX_EPOCH.elapsed().unwrap().as_millis()), bytes);
    }
}

trait UsizeIO {
    fn write_usize(&mut self, v: usize);
    fn read_usize(&mut self) -> usize;
}

impl UsizeIO for Vec<u8> {
    fn write_usize(&mut self, v: usize) {
        v.to_be_bytes().iter().for_each(|b| self.push(*b));
    }

    fn read_usize(&mut self) -> usize {
        let mut biases_len: [u8; 8] = [0u8; 8];
        let mut i = 0;
        for byte in self.drain(0..8) {
            biases_len[i] = byte;
            i += 1;
        }
        usize::from_be_bytes(biases_len)
    }
}

trait F32IO {
    fn write_f32(&mut self, v: f32);
    fn read_f32(&mut self) -> f32;
}

impl F32IO for Vec<u8> {
    fn write_f32(&mut self, v: f32) {
        v.to_be_bytes().iter().for_each(|b| self.push(*b));
    }

    fn read_f32(&mut self) -> f32 {
        let mut biases_len: [u8; 4] = [0u8; 4];
        let mut i = 0;
        for byte in self.drain(0..4) {
            biases_len[i] = byte;
            i += 1;
        }
        f32::from_be_bytes(biases_len)
    }
}