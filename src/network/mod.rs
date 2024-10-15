#![allow(unused)]

pub mod gpu;
pub mod cpu;

use std::thread;
use std::time::{Duration, Instant};
use num_format::Locale::{te, wo};

use ocl::{Buffer, OclPrm, ProQue, SpatialDims};
use ocl::enums::WriteSrc;
use rand::{random, Rng, thread_rng};
use crate::{cl_utils, gpu_math};
use crate::cl_utils::{calc_ws, execute_kernel};
use num_format::{Locale, ToFormattedString};
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
}