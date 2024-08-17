use std::collections::hash_map::Values;
use std::fmt::Pointer;
use ocl::{Buffer, ProQue, SpatialDims};
use crate::cl_utils;

pub fn mult(proque: &ProQue, first_offset: usize, first: &Buffer<f32>, second: &Buffer<f32>, target: &Buffer<f32>) {
    let max_wg = proque.max_wg_size().expect("Failed to get max workgroup size");

    let mult_kernel = proque
        .kernel_builder("multiply")
        .arg(first)
        .arg(second)
        .arg(target)
        .build()
        .expect("Failed to build multiply kernel");

    let work_size = cl_utils::calculate_worksize(max_wg, first.len());
    unsafe {
        mult_kernel
            .cmd()
            .global_work_offset(first_offset)
            .global_work_size(SpatialDims::from(first.len()))
            .local_work_size(SpatialDims::from(work_size))
            .enq().unwrap();
    }
}

pub fn mult_single(proque: &ProQue, first_offset: usize, first: &Buffer<f32>, second: f32, target: &Buffer<f32>) {
    let max_wg = proque.max_wg_size().expect("Failed to get max workgroup size");

    let mult_kernel = proque
        .kernel_builder("multiply_single")
        .arg(first)
        .arg(second)
        .arg(target)
        .build()
        .expect("Failed to build multiply kernel");

    let work_size = cl_utils::calculate_worksize(max_wg, first.len());
    unsafe {
        mult_kernel
            .cmd()
            .global_work_offset(first_offset)
            .global_work_size(SpatialDims::from(first.len()))
            .local_work_size(SpatialDims::from(work_size))
            .enq().unwrap();
    }
}

pub fn mtrx_combine_columns(proque: &ProQue, matrix: Buffer<f32>, x_len: i32, y_len: i32) -> Buffer<f32> {
    let max_wg = proque.max_wg_size().expect("Failed to get max workgroup size");
    let out: Buffer<f32> = Buffer::builder()
        .queue(proque.queue().clone())
        .len(x_len)
        .build()
        .expect("Failed to build output buffer");

    let kernel = proque
        .kernel_builder("flat_combine_matrix")
        .arg(&matrix)
        .arg(&out)
        .arg(x_len)
        .build()
        .expect("Failed to build kernel");

    unsafe {
        kernel
            .cmd()
            .global_work_size(SpatialDims::from((x_len, y_len)))
            .local_work_size(SpatialDims::from((cl_utils::calculate_worksize((max_wg as f32).sqrt() as usize, x_len as usize), cl_utils::calculate_worksize((max_wg as f32).sqrt() as usize, x_len as usize))))
            .enq()
            .expect("Failed to enq kernel")
    }

    out
}

pub fn load_buffer(buf: &Buffer<f32>) -> Vec<f32> {
    let mut val = vec![0.0; buf.len()];
    buf.read(&mut val).enq().expect("Failed to read buffer");
    val
}

pub fn activate_and_error_derivative(pro_que: &ProQue, values: &Buffer<f32>, target: &Buffer<f32>, out: &Buffer<f32>) {
    let max_wg = pro_que.max_wg_size().expect("Failed to get max workgroup size");

    let kernel = pro_que
        .kernel_builder("activate_and_error_derivative_calc")
        .arg(values)
        .arg(target)
        .arg(out)
        .build()
        .expect("Failed to build kernel");

    unsafe {
        kernel
            .cmd()
            .global_work_size(SpatialDims::from(values.len()))
            .local_work_size(SpatialDims::from(cl_utils::calculate_worksize(max_wg, values.len())))
            .enq()
            .expect("Failed to enq kernel")
    }
}