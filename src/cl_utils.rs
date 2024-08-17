use num_format::Locale::si;
use ocl::{Buffer, Kernel, ProQue, SpatialDims};
use ocl::enums::WriteSrc;
use ocl::traits::WorkDims;
use rand::{random, Rng};

pub fn new_buffer<T: ocl::OclPrm>(pro_que: &ProQue, size: usize) -> Buffer<T> {
    new_buffer_f(pro_que, size, T::default())
}

pub fn new_buffer_f<T: ocl::OclPrm>(pro_que: &ProQue, size: usize, fill: T) -> Buffer<T> {
    Buffer::builder()
        .queue(pro_que.queue().clone())
        .len(size)
        .fill_val(fill)
        .build()
        .expect("Failed to create buffer")
}

pub fn buf_read<T: ocl::OclPrm>(buffer: &Buffer<T>) -> Vec<T> {
    let mut target = vec![T::default(); buffer.len()];
    buffer.read(&mut target).enq().expect("Failed to read buffer");
    target
}

pub fn buf_write<T: ocl::OclPrm, W:  for<'a> Into<WriteSrc<'a, T>>>(buffer: &Buffer<T>, values: W) {
    buffer.write(values).enq().expect("Failed to write network inputs");
}

pub fn randomize_buffer(buffer: &Buffer<f32>, max_work_size: u32, div: f32, pro_que: &ProQue) {
    let rnd_kernel = pro_que
        .kernel_builder("random_buf")
        .arg(buffer)
        .arg(random::<u64>())
        .arg(div)
        .build()
        .expect("Failed to build rnd_kernel");

    unsafe {
        rnd_kernel
            .cmd()
            .global_work_size(buffer.len())
            .local_work_size(calculate_worksize(max_work_size as usize, buffer.len()))
            .enq()
            .expect("Failed to enq rnd_kernel")
    }
}

pub fn calculate_worksize(max: usize, size: usize) -> usize {
    let mut calc = 1;
    for i in (1..max+1).rev() {
        if (size as f32 / i as f32) % 1.0 == 0.0 {
            calc = i;
            break;
        }
    }
    calc
}

pub unsafe fn execute_kernel<SD: Into<SpatialDims>>(pro_que: &ProQue, kernel: &Kernel, size: SD) {
    let max_wg = pro_que.max_wg_size().expect("Failed to get max workgroup size");
    let size = size.into();
    let sizes = size.to_work_size().expect("Failed to convert SpatialDims to work sizes");
    let wg_size = match size.dim_count() {
        1 => {
            SpatialDims::One(calculate_worksize(max_wg, sizes[0]))
        },
        2 => {
            let max_wg = (max_wg as f32).sqrt() as usize;
            SpatialDims::Two(calculate_worksize(max_wg, sizes[0]), calculate_worksize(max_wg, sizes[1]))
        }
        3 => {
            let max_wg = (max_wg as f32).cbrt() as usize;
            SpatialDims::Three(calculate_worksize(max_wg, sizes[0]), calculate_worksize(max_wg, sizes[1]), calculate_worksize(max_wg, sizes[2]))
        }
        _ => SpatialDims::Unspecified
    };
    kernel
        .cmd()
        .global_work_size(size)
        .local_work_size(wg_size)
        .enq()
        .expect("Failed to enqueue activation kernel");
}