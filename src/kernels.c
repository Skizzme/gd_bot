void atomicAdd_g_f(volatile __global double *addr, double val) {
    union {
        ulong u64;
        double f64;
    } next, expected, current;
    current.f64 = *addr;
    do {
        expected.f64 = current.f64;
        next.f64 = expected.f64 + val;
        current.u64 = atom_cmpxchg((volatile __global ulong *)addr, expected.u64, next.u64);
    } while( current.u64 != expected.u64 );
}

double sigmoid(double val) {
    return 1 / (1 + exp(-val));
}

__kernel void forward(ulong input_length, ulong layer_len, ulong weights_offset, ulong biases_offset, __global int* counter, __constant double* weights, __constant double* biases, __constant double* input, __global double* output) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    atomicAdd_g_f(&output[x], input[y]*weights[(input_length*x)+y + weights_offset]);
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (y == 0) {
        atomicAdd_g_f(&output[x], biases[y]);
    }
//    int added = atom_add(counter, 1);
//    int comp = input_length*layer_len-layer_len;
//    if (added >= comp) {
////        printf("x %d y %d la %d in %d", x, y, layer_len, input_length);
//        output[x] = sigmoid(output[x]);
//    }
}

__kernel void set_biases(__global double* buffer, __constant double* biases, int offset) {
    int i = get_global_id(0);
    buffer[i] = biases[i+offset];
}

__kernel void random_buf(__global double* buffer, ulong randoms) {
    int i = get_global_id(0);
    ulong result = (((randoms + i*0xFF9D2D) * 0x5DEECE66DL + 0xBL) & ((1L << 48) -1)) >> 16;
    float res = ((float) result) / 4294967295.0;
    res = (res - 0.5) / 5.0;
    buffer[i] = res;
}

__kernel void activation(__global double* values) {
    int i = get_global_id(0);
    values[i] = sigmoid(values[i]);
}

__kernel void error(__constant double* values, __constant double* target, __global double* error) {
    int i = get_global_id(0);
    error[i] = target[i]-values[i];
}