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
    return 1 / (1 + exp(-val)) * 2 - 1;
}

__kernel void forward(int apply_activations_in, ulong input_length, ulong layer_len, ulong weights_offset, ulong biases_offset, __constant double* weights, __constant double* biases, __constant double* input, __global double* output) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    double in = input[y];
    if (apply_activations_in == 1) {
        in = sigmoid(in);
    }
    int w_ind =(input_length*x)+y + weights_offset;
    double value = input[y]*weights[w_ind];
//    printf("x %zu y %zu wv %f bi %zu bv %f wi %zu v %f\n", x, y, weights[w_ind], (uint) (y+biases_offset), biases[x+biases_offset], w_ind, value);
//    printf("in %f activated %zu\n", in, apply_activations_in);
    atomicAdd_g_f(&output[x], value);
//    if (x % 1000 == 0 && y % 100 == 0) {
//        printf("w %lu b %lu wo %lu bo %lu %f\n", w_ind, x+biases_offset, weights_offset, biases_offset, weights[w_ind]);
//    }
//    barrier(CLK_GLOBAL_MEM_FENCE);
    if (y == 0) {
        atomicAdd_g_f(&output[x], biases[x+biases_offset]);
    }
}

__kernel void set_biases(__global double* buffer, __constant double* biases, int offset) {
    int i = get_global_id(0);
    buffer[i] = biases[i+offset];
}

__kernel void random_buf(__global double* buffer, ulong randoms) {
    int i = get_global_id(0);
    ulong result = (((randoms + i*0xFF9D2D) * 0x5DEECE66DL + 0xBL) & ((1L << 48) -1)) >> 16;
    float res = ((float) result) / 4294967295.0;
    res = (res * 2.0 - 1.0) / 50.0; // This value division depends on the network size. If it's a big network it must be smaller, smaller network it must be larger.
    buffer[i] = res;
}

__kernel void activation(__global double* values, __global double* target) {
    int i = get_global_id(0);
    target[i] = sigmoid(values[i]);
}

__kernel void cost(__constant double* values, __constant double* target, __global double* output) {
    int i = get_global_id(0);
    atomicAdd_g_f(&output[0], pow(values[i]-target[i], 2.0));
//    error[i] = target[i]-values[i];
}

double error_derivative(double actual, double desired) {
    return 2.0 * (actual - desired);
}

double sigmoid_derivative(double value) {
    return (2.0*exp(value)) / pow(exp(value) + 1.0, 2.0);
//return 1.0;
}

__kernel void backward(ulong input_length, ulong weights_offset, ulong biases_offset, double learn_rate, __global double* inputs, __global double* layer_output, __global double* sensitivities, __global double* weights, __global double* biases, __global double* gradients_out) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    ulong weight_index = (input_length*x)+y + weights_offset;
    ulong bias_index = x+biases_offset;
    double gradient = sigmoid_derivative(layer_output[x]) * sensitivities[x];
    atomicAdd_g_f(&gradients_out[y], weights[weight_index]*gradient);
    double new_weight = weights[weight_index] - learn_rate * sigmoid(inputs[y]) * gradient;
    if (!isnan(new_weight)) {
        weights[weight_index] = new_weight;
    } else {
        weights[weight_index] = 0.0;
    }

    if (y == 0) {
        double new_bias = biases[bias_index] - learn_rate * gradient;
        if (!isnan(new_bias)) {
            biases[bias_index] = new_bias;
        } else {
            biases[bias_index] = 0.0;
        }
    }
}

__kernel void multiply(__global double* first, __global double* second, __global double* target) {
    int index = get_global_id(0);
    target[index] = first[index] * second[index];
}

__kernel void multiply_single(__global double* first, double second, __global double* target) {
     int index = get_global_id(0);
     target[index] = first[index] * second;
 }

__kernel void flat_combine_matrix(__global double* matrix, __global double* out, int x_len) {
    int x = get_global_id(0);
    atomicAdd_g_f(&out[x], matrix[(x_len*get_global_id(1))+x]);
}

__kernel void list_divide_inplace(__global double* top, double bottom) {
    int i = get_global_id(0);
    top[i] = top[i]/bottom;
}

__kernel void activate_and_error_derivative_calc(__global double* values, __global double* desired, __global double* out) {
    int i = get_global_id(0);
    out[i] = error_derivative(sigmoid(values[i]), desired[i]);
}

// gradient = input * error_derivative(actual_output - desired_output)
// new_weight = original_weight - learn_rate * gradient