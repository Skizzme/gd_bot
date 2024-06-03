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
    int w_ind =(input_length*y)+x + weights_offset;
    double value = input[y]*weights[w_ind];
//    printf("x %zu y %zu wv %f bi %zu bv %f wi %zu v %f\n", x, y, weights[w_ind], (uint) (y+biases_offset), biases[x+biases_offset], w_ind, value);
//    printf("in %f activated %zu\n", in, apply_activations_in);
    atomicAdd_g_f(&output[x], value);
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
    res = (res - 0.5) / 5.0;
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

__kernel void backward(int layer, ulong input_length, ulong weights_offset, ulong biases_offset, double learn_rate, __global double* inputs, __global double* layer_output, __global double* layer_target, __global double* weights, __global double* biases, __global double* gradients_out) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int weight_index = (input_length*y)+x + weights_offset;
    int bias_index = x+biases_offset;
    double output = layer_output[x];
    double activated_output = sigmoid(output);
    double target = layer_target[x];
    double input = inputs[y];
    double activated_input = sigmoid(input);
    // sensitivity of unactivated to activated * (sensitivity of cost to activated || sensitivity of layer )
    double gradient = sigmoid_derivative(output) * error_derivative(activated_output, target);
    printf("l %zu x %zu y %zu w %zu b %zu wv %f bv %f o %f ao %f t %f i %f ai %f g %f wd %f bd %f\n", layer, x, y, weight_index, bias_index, weights[weight_index], biases[bias_index], output, activated_output, target, input, activated_input, gradient, -1 * learn_rate * activated_input * gradient, -1 * learn_rate * gradient);
    weights[weight_index] = weights[weight_index] - learn_rate * activated_input * gradient;
    if (y == 0) {
        biases[bias_index] = biases[bias_index] - learn_rate * gradient;
    }
    atomicAdd_g_f(&gradients_out[x], gradient);
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

__kernel void error_derivative(__global double* values, __global double* desired) {
    int i = get_global_id(0);
    values[i] = error_derivative(values[i], desired[i])
}

// gradient = input * error_derivative(actual_output - desired_output)
// new_weight = original_weight - learn_rate * gradient