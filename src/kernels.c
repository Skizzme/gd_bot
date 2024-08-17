//void atomicAdd_g_f(volatile __global float *addr, float val) {
//    union {
//        uint u32;
//        float f32;
//    } next, expected, current;
//    current.f32 = *addr;
//    do {
//        expected.f32 = current.f32;
//        next.f32 = expected.f32 + val;
//        current.u32 = atom_cmpxchg((volatile __global uint *)addr, expected.u32, next.u32);
//    } while( current.u32 != expected.u32 );
//}
void atomicAdd_g_f(volatile __global float *addr, float val)
{
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg( (volatile __global unsigned int *)addr, expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}

float sigmoid(float val) {
    return val;
//    return 1 / (1 + exp(-val)) * 2 - 1;
}

float sigmoid_derivative(float value) {
//    return (2.0*exp(value)) / pow(exp(value) + 1.0, 2.0);
    return 1.0;
}

__kernel void forward(int apply_activations_in, ulong input_length, ulong layer_len, ulong weights_offset, ulong biases_offset, __constant float* weights, __constant float* biases, __constant float* input, __global float* output) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    float in = input[y];
    if (apply_activations_in == 1) {
        in = sigmoid(in);
    }
    int w_ind =(input_length*x)+y + weights_offset;
    float value = input[y]*weights[w_ind];
    printf("before %f\n", output[x]);
    atomicAdd_g_f(&output[x], 0.2327);
    printf("after %f\n", output[x]);
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

__kernel void set_biases(__global float* buffer, __constant float* biases, int offset) {
    int i = get_global_id(0);
    buffer[i] = biases[i+offset];
}

__kernel void random_buf(__global float* buffer, ulong randoms) {
    int i = get_global_id(0);
    ulong result = (((randoms + i*0xFF9D2D) * 0x5DEECE66DL + 0xBL) & ((1L << 48) -1)) >> 16;
    float res = ((float) result) / 4294967295.0;
    res = (res * 2.0 - 1.0) / 50.0; // This value division depends on the network size. If it's a big network it must be smaller, smaller network it must be larger.
    buffer[i] = res;
}

__kernel void activation(__global float* values, __global float* target) {
    int i = get_global_id(0);
    target[i] = sigmoid(values[i]);
}

__kernel void cost(__constant float* values, __constant float* target, __global float* output) {
    int i = get_global_id(0);
    atomicAdd_g_f(&output[0], pow(values[i]-target[i], 2.0f));
//    error[i] = target[i]-values[i];
}

float error_derivative(float actual, float desired) {
    return 2.0 * (actual - desired);
}

__kernel void backward(ulong input_length, ulong weights_offset, ulong biases_offset, float learn_rate, __global float* inputs, __global float* layer_output, __global float* sensitivities, __global float* weights, __global float* biases, __global float* gradients_out) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    ulong weight_index = (input_length*x)+y + weights_offset;
    ulong bias_index = x+biases_offset;
    float gradient = sigmoid_derivative(layer_output[x]) * sensitivities[x];
    atomicAdd_g_f(&gradients_out[y], weights[weight_index]*gradient);
    float new_weight = weights[weight_index] - learn_rate * sigmoid(inputs[y]) * gradient;
    if (!isnan(new_weight)) {
        weights[weight_index] = new_weight;
    } else {
        weights[weight_index] = 0.0;
    }

    if (y == 0) {
        float new_bias = biases[bias_index] - learn_rate * gradient;
        if (!isnan(new_bias)) {
            biases[bias_index] = new_bias;
        } else {
            biases[bias_index] = 0.0;
        }
    }
}

__kernel void multiply(__global float* first, __global float* second, __global float* target) {
    int index = get_global_id(0);
    target[index] = first[index] * second[index];
}

__kernel void multiply_single(__global float* first, float second, __global float* target) {
     int index = get_global_id(0);
     target[index] = first[index] * second;
 }

__kernel void flat_combine_matrix(__global float* matrix, __global float* out, int x_len) {
    int x = get_global_id(0);
    atomicAdd_g_f(&out[x], matrix[(x_len*get_global_id(1))+x]);
}

__kernel void list_divide_inplace(__global float* top, float bottom) {
    int i = get_global_id(0);
    top[i] = top[i]/bottom;
}

__kernel void activate_and_error_derivative_calc(__global float* values, __global float* desired, __global float* out) {
    int i = get_global_id(0);
    out[i] = error_derivative(sigmoid(values[i]), desired[i]);
}

// gradient = input * error_derivative(actual_output - desired_output)
// new_weight = original_weight - learn_rate * gradient