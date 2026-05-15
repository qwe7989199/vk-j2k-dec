#version 450

// One inverse irreversible 9/7 lifting pass over independent float 1D lines.
// The stride contract matches idwt53_1d.comp.glsl so rows and columns can share it.

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer LowBand {
    float low_band[];
};

layout(std430, set = 0, binding = 1) readonly buffer HighBand {
    float high_band[];
};

layout(std430, set = 0, binding = 2) writeonly buffer OutputBand {
    float output_band[];
};

layout(push_constant) uniform PushConstants {
    uint low_count;
    uint high_count;
    uint line_count;
    uint low_line_stride;
    uint low_sample_stride;
    uint high_line_stride;
    uint high_sample_stride;
    uint output_line_stride;
    uint output_sample_stride;
} pc;

const float DWT97_ALPHA = -1.5861343;
const float DWT97_BETA = -0.05298012;
const float DWT97_GAMMA = 0.8829111;
const float DWT97_DELTA = 0.44350687;
const float DWT97_K = 1.2301741;

float unscaled_high_at(uint line, uint index) {
    if (pc.high_count == 0u) {
        return 0.0;
    }
    uint clamped = min(index, pc.high_count - 1u);
    return high_band[line * pc.high_line_stride + clamped * pc.high_sample_stride] * DWT97_K;
}

float low_unscaled_at(uint line, uint index) {
    return low_band[line * pc.low_line_stride + index * pc.low_sample_stride] / DWT97_K;
}

float after_delta_low_at(uint line, uint index) {
    float low = low_unscaled_at(line, index);
    float left = index == 0u ? unscaled_high_at(line, 0u) : unscaled_high_at(line, index - 1u);
    float right = unscaled_high_at(line, index);
    return low - DWT97_DELTA * (left + right);
}

float after_gamma_high_at(uint line, uint index) {
    if (pc.high_count == 0u) {
        return 0.0;
    }
    uint clamped = min(index, pc.high_count - 1u);
    float high = unscaled_high_at(line, clamped);
    float left = after_delta_low_at(line, clamped);
    float right = after_delta_low_at(line, min(clamped + 1u, pc.low_count - 1u));
    return high - DWT97_GAMMA * (left + right);
}

float after_beta_low_at(uint line, uint index) {
    float low = after_delta_low_at(line, index);
    float left = index == 0u ? after_gamma_high_at(line, 0u) : after_gamma_high_at(line, index - 1u);
    float right = after_gamma_high_at(line, index);
    return low - DWT97_BETA * (left + right);
}

float after_alpha_high_at(uint line, uint index) {
    float high = after_gamma_high_at(line, index);
    float left = after_beta_low_at(line, index);
    float right = after_beta_low_at(line, min(index + 1u, pc.low_count - 1u));
    return high - DWT97_ALPHA * (left + right);
}

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint line = gl_GlobalInvocationID.y;
    uint output_count = pc.low_count + pc.high_count;
    if (line >= pc.line_count || x >= output_count) {
        return;
    }

    float value = (x & 1u) == 0u
        ? after_beta_low_at(line, x >> 1u)
        : after_alpha_high_at(line, x >> 1u);

    output_band[line * pc.output_line_stride + x * pc.output_sample_stride] = value;
}
