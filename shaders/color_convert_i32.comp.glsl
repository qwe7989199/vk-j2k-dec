#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

const uint COLOR_NONE = 0u;
const uint COLOR_RCT = 1u;

layout(std430, set = 0, binding = 0) readonly buffer Component0 {
    int c0[];
};

layout(std430, set = 0, binding = 1) readonly buffer Component1 {
    int c1[];
};

layout(std430, set = 0, binding = 2) readonly buffer Component2 {
    int c2[];
};

layout(std430, set = 0, binding = 3) writeonly buffer OutputRgb16Packed {
    uint rgb16_packed[];
};

layout(push_constant) uniform PushConstants {
    uint sample_count;
    uint color_transform;
    uint output_max;
    float level_shift;
} pc;

int floor_div_pow2(int value, uint shift) {
    int divisor = int(1u << shift);
    if (value >= 0) {
        return value / divisor;
    }
    return -((-value + divisor - 1) / divisor);
}

uint clamp_to_u16(int value) {
    int shifted = value + int(pc.level_shift);
    return uint(clamp(shifted, 0, int(pc.output_max)));
}

uint rgb_sample_at(uint sample_index) {
    uint index = sample_index / 3u;
    uint channel = sample_index - index * 3u;
    int r = c0[index];
    int g = c1[index];
    int b = c2[index];

    if (pc.color_transform == COLOR_RCT) {
        int y = c0[index];
        int db = c1[index];
        int dr = c2[index];
        g = y - floor_div_pow2(db + dr, 2u);
        r = dr + g;
        b = db + g;
    }

    if (channel == 0u) {
        return clamp_to_u16(r);
    }
    if (channel == 1u) {
        return clamp_to_u16(g);
    }
    return clamp_to_u16(b);
}

void main() {
    uint packed_index = gl_GlobalInvocationID.x;
    uint low_sample = packed_index * 2u;
    uint total_samples = pc.sample_count * 3u;
    if (low_sample >= total_samples) {
        return;
    }

    uint low = rgb_sample_at(low_sample);
    uint high = low_sample + 1u < total_samples ? rgb_sample_at(low_sample + 1u) : 0u;
    rgb16_packed[packed_index] = low | (high << 16u);
}
