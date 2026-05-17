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

layout(std430, set = 0, binding = 3) writeonly buffer OutputRgb16 {
    uint rgb16[];
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

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= pc.sample_count) {
        return;
    }

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

    uint base = index * 3u;
    rgb16[base + 0u] = clamp_to_u16(r);
    rgb16[base + 1u] = clamp_to_u16(g);
    rgb16[base + 2u] = clamp_to_u16(b);
}
