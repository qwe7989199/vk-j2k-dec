#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

const uint COLOR_NONE = 0u;
const uint COLOR_RCT = 1u;
const uint COLOR_ICT = 2u;

layout(std430, set = 0, binding = 0) readonly buffer Component0 {
    float c0[];
};

layout(std430, set = 0, binding = 1) readonly buffer Component1 {
    float c1[];
};

layout(std430, set = 0, binding = 2) readonly buffer Component2 {
    float c2[];
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

uint clamp_to_u16(float value) {
    float rounded = floor(value + 0.5);
    return uint(clamp(rounded, 0.0, float(pc.output_max)));
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= pc.sample_count) {
        return;
    }

    float r = c0[index];
    float g = c1[index];
    float b = c2[index];

    if (pc.color_transform == COLOR_RCT) {
        float y = c0[index];
        float db = c1[index];
        float dr = c2[index];
        g = y - floor((db + dr) * 0.25);
        r = dr + g;
        b = db + g;
    } else if (pc.color_transform == COLOR_ICT) {
        float y = c0[index];
        float cb = c1[index];
        float cr = c2[index];
        r = y + 1.402 * cr;
        g = y - 0.344136 * cb - 0.714136 * cr;
        b = y + 1.772 * cb;
    }

    uint base = index * 3u;
    rgb16[base + 0u] = clamp_to_u16(r + pc.level_shift);
    rgb16[base + 1u] = clamp_to_u16(g + pc.level_shift);
    rgb16[base + 2u] = clamp_to_u16(b + pc.level_shift);
}
