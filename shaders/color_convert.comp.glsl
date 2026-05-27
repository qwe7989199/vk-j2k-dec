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
    uint pair_index = gl_GlobalInvocationID.x;
    uint pixel0 = pair_index * 2u;
    if (pixel0 >= pc.sample_count) {
        return;
    }

    float r0, g0, b0;
    {
        float y = c0[pixel0];
        float cb = c1[pixel0];
        float cr = c2[pixel0];
        if (pc.color_transform == COLOR_ICT) {
            r0 = y + 1.402 * cr;
            g0 = y - 0.344136 * cb - 0.714136 * cr;
            b0 = y + 1.772 * cb;
        } else if (pc.color_transform == COLOR_RCT) {
            g0 = y - floor((cb + cr) * 0.25);
            r0 = cr + g0;
            b0 = cb + g0;
        } else {
            r0 = y; g0 = cb; b0 = cr;
        }
    }

    uint r0u = clamp_to_u16(r0 + pc.level_shift);
    uint g0u = clamp_to_u16(g0 + pc.level_shift);
    uint b0u = clamp_to_u16(b0 + pc.level_shift);

    uint r1u = 0u;
    uint g1u = 0u;
    uint b1u = 0u;
    uint pixel1 = pixel0 + 1u;
    if (pixel1 < pc.sample_count) {
        float r1, g1, b1;
        float y = c0[pixel1];
        float cb = c1[pixel1];
        float cr = c2[pixel1];
        if (pc.color_transform == COLOR_ICT) {
            r1 = y + 1.402 * cr;
            g1 = y - 0.344136 * cb - 0.714136 * cr;
            b1 = y + 1.772 * cb;
        } else if (pc.color_transform == COLOR_RCT) {
            g1 = y - floor((cb + cr) * 0.25);
            r1 = cr + g1;
            b1 = cb + g1;
        } else {
            r1 = y; g1 = cb; b1 = cr;
        }
        r1u = clamp_to_u16(r1 + pc.level_shift);
        g1u = clamp_to_u16(g1 + pc.level_shift);
        b1u = clamp_to_u16(b1 + pc.level_shift);
    }

    uint output_base = pair_index * 3u;
    rgb16[output_base] = r0u | (g0u << 16u);
    rgb16[output_base + 1u] = b0u | (r1u << 16u);
    rgb16[output_base + 2u] = g1u | (b1u << 16u);
}
