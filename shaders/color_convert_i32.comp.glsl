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
    uint output_shift;
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
    uint native = uint(clamp(shifted, 0, int(pc.output_max)));
    return native << pc.output_shift;
}

void rgb_pixel_at(uint index, out uint r_out, out uint g_out, out uint b_out) {
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

    r_out = clamp_to_u16(r);
    g_out = clamp_to_u16(g);
    b_out = clamp_to_u16(b);
}

void main() {
    uint pair_index = gl_GlobalInvocationID.x;
    uint pixel0 = pair_index * 2u;
    if (pixel0 >= pc.sample_count) {
        return;
    }

    uint r0;
    uint g0;
    uint b0;
    rgb_pixel_at(pixel0, r0, g0, b0);

    uint r1 = 0u;
    uint g1 = 0u;
    uint b1 = 0u;
    uint pixel1 = pixel0 + 1u;
    if (pixel1 < pc.sample_count) {
        rgb_pixel_at(pixel1, r1, g1, b1);
    }

    uint output_base = pair_index * 3u;
    rgb16_packed[output_base] = r0 | (g0 << 16u);
    rgb16_packed[output_base + 1u] = b0 | (r1 << 16u);
    rgb16_packed[output_base + 2u] = g1 | (b1 << 16u);
}
