#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, set = 0, binding = 0) readonly buffer Rgb48Input {
    uint data[];
} rgb48_buf;

layout(rgba8, set = 0, binding = 1) writeonly uniform image2D output_image;

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
    uint transfer_func;
    uint _pad;
    float m0, m1, m2, m3, m4, m5, m6, m7, m8;
    float off0, off1, off2;
    float input_scale;
} pc;

float linear_to_srgb(float c) {
    if (c <= 0.0031308)
        return c * 12.92;
    else
        return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

uint read_u16(uint sample_index) {
    uint word_index = sample_index >> 1;
    uint word = rgb48_buf.data[word_index];
    return ((sample_index & 1u) == 0u) ? (word & 0xFFFFu) : (word >> 16);
}

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= pc.width || y >= pc.height) return;

    uint pixel_index = y * pc.width + x;
    uint sample_base = pixel_index * 3u;

    float r = float(read_u16(sample_base));
    float g = float(read_u16(sample_base + 1u));
    float b = float(read_u16(sample_base + 2u));

    vec3 linear = (vec3(r, g, b) + vec3(pc.off0, pc.off1, pc.off2)) * pc.input_scale;
    vec3 col;
    col.r = pc.m0 * linear.r + pc.m1 * linear.g + pc.m2 * linear.b;
    col.g = pc.m3 * linear.r + pc.m4 * linear.g + pc.m5 * linear.b;
    col.b = pc.m6 * linear.r + pc.m7 * linear.g + pc.m8 * linear.b;
    col = clamp(col, 0.0, 1.0);

    vec3 out_color;
    if (pc.transfer_func == 1u) {
        out_color = vec3(linear_to_srgb(col.r), linear_to_srgb(col.g), linear_to_srgb(col.b));
    } else if (pc.transfer_func == 2u) {
        // DCI: input is gamma 2.6 encoded XYZ, decode then re-encode to sRGB
        vec3 linear_col = vec3(pow(col.r, 2.6), pow(col.g, 2.6), pow(col.b, 2.6));
        out_color = vec3(linear_to_srgb(linear_col.r), linear_to_srgb(linear_col.g), linear_to_srgb(linear_col.b));
    } else {
        out_color = col;
    }

    imageStore(output_image, ivec2(x, y), vec4(out_color, 1.0));
}
