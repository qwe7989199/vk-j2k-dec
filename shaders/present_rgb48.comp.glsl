#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, set = 0, binding = 0) readonly buffer Rgb48Input {
    uint rgb48_packed[];
};

layout(rgba8, set = 0, binding = 1) writeonly uniform image2D output_image;

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
    uint transfer_func; // 0=linear, 1=sRGB gamma, 2=PQ (ST 2084)
    uint _pad;
    mat3 color_matrix;  // row-major 3x3
    vec3 offset;        // pre-matrix offset (level shift)
    float input_scale;  // normalization (e.g. 1.0/4095.0 or 1.0/65535.0)
} pc;

float linear_to_srgb(float c) {
    if (c <= 0.0031308)
        return c * 12.92;
    else
        return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

float linear_to_pq(float c) {
    float y = clamp(c, 0.0, 1.0);
    float ym = pow(y, 0.1593017578125);
    float num = 0.8359375 + 18.8515625 * ym;
    float den = 1.0 + 18.6875 * ym;
    return pow(num / den, 78.84375);
}

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= pc.width || y >= pc.height) return;

    uint pixel_index = y * pc.width + x;
    // RGB48 packed as 2 pixels per 3 u32s (6 u16 samples = 3 u32 words)
    uint pair_index = pixel_index / 2;
    uint pair_offset = pair_index * 3;
    uint word0 = rgb48_packed[pair_offset + 0];
    uint word1 = rgb48_packed[pair_offset + 1];
    uint word2 = rgb48_packed[pair_offset + 2];

    float r, g, b;
    if ((pixel_index & 1u) == 0u) {
        r = float(word0 & 0xFFFFu);
        g = float(word0 >> 16);
        b = float(word1 & 0xFFFFu);
    } else {
        r = float(word1 >> 16);
        g = float(word2 & 0xFFFFu);
        b = float(word2 >> 16);
    }

    vec3 linear = (vec3(r, g, b) + pc.offset) * pc.input_scale;
    linear = pc.color_matrix * linear;
    linear = clamp(linear, 0.0, 1.0);

    vec3 out_color;
    if (pc.transfer_func == 1u) {
        out_color = vec3(linear_to_srgb(linear.r), linear_to_srgb(linear.g), linear_to_srgb(linear.b));
    } else if (pc.transfer_func == 2u) {
        out_color = vec3(linear_to_pq(linear.r), linear_to_pq(linear.g), linear_to_pq(linear.b));
    } else {
        out_color = linear;
    }

    imageStore(output_image, ivec2(x, y), vec4(out_color, 1.0));
}
