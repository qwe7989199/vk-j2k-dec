#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer LowBand {
    int low_band[];
};

layout(std430, set = 0, binding = 1) readonly buffer HighBand {
    int high_band[];
};

layout(std430, set = 0, binding = 2) writeonly buffer OutputBand {
    int output_band[];
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
    uint low_offset;
    uint high_offset;
    uint output_offset;
} pc;

int high_at(uint line, uint index) {
    if (pc.high_count == 0u) {
        return 0;
    }
    uint clamped = min(index, pc.high_count - 1u);
    return high_band[pc.high_offset + line * pc.high_line_stride + clamped * pc.high_sample_stride];
}

int updated_low_at(uint line, uint index) {
    int low = low_band[pc.low_offset + line * pc.low_line_stride + index * pc.low_sample_stride];
    int left = index == 0u ? high_at(line, 0u) : high_at(line, index - 1u);
    int right = high_at(line, index);
    return low - ((left + right + 2) >> 2);
}

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint line = gl_GlobalInvocationID.y;
    uint output_count = pc.low_count + pc.high_count;
    if (line >= pc.line_count || x >= output_count) {
        return;
    }

    int value;
    if ((x & 1u) == 0u) {
        value = updated_low_at(line, x >> 1u);
    } else {
        uint high_index = x >> 1u;
        int high = high_band[pc.high_offset + line * pc.high_line_stride + high_index * pc.high_sample_stride];
        int left = updated_low_at(line, high_index);
        int right = updated_low_at(line, min(high_index + 1u, pc.low_count - 1u));
        value = high + ((left + right) >> 1);
    }

    output_band[pc.output_offset + line * pc.output_line_stride + x * pc.output_sample_stride] = value;
}
