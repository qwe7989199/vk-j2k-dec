#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

struct Tier1Codeblock {
    uint component;
    uint resolution;
    uint subband;
    uint codeblock_x;
    uint codeblock_y;
    uint width;
    uint height;
    uint magnitude_bitplanes;
    uint zero_bit_planes;
    uint byte_offset;
    uint byte_len;
    uint coefficient_offset;
    uint coefficient_count;
    uint pass_offset;
    uint pass_count;
    uint padding;
};

layout(std430, set = 0, binding = 0) readonly buffer Codeblocks {
    Tier1Codeblock codeblocks[];
};

layout(std430, set = 0, binding = 1) readonly buffer CompressedWords {
    uint words[];
};

layout(std430, set = 0, binding = 2) writeonly buffer Output {
    uvec4 summaries[];
};

layout(push_constant) uniform PushConstants {
    uint codeblock_count;
} pc;

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= pc.codeblock_count) {
        return;
    }

    Tier1Codeblock c = codeblocks[index];
    uint byte_sum = 0;
    for (uint byte_index = 0; byte_index < c.byte_len; byte_index++) {
        uint absolute = c.byte_offset + byte_index;
        uint word = words[absolute >> 2];
        byte_sum += (word >> ((absolute & 3) * 8)) & 0xff;
    }

    summaries[index] = uvec4(
        c.byte_offset + c.byte_len,
        c.coefficient_offset + c.coefficient_count,
        c.pass_offset + c.pass_count,
        byte_sum
    );
}
