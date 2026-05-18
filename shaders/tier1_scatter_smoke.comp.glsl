#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

struct Tier1ScatterRecord {
    uint coefficient_offset;
    uint coefficient_count;
    uint width;
    uint height;
    uint plane_offset;
    uint plane_width;
    uint plane_x0;
    uint plane_y0;
};

layout(std430, set = 0, binding = 0) readonly buffer ScatterRecords {
    Tier1ScatterRecord records[];
};

layout(std430, set = 0, binding = 1) readonly buffer FlatCoefficients {
    int flat_coefficients[];
};

layout(std430, set = 0, binding = 2) buffer PlaneCoefficients {
    int plane_coefficients[];
};

layout(push_constant) uniform PushConstants {
    uint codeblock_count;
} pc;

void main() {
    uint codeblock_index = gl_GlobalInvocationID.y;
    if (codeblock_index >= pc.codeblock_count) {
        return;
    }

    Tier1ScatterRecord record = records[codeblock_index];
    uint coefficient_index = gl_GlobalInvocationID.x;
    if (coefficient_index >= record.coefficient_count) {
        return;
    }

    uint x = coefficient_index % record.width;
    uint y = coefficient_index / record.width;
    uint source_offset = record.coefficient_offset + coefficient_index;
    uint destination_offset = record.plane_offset
        + (record.plane_y0 + y) * record.plane_width
        + record.plane_x0
        + x;
    plane_coefficients[destination_offset] = flat_coefficients[source_offset];
}
