#version 450

// Generic coefficient-to-float staging. The CPU side computes the final per-subband
// scale from QCD/QCC, component precision, transform type, and guard bits.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

struct DequantStep {
    float scale;
};

layout(std430, set = 0, binding = 0) readonly buffer Coefficients {
    int coeffs[];
};

layout(std430, set = 0, binding = 1) readonly buffer SampleStepIndices {
    uint step_indices[];
};

layout(std430, set = 0, binding = 2) readonly buffer DequantSteps {
    DequantStep steps[];
};

layout(std430, set = 0, binding = 3) writeonly buffer Dequantized {
    float samples[];
};

layout(push_constant) uniform PushConstants {
    uint sample_count;
} pc;

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= pc.sample_count) {
        return;
    }

    uint step_index = step_indices[index];
    samples[index] = float(coeffs[index]) * steps[step_index].scale;
}
