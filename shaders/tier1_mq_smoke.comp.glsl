#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

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
    uint output_words[];
};

layout(push_constant) uniform PushConstants {
    uint codeblock_index;
    uint bit_count;
    uint context_index;
} pc;

const uint MQ_CONTEXT_COUNT = 19u;
const uint MQ_QE[47] = uint[](
    0x5601u, 0x3401u, 0x1801u, 0x0ac1u, 0x0521u, 0x0221u,
    0x5601u, 0x5401u, 0x4801u, 0x3801u, 0x3001u, 0x2401u,
    0x1c01u, 0x1601u, 0x5601u, 0x5401u, 0x5101u, 0x4801u,
    0x3801u, 0x3401u, 0x3001u, 0x2801u, 0x2401u, 0x2201u,
    0x1c01u, 0x1801u, 0x1601u, 0x1401u, 0x1201u, 0x1101u,
    0x0ac1u, 0x09c1u, 0x08a1u, 0x0521u, 0x0441u, 0x02a1u,
    0x0221u, 0x0141u, 0x0111u, 0x0085u, 0x0049u, 0x0025u,
    0x0015u, 0x0009u, 0x0005u, 0x0001u, 0x5601u
);
const uint MQ_NEXT_MPS[47] = uint[](
    1u, 2u, 3u, 4u, 5u, 38u, 7u, 8u, 9u, 10u, 11u, 12u,
    13u, 29u, 15u, 16u, 17u, 18u, 19u, 20u, 21u, 22u,
    23u, 24u, 25u, 26u, 27u, 28u, 29u, 30u, 31u, 32u,
    33u, 34u, 35u, 36u, 37u, 38u, 39u, 40u, 41u, 42u,
    43u, 44u, 45u, 45u, 46u
);
const uint MQ_NEXT_LPS[47] = uint[](
    1u, 6u, 9u, 12u, 29u, 33u, 6u, 14u, 14u, 14u, 17u, 18u,
    20u, 21u, 14u, 14u, 15u, 16u, 17u, 18u, 19u, 19u,
    20u, 21u, 22u, 23u, 24u, 25u, 26u, 27u, 28u, 29u,
    30u, 31u, 32u, 33u, 34u, 35u, 36u, 37u, 38u, 39u,
    40u, 41u, 42u, 43u, 46u
);
const uint MQ_SWITCH_MPS[47] = uint[](
    1u, 0u, 0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u,
    0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
);

uint context_state[19];
uint context_mps[19];
Tier1Codeblock codeblock;
uint byte_pos;
uint c_reg;
uint a_reg;
uint ct_reg;

uint byte_at(uint index) {
    if (index >= codeblock.byte_len) {
        return 0xffu;
    }
    uint absolute = codeblock.byte_offset + index;
    uint word = words[absolute >> 2];
    return (word >> ((absolute & 3u) * 8u)) & 0xffu;
}

void byte_in() {
    uint current = byte_at(byte_pos);
    uint next = byte_at(byte_pos + 1u);
    if (current == 0xffu) {
        if (next > 0x8fu) {
            c_reg += 0xff00u;
            ct_reg = 8u;
        } else {
            byte_pos += 1u;
            c_reg += next << 9u;
            ct_reg = 7u;
        }
    } else {
        byte_pos += 1u;
        c_reg += next << 8u;
        ct_reg = 8u;
    }
}

void renormalize_once() {
    if (ct_reg == 0u) {
        byte_in();
    }
    a_reg <<= 1u;
    c_reg <<= 1u;
    ct_reg -= 1u;
}

void renormalize() {
    while (a_reg < 0x8000u) {
        renormalize_once();
    }
}

uint decode_bit(uint context_index) {
    uint state = context_state[context_index];
    uint mps = context_mps[context_index];
    uint qe = MQ_QE[state];
    a_reg -= qe;

    uint decoded;
    if ((c_reg >> 16u) < qe) {
        if (a_reg < qe) {
            a_reg = qe;
            context_state[context_index] = MQ_NEXT_MPS[state];
            decoded = mps;
        } else {
            a_reg = qe;
            if (MQ_SWITCH_MPS[state] != 0u) {
                context_mps[context_index] = 1u - mps;
            }
            context_state[context_index] = MQ_NEXT_LPS[state];
            decoded = 1u - mps;
        }
        renormalize();
    } else {
        c_reg -= qe << 16u;
        if ((a_reg & 0x8000u) == 0u) {
            if (a_reg < qe) {
                if (MQ_SWITCH_MPS[state] != 0u) {
                    context_mps[context_index] = 1u - mps;
                }
                context_state[context_index] = MQ_NEXT_LPS[state];
                decoded = 1u - mps;
            } else {
                context_state[context_index] = MQ_NEXT_MPS[state];
                decoded = mps;
            }
            renormalize();
        } else {
            decoded = mps;
        }
    }
    return decoded;
}

void main() {
    codeblock = codeblocks[pc.codeblock_index];
    for (uint i = 0u; i < MQ_CONTEXT_COUNT; i++) {
        context_state[i] = 0u;
        context_mps[i] = 0u;
    }
    context_state[0] = 4u;
    context_state[17] = 3u;
    context_state[18] = 46u;

    byte_pos = 0u;
    c_reg = byte_at(0u) << 16u;
    ct_reg = 0u;
    byte_in();
    c_reg <<= 7u;
    ct_reg -= 7u;
    a_reg = 0x8000u;

    uint context_index = min(pc.context_index, MQ_CONTEXT_COUNT - 1u);
    uint bits = 0u;
    uint count = min(pc.bit_count, 32u);
    for (uint i = 0u; i < count; i++) {
        bits = (bits << 1u) | decode_bit(context_index);
    }

    output_words[0] = bits;
    output_words[1] = byte_pos;
    output_words[2] = a_reg;
    output_words[3] = c_reg;
    output_words[4] = ct_reg;
    output_words[5] = context_state[context_index] | (context_mps[context_index] << 8u);
}
