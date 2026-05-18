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
} pc;

const uint MAX_COEFFS = 4096u;
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
uint significant[4096];
uint negative_sign[4096];
int coefficients[4096];
Tier1Codeblock codeblock;
uint byte_pos;
uint c_reg;
uint a_reg;
uint ct_reg;
uint significance_bits;
uint sign_bits;
uint run_length_bits;
uint uniform_bits;
uint became_significant;
uint visited;

uint index_at(uint x, uint y) {
    return y * codeblock.width + x;
}

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

uint sig_at(int x, int y) {
    if (x < 0 || y < 0 || x >= int(codeblock.width) || y >= int(codeblock.height)) {
        return 0u;
    }
    return significant[index_at(uint(x), uint(y))];
}

uint sign_at(int x, int y) {
    if (x < 0 || y < 0 || x >= int(codeblock.width) || y >= int(codeblock.height)) {
        return 0u;
    }
    return negative_sign[index_at(uint(x), uint(y))];
}

uvec3 neighbor_counts(uint x, uint y) {
    int ix = int(x);
    int iy = int(y);
    uint horizontal = sig_at(ix - 1, iy) + sig_at(ix + 1, iy);
    uint vertical = sig_at(ix, iy - 1) + sig_at(ix, iy + 1);
    uint diagonal =
        sig_at(ix - 1, iy - 1) + sig_at(ix + 1, iy - 1) +
        sig_at(ix - 1, iy + 1) + sig_at(ix + 1, iy + 1);
    return uvec3(horizontal, vertical, diagonal);
}

uint lh_or_ll_context(uint primary, uint secondary, uint diagonal) {
    if (primary == 2u) { return 8u; }
    if (primary == 1u && secondary >= 1u) { return 7u; }
    if (primary == 1u && diagonal >= 1u) { return 6u; }
    if (primary == 1u) { return 5u; }
    if (secondary == 2u) { return 4u; }
    if (secondary == 1u) { return 3u; }
    if (diagonal >= 2u) { return 2u; }
    if (diagonal == 1u) { return 1u; }
    return 0u;
}

uint hh_context(uvec3 n) {
    uint hv = n.x + n.y;
    if (n.z >= 3u) { return 8u; }
    if (hv >= 1u && n.z == 2u) { return 7u; }
    if (hv == 0u && n.z == 2u) { return 6u; }
    if (hv >= 2u && n.z == 1u) { return 5u; }
    if (hv == 1u && n.z == 1u) { return 4u; }
    if (hv == 0u && n.z == 1u) { return 3u; }
    if (hv >= 2u) { return 2u; }
    if (hv == 1u) { return 1u; }
    return 0u;
}

uint significance_context(uint x, uint y) {
    uvec3 n = neighbor_counts(x, y);
    if (codeblock.subband == 1u) {
        return lh_or_ll_context(n.y, n.x, n.z);
    }
    if (codeblock.subband == 3u) {
        return hh_context(n);
    }
    return lh_or_ll_context(n.x, n.y, n.z);
}

int sign_contribution(uint x, uint y, int dx0, int dy0, int dx1, int dy1) {
    int ix = int(x);
    int iy = int(y);
    bool has_positive = false;
    bool has_negative = false;
    int xs[2] = int[](ix + dx0, ix + dx1);
    int ys[2] = int[](iy + dy0, iy + dy1);
    for (uint i = 0u; i < 2u; i++) {
        if (sig_at(xs[i], ys[i]) == 0u) {
            continue;
        }
        if (sign_at(xs[i], ys[i]) != 0u) {
            has_negative = true;
        } else {
            has_positive = true;
        }
    }
    if (has_positive && !has_negative) { return 1; }
    if (!has_positive && has_negative) { return -1; }
    return 0;
}

uvec2 sign_context(uint x, uint y) {
    int horizontal = sign_contribution(x, y, -1, 0, 1, 0);
    int vertical = sign_contribution(x, y, 0, -1, 0, 1);
    if (horizontal == 1 && vertical == 1) { return uvec2(13u, 0u); }
    if (horizontal == 1 && vertical == 0) { return uvec2(12u, 0u); }
    if (horizontal == 1 && vertical == -1) { return uvec2(11u, 0u); }
    if (horizontal == 0 && vertical == 1) { return uvec2(10u, 0u); }
    if (horizontal == 0 && vertical == 0) { return uvec2(9u, 0u); }
    if (horizontal == 0 && vertical == -1) { return uvec2(10u, 1u); }
    if (horizontal == -1 && vertical == 1) { return uvec2(11u, 1u); }
    if (horizontal == -1 && vertical == 0) { return uvec2(12u, 1u); }
    return uvec2(13u, 1u);
}

bool cleanup_run_length_candidate(uint x, uint stripe_y) {
    for (uint y_offset = 0u; y_offset < 4u; y_offset++) {
        uint y = stripe_y + y_offset;
        uint index = index_at(x, y);
        uvec3 n = neighbor_counts(x, y);
        if (significant[index] != 0u || (n.x + n.y + n.z) != 0u) {
            return false;
        }
    }
    return true;
}

void decode_significant_sample(uint x, uint y, uint bitplane) {
    uint index = index_at(x, y);
    uvec2 sc = sign_context(x, y);
    uint sign_bit = decode_bit(sc.x) ^ sc.y;
    int magnitude = int(1u << bitplane);
    coefficients[index] = sign_bit != 0u ? -magnitude : magnitude;
    significant[index] = 1u;
    negative_sign[index] = sign_bit;
    sign_bits += 1u;
    significance_bits += 1u;
    became_significant += 1u;
    visited += 1u;
}

void decode_cleanup_sample(uint x, uint y, uint bitplane) {
    uint index = index_at(x, y);
    if (significant[index] != 0u) {
        return;
    }
    uint is_significant = decode_bit(significance_context(x, y));
    if (is_significant != 0u) {
        decode_significant_sample(x, y, bitplane);
    } else {
        significance_bits += 1u;
        visited += 1u;
    }
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
    for (uint i = 0u; i < MAX_COEFFS; i++) {
        significant[i] = 0u;
        negative_sign[i] = 0u;
        coefficients[i] = 0;
    }

    byte_pos = 0u;
    c_reg = byte_at(0u) << 16u;
    ct_reg = 0u;
    byte_in();
    c_reg <<= 7u;
    ct_reg -= 7u;
    a_reg = 0x8000u;
    significance_bits = 0u;
    sign_bits = 0u;
    run_length_bits = 0u;
    uniform_bits = 0u;
    became_significant = 0u;
    visited = 0u;

    uint bitplane = codeblock.magnitude_bitplanes - codeblock.zero_bit_planes - 1u;
    uint stripe_y = 0u;
    while (stripe_y < codeblock.height) {
        uint stripe_height = min(4u, codeblock.height - stripe_y);
        for (uint x = 0u; x < codeblock.width; x++) {
            if (stripe_height == 4u && cleanup_run_length_candidate(x, stripe_y)) {
                uint any_significant = decode_bit(17u);
                run_length_bits += 1u;
                if (any_significant == 0u) {
                    significance_bits += 4u;
                    visited += 4u;
                    continue;
                }
                uint run = (decode_bit(18u) << 1u) | decode_bit(18u);
                uniform_bits += 2u;
                significance_bits += run;
                visited += run;
                decode_significant_sample(x, stripe_y + run, bitplane);
                for (uint y_offset = run + 1u; y_offset < stripe_height; y_offset++) {
                    decode_cleanup_sample(x, stripe_y + y_offset, bitplane);
                }
            } else {
                for (uint y_offset = 0u; y_offset < stripe_height; y_offset++) {
                    decode_cleanup_sample(x, stripe_y + y_offset, bitplane);
                }
            }
        }
        stripe_y += 4u;
    }

    uint nonzero = 0u;
    uint checksum = 0u;
    for (uint i = 0u; i < codeblock.coefficient_count; i++) {
        if (coefficients[i] != 0) {
            nonzero += 1u;
            checksum += uint(coefficients[i]);
        }
    }
    output_words[0] = nonzero;
    output_words[1] = checksum;
    output_words[2] = byte_pos;
    output_words[3] = significance_bits;
    output_words[4] = sign_bits;
    output_words[5] = run_length_bits;
    output_words[6] = uniform_bits;
    output_words[7] = became_significant;
    output_words[8] = visited;
}
