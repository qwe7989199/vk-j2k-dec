#version 450

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

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

layout(std430, set = 0, binding = 3) buffer CoefficientsOut {
    int output_coefficients[];
};

layout(push_constant) uniform PushConstants {
    uint codeblock_count;
} pc;

const uint MAX_COEFFS = 4096u;
const uint MQ_CONTEXT_COUNT = 19u;
const uint STATE_SIGNIFICANT = 1u;
const uint STATE_NEGATIVE = 2u;
const uint STATE_REFINED = 4u;
const uint STATE_VISIT_SHIFT = 8u;
const uint STATE_VISIT_MASK = 0xff00u;
const uint STATE_NEIGHBOR_SHIFT = 16u;
const uint STATE_NEIGHBOR_MASK = 0x0fff0000u;
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

const uvec2 SIGN_LUT[9] = uvec2[](
    uvec2(13u, 1u), // h=-1, v=-1
    uvec2(12u, 1u), // h=-1, v=0
    uvec2(11u, 1u), // h=-1, v=1
    uvec2(10u, 1u), // h=0, v=-1
    uvec2(9u, 0u),  // h=0, v=0
    uvec2(10u, 0u), // h=0, v=1
    uvec2(11u, 0u), // h=1, v=-1
    uvec2(12u, 0u), // h=1, v=0
    uvec2(13u, 0u)  // h=1, v=1
);

uint context_packed[19];
shared uint coefficient_state[4096];
Tier1Codeblock codeblock;
uint byte_pos;
uint c_reg;
uint a_reg;
uint ct_reg;
uint significance_bits;
uint sign_bits;
uint refinement_bits;
uint run_length_bits;
uint uniform_bits;
uint became_significant;
uint visited;
uint decoded_passes;
uint visit_word;

uint index_at(uint x, uint y) {
    return y * codeblock.width + x;
}

uint coeff_offset(uint index) {
    return codeblock.coefficient_offset + index;
}

uint state_at_index(uint index) {
    return coefficient_state[index];
}

bool state_is_significant(uint state) {
    return (state & STATE_SIGNIFICANT) != 0u;
}

bool state_is_negative(uint state) {
    return (state & STATE_NEGATIVE) != 0u;
}

bool state_is_refined(uint state) {
    return (state & STATE_REFINED) != 0u;
}

bool state_is_visited(uint state) {
    return (state & STATE_VISIT_MASK) == visit_word;
}

uvec3 neighbors_from_state(uint state) {
    return uvec3((state >> 16u) & 0xfu, (state >> 20u) & 0xfu, (state >> 24u) & 0xfu);
}

bool is_significant_index(uint index) {
    return state_is_significant(state_at_index(index));
}

bool is_negative_index(uint index) {
    return state_is_negative(state_at_index(index));
}

bool is_refined_index(uint index) {
    return state_is_refined(state_at_index(index));
}

bool is_visited_index(uint index) {
    return state_is_visited(state_at_index(index));
}

void mark_visited_index(uint index) {
    coefficient_state[index] = (coefficient_state[index] & ~STATE_VISIT_MASK)
        | visit_word;
}

uint pack_neighbors(uint horizontal, uint vertical, uint diagonal) {
    return horizontal | (vertical << 4u) | (diagonal << 8u);
}

uvec3 unpack_neighbors(uint packed) {
    return uvec3(packed & 0xfu, (packed >> 4u) & 0xfu, (packed >> 8u) & 0xfu);
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
    uint packed = context_packed[context_index];
    uint state = packed & 0x7fu;
    uint mps = packed >> 7u;
    uint qe = MQ_QE[state];
    a_reg -= qe;
    uint decoded;
    if ((c_reg >> 16u) < qe) {
        if (a_reg < qe) {
            a_reg = qe;
            context_packed[context_index] = MQ_NEXT_MPS[state] | (mps << 7u);
            decoded = mps;
        } else {
            a_reg = qe;
            uint next_mps = mps ^ MQ_SWITCH_MPS[state];
            context_packed[context_index] = MQ_NEXT_LPS[state] | (next_mps << 7u);
            decoded = 1u - mps;
        }
        renormalize();
    } else {
        c_reg -= qe << 16u;
        if ((a_reg & 0x8000u) == 0u) {
            if (a_reg < qe) {
                uint next_mps = mps ^ MQ_SWITCH_MPS[state];
                context_packed[context_index] = MQ_NEXT_LPS[state] | (next_mps << 7u);
                decoded = 1u - mps;
            } else {
                context_packed[context_index] = MQ_NEXT_MPS[state] | (mps << 7u);
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
    return is_significant_index(index_at(uint(x), uint(y))) ? 1u : 0u;
}

uint sign_at(int x, int y) {
    if (x < 0 || y < 0 || x >= int(codeblock.width) || y >= int(codeblock.height)) {
        return 0u;
    }
    return is_negative_index(index_at(uint(x), uint(y))) ? 1u : 0u;
}

uvec3 neighbor_counts(uint x, uint y) {
    return neighbors_from_state(coefficient_state[index_at(x, y)]);
}

void increment_neighbor(int x, int y, uint bit_shift) {
    if (x < 0 || y < 0 || x >= int(codeblock.width) || y >= int(codeblock.height)) {
        return;
    }
    coefficient_state[index_at(uint(x), uint(y))] += 1u << (STATE_NEIGHBOR_SHIFT + bit_shift);
}

void update_neighbors_for_significant(uint x, uint y) {
    if (x > 0u && y > 0u && x + 1u < codeblock.width && y + 1u < codeblock.height) {
        uint index = index_at(x, y);
        uint row_stride = codeblock.width;
        coefficient_state[index - 1u] += 1u << STATE_NEIGHBOR_SHIFT;
        coefficient_state[index + 1u] += 1u << STATE_NEIGHBOR_SHIFT;
        coefficient_state[index - row_stride] += 1u << (STATE_NEIGHBOR_SHIFT + 4u);
        coefficient_state[index + row_stride] += 1u << (STATE_NEIGHBOR_SHIFT + 4u);
        coefficient_state[index - row_stride - 1u] += 1u << (STATE_NEIGHBOR_SHIFT + 8u);
        coefficient_state[index - row_stride + 1u] += 1u << (STATE_NEIGHBOR_SHIFT + 8u);
        coefficient_state[index + row_stride - 1u] += 1u << (STATE_NEIGHBOR_SHIFT + 8u);
        coefficient_state[index + row_stride + 1u] += 1u << (STATE_NEIGHBOR_SHIFT + 8u);
        return;
    }
    int ix = int(x);
    int iy = int(y);
    increment_neighbor(ix - 1, iy, 0u);
    increment_neighbor(ix + 1, iy, 0u);
    increment_neighbor(ix, iy - 1, 4u);
    increment_neighbor(ix, iy + 1, 4u);
    increment_neighbor(ix - 1, iy - 1, 8u);
    increment_neighbor(ix + 1, iy - 1, 8u);
    increment_neighbor(ix - 1, iy + 1, 8u);
    increment_neighbor(ix + 1, iy + 1, 8u);
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

uint significance_context_from_neighbors(uvec3 n) {
    if (codeblock.subband == 1u) {
        return lh_or_ll_context(n.y, n.x, n.z);
    }
    if (codeblock.subband == 3u) {
        return hh_context(n);
    }
    return lh_or_ll_context(n.x, n.y, n.z);
}

uint significance_context(uint x, uint y) {
    return significance_context_from_neighbors(neighbor_counts(x, y));
}

uvec2 sign_context(uint x, uint y) {
    uint state_left   = (x > 0u) ? state_at_index(index_at(x - 1u, y)) : 0u;
    uint state_right  = (x + 1u < codeblock.width) ? state_at_index(index_at(x + 1u, y)) : 0u;
    uint state_top    = (y > 0u) ? state_at_index(index_at(x, y - 1u)) : 0u;
    uint state_bottom = (y + 1u < codeblock.height) ? state_at_index(index_at(x, y + 1u)) : 0u;

    uint has_pos_h = (((state_left & 3u) == 1u) || ((state_right & 3u) == 1u)) ? 1u : 0u;
    uint has_neg_h = (((state_left & 3u) == 3u) || ((state_right & 3u) == 3u)) ? 1u : 0u;
    int contrib_h = int(has_pos_h) - int(has_neg_h);

    uint has_pos_v = (((state_top & 3u) == 1u) || ((state_bottom & 3u) == 1u)) ? 1u : 0u;
    uint has_neg_v = (((state_top & 3u) == 3u) || ((state_bottom & 3u) == 3u)) ? 1u : 0u;
    int contrib_v = int(has_pos_v) - int(has_neg_v);

    uint index = uint((contrib_h + 1) * 3 + (contrib_v + 1));
    return SIGN_LUT[index];
}

bool cleanup_run_length_candidate(uint x, uint stripe_y) {
    for (uint y_offset = 0u; y_offset < 4u; y_offset++) {
        uint y = stripe_y + y_offset;
        uint state = state_at_index(index_at(x, y));
        if (state_is_visited(state) || state_is_significant(state) || (state & STATE_NEIGHBOR_MASK) != 0u) {
            return false;
        }
    }
    return true;
}

void decode_significant_sample(uint x, uint y, uint bitplane, bool mark_visit) {
    uint index = index_at(x, y);
    uvec2 sc = sign_context(x, y);
    uint sign_bit = decode_bit(sc.x) ^ sc.y;
    int magnitude = int(1u << bitplane);
    output_coefficients[coeff_offset(index)] = sign_bit != 0u ? -magnitude : magnitude;
    update_neighbors_for_significant(x, y);
    coefficient_state[index] = (coefficient_state[index] & STATE_NEIGHBOR_MASK)
        | STATE_SIGNIFICANT
        | (sign_bit != 0u ? STATE_NEGATIVE : 0u);
    if (mark_visit) {
        mark_visited_index(index);
    }
    sign_bits += 1u;
    significance_bits += 1u;
    became_significant += 1u;
    visited += 1u;
}

void decode_cleanup_sample(uint x, uint y, uint bitplane) {
    uint index = index_at(x, y);
    uint state = state_at_index(index);
    if (state_is_visited(state) || state_is_significant(state)) {
        return;
    }
    uint context;
    if ((state & STATE_NEIGHBOR_MASK) == 0u) {
        context = 0u;
    } else {
        context = significance_context_from_neighbors(neighbors_from_state(state));
    }
    uint is_significant = decode_bit(context);
    if (is_significant != 0u) {
        decode_significant_sample(x, y, bitplane, false);
    } else {
        significance_bits += 1u;
        visited += 1u;
    }
}

void clear_bitplane_visits() {
    visit_word += 0x100u;
}

void decode_cleanup_pass(uint bitplane) {
    // Cleanup is the final pass for a bitplane; cleanup-only visits are counted but not reread.
    uint stripe_y = 0u;
    while (stripe_y < codeblock.height) {
        uint stripe_height = min(4u, codeblock.height - stripe_y);
        if (stripe_height == 4u) {
            for (uint x = 0u; x < codeblock.width; x++) {
                if (cleanup_run_length_candidate(x, stripe_y)) {
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
                    decode_significant_sample(x, stripe_y + run, bitplane, false);
                    for (uint y_offset = run + 1u; y_offset < 4u; y_offset++) {
                        decode_cleanup_sample(x, stripe_y + y_offset, bitplane);
                    }
                } else {
                    for (uint y_offset = 0u; y_offset < 4u; y_offset++) {
                        decode_cleanup_sample(x, stripe_y + y_offset, bitplane);
                    }
                }
            }
        } else {
            for (uint x = 0u; x < codeblock.width; x++) {
                for (uint y_offset = 0u; y_offset < stripe_height; y_offset++) {
                    decode_cleanup_sample(x, stripe_y + y_offset, bitplane);
                }
            }
        }
        stripe_y += 4u;
    }
    decoded_passes += 1u;
}

void decode_significance_pass(uint bitplane) {
    uint stripe_y = 0u;
    while (stripe_y < codeblock.height) {
        uint stripe_height = min(4u, codeblock.height - stripe_y);
        if (stripe_height == 4u) {
            for (uint x = 0u; x < codeblock.width; x++) {
                for (uint y_offset = 0u; y_offset < 4u; y_offset++) {
                    uint y = stripe_y + y_offset;
                    uint index = index_at(x, y);
                    uint state = state_at_index(index);
                    if (state_is_visited(state) || state_is_significant(state) || (state & STATE_NEIGHBOR_MASK) == 0u) {
                        continue;
                    }
                    uvec3 n = neighbors_from_state(state);
                    uint is_significant = decode_bit(significance_context_from_neighbors(n));
                    if (is_significant != 0u) {
                        decode_significant_sample(x, y, bitplane, true);
                    } else {
                        mark_visited_index(index);
                        significance_bits += 1u;
                        visited += 1u;
                    }
                }
            }
        } else {
            for (uint x = 0u; x < codeblock.width; x++) {
                for (uint y_offset = 0u; y_offset < stripe_height; y_offset++) {
                    uint y = stripe_y + y_offset;
                    uint index = index_at(x, y);
                    uint state = state_at_index(index);
                    if (state_is_visited(state) || state_is_significant(state) || (state & STATE_NEIGHBOR_MASK) == 0u) {
                        continue;
                    }
                    uvec3 n = neighbors_from_state(state);
                    uint is_significant = decode_bit(significance_context_from_neighbors(n));
                    if (is_significant != 0u) {
                        decode_significant_sample(x, y, bitplane, true);
                    } else {
                        mark_visited_index(index);
                        significance_bits += 1u;
                        visited += 1u;
                    }
                }
            }
        }
        stripe_y += 4u;
    }
    decoded_passes += 1u;
}

void decode_refinement_pass(uint bitplane) {
    uint stripe_y = 0u;
    while (stripe_y < codeblock.height) {
        uint stripe_height = min(4u, codeblock.height - stripe_y);
        if (stripe_height == 4u) {
            for (uint x = 0u; x < codeblock.width; x++) {
                for (uint y_offset = 0u; y_offset < 4u; y_offset++) {
                    uint y = stripe_y + y_offset;
                    uint index = index_at(x, y);
                    uint state = state_at_index(index);
                    if (state_is_visited(state) || !state_is_significant(state)) {
                        continue;
                    }
                    uint context = state_is_refined(state) ? 16u : (((state & STATE_NEIGHBOR_MASK) != 0u) ? 15u : 14u);
                    uint bit = decode_bit(context);
                    int delta = bit != 0u ? int(1u << bitplane) : 0;
                    uint offset = coeff_offset(index);
                    output_coefficients[offset] += state_is_negative(state) ? -delta : delta;
                    coefficient_state[index] = ((state | STATE_REFINED) & ~STATE_VISIT_MASK) | visit_word;
                    refinement_bits += 1u;
                    visited += 1u;
                }
            }
        } else {
            for (uint x = 0u; x < codeblock.width; x++) {
                for (uint y_offset = 0u; y_offset < stripe_height; y_offset++) {
                    uint y = stripe_y + y_offset;
                    uint index = index_at(x, y);
                    uint state = state_at_index(index);
                    if (state_is_visited(state) || !state_is_significant(state)) {
                        continue;
                    }
                    uint context = state_is_refined(state) ? 16u : (((state & STATE_NEIGHBOR_MASK) != 0u) ? 15u : 14u);
                    uint bit = decode_bit(context);
                    int delta = bit != 0u ? int(1u << bitplane) : 0;
                    uint offset = coeff_offset(index);
                    output_coefficients[offset] += state_is_negative(state) ? -delta : delta;
                    coefficient_state[index] = ((state | STATE_REFINED) & ~STATE_VISIT_MASK) | visit_word;
                    refinement_bits += 1u;
                    visited += 1u;
                }
            }
        }
        stripe_y += 4u;
    }
    decoded_passes += 1u;
}

void main() {
    uint codeblock_index = gl_WorkGroupID.x;
    if (codeblock_index >= pc.codeblock_count) {
        return;
    }

    codeblock = codeblocks[codeblock_index];
    
    // Collaborative shared memory zero-initialization
    for (uint i = gl_LocalInvocationID.x; i < codeblock.coefficient_count; i += 32u) {
        coefficient_state[i] = 0u;
    }
    barrier();

    // Only thread 0 performs MQ arithmetic decoding
    if (gl_LocalInvocationID.x == 0u) {
        for (uint i = 0u; i < MQ_CONTEXT_COUNT; i++) {
            context_packed[i] = 0u;
        }
        context_packed[0] = 4u;
        context_packed[17] = 3u;
        context_packed[18] = 46u;

        byte_pos = 0u;
        c_reg = byte_at(0u) << 16u;
        ct_reg = 0u;
        byte_in();
        c_reg <<= 7u;
        ct_reg -= 7u;
        a_reg = 0x8000u;
        significance_bits = 0u;
        sign_bits = 0u;
        refinement_bits = 0u;
        run_length_bits = 0u;
        uniform_bits = 0u;
        became_significant = 0u;
        visited = 0u;
        decoded_passes = 0u;
        visit_word = 0x100u;

        if (codeblock.pass_count > 0u) {
            uint first_bitplane = codeblock.magnitude_bitplanes - codeblock.zero_bit_planes - 1u;
            decode_cleanup_pass(first_bitplane);

            uint pass_index = 1u;
            uint bitplane = first_bitplane - 1u;
            while (pass_index < codeblock.pass_count) {
                clear_bitplane_visits();
                decode_significance_pass(bitplane);
                pass_index += 1u;
                if (pass_index >= codeblock.pass_count) {
                    break;
                }

                decode_refinement_pass(bitplane);
                pass_index += 1u;
                if (pass_index >= codeblock.pass_count) {
                    break;
                }

                decode_cleanup_pass(bitplane);
                pass_index += 1u;
                if (bitplane > 0u) {
                    bitplane -= 1u;
                }
            }
        }

        uint nonzero = 0u;
        uint checksum = 0u;
        for (uint i = 0u; i < codeblock.coefficient_count; i++) {
            uint offset = codeblock.coefficient_offset + i;
            if (is_significant_index(i)) {
                int coefficient = output_coefficients[offset];
                nonzero += 1u;
                checksum += uint(coefficient);
            } else {
                output_coefficients[offset] = 0;
            }
        }
        uint output_offset = codeblock_index * 9u;
        output_words[output_offset + 0u] = nonzero;
        output_words[output_offset + 1u] = checksum;
        output_words[output_offset + 2u] = byte_pos;
        output_words[output_offset + 3u] = decoded_passes;
        output_words[output_offset + 4u] = significance_bits;
        output_words[output_offset + 5u] = sign_bits;
        output_words[output_offset + 6u] = refinement_bits;
        output_words[output_offset + 7u] = became_significant;
        output_words[output_offset + 8u] = visited;
    }
}
