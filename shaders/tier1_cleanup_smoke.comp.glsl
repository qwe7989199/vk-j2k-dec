#version 450

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

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

// Packed MQ table: bits [15:0]=QE, [21:16]=NEXT_MPS, [27:22]=NEXT_LPS, [28]=SWITCH_MPS
const uint MQ_TABLE[47] = uint[](
    0x10415601u, 0x01823401u, 0x02431801u, 0x03040ac1u, 0x07450521u, 0x08660221u,
    0x11875601u, 0x03885401u, 0x03894801u, 0x038a3801u, 0x044b3001u, 0x048c2401u,
    0x050d1c01u, 0x055d1601u, 0x138f5601u, 0x03905401u, 0x03d15101u, 0x04124801u,
    0x04533801u, 0x04943401u, 0x04d53001u, 0x04d62801u, 0x05172401u, 0x05582201u,
    0x05991c01u, 0x05da1801u, 0x061b1601u, 0x065c1401u, 0x069d1201u, 0x06de1101u,
    0x071f0ac1u, 0x076009c1u, 0x07a108a1u, 0x07e20521u, 0x08230441u, 0x086402a1u,
    0x08a50221u, 0x08e60141u, 0x09270111u, 0x09680085u, 0x09a90049u, 0x09ea0025u,
    0x0a2b0015u, 0x0a6c0009u, 0x0aad0005u, 0x0aed0001u, 0x0bae5601u
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

layout(std430, set = 0, binding = 4) buffer CoefficientState {
    uint state_data[];
};

const uint CODEBLOCKS_PER_WORKGROUP = 4u;

shared uint s_context_packed[4 * 19];
shared uint s_reduction[128];

uint warp_id;
uint lane_id;
uint ctx_base;

#define ctx_at(i) s_context_packed[ctx_base + (i)]

Tier1Codeblock codeblock;
uint state_base;
uint byte_pos;
uint c_reg;
uint a_reg;
uint ct_reg;
uint cached_word;
uint cached_word_index;
uint cur_byte;
uint next_byte;
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
    return state_data[state_base + index];
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
    state_data[state_base + index] = (state_data[state_base + index] & ~STATE_VISIT_MASK)
        | visit_word;
}

uint pack_neighbors(uint horizontal, uint vertical, uint diagonal) {
    return horizontal | (vertical << 4u) | (diagonal << 8u);
}

uvec3 unpack_neighbors(uint packed) {
    return uvec3(packed & 0xfu, (packed >> 4u) & 0xfu, (packed >> 8u) & 0xfu);
}

uint byte_at(uint index) {
    uint absolute = codeblock.byte_offset + index;
    uint word_idx = absolute >> 2u;
    uint word;
    if (word_idx == cached_word_index) {
        word = cached_word;
    } else {
        word = words[word_idx];
        cached_word = word;
        cached_word_index = word_idx;
    }
    return (word >> ((absolute & 3u) * 8u)) & 0xffu;
}

void byte_in() {
    if (cur_byte == 0xffu) {
        if (next_byte > 0x8fu) {
            c_reg += 0xff00u;
            ct_reg = 8u;
        } else {
            byte_pos += 1u;
            c_reg += next_byte << 9u;
            ct_reg = 7u;
            cur_byte = next_byte;
            next_byte = byte_at(byte_pos + 1u);
        }
    } else {
        byte_pos += 1u;
        c_reg += next_byte << 8u;
        ct_reg = 8u;
        cur_byte = next_byte;
        next_byte = byte_at(byte_pos + 1u);
    }
}

void renormalize() {
    if (a_reg >= 0x8000u) {
        return;
    }
    uint k = 15u - uint(findMSB(a_reg));
    while (k >= ct_reg) {
        k -= ct_reg;
        a_reg <<= ct_reg;
        c_reg <<= ct_reg;
        byte_in();
    }
    a_reg <<= k;
    c_reg <<= k;
    ct_reg -= k;
}

uint decode_bit(uint context_index) {
    uint ctx_packed = ctx_at(context_index);
    uint state = ctx_packed & 0x7fu;
    uint mps = ctx_packed >> 7u;

    uint mq_entry = MQ_TABLE[state];
    uint qe = mq_entry & 0xffffu;
    uint next_mps_val = (mq_entry >> 16u) & 0x3fu;
    uint next_lps_val = (mq_entry >> 22u) & 0x3fu;
    uint switch_mps_val = (mq_entry >> 28u) & 1u;

    a_reg -= qe;
    bool cond_lps = (c_reg >> 16u) < qe;
    bool choose_mps = (cond_lps == (a_reg < qe));

    uint decoded;
    if (cond_lps) {
        a_reg = qe;
    } else {
        c_reg -= qe << 16u;
    }

    if (cond_lps || (a_reg < 0x8000u)) {
        uint new_state = choose_mps ? next_mps_val : next_lps_val;
        uint new_mps = choose_mps ? mps : (mps ^ switch_mps_val);
        ctx_at(context_index) = new_state | (new_mps << 7u);
        decoded = choose_mps ? mps : (1u - mps);
        renormalize();
    } else {
        decoded = mps;
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
    return neighbors_from_state(state_data[state_base + index_at(x, y)]);
}

void increment_neighbor(int x, int y, uint bit_shift) {
    if (x < 0 || y < 0 || x >= int(codeblock.width) || y >= int(codeblock.height)) {
        return;
    }
    uint idx = state_base + index_at(uint(x), uint(y));
    state_data[idx] += 1u << (STATE_NEIGHBOR_SHIFT + bit_shift);
}

void update_neighbors_for_significant(uint x, uint y) {
    if (x > 0u && y > 0u && x + 1u < codeblock.width && y + 1u < codeblock.height) {
        uint index = state_base + index_at(x, y);
        uint row_stride = codeblock.width;
        state_data[index - 1u] += 1u << STATE_NEIGHBOR_SHIFT;
        state_data[index + 1u] += 1u << STATE_NEIGHBOR_SHIFT;
        state_data[index - row_stride] += 1u << (STATE_NEIGHBOR_SHIFT + 4u);
        state_data[index + row_stride] += 1u << (STATE_NEIGHBOR_SHIFT + 4u);
        state_data[index - row_stride - 1u] += 1u << (STATE_NEIGHBOR_SHIFT + 8u);
        state_data[index - row_stride + 1u] += 1u << (STATE_NEIGHBOR_SHIFT + 8u);
        state_data[index + row_stride - 1u] += 1u << (STATE_NEIGHBOR_SHIFT + 8u);
        state_data[index + row_stride + 1u] += 1u << (STATE_NEIGHBOR_SHIFT + 8u);
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

const uint LH_LL_LUT_FAST[85] = uint[](
    0u, 1u, 2u, 2u, 2u, 0u, 0u, 0u, 3u, 3u, 3u, 3u, 3u, 0u, 0u, 0u,
    4u, 4u, 4u, 4u, 4u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    5u, 6u, 6u, 6u, 6u, 0u, 0u, 0u, 7u, 7u, 7u, 7u, 7u, 0u, 0u, 0u,
    7u, 7u, 7u, 7u, 7u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    8u, 8u, 8u, 8u, 8u, 0u, 0u, 0u, 8u, 8u, 8u, 8u, 8u, 0u, 0u, 0u,
    8u, 8u, 8u, 8u, 8u
);

const uint HH_LUT_FAST[37] = uint[](
    0u, 3u, 6u, 8u, 8u, 0u, 0u, 0u, 1u, 4u, 7u, 8u, 8u, 0u, 0u, 0u,
    2u, 5u, 7u, 8u, 8u, 0u, 0u, 0u, 2u, 5u, 7u, 8u, 8u, 0u, 0u, 0u,
    2u, 5u, 7u, 8u, 8u
);

uint significance_context_from_neighbors(uint state) {
    uint n_x = (state >> 16u) & 0xfu;
    uint n_y = (state >> 20u) & 0xfu;
    uint n_z = (state >> 24u) & 0xfu;

    uint diagonal = min(n_z, 4u);
    if (codeblock.subband == 3u) {
        uint hv = min(n_x + n_y, 4u);
        return HH_LUT_FAST[(hv << 3u) | diagonal];
    }
    uint primary = min(n_x, 2u);
    uint secondary = min(n_y, 2u);
    if (codeblock.subband == 1u) {
        uint temp = primary;
        primary = secondary;
        secondary = temp;
    }
    return LH_LL_LUT_FAST[(primary << 5u) | (secondary << 3u) | diagonal];
}

uint significance_context(uint x, uint y) {
    return significance_context_from_neighbors(state_data[state_base + index_at(x, y)]);
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
    state_data[state_base + index] = (state_data[state_base + index] & STATE_NEIGHBOR_MASK)
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
        context = significance_context_from_neighbors(state);
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
                    uint is_significant = decode_bit(significance_context_from_neighbors(state));
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
                    uint is_significant = decode_bit(significance_context_from_neighbors(state));
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
                    state_data[state_base + index] = ((state | STATE_REFINED) & ~STATE_VISIT_MASK) | visit_word;
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
                    state_data[state_base + index] = ((state | STATE_REFINED) & ~STATE_VISIT_MASK) | visit_word;
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
    warp_id = gl_LocalInvocationID.x >> 5u;
    lane_id = gl_LocalInvocationID.x & 31u;
    ctx_base = warp_id * 19u;

    uint codeblock_index = gl_WorkGroupID.x * CODEBLOCKS_PER_WORKGROUP + warp_id;
    if (codeblock_index >= pc.codeblock_count) {
        return;
    }

    codeblock = codeblocks[codeblock_index];
    state_base = codeblock.coefficient_offset;

    // Collaborative zero-initialization of global state region (32 threads per warp)
    for (uint i = lane_id; i < codeblock.coefficient_count; i += 32u) {
        state_data[state_base + i] = 0u;
    }
    barrier();

    // Only lane 0 of each warp performs MQ arithmetic decoding
    if (lane_id == 0u) {
        for (uint i = 0u; i < MQ_CONTEXT_COUNT; i++) {
            ctx_at(i) = 0u;
        }
        ctx_at(0) = 4u;
        ctx_at(17) = 3u;
        ctx_at(18) = 46u;

        cached_word_index = 0xffffffffu;

        byte_pos = 0u;
        cur_byte = byte_at(0u);
        next_byte = byte_at(1u);

        c_reg = cur_byte << 16u;
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

    } // End of lane 0 MQ arithmetic decoding block

    barrier();

    // Collaborative parallel write-back and local counting (per-warp)
    uint thread_nonzero = 0u;
    uint thread_checksum = 0u;
    uint coeff_count = codeblock.coefficient_count;
    uint coeff_offset_base = codeblock.coefficient_offset;

    for (uint i = lane_id; i < coeff_count; i += 32u) {
        uint offset = coeff_offset_base + i;
        if (is_significant_index(i)) {
            int coefficient = output_coefficients[offset];
            thread_nonzero += 1u;
            thread_checksum += uint(coefficient);
        } else {
            output_coefficients[offset] = 0;
        }
    }

    uint red_base = warp_id * 32u;
    s_reduction[red_base + lane_id] = thread_nonzero;
    barrier();

    if (lane_id == 0u) {
        uint nonzero = 0u;
        for (uint j = 0u; j < 32u; j++) {
            nonzero += s_reduction[red_base + j];
        }
        output_words[codeblock_index * 9u + 0u] = nonzero;
    }

    barrier();
    s_reduction[red_base + lane_id] = thread_checksum;
    barrier();

    if (lane_id == 0u) {
        uint checksum = 0u;
        for (uint j = 0u; j < 32u; j++) {
            checksum += s_reduction[red_base + j];
        }
        output_words[codeblock_index * 9u + 1u] = checksum;
        output_words[codeblock_index * 9u + 2u] = byte_pos;
        output_words[codeblock_index * 9u + 3u] = decoded_passes;
        output_words[codeblock_index * 9u + 4u] = significance_bits;
        output_words[codeblock_index * 9u + 5u] = sign_bits;
        output_words[codeblock_index * 9u + 6u] = refinement_bits;
        output_words[codeblock_index * 9u + 7u] = became_significant;
        output_words[codeblock_index * 9u + 8u] = visited;
    }
}