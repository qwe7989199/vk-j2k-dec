#version 450
#extension GL_KHR_shader_subgroup_basic : require

// Warp-per-codeblock Tier-1 decoder with packed stripe-column state.
//
// Same bindings, push constants and coefficient output as tier1_cleanup_smoke.comp.glsl built
// with TIER1_NO_STATS (the summary words in binding 2 are not written). Instead of one state
// word per sample holding neighbour counts, state is one word per 4-sample stripe column. Lane 0
// keeps the words of columns x-1, x and x+1 in registers while it scans a stripe, so context
// formation is bit arithmetic and making a sample significant touches at most one other word:
// the mirror copy in the adjacent stripe.
//
// Column word layout (stripe s, column x), "rows" relative to the stripe:
//   bits  0..5   significance of rows -1..4; rows -1 and 4 mirror the adjacent stripes
//   bits  8..13  sign of rows -1..4, set together with the significance bit
//   bits 16..19  refined (first refinement done) for rows 0..3
//   bits 20..23  visited in the current bitplane for rows 0..3, cleared by the cleanup pass
//
// A codeblock needs width * ceil(height / 4) words, which always fits in the
// coefficient_count-sized region the warp shader uses at coefficient_offset.

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
    uint reconstruction_mode;
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
    uint reconstruction_mode;
    uint codeblock_offset;
} pc;

layout(std430, set = 0, binding = 4) buffer CoefficientState {
    uint state_data[];
};

const uint MQ_CONTEXT_COUNT = 19u;
const uint CODEBLOCKS_PER_WORKGROUP = 4u;

const uint COL_SIG_MASK = 0x3fu;
const uint COL_SIGN_SHIFT = 8u;
const uint COL_REFINED_SHIFT = 16u;
const uint COL_VISITED_SHIFT = 20u;
const uint COL_VISITED_MASK = 0xfu << 20u;

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

shared uint s_context_packed[4 * 19];

uint warp_id;
uint lane_id;
uint ctx_base;

#define ctx_at(i) s_context_packed[ctx_base + (i)]

Tier1Codeblock codeblock;
uint state_base;
uint stripe_count;
uint byte_pos;
uint c_reg;
uint a_reg;
uint ct_reg;
uint cached_word;
uint cached_word_index;
uint cur_byte;
uint next_byte;

// Sliding window over the current stripe: column words x-1, x, x+1, zero outside the codeblock.
uint col_left;
uint col_cur;
uint col_right;
uint recon_mode;
// Bit s set once stripe s holds a significant sample. Only tracked when every stripe index
// fits (codeblocks can be up to 1024 rows tall); otherwise no stripe is skipped.
uint sig_stripes;
bool stripe_skip;

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

// Each context holds MQ_TABLE[state] with the MPS in bit 31, so Qe needs no table lookup;
// the table is read only when the state changes.
uint mq_context(uint state, uint mps) {
    return MQ_TABLE[state] | (mps << 31u);
}

uint decode_bit(uint context_index) {
    uint mq_entry = ctx_at(context_index);
    uint mps = mq_entry >> 31u;
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
        ctx_at(context_index) = mq_context(new_state, new_mps);
        decoded = choose_mps ? mps : (1u - mps);
        renormalize();
    } else {
        decoded = mps;
    }
    return decoded;
}

// After `word >> r`, bits 0/1/2 hold rows r-1/r/r+1 of that column (sign at bits 8/9/10).

bool row_skipped(uint r) {
    // Already significant, or already visited in this bitplane.
    return (((col_cur >> (r + 1u)) | (col_cur >> (COL_VISITED_SHIFT + r))) & 1u) != 0u;
}

bool has_neighbor(uint r) {
    return (((((col_left | col_right) >> r) & 7u) | ((col_cur >> r) & 5u))) != 0u;
}

// Bit r set when row r has at least one significant neighbour.
uint neighbor_rows() {
    uint sides = (col_left | col_right) & COL_SIG_MASK;
    uint center = col_cur & COL_SIG_MASK;
    return (sides | (sides >> 1u) | (sides >> 2u) | center | (center >> 2u)) & 0xfu;
}

uint significance_context(uint r) {
    uint lw = col_left >> r;
    uint cw = col_cur >> r;
    uint rw = col_right >> r;
    uint n_x = ((lw >> 1u) & 1u) + ((rw >> 1u) & 1u);
    uint n_y = (cw & 1u) + ((cw >> 2u) & 1u);
    uint n_z = (lw & 1u) + ((lw >> 2u) & 1u) + (rw & 1u) + ((rw >> 2u) & 1u);

    // Same values as HH_LUT_FAST / LH_LL_LUT_FAST, computed instead of looked up.
    if (codeblock.subband == 3u) {
        uint hv = min(n_x + n_y, 2u);
        if (n_z >= 3u) {
            return 8u;
        }
        if (n_z == 2u) {
            return hv == 0u ? 6u : 7u;
        }
        return n_z * 3u + hv;
    }
    uint primary = n_x;
    uint secondary = n_y;
    if (codeblock.subband == 1u) {
        primary = n_y;
        secondary = n_x;
    }
    if (primary == 2u) {
        return 8u;
    }
    if (primary == 1u) {
        return secondary != 0u ? 7u : (n_z != 0u ? 6u : 5u);
    }
    if (secondary != 0u) {
        return secondary + 2u;
    }
    return min(n_z, 2u);
}

uvec2 sign_context(uint r) {
    uint lw = col_left >> r;
    uint cw = col_cur >> r;
    uint rw = col_right >> r;
    // 2-bit states as in the per-sample shader: bit 0 significant, bit 1 negative.
    uint state_left   = ((lw >> 1u) & 1u) | ((lw >> 8u) & 2u);
    uint state_right  = ((rw >> 1u) & 1u) | ((rw >> 8u) & 2u);
    uint state_top    = (cw & 1u) | ((cw >> 7u) & 2u);
    uint state_bottom = ((cw >> 2u) & 1u) | ((cw >> 9u) & 2u);

    uint has_pos_h = ((state_left == 1u) || (state_right == 1u)) ? 1u : 0u;
    uint has_neg_h = ((state_left == 3u) || (state_right == 3u)) ? 1u : 0u;
    int contrib_h = int(has_pos_h) - int(has_neg_h);

    uint has_pos_v = ((state_top == 1u) || (state_bottom == 1u)) ? 1u : 0u;
    uint has_neg_v = ((state_top == 3u) || (state_bottom == 3u)) ? 1u : 0u;
    int contrib_v = int(has_pos_v) - int(has_neg_v);

    // Same values as SIGN_LUT[(h + 1) * 3 + (v + 1)].
    uint context = contrib_h == 0 ? 9u + uint(abs(contrib_v)) : uint(12 + contrib_h * contrib_v);
    uint flip = (contrib_h < 0 || (contrib_h == 0 && contrib_v < 0)) ? 1u : 0u;
    return uvec2(context, flip);
}

void make_significant(uint x, uint s, uint r, uint bitplane) {
    uvec2 sc = sign_context(r);
    uint sign_bit = decode_bit(sc.x) ^ sc.y;
    int magnitude;
    if (recon_mode == 2u) {
        uint shifted = bitplane + 8u;
        magnitude = int((1u << shifted) | ((1u << shifted) >> 1u));
    } else if (recon_mode == 1u) {
        magnitude = int((1u << bitplane) | ((1u << bitplane) >> 1u));
        if (codeblock.resolution == 0u && codeblock.subband == 0u) {
            magnitude -= 1;
        }
    } else {
        magnitude = int(1u << bitplane);
    }
    uint index = (s * 4u + r) * codeblock.width + x;
    output_coefficients[codeblock.coefficient_offset + index] = sign_bit != 0u ? -magnitude : magnitude;
    col_cur |= (2u << r) | (sign_bit << (COL_SIGN_SHIFT + 1u + r));
    if (stripe_skip) {
        sig_stripes |= 1u << s;
    }
    // Mirror into the adjacent stripe: row 0 is that stripe's row 4, row 3 its row -1.
    if (r == 0u && s > 0u) {
        atomicOr(state_data[state_base + (s - 1u) * codeblock.width + x],
            (1u << 5u) | (sign_bit << (COL_SIGN_SHIFT + 5u)));
    }
    if (r == 3u && s + 1u < stripe_count) {
        atomicOr(state_data[state_base + (s + 1u) * codeblock.width + x], 1u | (sign_bit << COL_SIGN_SHIFT));
    }
}

void load_stripe(uint base) {
    col_left = 0u;
    col_cur = state_data[base];
    col_right = (codeblock.width > 1u) ? state_data[base + 1u] : 0u;
}

void advance_column(uint base, uint x) {
    col_left = col_cur;
    col_cur = col_right;
    col_right = (x + 2u < codeblock.width) ? state_data[base + x + 2u] : 0u;
}

void decode_cleanup_sample(uint x, uint s, uint r, uint bitplane) {
    if (!row_skipped(r)) {
        if (decode_bit(significance_context(r)) != 0u) {
            make_significant(x, s, r, bitplane);
        }
    }
}

void decode_cleanup_pass(uint bitplane) {
    for (uint s = 0u; s < stripe_count; s++) {
        uint stripe_height = min(4u, codeblock.height - s * 4u);
        uint base = state_base + s * codeblock.width;
        load_stripe(base);
        for (uint x = 0u; x < codeblock.width; x++) {
            uint original = col_cur;
            bool run_length = stripe_height == 4u
                && ((col_left | col_cur | col_right) & COL_SIG_MASK) == 0u
                && (col_cur & COL_VISITED_MASK) == 0u;
            if (run_length) {
                if (decode_bit(17u) != 0u) {
                    uint run = (decode_bit(18u) << 1u) | decode_bit(18u);
                    make_significant(x, s, run, bitplane);
                    for (uint r = run + 1u; r < 4u; r++) {
                        decode_cleanup_sample(x, s, r, bitplane);
                    }
                }
            } else {
                for (uint r = 0u; r < stripe_height; r++) {
                    decode_cleanup_sample(x, s, r, bitplane);
                }
            }
            // Cleanup ends the bitplane: drop this column's visited flags for the next one.
            col_cur &= ~COL_VISITED_MASK;
            if (col_cur != original) {
                state_data[base + x] = col_cur;
            }
            advance_column(base, x);
        }
    }
}

void decode_significance_pass(uint bitplane) {
    for (uint s = 0u; s < stripe_count; s++) {
        // Candidates need a significant neighbour, which can only sit in this stripe or the
        // stripes above and below it.
        if (stripe_skip && ((((sig_stripes << 1u) | sig_stripes | (sig_stripes >> 1u)) >> s) & 1u) == 0u) {
            continue;
        }
        uint stripe_height = min(4u, codeblock.height - s * 4u);
        uint valid_rows = (1u << stripe_height) - 1u;
        uint base = state_base + s * codeblock.width;
        load_stripe(base);
        for (uint x = 0u; x < codeblock.width; x++) {
            uint original = col_cur;
            uint skipped_rows = (col_cur >> 1u) | (col_cur >> COL_VISITED_SHIFT);
            // A column with no candidate cannot gain one: that takes a significance in it.
            if ((neighbor_rows() & ~skipped_rows & valid_rows) != 0u) {
                for (uint r = 0u; r < stripe_height; r++) {
                    if (!row_skipped(r) && has_neighbor(r)) {
                        if (decode_bit(significance_context(r)) != 0u) {
                            make_significant(x, s, r, bitplane);
                        }
                        col_cur |= 1u << (COL_VISITED_SHIFT + r);
                    }
                }
            }
            if (col_cur != original) {
                state_data[base + x] = col_cur;
            }
            advance_column(base, x);
        }
    }
}

void decode_refinement_pass(uint bitplane) {
    for (uint s = 0u; s < stripe_count; s++) {
        // Only significant samples are refined.
        if (stripe_skip && ((sig_stripes >> s) & 1u) == 0u) {
            continue;
        }
        uint stripe_height = min(4u, codeblock.height - s * 4u);
        uint valid_rows = (1u << stripe_height) - 1u;
        uint base = state_base + s * codeblock.width;
        load_stripe(base);
        for (uint x = 0u; x < codeblock.width; x++) {
            uint original = col_cur;
            // Significant and not visited; refinement never changes significance.
            uint rows = (col_cur >> 1u) & ~(col_cur >> COL_VISITED_SHIFT) & valid_rows;
            if (rows != 0u) {
                for (uint r = 0u; r < stripe_height; r++) {
                    if (((rows >> r) & 1u) != 0u) {
                        bool refined = ((col_cur >> (COL_REFINED_SHIFT + r)) & 1u) != 0u;
                        uint context = refined ? 16u : (has_neighbor(r) ? 15u : 14u);
                        uint bit = decode_bit(context);
                        int delta;
                        if (recon_mode == 2u) {
                            int half_step = int((1u << (bitplane + 8u)) >> 1u);
                            delta = bit != 0u ? half_step : -half_step;
                        } else if (recon_mode == 1u) {
                            int half_step = int((1u << bitplane) >> 1u);
                            delta = bit != 0u ? half_step : -half_step;
                        } else {
                            delta = bit != 0u ? int(1u << bitplane) : 0;
                        }
                        bool negative = ((col_cur >> (COL_SIGN_SHIFT + 1u + r)) & 1u) != 0u;
                        uint offset = codeblock.coefficient_offset + (s * 4u + r) * codeblock.width + x;
                        if (delta != 0) {
                            atomicAdd(output_coefficients[offset], negative ? -delta : delta);
                        }
                        col_cur |= (1u << (COL_REFINED_SHIFT + r)) | (1u << (COL_VISITED_SHIFT + r));
                    }
                }
            }
            if (col_cur != original) {
                state_data[base + x] = col_cur;
            }
            advance_column(base, x);
        }
    }
}

void main() {
    warp_id = gl_LocalInvocationID.x >> 5u;
    lane_id = gl_LocalInvocationID.x & 31u;
    ctx_base = warp_id * 19u;

    uint codeblock_index = gl_WorkGroupID.x * CODEBLOCKS_PER_WORKGROUP + warp_id;
    if (codeblock_index >= pc.codeblock_count) {
        return;
    }
    codeblock_index += pc.codeblock_offset;

    codeblock = codeblocks[codeblock_index];
    state_base = codeblock.coefficient_offset;
    stripe_count = (codeblock.height + 3u) >> 2u;
    sig_stripes = 0u;
    stripe_skip = stripe_count <= 31u;
    recon_mode = codeblocks[0].reconstruction_mode;

    // All 32 lanes clear the column words and the coefficients; only significant samples are
    // written afterwards.
    uint state_words = stripe_count * codeblock.width;
    for (uint i = lane_id; i < state_words; i += 32u) {
        state_data[state_base + i] = 0u;
    }
    for (uint i = lane_id; i < codeblock.coefficient_count; i += 32u) {
        output_coefficients[codeblock.coefficient_offset + i] = 0;
    }
    subgroupBarrier();

    if (lane_id == 0u) {
        for (uint i = 0u; i < MQ_CONTEXT_COUNT; i++) {
            ctx_at(i) = mq_context(0u, 0u);
        }
        ctx_at(0) = mq_context(4u, 0u);
        ctx_at(17) = mq_context(3u, 0u);
        ctx_at(18) = mq_context(46u, 0u);

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

        if (codeblock.pass_count > 0u && codeblock.width > 0u && codeblock.height > 0u) {
            uint first_bitplane = codeblock.magnitude_bitplanes - codeblock.zero_bit_planes - 1u;
            decode_cleanup_pass(first_bitplane);

            uint pass_index = 1u;
            uint bitplane = first_bitplane - 1u;
            while (pass_index < codeblock.pass_count) {
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
                if (recon_mode == 2u || bitplane > 0u) {
                    bitplane -= 1u;
                }
            }
        }
    }
}
