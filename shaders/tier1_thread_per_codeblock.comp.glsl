#version 450

// Thread-per-codeblock Tier-1 decoder: each thread independently decodes one codeblock.
// This maximizes GPU utilization by keeping ALL threads active (vs the warp-per-codeblock
// approach where only lane 0 works and 31 lanes are idle).

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

layout(std430, set = 0, binding = 2) buffer Output {
    uint output_words[];
};

layout(std430, set = 0, binding = 3) buffer CoefficientsOut {
    int output_coefficients[];
};

layout(push_constant) uniform PushConstants {
    uint codeblock_count;
    uint reconstruction_mode;
} pc;

layout(std430, set = 0, binding = 4) buffer CoefficientState {
    uint state_data[];
};

// Per-thread MQ context (private memory, not shared)
uint ctx[19];

// Per-thread MQ state
uint byte_pos;
uint c_reg;
uint a_reg;
uint ct_reg;
uint cached_word;
uint cached_word_index;
uint cur_byte;
uint next_byte;

// Per-thread codeblock info
Tier1Codeblock cb;
uint state_base;
uint visit_word;

// Constants
const uint STATE_SIGNIFICANT = 1u;
const uint STATE_NEGATIVE = 2u;
const uint STATE_REFINED = 4u;
const uint STATE_VISIT_MASK = 0xff00u;
const uint STATE_NEIGHBOR_SHIFT = 16u;
const uint STATE_NEIGHBOR_MASK = 0x0fff0000u;

// Packed MQ table
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
    uvec2(13u, 1u), uvec2(12u, 1u), uvec2(11u, 1u),
    uvec2(10u, 1u), uvec2(9u, 0u),  uvec2(10u, 0u),
    uvec2(11u, 0u), uvec2(12u, 0u), uvec2(13u, 0u)
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

// --- Helper functions (same logic as smoke shader, but using thread-private state) ---

uint index_at(uint x, uint y) { return y * cb.width + x; }
uint coeff_offset(uint index) { return cb.coefficient_offset + index; }
uint state_at_index(uint index) { return state_data[state_base + index]; }
bool state_is_significant(uint state) { return (state & STATE_SIGNIFICANT) != 0u; }
bool state_is_negative(uint state) { return (state & STATE_NEGATIVE) != 0u; }
bool state_is_refined(uint state) { return (state & STATE_REFINED) != 0u; }
bool state_is_visited(uint state) { return (state & STATE_VISIT_MASK) == visit_word; }

uint byte_at(uint index) {
    uint absolute = cb.byte_offset + index;
    uint word_idx = absolute >> 2u;
    uint word;
    if (word_idx == cached_word_index) { word = cached_word; }
    else { word = words[word_idx]; cached_word = word; cached_word_index = word_idx; }
    return (word >> ((absolute & 3u) * 8u)) & 0xffu;
}

void byte_in() {
    if (cur_byte == 0xffu) {
        if (next_byte > 0x8fu) { c_reg += 0xff00u; ct_reg = 8u; }
        else { byte_pos += 1u; c_reg += next_byte << 9u; ct_reg = 7u; cur_byte = next_byte; next_byte = byte_at(byte_pos + 1u); }
    } else {
        byte_pos += 1u; c_reg += next_byte << 8u; ct_reg = 8u; cur_byte = next_byte; next_byte = byte_at(byte_pos + 1u);
    }
}

void renormalize() {
    if (a_reg >= 0x8000u) return;
    uint k = 15u - uint(findMSB(a_reg));
    while (k >= ct_reg) { k -= ct_reg; a_reg <<= ct_reg; c_reg <<= ct_reg; byte_in(); }
    a_reg <<= k; c_reg <<= k; ct_reg -= k;
}

uint decode_bit(uint context_index) {
    uint ctx_packed = ctx[context_index];
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
    if (cond_lps) { a_reg = qe; } else { c_reg -= qe << 16u; }
    if (cond_lps || (a_reg < 0x8000u)) {
        uint new_state = choose_mps ? next_mps_val : next_lps_val;
        uint new_mps = choose_mps ? mps : (mps ^ switch_mps_val);
        ctx[context_index] = new_state | (new_mps << 7u);
        decoded = choose_mps ? mps : (1u - mps);
        renormalize();
    } else { decoded = mps; }
    return decoded;
}

uint significance_context_from_state(uint st) {
    uint n_x = (st >> 16u) & 0xfu;
    uint n_y = (st >> 20u) & 0xfu;
    uint n_z = (st >> 24u) & 0xfu;
    uint diagonal = min(n_z, 4u);
    if (cb.subband == 3u) { return HH_LUT_FAST[(min(n_x + n_y, 4u) << 3u) | diagonal]; }
    uint primary = min(n_x, 2u);
    uint secondary = min(n_y, 2u);
    if (cb.subband == 1u) { uint t = primary; primary = secondary; secondary = t; }
    return LH_LL_LUT_FAST[(primary << 5u) | (secondary << 3u) | diagonal];
}

uvec2 sign_context(uint x, uint y) {
    uint sl = (x > 0u) ? state_at_index(index_at(x-1u, y)) : 0u;
    uint sr = (x+1u < cb.width) ? state_at_index(index_at(x+1u, y)) : 0u;
    uint st = (y > 0u) ? state_at_index(index_at(x, y-1u)) : 0u;
    uint sb = (y+1u < cb.height) ? state_at_index(index_at(x, y+1u)) : 0u;
    uint hp = (((sl&3u)==1u)||((sr&3u)==1u)) ? 1u : 0u;
    uint hn = (((sl&3u)==3u)||((sr&3u)==3u)) ? 1u : 0u;
    int ch = int(hp) - int(hn);
    uint vp = (((st&3u)==1u)||((sb&3u)==1u)) ? 1u : 0u;
    uint vn = (((st&3u)==3u)||((sb&3u)==3u)) ? 1u : 0u;
    int cv = int(vp) - int(vn);
    return SIGN_LUT[uint((ch+1)*3+(cv+1))];
}

void increment_neighbor(int x, int y, uint bit_shift) {
    if (x < 0 || y < 0 || x >= int(cb.width) || y >= int(cb.height)) return;
    state_data[state_base + uint(y) * cb.width + uint(x)] += 1u << (STATE_NEIGHBOR_SHIFT + bit_shift);
}

void update_neighbors(uint x, uint y) {
    int ix = int(x), iy = int(y);
    increment_neighbor(ix-1, iy, 0u); increment_neighbor(ix+1, iy, 0u);
    increment_neighbor(ix, iy-1, 4u); increment_neighbor(ix, iy+1, 4u);
    increment_neighbor(ix-1, iy-1, 8u); increment_neighbor(ix+1, iy-1, 8u);
    increment_neighbor(ix-1, iy+1, 8u); increment_neighbor(ix+1, iy+1, 8u);
}

void decode_significant_sample(uint x, uint y, uint bitplane) {
    uint index = index_at(x, y);
    uvec2 sc = sign_context(x, y);
    uint sign_bit = decode_bit(sc.x) ^ sc.y;
    int magnitude;
    if (cb.reconstruction_mode == 1u) {
        magnitude = int((1u << bitplane) | ((1u << bitplane) >> 1u));
        if (cb.resolution == 0u && cb.subband == 0u) magnitude -= 1;
    } else {
        magnitude = int(1u << bitplane);
    }
    output_coefficients[coeff_offset(index)] = sign_bit != 0u ? -magnitude : magnitude;
    update_neighbors(x, y);
    state_data[state_base + index] = (state_data[state_base + index] & STATE_NEIGHBOR_MASK)
        | STATE_SIGNIFICANT | (sign_bit != 0u ? STATE_NEGATIVE : 0u);
}

// --- Main decode function ---
void main() {
    uint codeblock_index = gl_GlobalInvocationID.x;
    if (codeblock_index >= pc.codeblock_count) return;

    cb = codeblocks[codeblock_index];
    state_base = cb.coefficient_offset;
    visit_word = 0x100u;

    // Zero-init state
    for (uint i = 0u; i < cb.coefficient_count; i++) {
        state_data[state_base + i] = 0u;
    }

    // Init MQ contexts
    for (uint i = 0u; i < 19u; i++) ctx[i] = 0u;
    ctx[0] = 4u; ctx[17] = 3u; ctx[18] = 46u;

    // Init MQ decoder
    cached_word_index = 0xFFFFFFFFu;
    byte_pos = 0u;
    cur_byte = byte_at(0u);
    next_byte = byte_at(1u);
    c_reg = uint(cur_byte) << 16u;
    ct_reg = 0u;
    byte_in();
    c_reg <<= 7u;
    ct_reg -= 7u;
    a_reg = 0x8000u;

    if (cb.pass_count == 0u) {
        // Write zeros and summary
        for (uint i = 0u; i < cb.coefficient_count; i++)
            output_coefficients[cb.coefficient_offset + i] = 0;
        for (uint i = 0u; i < 9u; i++)
            output_words[codeblock_index * 9u + i] = 0u;
        return;
    }

    uint first_bitplane = cb.magnitude_bitplanes - cb.zero_bit_planes - 1u;

    // === Cleanup pass (first) ===
    uint stripe_y = 0u;
    while (stripe_y < cb.height) {
        uint stripe_height = min(4u, cb.height - stripe_y);
        for (uint x = 0u; x < cb.width; x++) {
            bool run_candidate = (stripe_height == 4u);
            if (run_candidate) {
                for (uint yo = 0u; yo < 4u; yo++) {
                    uint st = state_at_index(index_at(x, stripe_y + yo));
                    if (state_is_visited(st) || state_is_significant(st) || (st & STATE_NEIGHBOR_MASK) != 0u) {
                        run_candidate = false; break;
                    }
                }
            }
            if (run_candidate) {
                uint any_sig = decode_bit(17u);
                if (any_sig == 0u) { continue; }
                uint run = (decode_bit(18u) << 1u) | decode_bit(18u);
                decode_significant_sample(x, stripe_y + run, first_bitplane);
                for (uint yo = run + 1u; yo < 4u; yo++) {
                    uint idx = index_at(x, stripe_y + yo);
                    uint st = state_at_index(idx);
                    if (state_is_visited(st) || state_is_significant(st)) continue;
                    uint ctx_idx = significance_context_from_state(st);
                    uint is_sig = decode_bit(ctx_idx);
                    if (is_sig != 0u) { decode_significant_sample(x, stripe_y + yo, first_bitplane); }
                }
            } else {
                for (uint yo = 0u; yo < stripe_height; yo++) {
                    uint idx = index_at(x, stripe_y + yo);
                    uint st = state_at_index(idx);
                    if (state_is_visited(st) || state_is_significant(st)) continue;
                    uint ctx_idx = significance_context_from_state(st);
                    uint is_sig = decode_bit(ctx_idx);
                    if (is_sig != 0u) { decode_significant_sample(x, stripe_y + yo, first_bitplane); }
                }
            }
        }
        stripe_y += 4u;
    }

    // === Remaining passes (sig + ref + cleanup triplets) ===
    uint pass_index = 1u;
    uint bitplane = first_bitplane - 1u;
    while (pass_index < cb.pass_count) {
        // Clear visits
        visit_word += 0x100u;

        // Significance pass
        stripe_y = 0u;
        while (stripe_y < cb.height) {
            uint stripe_height = min(4u, cb.height - stripe_y);
            for (uint x = 0u; x < cb.width; x++) {
                for (uint yo = 0u; yo < stripe_height; yo++) {
                    uint y = stripe_y + yo;
                    uint idx = index_at(x, y);
                    uint st = state_at_index(idx);
                    if (state_is_visited(st) || state_is_significant(st) || (st & STATE_NEIGHBOR_MASK) == 0u) continue;
                    uint ctx_idx = significance_context_from_state(st);
                    uint is_sig = decode_bit(ctx_idx);
                    if (is_sig != 0u) {
                        decode_significant_sample(x, y, bitplane);
                        state_data[state_base + idx] = (state_data[state_base + idx] & ~STATE_VISIT_MASK) | visit_word;
                    } else {
                        state_data[state_base + idx] = (state_data[state_base + idx] & ~STATE_VISIT_MASK) | visit_word;
                    }
                }
            }
            stripe_y += 4u;
        }
        pass_index += 1u;
        if (pass_index >= cb.pass_count) break;

        // Refinement pass
        stripe_y = 0u;
        while (stripe_y < cb.height) {
            uint stripe_height = min(4u, cb.height - stripe_y);
            for (uint x = 0u; x < cb.width; x++) {
                for (uint yo = 0u; yo < stripe_height; yo++) {
                    uint y = stripe_y + yo;
                    uint idx = index_at(x, y);
                    uint st = state_at_index(idx);
                    if (state_is_visited(st) || !state_is_significant(st)) continue;
                    uint context = state_is_refined(st) ? 16u : (((st & STATE_NEIGHBOR_MASK) != 0u) ? 15u : 14u);
                    uint bit = decode_bit(context);
                    int delta;
                    if (cb.reconstruction_mode == 1u) {
                        int half_step = int((1u << bitplane) >> 1u);
                        delta = bit != 0u ? half_step : -half_step;
                    } else {
                        delta = bit != 0u ? int(1u << bitplane) : 0;
                    }
                    uint offset = coeff_offset(idx);
                    output_coefficients[offset] += state_is_negative(st) ? -delta : delta;
                    state_data[state_base + idx] = ((st | STATE_REFINED) & ~STATE_VISIT_MASK) | visit_word;
                }
            }
            stripe_y += 4u;
        }
        pass_index += 1u;
        if (pass_index >= cb.pass_count) break;

        // Cleanup pass
        stripe_y = 0u;
        while (stripe_y < cb.height) {
            uint stripe_height = min(4u, cb.height - stripe_y);
            for (uint x = 0u; x < cb.width; x++) {
                bool run_candidate = (stripe_height == 4u);
                if (run_candidate) {
                    for (uint yo = 0u; yo < 4u; yo++) {
                        uint st = state_at_index(index_at(x, stripe_y + yo));
                        if (state_is_visited(st) || state_is_significant(st) || (st & STATE_NEIGHBOR_MASK) != 0u) {
                            run_candidate = false; break;
                        }
                    }
                }
                if (run_candidate) {
                    uint any_sig = decode_bit(17u);
                    if (any_sig == 0u) { continue; }
                    uint run = (decode_bit(18u) << 1u) | decode_bit(18u);
                    decode_significant_sample(x, stripe_y + run, bitplane);
                    for (uint yo = run + 1u; yo < 4u; yo++) {
                        uint idx = index_at(x, stripe_y + yo);
                        uint st = state_at_index(idx);
                        if (state_is_visited(st) || state_is_significant(st)) continue;
                        uint ctx_idx = significance_context_from_state(st);
                        uint is_sig = decode_bit(ctx_idx);
                        if (is_sig != 0u) { decode_significant_sample(x, stripe_y + yo, bitplane); }
                    }
                } else {
                    for (uint yo = 0u; yo < stripe_height; yo++) {
                        uint idx = index_at(x, stripe_y + yo);
                        uint st = state_at_index(idx);
                        if (state_is_visited(st) || state_is_significant(st)) continue;
                        uint ctx_idx = significance_context_from_state(st);
                        uint is_sig = decode_bit(ctx_idx);
                        if (is_sig != 0u) { decode_significant_sample(x, stripe_y + yo, bitplane); }
                    }
                }
            }
            stripe_y += 4u;
        }
        pass_index += 1u;
        if (bitplane > 0u) bitplane -= 1u;
    }

    // Write summary and zero non-significant coefficients
    uint nonzero = 0u;
    uint checksum = 0u;
    for (uint i = 0u; i < cb.coefficient_count; i++) {
        if ((state_data[state_base + i] & STATE_SIGNIFICANT) != 0u) {
            nonzero += 1u;
            checksum += uint(output_coefficients[cb.coefficient_offset + i]);
        } else {
            output_coefficients[cb.coefficient_offset + i] = 0;
        }
    }
    output_words[codeblock_index * 9u + 0u] = nonzero;
    output_words[codeblock_index * 9u + 1u] = checksum;
    output_words[codeblock_index * 9u + 2u] = byte_pos;
    output_words[codeblock_index * 9u + 3u] = cb.pass_count;
    output_words[codeblock_index * 9u + 4u] = 0u;
    output_words[codeblock_index * 9u + 5u] = 0u;
    output_words[codeblock_index * 9u + 6u] = 0u;
    output_words[codeblock_index * 9u + 7u] = 0u;
    output_words[codeblock_index * 9u + 8u] = 0u;
}
