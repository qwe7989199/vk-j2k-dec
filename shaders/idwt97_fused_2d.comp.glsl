#version 450

// One level of the irreversible 9/7 inverse DWT for three components in a single dispatch.
//
// Each workgroup reconstructs a TILE_X x TILE_Y output tile of one component (gl_WorkGroupID.z):
// it stages the LL/HL/LH/HH samples the tile depends on (including the lifting halo) in shared
// memory, reconstructs every staged row horizontally, then the tile's columns vertically. Each
// work item produces one low/high output pair straight from the staged samples, running all four
// lifting steps on a small register window (lows k-1..k+2, highs k-2..k+2), so a pass needs no
// barrier between steps. Pairs whose window crosses a band edge use the clamped formulas instead.
// The float formulas, the K scaling of high samples and the edge clamping are those of
// idwt97_offset_1d.comp.glsl (horizontal pass, then vertical).
//
// Subbands start at even coordinates (low sample first), as in idwt97_offset_1d.
//
// TILE_X / TILE_Y default to 32 and must be even; the host dispatches
// ceil(width / TILE_X) x ceil(height / TILE_Y) x 3 workgroups.

#ifndef TILE_X
#define TILE_X 32u
#endif
#ifndef TILE_Y
#define TILE_Y 32u
#endif

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Current LL: the subband plane for the first level, the previous level's output afterwards.
layout(std430, set = 0, binding = 0) readonly buffer LowLowBand {
    float ll_band[];
};

// HL, LH and HH subband planes.
layout(std430, set = 0, binding = 1) readonly buffer SubbandPlanes {
    float planes[];
};

layout(std430, set = 0, binding = 2) writeonly buffer OutputBand {
    float output_band[];
};

layout(push_constant) uniform PushConstants {
    uint low_width;   // LL / LH width
    uint low_height;  // LL / HL height
    uint high_width;  // HL / HH width
    uint high_height; // LH / HH height
    uint ll_stride;
    uint hl_stride;
    uint lh_stride;
    uint hh_stride;
    uint ll_offset0;
    uint ll_offset1;
    uint ll_offset2;
    uint hl_offset0;
    uint hl_offset1;
    uint hl_offset2;
    uint lh_offset0;
    uint lh_offset1;
    uint lh_offset2;
    uint hh_offset0;
    uint hh_offset1;
    uint hh_offset2;
    uint out_offset0;
    uint out_offset1;
    uint out_offset2;
} pc;

const float DWT97_ALPHA = -1.5861343;
const float DWT97_BETA = -0.05298012;
const float DWT97_GAMMA = 0.8829111;
const float DWT97_DELTA = 0.44350687;
const float DWT97_K = 1.2301741;

const uint THREADS = 256u;
// A tile of 2n output samples starting at low index a needs low samples a-1..a+n+1 and high
// samples a-2..a+n+1 along each axis (four lifting steps, one sample of reach each).
const uint LOW_X = TILE_X / 2u + 3u;
const uint HIGH_X = TILE_X / 2u + 4u;
const uint LOW_Y = TILE_Y / 2u + 3u;
const uint HIGH_Y = TILE_Y / 2u + 4u;
const int LOW_BASE = -1;
const int HIGH_BASE = -2;
// Staged rows: LOW_Y rows from LL/HL (vertical lows), then HIGH_Y rows from LH/HH.
const uint ROWS = LOW_Y + HIGH_Y;
const uint HIGH_ROW0 = LOW_Y * TILE_X;
const uint PAIRS_X = TILE_X / 2u;
const uint PAIRS_Y = TILE_Y / 2u;

shared float s_low[ROWS * LOW_X];
shared float s_high[ROWS * HIGH_X];
// Horizontally reconstructed rows, one entry per tile column, in the same row order.
shared float s_rows[ROWS * TILE_X];

// First low sample of the tile along x / y.
uint ax;
uint ay;

uint pick(uint c, uint v0, uint v1, uint v2) {
    return c == 0u ? v0 : (c == 1u ? v1 : v2);
}

// Band index held by `slot` of an axis whose slots start at `origin + base`.
uint slot_index(uint slot, uint origin, int base, uint count) {
    return uint(clamp(int(origin) + base + int(slot), 0, int(count) - 1));
}

// Slot of band index `index` (always inside the staged range for the samples a pair reads).
uint low_slot(uint index, uint origin) {
    return uint(int(index) - int(origin) - LOW_BASE);
}

uint high_slot(uint index, uint origin) {
    return uint(int(index) - int(origin) - HIGH_BASE);
}

// High neighbours of low sample `index`: index-1 (index 0 uses 0) and index, clamped to the band.
uvec2 high_neighbors(uint index, uint high_count) {
    return uvec2(index == 0u ? 0u : min(index - 1u, high_count - 1u), min(index, high_count - 1u));
}

// Low neighbours of high sample `index` (already clamped): index and index+1, clamped.
uvec2 low_neighbors(uint index, uint low_count) {
    return uvec2(index, min(index + 1u, low_count - 1u));
}

// All four lifting steps for the output pair at index k, from lows k-1..k+2 and highs k-2..k+2
// with no clamping inside the window. Returns (low k, high k).
vec2 lift_pair(float l0, float l1, float l2, float l3, float h0, float h1, float h2, float h3, float h4) {
    float d0 = l0 - DWT97_DELTA * (h0 * DWT97_K + h1 * DWT97_K);
    float d1 = l1 - DWT97_DELTA * (h1 * DWT97_K + h2 * DWT97_K);
    float d2 = l2 - DWT97_DELTA * (h2 * DWT97_K + h3 * DWT97_K);
    float d3 = l3 - DWT97_DELTA * (h3 * DWT97_K + h4 * DWT97_K);
    float g0 = h1 * DWT97_K - DWT97_GAMMA * (d0 + d1);
    float g1 = h2 * DWT97_K - DWT97_GAMMA * (d1 + d2);
    float g2 = h3 * DWT97_K - DWT97_GAMMA * (d2 + d3);
    float b0 = d1 - DWT97_BETA * (g0 + g1);
    float b1 = d2 - DWT97_BETA * (g1 + g2);
    return vec2(b0, g1 - DWT97_ALPHA * (b0 + b1));
}

// Clamped lifting along a staged row r, for pairs near a band edge. Indices are band indices.
float row_low(uint r, uint index) {
    return s_low[r * LOW_X + low_slot(index, ax)];
}

float row_high(uint r, uint index) {
    return s_high[r * HIGH_X + high_slot(index, ax)] * DWT97_K;
}

float row_delta(uint r, uint index) {
    uvec2 h = high_neighbors(index, pc.high_width);
    return row_low(r, index) - DWT97_DELTA * (row_high(r, h.x) + row_high(r, h.y));
}

float row_gamma(uint r, uint index) {
    uvec2 l = low_neighbors(index, pc.low_width);
    return row_high(r, index) - DWT97_GAMMA * (row_delta(r, l.x) + row_delta(r, l.y));
}

float row_beta(uint r, uint index) {
    uvec2 h = high_neighbors(index, pc.high_width);
    return row_delta(r, index) - DWT97_BETA * (row_gamma(r, h.x) + row_gamma(r, h.y));
}

float row_alpha(uint r, uint index) {
    uvec2 l = low_neighbors(index, pc.low_width);
    return row_gamma(r, index) - DWT97_ALPHA * (row_beta(r, l.x) + row_beta(r, l.y));
}

// The same down tile column `col` of the reconstructed rows.
float col_low(uint index, uint col) {
    return s_rows[low_slot(index, ay) * TILE_X + col];
}

float col_high(uint index, uint col) {
    return s_rows[HIGH_ROW0 + high_slot(index, ay) * TILE_X + col] * DWT97_K;
}

float col_delta(uint index, uint col) {
    uvec2 h = high_neighbors(index, pc.high_height);
    return col_low(index, col) - DWT97_DELTA * (col_high(h.x, col) + col_high(h.y, col));
}

float col_gamma(uint index, uint col) {
    uvec2 l = low_neighbors(index, pc.low_height);
    return col_high(index, col) - DWT97_GAMMA * (col_delta(l.x, col) + col_delta(l.y, col));
}

float col_beta(uint index, uint col) {
    uvec2 h = high_neighbors(index, pc.high_height);
    return col_delta(index, col) - DWT97_BETA * (col_gamma(h.x, col) + col_gamma(h.y, col));
}

float col_alpha(uint index, uint col) {
    uvec2 l = low_neighbors(index, pc.low_height);
    return col_gamma(index, col) - DWT97_ALPHA * (col_beta(l.x, col) + col_beta(l.y, col));
}

void main() {
    uint component = gl_WorkGroupID.z;
    uint x0 = gl_WorkGroupID.x * TILE_X;
    uint y0 = gl_WorkGroupID.y * TILE_Y;
    uint out_width = pc.low_width + pc.high_width;
    ax = x0 >> 1u;
    ay = y0 >> 1u;
    uint lid = gl_LocalInvocationIndex;

    uint ll_offset = pick(component, pc.ll_offset0, pc.ll_offset1, pc.ll_offset2);
    uint hl_offset = pick(component, pc.hl_offset0, pc.hl_offset1, pc.hl_offset2);
    uint lh_offset = pick(component, pc.lh_offset0, pc.lh_offset1, pc.lh_offset2);
    uint hh_offset = pick(component, pc.hh_offset0, pc.hh_offset1, pc.hh_offset2);
    uint out_offset = pick(component, pc.out_offset0, pc.out_offset1, pc.out_offset2);

    bool has_high_x = pc.high_width > 0u;
    bool has_high_y = pc.high_height > 0u;
    uint rows = has_high_y ? ROWS : LOW_Y;

    for (uint i = lid; i < rows * LOW_X; i += THREADS) {
        uint r = i / LOW_X;
        uint col = slot_index(i % LOW_X, ax, LOW_BASE, pc.low_width);
        s_low[i] = r < LOW_Y
            ? ll_band[ll_offset + slot_index(r, ay, LOW_BASE, pc.low_height) * pc.ll_stride + col]
            : planes[lh_offset + slot_index(r - LOW_Y, ay, HIGH_BASE, pc.high_height) * pc.lh_stride + col];
    }
    if (has_high_x) {
        for (uint i = lid; i < rows * HIGH_X; i += THREADS) {
            uint r = i / HIGH_X;
            uint col = slot_index(i % HIGH_X, ax, HIGH_BASE, pc.high_width);
            s_high[i] = r < LOW_Y
                ? planes[hl_offset + slot_index(r, ay, LOW_BASE, pc.low_height) * pc.hl_stride + col]
                : planes[hh_offset + slot_index(r - LOW_Y, ay, HIGH_BASE, pc.high_height) * pc.hh_stride + col];
        }
    }
    barrier();

    // Horizontal: output pair (columns 2k, 2k+1) of every staged row. Columns past the band are
    // zero. Without high samples the lows pass through unchanged.
    for (uint i = lid; i < rows * PAIRS_X; i += THREADS) {
        uint r = i / PAIRS_X;
        uint pair = i % PAIRS_X;
        uint k = ax + pair;
        vec2 value = vec2(0.0);
        if (k < pc.low_width) {
            if (!has_high_x) {
                value.x = row_low(r, k);
            } else if (k >= 2u && k + 2u < pc.high_width) {
                uint l = r * LOW_X + low_slot(k, ax);
                uint h = r * HIGH_X + high_slot(k, ax);
                value = lift_pair(s_low[l - 1u], s_low[l], s_low[l + 1u], s_low[l + 2u],
                    s_high[h - 2u], s_high[h - 1u], s_high[h], s_high[h + 1u], s_high[h + 2u]);
            } else {
                value.x = row_beta(r, k);
                if (k < pc.high_width) {
                    value.y = row_alpha(r, k);
                }
            }
        }
        s_rows[r * TILE_X + 2u * pair] = value.x;
        s_rows[r * TILE_X + 2u * pair + 1u] = value.y;
    }
    barrier();

    // Vertical: output pair (rows 2k, 2k+1) of every tile column, written out.
    for (uint i = lid; i < PAIRS_Y * TILE_X; i += THREADS) {
        uint col = i % TILE_X;
        uint k = ay + i / TILE_X;
        uint x = x0 + col;
        if (x >= out_width || k >= pc.low_height) {
            continue;
        }
        bool has_high = k < pc.high_height;
        vec2 value = vec2(0.0);
        if (!has_high_y) {
            value.x = col_low(k, col);
        } else if (k >= 2u && k + 2u < pc.high_height) {
            uint l = low_slot(k, ay) * TILE_X + col;
            uint h = HIGH_ROW0 + high_slot(k, ay) * TILE_X + col;
            value = lift_pair(s_rows[l - TILE_X], s_rows[l], s_rows[l + TILE_X], s_rows[l + 2u * TILE_X],
                s_rows[h - 2u * TILE_X], s_rows[h - TILE_X], s_rows[h], s_rows[h + TILE_X], s_rows[h + 2u * TILE_X]);
        } else {
            value.x = col_beta(k, col);
            if (has_high) {
                value.y = col_alpha(k, col);
            }
        }
        uint y = 2u * k;
        output_band[out_offset + y * out_width + x] = value.x;
        if (has_high) {
            output_band[out_offset + (y + 1u) * out_width + x] = value.y;
        }
    }
}
