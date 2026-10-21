#version 450

// One level of the irreversible 9/7 inverse DWT for three components in a single dispatch.
//
// Each workgroup reconstructs a 32x32 output tile of one component (gl_WorkGroupID.z): it stages
// the LL/HL/LH/HH samples the tile depends on (including the lifting halo) in shared memory,
// runs the horizontal lifting for every row the tile needs, then the vertical lifting. Each
// lifting step updates one parity in place from the other, so a step reads values the previous
// step finished. The float formulas, the K scaling of high samples and the edge clamping are
// those of idwt97_offset_1d.comp.glsl (horizontal pass, then vertical).
//
// Subbands start at even coordinates (low sample first), as in idwt97_offset_1d.

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

const uint TILE = 32u;
const uint THREADS = 256u;
// A tile of 2n output samples starting at low index a needs low samples a-1..a+n+1 and high
// samples a-2..a+n+1 along each axis (four lifting steps, one sample of reach each).
const uint LOW_SLOTS = TILE / 2u + 3u;
const uint HIGH_SLOTS = TILE / 2u + 4u;
const int LOW_BASE = -1;
const int HIGH_BASE = -2;
// Staged rows: LOW_SLOTS rows from LL/HL (vertical lows), then HIGH_SLOTS rows from LH/HH.
const uint ROWS = LOW_SLOTS + HIGH_SLOTS;

shared float s_low[ROWS * LOW_SLOTS];
shared float s_high[ROWS * HIGH_SLOTS];
// Horizontally reconstructed rows, one entry per tile column, in the same row order.
shared float s_rows[ROWS * TILE];

uint pick(uint c, uint v0, uint v1, uint v2) {
    return c == 0u ? v0 : (c == 1u ? v1 : v2);
}

// Band index held by `slot` of an axis whose slots start at `origin + base`.
uint slot_index(uint slot, uint origin, int base, uint count) {
    return uint(clamp(int(origin) + base + int(slot), 0, int(count) - 1));
}

// Slot of band index `index` (always inside the staged range for the slots a step reads).
uint low_slot(uint index, uint origin) {
    return uint(int(index) - int(origin) - LOW_BASE);
}

uint high_slot(uint index, uint origin) {
    return uint(int(index) - int(origin) - HIGH_BASE);
}

// High neighbours of low sample `index`: index-1 (index 0 uses 0) and index, clamped to the band.
uvec2 high_neighbor_slots(uint index, uint high_count, uint origin) {
    uint left = index == 0u ? 0u : min(index - 1u, high_count - 1u);
    uint right = min(index, high_count - 1u);
    return uvec2(high_slot(left, origin), high_slot(right, origin));
}

// Low neighbours of high sample `index` (already clamped): index and index+1, clamped.
uvec2 low_neighbor_slots(uint index, uint low_count, uint origin) {
    return uvec2(low_slot(index, origin), low_slot(min(index + 1u, low_count - 1u), origin));
}

void main() {
    uint component = gl_WorkGroupID.z;
    uint x0 = gl_WorkGroupID.x * TILE;
    uint y0 = gl_WorkGroupID.y * TILE;
    uint out_width = pc.low_width + pc.high_width;
    uint out_height = pc.low_height + pc.high_height;
    uint ax = x0 >> 1u;
    uint ay = y0 >> 1u;
    uint lid = gl_LocalInvocationIndex;

    uint ll_offset = pick(component, pc.ll_offset0, pc.ll_offset1, pc.ll_offset2);
    uint hl_offset = pick(component, pc.hl_offset0, pc.hl_offset1, pc.hl_offset2);
    uint lh_offset = pick(component, pc.lh_offset0, pc.lh_offset1, pc.lh_offset2);
    uint hh_offset = pick(component, pc.hh_offset0, pc.hh_offset1, pc.hh_offset2);
    uint out_offset = pick(component, pc.out_offset0, pc.out_offset1, pc.out_offset2);

    bool has_high_x = pc.high_width > 0u;
    bool has_high_y = pc.high_height > 0u;
    uint rows = has_high_y ? ROWS : LOW_SLOTS;

    for (uint i = lid; i < rows * LOW_SLOTS; i += THREADS) {
        uint r = i / LOW_SLOTS;
        uint col = slot_index(i % LOW_SLOTS, ax, LOW_BASE, pc.low_width);
        s_low[i] = r < LOW_SLOTS
            ? ll_band[ll_offset + slot_index(r, ay, LOW_BASE, pc.low_height) * pc.ll_stride + col]
            : planes[lh_offset + slot_index(r - LOW_SLOTS, ay, HIGH_BASE, pc.high_height) * pc.lh_stride + col];
    }
    if (has_high_x) {
        for (uint i = lid; i < rows * HIGH_SLOTS; i += THREADS) {
            uint r = i / HIGH_SLOTS;
            uint col = slot_index(i % HIGH_SLOTS, ax, HIGH_BASE, pc.high_width);
            s_high[i] = r < LOW_SLOTS
                ? planes[hl_offset + slot_index(r, ay, LOW_BASE, pc.low_height) * pc.hl_stride + col]
                : planes[hh_offset + slot_index(r - LOW_SLOTS, ay, HIGH_BASE, pc.high_height) * pc.hh_stride + col];
        }
    }
    barrier();

    // Horizontal lifting. Without high samples every step leaves the lows unchanged.
    if (has_high_x) {
        // Delta: lows from K-scaled highs, every low slot.
        for (uint i = lid; i < rows * LOW_SLOTS; i += THREADS) {
            uint r = i / LOW_SLOTS;
            uint index = slot_index(i % LOW_SLOTS, ax, LOW_BASE, pc.low_width);
            uvec2 h = high_neighbor_slots(index, pc.high_width, ax);
            float left = s_high[r * HIGH_SLOTS + h.x] * DWT97_K;
            float right = s_high[r * HIGH_SLOTS + h.y] * DWT97_K;
            s_low[i] = s_low[i] - DWT97_DELTA * (left + right);
        }
        barrier();
        // Gamma: highs, slots 1..HIGH_SLOTS-2.
        for (uint i = lid; i < rows * (HIGH_SLOTS - 2u); i += THREADS) {
            uint r = i / (HIGH_SLOTS - 2u);
            uint slot = i % (HIGH_SLOTS - 2u) + 1u;
            uint index = slot_index(slot, ax, HIGH_BASE, pc.high_width);
            uvec2 l = low_neighbor_slots(index, pc.low_width, ax);
            float high = s_high[r * HIGH_SLOTS + slot] * DWT97_K;
            s_high[r * HIGH_SLOTS + slot] =
                high - DWT97_GAMMA * (s_low[r * LOW_SLOTS + l.x] + s_low[r * LOW_SLOTS + l.y]);
        }
        barrier();
        // Beta: lows, slots 1..LOW_SLOTS-2.
        for (uint i = lid; i < rows * (LOW_SLOTS - 2u); i += THREADS) {
            uint r = i / (LOW_SLOTS - 2u);
            uint slot = i % (LOW_SLOTS - 2u) + 1u;
            uint index = slot_index(slot, ax, LOW_BASE, pc.low_width);
            uvec2 h = high_neighbor_slots(index, pc.high_width, ax);
            uint s = r * LOW_SLOTS + slot;
            s_low[s] = s_low[s] - DWT97_BETA * (s_high[r * HIGH_SLOTS + h.x] + s_high[r * HIGH_SLOTS + h.y]);
        }
        barrier();
    }

    // Horizontal alpha step and interleave: every tile column of each staged row.
    for (uint i = lid; i < rows * TILE; i += THREADS) {
        uint r = i / TILE;
        uint x = x0 + i % TILE;
        float value = 0.0;
        if (x < out_width) {
            uint index = x >> 1u;
            if ((x & 1u) == 0u) {
                value = s_low[r * LOW_SLOTS + low_slot(index, ax)];
            } else {
                uvec2 l = low_neighbor_slots(index, pc.low_width, ax);
                float high = s_high[r * HIGH_SLOTS + high_slot(index, ax)];
                value = high - DWT97_ALPHA * (s_low[r * LOW_SLOTS + l.x] + s_low[r * LOW_SLOTS + l.y]);
            }
        }
        s_rows[i] = value;
    }
    barrier();

    // Vertical lifting on the reconstructed rows: low rows first, then high rows.
    const uint HIGH_ROW0 = LOW_SLOTS * TILE;
    if (has_high_y) {
        for (uint i = lid; i < LOW_SLOTS * TILE; i += THREADS) {
            uint col = i % TILE;
            uint index = slot_index(i / TILE, ay, LOW_BASE, pc.low_height);
            uvec2 h = high_neighbor_slots(index, pc.high_height, ay);
            float top = s_rows[HIGH_ROW0 + h.x * TILE + col] * DWT97_K;
            float bottom = s_rows[HIGH_ROW0 + h.y * TILE + col] * DWT97_K;
            s_rows[i] = s_rows[i] - DWT97_DELTA * (top + bottom);
        }
        barrier();
        for (uint i = lid; i < (HIGH_SLOTS - 2u) * TILE; i += THREADS) {
            uint col = i % TILE;
            uint slot = i / TILE + 1u;
            uint index = slot_index(slot, ay, HIGH_BASE, pc.high_height);
            uvec2 l = low_neighbor_slots(index, pc.low_height, ay);
            uint s = HIGH_ROW0 + slot * TILE + col;
            float high = s_rows[s] * DWT97_K;
            s_rows[s] = high - DWT97_GAMMA * (s_rows[l.x * TILE + col] + s_rows[l.y * TILE + col]);
        }
        barrier();
        for (uint i = lid; i < (LOW_SLOTS - 2u) * TILE; i += THREADS) {
            uint col = i % TILE;
            uint slot = i / TILE + 1u;
            uint index = slot_index(slot, ay, LOW_BASE, pc.low_height);
            uvec2 h = high_neighbor_slots(index, pc.high_height, ay);
            uint s = slot * TILE + col;
            s_rows[s] = s_rows[s]
                - DWT97_BETA * (s_rows[HIGH_ROW0 + h.x * TILE + col] + s_rows[HIGH_ROW0 + h.y * TILE + col]);
        }
        barrier();
    }

    // Vertical alpha step and output.
    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    for (uint dy = 0u; dy < TILE; dy += 16u) {
        uint y = y0 + ty + dy;
        for (uint dx = 0u; dx < TILE; dx += 16u) {
            uint col = tx + dx;
            uint x = x0 + col;
            if (x < out_width && y < out_height) {
                uint index = y >> 1u;
                float value;
                if ((y & 1u) == 0u) {
                    value = s_rows[low_slot(index, ay) * TILE + col];
                } else {
                    uvec2 l = low_neighbor_slots(index, pc.low_height, ay);
                    float high = s_rows[HIGH_ROW0 + high_slot(index, ay) * TILE + col];
                    value = high - DWT97_ALPHA * (s_rows[l.x * TILE + col] + s_rows[l.y * TILE + col]);
                }
                output_band[out_offset + y * out_width + x] = value;
            }
        }
    }
}
