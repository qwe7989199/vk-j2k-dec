#version 450

// One level of the reversible 5/3 inverse DWT for three components in a single dispatch.
//
// Each workgroup reconstructs a 32x32 output tile of one component (gl_WorkGroupID.z): it stages
// the LL/HL/LH/HH samples the tile depends on (including the one-sample halo) in shared memory,
// runs the horizontal lifting for every row the tile needs, then the vertical lifting. The
// integer formulas and edge clamping are those of idwt53_offset_1d.comp.glsl, so the result is
// bit-identical to its horizontal-then-vertical three-dispatch sequence.
//
// Subbands start at even coordinates (low sample first), as in idwt53_offset_1d.

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Current LL: the subband plane for the first level, the previous level's output afterwards.
layout(std430, set = 0, binding = 0) readonly buffer LowLowBand {
    int ll_band[];
};

// HL, LH and HH subband planes.
layout(std430, set = 0, binding = 1) readonly buffer SubbandPlanes {
    int planes[];
};

layout(std430, set = 0, binding = 2) writeonly buffer OutputBand {
    int output_band[];
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

const uint TILE = 32u;
const uint THREADS = 256u;
// A tile of 2n output samples needs n+1 low and n+2 high samples along each axis.
const uint LOW_SLOTS = TILE / 2u + 1u;
const uint HIGH_SLOTS = TILE / 2u + 2u;

shared int s_ll[LOW_SLOTS * LOW_SLOTS];
shared int s_hl[LOW_SLOTS * HIGH_SLOTS];
shared int s_lh[HIGH_SLOTS * LOW_SLOTS];
shared int s_hh[HIGH_SLOTS * HIGH_SLOTS];
// Horizontally reconstructed rows, one entry per tile column.
shared int s_row_low[LOW_SLOTS * TILE];
shared int s_row_high[HIGH_SLOTS * TILE];

uint component;
// First low sample of the tile along x / y; high slots start one sample earlier.
uint ax;
uint ay;
int hx_base;
int hy_base;

uint pick(uint c, uint v0, uint v1, uint v2) {
    return c == 0u ? v0 : (c == 1u ? v1 : v2);
}

// Real subband indices held by each shared slot.
uint low_col(uint slot) {
    return min(ax + slot, pc.low_width - 1u);
}

uint high_col(uint slot) {
    return uint(clamp(hx_base + int(slot), 0, int(pc.high_width) - 1));
}

uint low_row(uint slot) {
    return min(ay + slot, pc.low_height - 1u);
}

uint high_row(uint slot) {
    return uint(clamp(hy_base + int(slot), 0, int(pc.high_height) - 1));
}

// Horizontal lifting of staged row `r` (an LL/HL row, or an LH/HH row when `high_row_band`).

int h_low(bool high_row_band, uint r, uint index) {
    uint slot = r * LOW_SLOTS + (index - ax);
    return high_row_band ? s_lh[slot] : s_ll[slot];
}

int h_high(bool high_row_band, uint r, uint index) {
    if (pc.high_width == 0u) {
        return 0;
    }
    uint slot = r * HIGH_SLOTS + uint(int(index) - hx_base);
    return high_row_band ? s_hh[slot] : s_hl[slot];
}

int h_high_at(bool high_row_band, uint r, uint index) {
    if (pc.high_width == 0u) {
        return 0;
    }
    return h_high(high_row_band, r, min(index, pc.high_width - 1u));
}

int h_updated_low(bool high_row_band, uint r, uint index) {
    int low = h_low(high_row_band, r, index);
    int left = index == 0u ? h_high_at(high_row_band, r, 0u) : h_high_at(high_row_band, r, index - 1u);
    int right = h_high_at(high_row_band, r, index);
    return low - ((left + right + 2) >> 2);
}

int horizontal(bool high_row_band, uint r, uint x) {
    if ((x & 1u) == 0u) {
        return h_updated_low(high_row_band, r, x >> 1u);
    }
    uint high_index = x >> 1u;
    int high = h_high(high_row_band, r, high_index);
    int left = h_updated_low(high_row_band, r, high_index);
    int right = h_updated_low(high_row_band, r, min(high_index + 1u, pc.low_width - 1u));
    return high + ((left + right) >> 1);
}

// Vertical lifting over the horizontally reconstructed rows, for tile column `col`.

int v_low(uint index, uint col) {
    return s_row_low[(index - ay) * TILE + col];
}

int v_high(uint index, uint col) {
    if (pc.high_height == 0u) {
        return 0;
    }
    return s_row_high[uint(int(index) - hy_base) * TILE + col];
}

int v_high_at(uint index, uint col) {
    if (pc.high_height == 0u) {
        return 0;
    }
    return v_high(min(index, pc.high_height - 1u), col);
}

int v_updated_low(uint index, uint col) {
    int low = v_low(index, col);
    int left = index == 0u ? v_high_at(0u, col) : v_high_at(index - 1u, col);
    int right = v_high_at(index, col);
    return low - ((left + right + 2) >> 2);
}

int vertical(uint y, uint col) {
    if ((y & 1u) == 0u) {
        return v_updated_low(y >> 1u, col);
    }
    uint high_index = y >> 1u;
    int high = v_high(high_index, col);
    int top = v_updated_low(high_index, col);
    int bottom = v_updated_low(min(high_index + 1u, pc.low_height - 1u), col);
    return high + ((top + bottom) >> 1);
}

void main() {
    component = gl_WorkGroupID.z;
    uint x0 = gl_WorkGroupID.x * TILE;
    uint y0 = gl_WorkGroupID.y * TILE;
    uint out_width = pc.low_width + pc.high_width;
    uint out_height = pc.low_height + pc.high_height;
    ax = x0 >> 1u;
    ay = y0 >> 1u;
    hx_base = int(ax) - 1;
    hy_base = int(ay) - 1;
    uint lid = gl_LocalInvocationIndex;

    uint ll_offset = pick(component, pc.ll_offset0, pc.ll_offset1, pc.ll_offset2);
    uint hl_offset = pick(component, pc.hl_offset0, pc.hl_offset1, pc.hl_offset2);
    uint lh_offset = pick(component, pc.lh_offset0, pc.lh_offset1, pc.lh_offset2);
    uint hh_offset = pick(component, pc.hh_offset0, pc.hh_offset1, pc.hh_offset2);
    uint out_offset = pick(component, pc.out_offset0, pc.out_offset1, pc.out_offset2);

    bool has_high_x = pc.high_width > 0u;
    bool has_high_y = pc.high_height > 0u;

    for (uint i = lid; i < LOW_SLOTS * LOW_SLOTS; i += THREADS) {
        uint r = i / LOW_SLOTS;
        uint c = i % LOW_SLOTS;
        s_ll[i] = ll_band[ll_offset + low_row(r) * pc.ll_stride + low_col(c)];
    }
    if (has_high_x) {
        for (uint i = lid; i < LOW_SLOTS * HIGH_SLOTS; i += THREADS) {
            uint r = i / HIGH_SLOTS;
            uint c = i % HIGH_SLOTS;
            s_hl[i] = planes[hl_offset + low_row(r) * pc.hl_stride + high_col(c)];
        }
    }
    if (has_high_y) {
        for (uint i = lid; i < HIGH_SLOTS * LOW_SLOTS; i += THREADS) {
            uint r = i / LOW_SLOTS;
            uint c = i % LOW_SLOTS;
            s_lh[i] = planes[lh_offset + high_row(r) * pc.lh_stride + low_col(c)];
        }
    }
    if (has_high_x && has_high_y) {
        for (uint i = lid; i < HIGH_SLOTS * HIGH_SLOTS; i += THREADS) {
            uint r = i / HIGH_SLOTS;
            uint c = i % HIGH_SLOTS;
            s_hh[i] = planes[hh_offset + high_row(r) * pc.hh_stride + high_col(c)];
        }
    }
    barrier();

    for (uint i = lid; i < LOW_SLOTS * TILE; i += THREADS) {
        uint x = x0 + i % TILE;
        s_row_low[i] = x < out_width ? horizontal(false, i / TILE, x) : 0;
    }
    if (has_high_y) {
        for (uint i = lid; i < HIGH_SLOTS * TILE; i += THREADS) {
            uint x = x0 + i % TILE;
            s_row_high[i] = x < out_width ? horizontal(true, i / TILE, x) : 0;
        }
    }
    barrier();

    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    for (uint dy = 0u; dy < TILE; dy += 16u) {
        uint y = y0 + ty + dy;
        for (uint dx = 0u; dx < TILE; dx += 16u) {
            uint col = tx + dx;
            uint x = x0 + col;
            if (x < out_width && y < out_height) {
                output_band[out_offset + y * out_width + x] = vertical(y, col);
            }
        }
    }
}
