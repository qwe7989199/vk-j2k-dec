// Optional benchmark backend. Each in-flight frame owns its CUDA stream and
// nvJPEG2000 decode state; every frame is parsed and decoded exactly once.
#include <cuda_runtime.h>
#include "nvjpeg2k.h"
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define EXPORT extern "C" __declspec(dllexport)
static thread_local std::string last_error;
static void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
static void check(nvjpeg2kStatus_t code) {
    if (code != NVJPEG2K_STATUS_SUCCESS) throw std::runtime_error("nvJPEG2000 status " + std::to_string(code));
}

__global__ void pack_rgb48(const uint16_t* r, const uint16_t* g, const uint16_t* b,
                           uint16_t* output, size_t pixels, unsigned shift) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < pixels) {
        output[3*i] = r[i] << shift;
        output[3*i+1] = g[i] << shift;
        output[3*i+2] = b[i] << shift;
    }
}

struct Lane {
    cudaStream_t stream = nullptr;
    nvjpeg2kHandle_t decoder = nullptr;
    nvjpeg2kDecodeState_t state = nullptr;
    nvjpeg2kStream_t parsed = nullptr;
    uint16_t* planes[3] = {};
    uint16_t* packed = nullptr;
    uint16_t* host = nullptr;
    std::vector<unsigned char> compressed;
    bool pending = false;
    ~Lane() {
        if (stream) cudaStreamSynchronize(stream);
        if (parsed) nvjpeg2kStreamDestroy(parsed);
        if (state) nvjpeg2kDecodeStateDestroy(state);
        if (decoder) nvjpeg2kDestroy(decoder);
        for (auto p : planes) if (p) cudaFree(p);
        if (packed) cudaFree(packed);
        if (host) cudaFreeHost(host);
        if (stream) cudaStreamDestroy(stream);
    }
};
struct Decoder {
    unsigned width, height, precision;
    size_t samples;
    std::string device;
    std::vector<std::unique_ptr<Lane>> lanes;
};

EXPORT Decoder* nvj_create(unsigned width, unsigned height, unsigned precision,
                          unsigned lanes, int device) {
    try {
        if (!width || !height || precision < 1 || precision > 16 || lanes < 1 || lanes > 16)
            throw std::runtime_error("invalid decoder dimensions/lanes");
        check(cudaSetDevice(device));
        auto result = std::make_unique<Decoder>();
        result->width = width; result->height = height; result->precision = precision;
        result->samples = size_t(width) * height * 3;
        cudaDeviceProp properties{}; check(cudaGetDeviceProperties(&properties, device));
        result->device = properties.name;
        for (unsigned i = 0; i < lanes; ++i) {
            auto lane = std::make_unique<Lane>();
            check(cudaStreamCreateWithFlags(&lane->stream, cudaStreamNonBlocking));
            check(nvjpeg2kCreateSimple(&lane->decoder));
            check(nvjpeg2kDecodeStateCreate(lane->decoder, &lane->state));
            check(nvjpeg2kStreamCreate(&lane->parsed));
            for (auto& p : lane->planes) check(cudaMalloc(&p, size_t(width) * height * 2));
            check(cudaMalloc(&lane->packed, result->samples * 2));
            check(cudaMallocHost(&lane->host, result->samples * 2));
            result->lanes.push_back(std::move(lane));
        }
        return result.release();
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

EXPORT int nvj_submit(Decoder* decoder, unsigned index, const unsigned char* data, size_t size) {
    try {
        if (index >= decoder->lanes.size()) throw std::runtime_error("invalid lane");
        auto& lane = *decoder->lanes[index];
        if (lane.pending) throw std::runtime_error("lane still in flight");
        lane.compressed.assign(data, data + size);
        check(nvjpeg2kStreamParse(lane.decoder, lane.compressed.data(), size, 0, 0, lane.parsed));
        nvjpeg2kImageInfo_t info{};
        check(nvjpeg2kStreamGetImageInfo(lane.parsed, &info));
        if (info.image_width != decoder->width || info.image_height != decoder->height || info.num_components != 3)
            throw std::runtime_error("frame geometry changed");
        for (unsigned i = 0; i < 3; ++i) {
            nvjpeg2kImageComponentInfo_t component{};
            check(nvjpeg2kStreamGetImageComponentInfo(lane.parsed, &component, i));
            if (component.component_width != info.image_width || component.component_height != info.image_height
                || component.precision != decoder->precision || component.sgn)
                throw std::runtime_error("unsupported component geometry/precision");
        }
        void* pixels[3] = {lane.planes[0], lane.planes[1], lane.planes[2]};
        size_t pitches[3] = {size_t(decoder->width)*2, size_t(decoder->width)*2, size_t(decoder->width)*2};
        nvjpeg2kImage_t output{pixels, pitches, NVJPEG2K_UINT16, 3};
        check(nvjpeg2kDecode(lane.decoder, lane.state, lane.parsed, &output, lane.stream));
        const size_t count = decoder->samples / 3;
        pack_rgb48<<<unsigned((count + 255) / 256), 256, 0, lane.stream>>>(
            lane.planes[0], lane.planes[1], lane.planes[2], lane.packed, count, 16-decoder->precision);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(lane.host, lane.packed, decoder->samples*2, cudaMemcpyDeviceToHost, lane.stream));
        lane.pending = true;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

EXPORT int nvj_wait(Decoder* decoder, unsigned index, const uint16_t** samples) {
    try {
        auto& lane = *decoder->lanes.at(index);
        if (!lane.pending) throw std::runtime_error("lane has no pending frame");
        check(cudaStreamSynchronize(lane.stream));
        lane.pending = false; *samples = lane.host;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
EXPORT const char* nvj_device(Decoder* decoder) { return decoder->device.c_str(); }
EXPORT void nvj_destroy(Decoder* decoder) { delete decoder; }
EXPORT const char* nvj_error() { return last_error.c_str(); }
