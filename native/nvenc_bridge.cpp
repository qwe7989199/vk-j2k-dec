// Optional Windows Vulkan -> CUDA -> FFmpeg h264_nvenc bridge.
// Uncompressed pictures stay on the GPU. Host fence/stream waits are intentional:
// Vulkan releases ownership before CUDA reads; CUDA finishes reading before reuse.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <cuda.h>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
}
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static thread_local std::string last_error;
static void cucheck(CUresult status, const char* name) {
    if (status == CUDA_SUCCESS) return;
    const char* text = nullptr;
    cuGetErrorString(status, &text);
    throw std::runtime_error(std::string(name) + ": " + (text ? text : "CUDA error"));
}
static void avcheck(int status, const char* name) {
    if (status >= 0) return;
    char text[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(status, text, sizeof(text));
    throw std::runtime_error(std::string(name) + ": " + text);
}
struct CurrentContext {
    explicit CurrentContext(CUcontext context) { cucheck(cuCtxPushCurrent(context), "cuCtxPushCurrent"); }
    ~CurrentContext() { CUcontext previous; cuCtxPopCurrent(&previous); }
};
struct SharedBuffer {
    uint64_t memory;
    uint64_t allocation_size;
    uint64_t size;
};
struct Import {
    CUexternalMemory external = nullptr;
    CUdeviceptr pointer = 0;
    uint64_t size = 0;
};
struct Encoder {
    AVBufferRef* device = nullptr;
    AVBufferRef* frames = nullptr;
    AVCodecContext* codec = nullptr;
    AVFormatContext* mux = nullptr;
    AVStream* video = nullptr;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUmodule module = nullptr;
    CUfunction convert = nullptr;
    std::vector<Import> imports;
    uint64_t hash = 0xcbf29ce484222325ULL;
    uint64_t submitted = 0, packets = 0;
    unsigned int width = 0, height = 0;
    bool finished = false;

    ~Encoder() {
        // NVENC must release its frame references before CUDA imports/contexts go away.
        if (context) cuCtxPushCurrent(context);
        if (context) cuStreamSynchronize(stream);
        avcodec_free_context(&codec);
        for (auto& item : imports) {
            if (item.pointer) cuMemFree(item.pointer);
            if (item.external) cuDestroyExternalMemory(item.external);
        }
        if (module) cuModuleUnload(module);
        if (context) { CUcontext previous; cuCtxPopCurrent(&previous); }
        av_buffer_unref(&frames);
        av_buffer_unref(&device);
        if (mux) {
            if (mux->pb) avio_closep(&mux->pb);
            avformat_free_context(mux);
        }
    }

    void drain(bool flushing = false) {
        AVPacket* raw = av_packet_alloc();
        if (!raw) throw std::bad_alloc();
        std::unique_ptr<AVPacket, void(*)(AVPacket*)> packet(raw, [](AVPacket* p) { av_packet_free(&p); });
        for (;;) {
            int status = avcodec_receive_packet(codec, raw);
            if (status == AVERROR_EOF) break;
            if (status == AVERROR(EAGAIN)) {
                if (flushing) throw std::runtime_error("encoder requested input after flush");
                break;
            }
            avcheck(status, "avcodec_receive_packet");
            // h264_nvenc does not propagate AVFrame.duration in all FFmpeg builds.
            // This bridge uses a fixed 24 fps time base, including the final packet.
            raw->duration = 1;
            av_packet_rescale_ts(raw, codec->time_base, video->time_base);
            raw->stream_index = video->index;
            avcheck(av_interleaved_write_frame(mux, raw), "av_interleaved_write_frame");
            ++packets;
            av_packet_unref(raw);
        }
    }
};

#define EXPORT extern "C" __declspec(dllexport)
EXPORT const char* nvenc_error() { return last_error.c_str(); }
EXPORT void nvenc_destroy(Encoder* encoder) { delete encoder; }

EXPORT Encoder* nvenc_create(
    VkInstance instance, VkPhysicalDevice physical, VkDevice vk_device,
    const SharedBuffer* buffers, uint32_t count, uint32_t width, uint32_t height,
    const char* output, const char* ptx, const char* preset, int qp) {
    try {
        if (!width || !height || width % 2 || height % 2 || !count)
            throw std::runtime_error("NV12 requires nonzero even frame dimensions and shared buffers");
        auto encoder = std::make_unique<Encoder>();
        encoder->width = width; encoder->height = height;
        cucheck(cuInit(0), "cuInit");
        // Match the physical GPU by UUID, never assume CUDA ordinal 0 matches Vulkan.
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props.pNext = &id;
        auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2"));
        if (!get_props) throw std::runtime_error("vkGetPhysicalDeviceProperties2 unavailable");
        get_props(physical, &props);
        int devices = 0, ordinal = -1;
        cucheck(cuDeviceGetCount(&devices), "cuDeviceGetCount");
        for (int i = 0; i < devices; ++i) {
            CUdevice candidate; CUuuid uuid;
            cucheck(cuDeviceGet(&candidate, i), "cuDeviceGet");
            cucheck(cuDeviceGetUuid(&uuid, candidate), "cuDeviceGetUuid");
            if (memcmp(uuid.bytes, id.deviceUUID, VK_UUID_SIZE) == 0) { ordinal = i; break; }
        }
        if (ordinal < 0) throw std::runtime_error("no CUDA GPU matches the Vulkan device UUID");
        std::string ordinal_text = std::to_string(ordinal);
        avcheck(av_hwdevice_ctx_create(&encoder->device, AV_HWDEVICE_TYPE_CUDA, ordinal_text.c_str(), nullptr, 0), "av_hwdevice_ctx_create");
        auto hw = reinterpret_cast<AVHWDeviceContext*>(encoder->device->data);
        auto cuda = reinterpret_cast<AVCUDADeviceContext*>(hw->hwctx);
        encoder->context = cuda->cuda_ctx;
        encoder->stream = cuda->stream;
        CurrentContext current(encoder->context);
        cucheck(cuModuleLoad(&encoder->module, ptx), "cuModuleLoad");
        cucheck(cuModuleGetFunction(&encoder->convert, encoder->module, "rgb48_to_nv12"), "cuModuleGetFunction");

        auto get_handle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(vk_device, "vkGetMemoryWin32HandleKHR"));
        if (!get_handle) throw std::runtime_error("vkGetMemoryWin32HandleKHR unavailable");
        encoder->imports.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            auto& item = encoder->imports[i];
            if (buffers[i].size < uint64_t(width) * height * 6)
                throw std::runtime_error("shared RGB48 buffer is too small");
            VkMemoryGetWin32HandleInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
            info.memory = reinterpret_cast<VkDeviceMemory>(buffers[i].memory);
            info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
            HANDLE handle = nullptr;
            if (get_handle(vk_device, &info, &handle) != VK_SUCCESS)
                throw std::runtime_error("vkGetMemoryWin32HandleKHR failed");
            CUDA_EXTERNAL_MEMORY_HANDLE_DESC desc{};
            desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
            desc.handle.win32.handle = handle;
            desc.size = buffers[i].allocation_size;
            desc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
            CUresult imported = cuImportExternalMemory(&item.external, &desc);
            CloseHandle(handle); // CUDA does not take ownership of Win32 handles.
            cucheck(imported, "cuImportExternalMemory");
            CUDA_EXTERNAL_MEMORY_BUFFER_DESC mapped{};
            mapped.size = buffers[i].size;
            cucheck(cuExternalMemoryGetMappedBuffer(&item.pointer, item.external, &mapped), "cuExternalMemoryGetMappedBuffer");
            item.size = mapped.size;
        }

        encoder->frames = av_hwframe_ctx_alloc(encoder->device);
        if (!encoder->frames) throw std::bad_alloc();
        auto pool = reinterpret_cast<AVHWFramesContext*>(encoder->frames->data);
        pool->format = AV_PIX_FMT_CUDA;
        pool->sw_format = AV_PIX_FMT_NV12;
        pool->width = width; pool->height = height;
        avcheck(av_hwframe_ctx_init(encoder->frames), "av_hwframe_ctx_init");
        const AVCodec* impl = avcodec_find_encoder_by_name("h264_nvenc");
        if (!impl) throw std::runtime_error("FFmpeg h264_nvenc encoder missing");
        encoder->codec = avcodec_alloc_context3(impl);
        if (!encoder->codec) throw std::bad_alloc();
        auto codec = encoder->codec;
        codec->width = width; codec->height = height;
        codec->pix_fmt = AV_PIX_FMT_CUDA;
        codec->time_base = {1, 24}; codec->framerate = {24, 1};
        codec->gop_size = 48; codec->max_b_frames = 0;
        codec->color_range = AVCOL_RANGE_MPEG;
        codec->colorspace = AVCOL_SPC_BT709;
        codec->color_primaries = AVCOL_PRI_BT709;
        codec->color_trc = AVCOL_TRC_BT709;
        codec->hw_frames_ctx = av_buffer_ref(encoder->frames);
        if (!codec->hw_frames_ctx) throw std::bad_alloc();
        avcheck(avformat_alloc_output_context2(&encoder->mux, nullptr, "mp4", output), "avformat_alloc_output_context2");
        if (encoder->mux->oformat->flags & AVFMT_GLOBALHEADER) codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        avcheck(av_opt_set(codec->priv_data, "preset", preset, 0), "preset");
        avcheck(av_opt_set(codec->priv_data, "tune", "hq", 0), "tune");
        avcheck(av_opt_set(codec->priv_data, "rc", "constqp", 0), "rc");
        avcheck(av_opt_set_int(codec->priv_data, "qp", qp, 0), "qp");
        avcheck(av_opt_set_int(codec->priv_data, "rc-lookahead", 0, 0), "lookahead");
        avcheck(avcodec_open2(codec, impl, nullptr), "avcodec_open2");
        encoder->video = avformat_new_stream(encoder->mux, nullptr);
        if (!encoder->video) throw std::bad_alloc();
        encoder->video->time_base = codec->time_base;
        avcheck(avcodec_parameters_from_context(encoder->video->codecpar, codec), "avcodec_parameters_from_context");
        avcheck(avio_open(&encoder->mux->pb, output, AVIO_FLAG_WRITE), "avio_open");
        avcheck(avformat_write_header(encoder->mux, nullptr), "avformat_write_header");
        return encoder.release();
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

EXPORT int nvenc_submit_batch(Encoder* encoder, uint32_t buffer_index, uint32_t count, int64_t pts, int verify) {
    try {
        if (encoder->finished || !count || uint64_t(buffer_index) + count > encoder->imports.size()) throw std::runtime_error("invalid encode state/index");
        CurrentContext current(encoder->context);
        using FramePtr = std::unique_ptr<AVFrame, void(*)(AVFrame*)>;
        std::vector<FramePtr> batch;
        batch.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            auto& item = encoder->imports[buffer_index + i];
            if (verify) {
                // Explicit correctness mode only; disabled in all performance runs.
                std::vector<uint16_t> samples(size_t(encoder->width) * encoder->height * 3);
                cucheck(cuMemcpyDtoH(samples.data(), item.pointer, samples.size() * 2), "verification readback");
                for (uint16_t sample : samples) encoder->hash = (encoder->hash ^ sample) * 0x100000001b3ULL;
            }
            AVFrame* raw = av_frame_alloc();
            if (!raw) throw std::bad_alloc();
            FramePtr frame(raw, [](AVFrame* p) { av_frame_free(&p); });
            avcheck(av_hwframe_get_buffer(encoder->frames, raw, 0), "av_hwframe_get_buffer");
            raw->pts = pts + i;
            raw->duration = 1;
            raw->color_range = encoder->codec->color_range;
            raw->colorspace = encoder->codec->colorspace;
            raw->color_primaries = encoder->codec->color_primaries;
            raw->color_trc = encoder->codec->color_trc;
            CUdeviceptr y = reinterpret_cast<CUdeviceptr>(raw->data[0]);
            CUdeviceptr uv = reinterpret_cast<CUdeviceptr>(raw->data[1]);
            unsigned int yp = raw->linesize[0], uvp = raw->linesize[1];
            void* args[] = {&item.pointer, &y, &uv, &yp, &uvp, &encoder->width, &encoder->height};
            cucheck(cuLaunchKernel(encoder->convert, (encoder->width + 31) / 32, (encoder->height + 15) / 16, 1,
                16, 8, 1, 0, encoder->stream, args, nullptr), "cuLaunchKernel");
            batch.push_back(std::move(frame));
        }
        // Makes NV12 visible to NVENC and returns shared RGB ownership to Vulkan safely.
        cucheck(cuStreamSynchronize(encoder->stream), "cuStreamSynchronize");
        for (auto& frame : batch) {
        AVFrame* raw = frame.get();
        int status = avcodec_send_frame(encoder->codec, raw);
        if (status == AVERROR(EAGAIN)) { encoder->drain(); status = avcodec_send_frame(encoder->codec, raw); }
        avcheck(status, "avcodec_send_frame");
        ++encoder->submitted;
        encoder->drain();
        }
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
EXPORT int nvenc_finish(Encoder* encoder, uint64_t* frames, uint64_t* packets, uint64_t* hash) {
    try {
        if (!encoder->finished) {
            CurrentContext current(encoder->context);
            avcheck(avcodec_send_frame(encoder->codec, nullptr), "flush encoder");
            encoder->drain(true);
            avcheck(av_write_trailer(encoder->mux), "av_write_trailer");
            avcheck(avio_closep(&encoder->mux->pb), "avio_closep");
            encoder->finished = true;
        }
        *frames = encoder->submitted; *packets = encoder->packets; *hash = encoder->hash;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
