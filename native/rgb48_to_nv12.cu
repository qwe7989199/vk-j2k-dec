// RGB48 (full range, MSB-aligned) -> NV12, BT.709 limited range.
// One thread handles a 2x2 block; chroma is box-filtered before quantization.
__device__ unsigned char quantize(float x) {
    return (unsigned char)__float2int_rn(fminf(255.0f, fmaxf(0.0f, x)));
}
extern "C" __global__ void rgb48_to_nv12(
    const unsigned short* rgb, unsigned char* yplane, unsigned char* uvplane,
    unsigned int ypitch, unsigned int uvpitch, unsigned int width, unsigned int height) {
    unsigned int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    unsigned int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (x >= width || y >= height) return;
    float ravg = 0, gavg = 0, bavg = 0;
    for (unsigned int dy = 0; dy < 2; ++dy) {
        for (unsigned int dx = 0; dx < 2; ++dx) {
            unsigned int i = ((y + dy) * width + x + dx) * 3;
            float r = rgb[i] / 65535.0f;
            float g = rgb[i + 1] / 65535.0f;
            float b = rgb[i + 2] / 65535.0f;
            yplane[(y + dy) * ypitch + x + dx] = quantize(16.0f + 219.0f *
                (0.2126f * r + 0.7152f * g + 0.0722f * b));
            ravg += r * 0.25f; gavg += g * 0.25f; bavg += b * 0.25f;
        }
    }
    float luma = 0.2126f * ravg + 0.7152f * gavg + 0.0722f * bavg;
    uvplane[(y / 2) * uvpitch + x] = quantize(128.0f + 224.0f * (bavg - luma) / 1.8556f);
    uvplane[(y / 2) * uvpitch + x + 1] = quantize(128.0f + 224.0f * (ravg - luma) / 1.5748f);
}
