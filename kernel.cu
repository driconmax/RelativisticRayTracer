#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <fstream>
#include <algorithm>

struct uchar3_ { unsigned char x, y, z; };

__global__ void renderGradient(uchar3_* out, int w, int h)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    float u = (x + 0.5f) / (float)w;
    float v = (y + 0.5f) / (float)h;

    // Simple gradient + vignette-ish
    float r = u;
    float g = v;
    float b = 0.2f + 0.8f * (1.0f - (u - 0.5f) * (u - 0.5f) - (v - 0.5f) * (v - 0.5f));

    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);

    int idx = y * w + x;
    out[idx].x = (unsigned char)(255.0f * r);
    out[idx].y = (unsigned char)(255.0f * g);
    out[idx].z = (unsigned char)(255.0f * b);
}

static void writePPM(const char* path, const std::vector<uchar3_>& pixels, int w, int h)
{
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(pixels.data()), (std::streamsize)pixels.size() * 3);
    f.close();
}

int main()
{
    int w = 800, h = 450;

    // CUDA device check (optional, but nice)
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count <= 0) {
        printf("No CUDA devices found.\n");
        return 1;
    }
    cudaSetDevice(0);

    // Allocate device output
    uchar3_* d_out = nullptr;
    size_t bytes = (size_t)w * (size_t)h * sizeof(uchar3_);
    cudaMalloc((void**)&d_out, bytes);

    // Launch
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    renderGradient <<<grid, block >>> (d_out, w, h);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    // Copy back and write
    std::vector<uchar3_> host((size_t)w * (size_t)h);
    cudaMemcpy(host.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    writePPM("output.ppm", host, w, h);
    printf("Wrote output.ppm (%dx%d)\n", w, h);
    return 0;
}
