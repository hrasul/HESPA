#include <cuda_runtime.h>
#include <stdio.h>
#include <utility>

#include "../util.h"
#include "stream-util.h"

__global__ void copyOnGPU(size_t nx, const double *__restrict__ src, double *__restrict__ dest) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
        dest[i] = src[i] + 1.0;
    }
}

int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    int blockSize = 256;
    if (argc > 4) {
        blockSize = atoi(argv[4]);
    }

    size_t size = nx * sizeof(double);

    double *src, *dest;
    cudaMallocHost(&src, size);
    cudaMallocHost(&dest, size);

    double *d_src, *d_dest;
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dest, size);

    initStream(src, nx);
    cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice);

    int numBlocks = (nx + blockSize - 1) / blockSize;

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        copyOnGPU<<<numBlocks, blockSize>>>(nx, d_src, d_dest);
        std::swap(d_src, d_dest);
    }
    cudaDeviceSynchronize();

    // measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (size_t i = 0; i < nIt; ++i) {
        copyOnGPU<<<numBlocks, blockSize>>>(nx, d_src, d_dest);
        std::swap(d_src, d_dest);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(src, d_src, size, cudaMemcpyDeviceToHost);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    double seconds = ms / 1000.0;
    double mlups = (double(nx) * double(nIt)) / seconds / 1.0e6;
    double bandwidth = (double(nx) * sizeof(double) * 2.0 * double(nIt)) / seconds / 1.0e9;

    printf("#cells / #it:  %zu / %zu\n", nx, nIt);
    printf("elapsed time:  %.6f ms\n", ms);
    printf("per iteration: %.6f ms\n", ms / nIt);
    printf("MLUP/s:        %.2f\n", mlups);
    printf("bandwidth:     %.4f GB/s\n", bandwidth);

    checkSolutionStream(src, nx, nIt + nItWarmUp);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dest);
    cudaFreeHost(src);
    cudaFreeHost(dest);

    return 0;
}