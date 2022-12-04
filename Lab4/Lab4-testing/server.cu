#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

// #define N 255
#define N 255
#define MAX_ERR 1e-6
#define I 1000
#define BLOCK_SIZE 16
#define GRID_SIZE 1

__global__ void updateGrid(float *d_G[N + 2], float *d_H[N + 2])
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N + 1 && j < N + 1 && i > 0 && j > 0)
    {
        d_G[i][j] = 0.25 * (d_H[i - 1][j] + d_H[i + 1][j] +
                            d_H[i][j - 1] + d_H[i][j + 1]);
    }

    return;
}

int main()
{

    // float *g, *h;
    float *d_G, *d_H;

    // Allocate host memory
    float *g[N + 2];
    float *h[N + 2];

    for (int i = 0; i < N + 2; i++)
    {
        g[i] = (float *)malloc(N + 2 * sizeof(float));
        h[i] = (float *)malloc(N + 2 * sizeof(float));
    }

    // Allocate device memory
    cudaMalloc((void **)&d_G, sizeof(float[N + 2][N + 2]));
    cudaMalloc((void **)&d_H, sizeof(float[N + 2][N + 2]));

    for (int i = 0; i < N + 2; i++)
    {
        for (int j = 0; j < N + 2; j++)
        {
            printf("%f ", h[i][j]);
        }
        printf("\n");
    }

    // 0 - corner
    // 1->N+1 interior points
    // N+2 - corner

    for (int i = 0; i < N + 2; i++)
    {
        h[0][i] = 20.0f;
        h[i][0] = 20.0f;
        h[i][N + 1] = 20.0f;
        h[N + 1][i] = 20.0f;
    }

    // fix this
    int offset = (N / 5);

    for (int i = offset * 2; i < offset * 4; i++)
    {

        h[0][i] = 100.0f;
    }

    for (int i = 0; i < N + 2; i++)
    {
        for (int j = 0; j < N + 2; j++)
        {
            printf("%f ", h[i][j]);
        }
        printf("\n");
    }

    // Transfer data from host to device memory
    cudaMemcpy(d_G, g, sizeof(float) * N + 2 * N + 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, h, sizeof(float) * N + 2 * N + 2, cudaMemcpyHostToDevice);

    // Kernel invocation
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Executing kernel
    // int block_size = 1024;
    // int grid_size = ((N + block_size) / block_size);

    for (int i = 0; i < I; i++)
    {
        if (i % 2 == 0)
        {
            updateGrid<<<numBlocks, threadsPerBlock>>>(&d_G, &d_H);
        }
        else
        {
            updateGrid<<<numBlocks, threadsPerBlock>>>(&d_H, &d_G);
        }
    }

    // Transfer data back to host memory
    cudaMemcpy(g, d_G, sizeof(float) * N + 2 * N + 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h, d_H, sizeof(float) * N + 2 * N + 2, cudaMemcpyDeviceToHost);

    FILE *fptr;
    fptr = fopen("finalTemperatures.csv", "w");

    if (fptr == NULL)
    {
        printf("Error!");
        exit(1);
    }

    for (i = 0; i < lines; i++)
    {
        for (j = 0; j < num; j++)
        {
            fprintf(fptr, "%d ", array[i][j]);
        }
        fprintf(fptr, "\n");
    }

    for (int i = 0; i < N + 2; i++)
    {
        for (int j = 0; j < N + 2; j++)
        {
            printf("%f ", h[i][j]);
        }
        printf("\n");
    }

    // Deallocate device memory
    cudaFree(d_G);
    cudaFree(d_H);

    // Deallocate host memory
    // free(g);
    // free(h);

    return 0;
}