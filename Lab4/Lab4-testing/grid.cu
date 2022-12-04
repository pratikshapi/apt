/*
Author: Pratiksha Pai
Date last modified: 11/12/2022
Organization: ECE6122

Description:

Simple cuda implementation.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

// #define N 255
// #define I 1000

__global__ void updateGrid(double *d_G, double *d_H, int len)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i < (len - 1) && j < (len - 1) && i > 0 && j > 0)
    {
        d_G[(len * i) + j] = 0.25f * (d_H[(len * (i - 1)) + j] + d_H[(len * (i + 1)) + j] +
                                      d_H[(len * i) + (j - 1)] + d_H[(len * i) + (j + 1)]);
    }
}

bool isMiddle(const int index, const int len)
{
    return (index + 1) > round(0.3 * len) && (index + 1) <= round(0.7 * len);
}

int main(int argc, char **argv)
{

    if (argc != 5)
    {
        cout << "Invalid Inputs...";
        exit(1);
    }
    int N, I;
    for (int i = 1; i < argc; i += 2)
    {
        if (!strcmp(argv[i], "-n"))
        {
            N = stoi(argv[i + 1]);
        }
        else if (!strcmp(argv[i], "-I"))
        {
            I = stoi(argv[i + 1]);
        }
        else
        {
            cout << "Invalid Inputs...";
            exit(1);
        }
    }

    int len = N + 2;
    int sizeGrid = sizeof(double) * len * len;
    double *h;
    h = (double *)calloc(len * len, sizeof(double));

    // initialise the input matrix
    for (int i = 0; i < len; i++)
    {
        if (isMiddle(i, len))
        {
            h[i] = 100;
        }
        else
        {
            h[i] = 20;
        }

        h[(len - 1) * len + i] = 20;
        h[i * len] = 20;
        h[((i + 1) * len) - 1] = 20;
    }

    // Kernel dim3 initialisation
    int threadsPerBlock;
    if (len < 32)
    {
        threadsPerBlock = len;
    }
    else
    {
        threadsPerBlock = 32;
    }

    int blocksPerGrid = (int)(len + 31) / 32;
    dim3 dimBlock(threadsPerBlock, threadsPerBlock);
    dim3 dimGrid(blocksPerGrid, blocksPerGrid);

    // Clock the process
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(stop, 0);
    cudaEventRecord(start, 0);

    // Allocate device memory
    double *d_G, *d_H;
    cudaMallocManaged((void **)&d_G, sizeGrid);
    cudaMallocManaged((void **)&d_H, sizeGrid);

    // do cuda operations
    cudaMemcpy(d_H, h, sizeGrid, cudaMemcpyHostToDevice);
    cudaMemcpy(d_G, h, sizeGrid, cudaMemcpyHostToDevice);

    for (int i = 0; i < I; i++)
    {
        if (i % 2 == 0)
        {
            updateGrid<<<dimGrid, dimBlock>>>(d_G, d_H, len);
            cudaDeviceSynchronize();
        }
        else
        {
            updateGrid<<<dimGrid, dimBlock>>>(d_H, d_G, len);
            cudaDeviceSynchronize();
        }
    }

    // Transfer data back to host memory
    if (I % 2 == 0)
    {
        cudaMemcpy(h, d_H, sizeGrid, cudaMemcpyDeviceToHost);
    }
    else
    {
        cudaMemcpy(h, d_G, sizeGrid, cudaMemcpyDeviceToHost);
    }

    // get elapsed time
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to generate:  %f ms\n", elapsedTime);

    // dump the result matrix into output file
    std::ofstream outFile;
    string outName = "finalTemperatures.csv";
    outFile.open(outName);
    for (int i = 0; i < len; ++i)
    {
        for (int j = 0; j < len - 1; ++j)
        {
            outFile << h[(len * i) + j] << ",";
        }
        outFile << h[(len * (i + 1)) - 1] << endl;
    }
    outFile.close();

    return 0;
}
