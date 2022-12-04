/*
Author: Muhammed Saneens Bin Zubair
Class: ECE6122 (section)
Last Date Modified: 11/12/2022
Description: Using laplace equation to find the steady state heat 
conduction in a  2D thin place
*/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <iomanip>

using namespace::std;
///////////////////////////////////////////////////////////////////////
// Function to handle error from CUDA
///////////////////////////////////////////////////////////////////////
inline cudaError_t HANDLE_ERROR(cudaError_t result)
{
    #if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) 
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    #endif
    return result;
}
///////////////////////////////////////////////////////////////////////
// Exit if incorrect command line arguments are passed
///////////////////////////////////////////////////////////////////////
void dieUsage()
{
    printf("Invalid Inputs passed!!\n");
    exit(0);
}
///////////////////////////////////////////////////////////////////////
// Get command line options
///////////////////////////////////////////////////////////////////////
void getParams(int argc, char** argv, int* n, int* iterations)
{
    if(argc < 5)
    {
        dieUsage();
    }
    for (int i = 1; i < argc; i+=2)
    {
        if (!strcmp(argv[i], "-n")) 
        {
            *n = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "-I")) 
        {
            *iterations = atoi(argv[i+1]);
        }
        else
        {
            dieUsage();
        }
    }
}
///////////////////////////////////////////////////////////////////////
// check if this index is middle 40%
///////////////////////////////////////////////////////////////////////
bool isMiddle40Percentage(const int index, const int width)
{
    return (index + 1) > round(0.3 * width) && (index + 1) <= round(0.7 * width);
}
///////////////////////////////////////////////////////////////////////
// Kernal call
///////////////////////////////////////////////////////////////////////
__global__ void calculateHeat(double* Hd, double* Gd, int width)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x; 
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > 0 && x < (width-1) && y > 0 && y < (width-1))
    {
        Gd[x * width + y] = 0.25 * (Hd[(x - 1) * width + y] + Hd[(x + 1) * width + y] + 
        Hd[x * width + (y - 1)] + Hd[x * width + (y + 1)]);
    }
}
///////////////////////////////////////////////////////////////////////
// Main Function
///////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    int width, iterations;
    int size;
    double *G, *H;
    getParams(argc, argv, &width, &iterations);
    
    width = width + 2;
    size = width * width * sizeof(double);
    
    // allocate memory on the CPU
    H = (double*)calloc(width * width, sizeof(double));
    G = (double*)calloc(width * width, sizeof(double));
    
    // initialize the edges to 20 and middle 40% top to 100
    for (int x = 0; x < width; x++) 
    {
        // Top Row
        if(isMiddle40Percentage(x, width))
        {
            H[x] = 100;
        }
        else
        {
            H[x] = 20;
        }
        // Bottom row
        H[(width-1) * width + x] = 20;
        // Left Column
        H[x * width] = 20;
        // Right Column
        H[((x + 1) * width) - 1] = 20;
    }
    
    // Cuda
    double *Gd, *Hd;
    // Decide on the block and thread dimensions
    int numThreads = min(32, width);
    int numBlocks = (int) (width + 31)/32;
    dim3 dimBlock(numThreads, numThreads);
    dim3 dimGrid(numBlocks, numBlocks);

    // capture start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    
    // allocate memory on the GPU
    HANDLE_ERROR(cudaMallocManaged((void**)&Gd, size));
    HANDLE_ERROR(cudaMallocManaged((void**)&Hd, size));
    
    // transfer G to device memory
    HANDLE_ERROR(cudaMemcpy(Hd, H, size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(Gd, H, size, cudaMemcpyHostToDevice));
    
    // Do the calculations
    for (int i = 0; i < iterations/2; i++)
    {
        calculateHeat <<<dimGrid, dimBlock>>> (Hd, Gd, width);
        cudaDeviceSynchronize();
        calculateHeat <<<dimGrid, dimBlock>>> (Gd, Hd, width);
        cudaDeviceSynchronize();
    }

    // Get the results from the device
    HANDLE_ERROR(cudaMemcpy(H, Hd, size, cudaMemcpyDeviceToHost));

    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );
    
    
    
    // Write the results into a csv file
    ofstream myFile;
    myFile.open ("finalTemperatures.csv");
    myFile << std::fixed << setprecision(15);
    for (int x = 0; x < width; x++) 
    {
        for (int y = 0; y < width-1; y++)
        {
            myFile << H[x*width + y] << ",";
        }
        myFile << H[(x + 1)*width - 1] <<"\n";
    }
    myFile.close();
    
    // destroy events to free memory
    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );
    // free the memory allocated on the GPU
    HANDLE_ERROR(cudaFree(Gd));
    HANDLE_ERROR(cudaFree(Hd));
    
    // Free the memory allocated on CPU
    free(G);
    free(H);
    
    return 0;
}