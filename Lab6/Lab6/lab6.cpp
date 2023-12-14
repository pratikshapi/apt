/*
Author: Pratiksha Pai
Date last modified: 12/04/2022
Organization: ECE6122

Description:

Simple MPI implementation for 2d steady state heat conduction.
*/

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

using namespace std;

// check if it is the middle portion of the matrix
// if it is the middle portion of the matrix fill in 100s instead of 20
bool isMiddle40Percentage(const int index, const int width)
{
    return (index + 1) > round(0.3 * width) && (index + 1) <= round(0.7 * width);
}

// driver code to run the iterations
int main(int argc, char **argv)
{
    int WIDTH, numInterations;
    int NUM_SIZE = WIDTH * WIDTH;

    // input argument check
    if (argc < 5)
    {
        printf("Invalid Inputs passed!!\n");
        exit(0);
    }
    // parsing the input arguments
    for (int i = 1; i < argc; i += 2)
    {
        if (!strcmp(argv[i], "-n"))
        {
            WIDTH = atoi(argv[i + 1]);
            WIDTH += 2;
        }
        else if (!strcmp(argv[i], "-I"))
        {
            numInterations = atoi(argv[i + 1]);
        }
    }

    int rank, size, ii, jj, kk;
    int index, up, down, left, right;
    int rank_prev, rank_next;

    // Initialise MPI related variables
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // initialise the array with the modfied width
    int newWidth = size * (int)std::ceil((double)WIDTH / (double)size);
    int sendCount = newWidth * newWidth / size;
    int recvCount = sendCount;
    double *pPrevious = new double[newWidth * newWidth];
    double *pNext = new double[newWidth * newWidth];

    int block_size = (newWidth / size);

    // initialise the edges of the array
    for (int ii = 0; ii < WIDTH; ii++)
    {
        // Top Row
        if (isMiddle40Percentage(ii, WIDTH))
        {
            pPrevious[ii] = 100.0;
            pNext[ii] = 100.0;
        }
        else
        {
            pPrevious[ii] = 20.0;
            pNext[ii] = 20.0;
        }
        // Bottom row
        pPrevious[((WIDTH - 1) * newWidth) + ii] = 20.0;
        pNext[((WIDTH - 1) * newWidth) + ii] = 20.0;
        // Left Column
        pPrevious[ii * newWidth] = 20.0;
        pNext[ii * newWidth] = 20.0;
        // Right Column
        pPrevious[(ii * newWidth) + (WIDTH - 1)] = 20.0;
        pNext[(ii * newWidth) + (WIDTH - 1)] = 20.0;
    }

    // calculate where to start the calculation

    int startHere = rank * block_size;

    // auto t1 = std::chrono::high_resolution_clock::now();

    // calculate the 2d heat distribution
    for (kk = 0; kk < numInterations; kk++)
    {
        for (ii = startHere; ii < startHere + block_size; ii++)
        {
            if (ii == 0 || ii >= WIDTH - 1)
            {
                continue;
            }
            for (jj = 0; jj < WIDTH; jj++)
            {
                if (jj == 0 || jj >= WIDTH - 1)
                {
                    continue;
                }

                index = (ii * newWidth) + jj;

                up = ((ii - 1) * newWidth) + jj;
                down = ((ii + 1) * newWidth) + jj;
                left = (ii * newWidth) + (jj - 1);
                right = (ii * newWidth) + (jj + 1);

                pNext[index] = 0.25 * (pPrevious[up] + pPrevious[down] + pPrevious[left] + pPrevious[right]);
            }
        }

        // gather everything into pPrevious
        MPI_Allgather(
            &pNext[startHere * newWidth],
            sendCount,
            MPI_DOUBLE,
            pPrevious,
            recvCount,
            MPI_DOUBLE,
            MPI_COMM_WORLD);
    }

    // if (rank == 0)
    // {
    //     auto t2 = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    //     std::cout << fp_ms.count() << " ms " << endl;
    // }

    // Writing to file
    if (rank == 0)
    {
        ofstream outfile;
        outfile.open("finalTemperatures.csv");

        for (int i = 0; i < newWidth; i++)
        {
            for (int j = 0; j < newWidth; j++)
            {
                if (i >= WIDTH || j >= WIDTH)
                {
                    continue;
                }
                outfile << std::fixed << std::setprecision(15) << pPrevious[i * newWidth + j] << ",";
            }
            outfile << "\n";
        }
        outfile.close();
    }
    MPI_Finalize();
    return 0;
}