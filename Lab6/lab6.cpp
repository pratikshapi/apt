#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "mpi.h"
#define WIDTH 10
#define NUM_SIZE WIDTH *WIDTH
int iteration_through_all = 1;

using namespace std;

bool isMiddle40Percentage(const int index, const int width)
{
    return (index + 1) > round(0.3 * width) && (index + 1) <= round(0.7 * width);
}

int main(int argc, char **argv)
{
    int rank, size, ii, jj, kk;
    int index, up, down, left, right;
    int numInterations = 1;
    // int numInterations = 10;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // cout <<"size "<<size<<endl;
    // cout <<"rank "<<rank<<endl;
    int newWidth = size * (int)std::ceil((double)WIDTH / (double)size);
    int sendCount = newWidth * newWidth / size;
    int recvCount = sendCount;
    double *pPrevious = new double[newWidth * newWidth];
    double *pNext = new double[newWidth * newWidth];

    int block_size = (newWidth / size);

    if (rank == 0)
    {
        cout << "Num Processors: " << size << endl;
        cout << "New Width: " << newWidth << endl;
        cout << "NumRows Per Processor: " << block_size << endl;
    }

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
        pPrevious[((WIDTH - 1) * WIDTH) + ii] = 20.0;
        pNext[((WIDTH - 1) * WIDTH) + ii] = 20.0;
        // Left Column
        pPrevious[ii * WIDTH] = 20.0;
        pNext[ii * WIDTH] = 20.0;
        // Right Column
        pPrevious[((ii + 1) * WIDTH) - 1] = 20.0;
        pNext[((ii + 1) * WIDTH) - 1] = 20.0;
    }

    // Writing to file
    ofstream iterfile;
    iterfile.open("iterations.csv");
    int startHere = rank * block_size;
    
    for (kk = 0; kk < numInterations; kk++)
    {
        // iterfile << rank << ", ";
        // iterfile << startHere << ", ";
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
                printf("rank:%d, index:%d\n", rank, index);
            }
        }
        
        MPI_Allgather(
            &pNext[startHere * newWidth],
            sendCount,
            MPI_DOUBLE,
            pPrevious,
            recvCount,
            MPI_DOUBLE,
            MPI_COMM_WORLD);
    }

    // cout << "iteration_through_all " << iteration_through_all << endl;
    // Writing to file
    if (rank == 0)
    {
        ofstream outfile;
        outfile.open("pPrevious_final.csv");

        for (int i = 0; i < newWidth; i++)
        {
            for (int j = 0; j < newWidth; j++)
            {
                if(i >= WIDTH || j >= WIDTH){
                    continue;
                }
                outfile << std::fixed << std::setprecision(0) << pPrevious[i * newWidth + j] << ",";
            }
            outfile << "\n";
        }
        outfile.close();
    }
    MPI_Finalize();
    return 0;
}