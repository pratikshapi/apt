/*
Author: Pratiksha Pai
Date last modified: 10/10/2022
Organization: ECE6122

Description:

Function to output multiplication of two matrices.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
using namespace std;


int main(int argc, char** argv)
{
    if (argc!=2){
        cout<<"No file passed as input, exiting...";
        return 1;
    }

    // define input/output streams
    string inFileName = argv[1];
    string outFileName = "MatrixOut.txt";
    std::fstream inFile;
    std::ofstream outFile;
    
    inFile.open(inFileName);

    // dimensions of matrix A
    string m_str, n_str;
    inFile >> m_str >> n_str;
    int a_m, a_n;
    a_m = stoi(m_str);
    a_n = stoi(n_str);
    
    // Prepare matrix A from the given input file
    vector< vector<double> > a_arr(a_m, std::vector<double>(a_n, 0));
    for (int i = 0 ; i < a_m ; ++i){
        for (int j = 0 ; j < a_n ; ++j){
            inFile >> a_arr[i][j];
        }
    }

    // dimensions of matrix B
    inFile >> m_str >> n_str;
    int b_m, b_n;
    b_m = stoi(m_str);
    b_n = stoi(n_str);

    // Prepare matrix B from the given input file
    vector< vector<double> > b_arr(b_m, std::vector<double>(b_n, 0));
    for (int i = 0 ; i < b_m ; ++i){
        for (int j = 0 ; j < b_n ; ++j){
            inFile >> b_arr[i][j];
        }
    }

    // Initializing elements of matrix result array to 0.
    vector< vector<double> > res_arr(a_m, std::vector<double>(b_n, 0));

    // Multiplying matrix A and B and storing in result array.
    #pragma omp parallel for 
    for(int i = 0; i < a_m; ++i){
        for(int j = 0; j < b_n; ++j){
            for(int k = 0; k < a_n; ++k){
                res_arr[i][j] += a_arr[i][k] * b_arr[k][j];
            }
        }
    }
    
    // dump the result matrix into output file
    outFile.open(outFileName);
    outFile << a_m <<" "<< b_n<<endl;
    for(int i = 0; i < a_m; ++i){
        for(int j = 0; j < b_n; ++j){
            outFile << res_arr[i][j]<<" ";
            if(j == b_n - 1)
                outFile << endl;
        }
    }
    outFile.close();

    //operations successful!
    return 0;
}

