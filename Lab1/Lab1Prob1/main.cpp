/*
Author: Pratiksha Pai
Date last modified: 10/01/2022
Organization: ECE6122

Description:

Function to output unique number of grid paths for a given grid size
The output file can be found at ./NumberPaths.txt
We use a custom cpp file - numberGridPaths to perform the logical operations.
*/

#include <iostream>
#include <fstream>
#include "numberGridPaths.cpp"
using namespace std;

int main(int argc, char *argv[]){
    string outFileName = "NumberPaths.txt";
    std::ofstream outFile;
    outFile.open(outFileName);

    if (argc != 3){
        outFile << "Invalid Input!";
        outFile.close();
        return 1;
    }
    int nRows = 0;
    int nCols = 0;
    try {
        nRows = stoi(argv[1]);
        nCols = stoi(argv[2]);
    }
    catch(std::invalid_argument& e){
        outFile << "Invalid Input!";
        outFile.close();
        return 1;
    }
    string nRowsString = argv[1];
    string nColsString = argv[2];

    if (nRowsString.rfind("0", 0) == 0 and nRowsString.length()>1){
        outFile << "Invalid Input!";
        outFile.close();
        return 1;
    }
    if (nColsString.rfind("0", 0) == 0 and nColsString.length()>1){
        outFile << "Invalid Input!";
        outFile.close();
        return 1;
    }
    uint64_t res = numberGridPaths(nRows, nCols);
    if (res==-1){
        outFile << "Invalid Input!";
    }else{
        outFile << "Total Number Paths: "<<res<<endl;
    }
    outFile.close();
    return 0;

}
 