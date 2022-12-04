#include <iostream>
#include <fstream>
#include "numberGridPaths.cpp"
using namespace std;

int main(int argc, char *argv[]){

    string outFileName = "NumberPaths.txt";
    std::ofstream outFile;
    outFile.open(outFileName);

    if (argc != 3){
        cout << "Invalid Input!";
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
        cout << "Invalid Input!";
        outFile << "Invalid Input!";
        outFile.close();
        return 1;
    }
    string nRowsString = argv[1];
    string nColsString = argv[2];

    if (nRowsString.rfind("0", 0) == 0 and nRowsString.length()>1){
        cout << "Invalid Input!";
        outFile << "Invalid Input!";
        outFile.close();
        return 1;
    }
    if (nColsString.rfind("0", 0) == 0 and nColsString.length()>1){
        cout << "Invalid Input!";
        outFile << "Invalid Input!";
        outFile.close();
        return 1;
    }

    cout<<nRows<<" "<<nCols<<endl;
    uint64_t res = numberGridPaths(nRows, nCols);
    if (res==-1){
        outFile << "Invalid Input!";
    }else{
        cout << "Total Number Paths: "<<res<<endl;
        outFile << "Total Number Paths: "<<res<<endl;
    }
    outFile.close();
    return 0;

}
 