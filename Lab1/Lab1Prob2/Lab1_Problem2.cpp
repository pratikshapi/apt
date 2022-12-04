/*
Author: Pratiksha Pai
Date last modified: 10/01/2022
Organization: ECE6122

Description:

Function to solve Sudoku Puzzle using multithreading.
*/

#include <iostream>
#include <mutex>
#include <fstream>
#include <thread>
#include <vector>
using namespace std;

std::mutex outFileMutex;
std::mutex inFileMutex;
std::fstream inFile;
std::ofstream outFile;
string inFileName;
string outFileName;
int N=9;

// Class which holds information regarding one 9x9 sudoku grid.
class SudokuGrid {
    private:
        std::string m_strGridName;
        unsigned char gridElement[9][9];
    public:
        friend fstream& operator>>(fstream&, SudokuGrid&);
        friend ofstream& operator<<(ofstream&, const SudokuGrid&);
        bool isSafe(int, int, int);
        bool solveSudoku(int, int);

};

// Member function of SudokuGrid that reads a single SudokuGrid object from a fstream file
fstream& operator>>(fstream& os, SudokuGrid& gridIn){
    // Read in the grid name
    std::getline(os, gridIn.m_strGridName);
    std::string strIn;
    for (int i = 0 ; i < 9 ; ++i){
        std::getline(os, strIn);
        for (int j = 0 ; j < 9 ; ++j){
            gridIn.gridElement[i][j] = strIn[j];
        }
    }
    return os;
}

// Member function of SudokuGrid that writes the SudokuGrid object to a file
ofstream& operator<<(ofstream& os, const SudokuGrid& gridOut){
    os << gridOut.m_strGridName << endl;
    for (int i = 0; i < 9; ++i){
        for (int j = 0; j < 9; ++j){
            os << gridOut.gridElement[i][j];
        }
        os << endl;
    }
    return os;
}

// Member function of SudokuGrid which checks if gridElement[row][col] filled with num is valid
bool SudokuGrid::isSafe(int row, int col, int num)
{
    for (int x = 0; x <= 8; x++)
        if (int(SudokuGrid::gridElement[row][x] - '0') == num)
            return false;

    for (int x = 0; x <= 8; x++)
        if (int(SudokuGrid::gridElement[x][col] - '0') == num)
            return false;

    int startRow = row - row % 3,
            startCol = col - col % 3;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (int(SudokuGrid::gridElement[i + startRow][j +
                                             startCol] - '0') == num)
                return false;

    return true;
}

// logic is from geeksforgeeks.org
// Member function of SudokuGrid that solves the 9x9 sudoku using backtracking
bool SudokuGrid::solveSudoku(int row, int col)
{
    if (row == N - 1 && col == N)
        return true;

    if (col == N) {
        row++;
        col = 0;
    }
    if (int(SudokuGrid::gridElement[row][col] - '0') > 0)
        return SudokuGrid::solveSudoku(row, col + 1);

    for (int num = 1; num <= N; num++)
    {
        if (SudokuGrid::isSafe(row, col, num))
        {
            SudokuGrid::gridElement[row][col] = char(num + 48);
            if (SudokuGrid::solveSudoku(row, col + 1))
                return true;
        }
        SudokuGrid::gridElement[row][col] = '0';
    }
    return false;
}

// Function that calls SudokuGrid::solveSudoku
// This function also handles the mutex lock/unlock functionality on the input and output files
void solveSudokuPuzzles(){
    SudokuGrid sudoku;
    do
    {
        inFileMutex.lock();
        if (inFile.eof()){
            break;
        }
        else{
            inFile >> sudoku;
        }
        inFileMutex.unlock();
        if (!sudoku.solveSudoku(0, 0))
            cout << "Solution does not exist!" << endl;
        outFileMutex.lock();
        outFile << sudoku;
        outFileMutex.unlock();
    } while (true);

    inFileMutex.unlock();
    outFileMutex.unlock();
    return;
}

//entrypoint
int main(int argc, char **argv){
    if (argc!=2){
        cout<<"No file passed as input, exiting...";
        return 1;
    }

    unsigned int numThreads = std::thread::hardware_concurrency();
    inFileName = argv[1];
    inFile.open(inFileName);

    if (inFile.fail()) {
        cout<<"File "<<inFileName<<" not found!"<<endl;
        return 1;
    }

    outFileName = "Lab2Prob2.txt";
    outFile.open(outFileName);
    std::vector<std::thread> threads;
    for (int i=0; i<numThreads; i++) {
        threads.push_back(std::thread(solveSudokuPuzzles));
    }
    for (auto& th : threads) th.join();
    inFile.close();
    outFile.close();

    return 0;
}