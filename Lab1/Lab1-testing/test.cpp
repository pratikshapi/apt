#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
using namespace std;
using namespace std::chrono;

std::mutex outFileMutex;
std::mutex inFileMutex;
std::fstream inFile;
std::ofstream outFile;
string inFileName;
string outFileName;
int N=9;

class SudokuGrid {
    private:
        std::string m_strGridName;
        unsigned char grid[9][9];
    public:
        friend fstream& operator>>(fstream&, SudokuGrid&);
        friend ofstream& operator<<(ofstream&, const SudokuGrid&);
        bool isSafe(int, int, int);
        bool solveSudoku(int, int);

};

fstream& operator>>(fstream& os, SudokuGrid& gridIn){
    // Read in the grid name
    std::getline(os, gridIn.m_strGridName);
    std::string strIn;
    for (int i = 0 ; i < 9 ; ++i){
        std::getline(os, strIn);
        for (int j = 0 ; j < 9 ; ++j){
            gridIn.grid[i][j] = strIn[j];
        }
    }
    return os;
}

ofstream& operator<<(ofstream& os, const SudokuGrid& gridOut){
    os << gridOut.m_strGridName << endl;
    for (int i = 0; i < 9; ++i){
        for (int j = 0; j < 9; ++j){
            os << gridOut.grid[i][j];
        }
        os << endl;
    }
    return os;
}

bool SudokuGrid::isSafe(int row, int col, int num)
{
    for (int x = 0; x <= 8; x++)
        if (int(SudokuGrid::grid[row][x] - '0') == num)
            return false;

    for (int x = 0; x <= 8; x++)
        if (int(SudokuGrid::grid[x][col] - '0') == num)
            return false;

    int startRow = row - row % 3,
            startCol = col - col % 3;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (int(SudokuGrid::grid[i + startRow][j +
                                             startCol] - '0') == num)
                return false;

    return true;
}

bool SudokuGrid::solveSudoku(int row, int col)
{
    if (row == N - 1 && col == N)
        return true;

    if (col == N) {
        row++;
        col = 0;
    }
    if (int(SudokuGrid::grid[row][col] - '0') > 0)
        return SudokuGrid::solveSudoku(row, col + 1);

    for (int num = 1; num <= N; num++)
    {
        if (SudokuGrid::isSafe(row, col, num))
        {
            SudokuGrid::grid[row][col] = char(num + 48);
            if (SudokuGrid::solveSudoku(row, col + 1))
                return true;
        }
        SudokuGrid::grid[row][col] = '0';
    }
    return false;
}

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
int main(int argc, char **argv){
    auto start = high_resolution_clock::now();

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

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>(stop - start);
    cout << duration.count() << endl;
    return 0;
}