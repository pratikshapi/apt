#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <vector>
using namespace std;

std::mutex outFileMutex;
std::mutex inFileMutex;
std::fstream inFile;
std::ofstream outFile;
string inFileName;
string outFileName;
int N=9;

class SudokuGrid {
private:
    // Data Members
    std::string m_strGridName;
    unsigned char grid[9][9];

public:
    // Member Functions()

    friend fstream& operator>>(fstream& os, SudokuGrid& gridIn){
        // Read in the grid name
        std::getline(os, gridIn.m_strGridName);
        cout<<"gridIn.m_strGridName"<<endl;
        cout<<gridIn.m_strGridName<<endl;
        std::string strIn;
        for (int i = 0 ; i < 9 ; ++i){
            std::getline(os, strIn);
            for (int j = 0 ; j < 9 ; ++j){
                gridIn.grid[i][j] = strIn[j];
//                    cout<<gridIn.grid[i][j]<<" ";
            }
            cout<<endl;
        }
        cout<<"gridIn.grid"<<endl;
        cout<<gridIn.grid<<endl;

        return os;
    }

    friend ofstream& operator<<(ofstream& os, const SudokuGrid& gridOut){
        os << gridOut.m_strGridName << endl;
        for (int i = 0; i < 9; ++i){
            for (int j = 0; j < 9; ++j){
                os << gridOut.grid[j][i]<<" ";
            }
            os << endl;
        }
        return os;
    }
    bool isPresentInCol(int col, int num){ //check whether num is present in col or not
        for (int row = 0; row < N; row++)
            if (grid[row][col] == num)
                return true;
        return false;
    }
    bool isPresentInRow(int row, int num){ //check whether num is present in row or not
        for (int col = 0; col < N; col++)
            if (grid[row][col] == num)
                return true;
        return false;
    }
    bool isPresentInBox(int boxStartRow, int boxStartCol, int num){
//check whether num is present in 3x3 box or not
        for (int row = 0; row < 3; row++)
            for (int col = 0; col < 3; col++)
                if (grid[row+boxStartRow][col+boxStartCol] == num)
                    return true;
        return false;
    }
    void sudokuGrid(){ //print the sudoku grid after solve
        for (int row = 0; row < N; row++){
            for (int col = 0; col < N; col++){
                if(col == 3 || col == 6)
                    cout << " | ";
                cout << grid[row][col] <<" ";
            }
            if(row == 2 || row == 5){
                cout << endl;
                for(int i = 0; i<N; i++)
                    cout << "---";
            }
            cout << endl;
        }
    }
    bool findEmptyPlace(int &row, int &col){ //get empty location and update row and column
        for (row = 0; row < N; row++)
            for (col = 0; col < N; col++)
                if (int(grid[row][col]) == 0) //marked with 0 is empty
                    return true;
        return false;
    }
    bool isValidPlace(int row, int col, int num){
        //when item not found in col, row and current 3x3 box
        return !isPresentInRow(row, num) && !isPresentInCol(col, num) && !isPresentInBox(row - row%3 ,
                                                                                         col - col%3, num);
    }
    bool solveSudoku(int row, int col){
        cout<<"inside solve sudoku class"<<endl;
        if (!findEmptyPlace(row, col))
            return true; //when all places are filled
        for (int num = 1; num <= 9; num++){ //valid numbers are 1 - 9
            if (isValidPlace(row, col, num)){ //check validation, if yes, put the number in the grid
                (*this).grid[row][col] = num;
                if (solveSudoku(row)) //recursively go for other rooms in the grid
                    return true;
                (*this).grid[row][col] = 0; //turn to unassigned space when conditions are not satisfied
            }
        }

        for (int i=0; i<9;i++){
            for (int j=0;j<9;j++){
                cout<<(*this).grid[i][j]<<" ";
            }
            cout<<endl;
        }
        return false;
    }
};

void solveSudokuPuzzles(){
    SudokuGrid sudoku;
    cout<<"inside solveSudokuPuzzles"<<endl;
    do
    {
//        inFileMutex.lock();
        if (inFile.eof()){
            break;
        }
        else{
            inFile >> sudoku;
        }
//        inFileMutex.unlock();
        if (sudoku.solveSudoku(0, 0))
            cout << "check lab2prob2 " << endl;
        else
            cout << "no solution  exists " << endl;
//        outFileMutex.lock();
        outFile << sudoku;
//        outFileMutex.unlock();
    } while (true);
    return;
}
int main() {
    //
    int numThreads = 27;
    inFileName = "input_sudoku.txt";
    cout << inFileName;
    inFile.open(inFileName);

    outFileName = "Lab2Prob2.txt";
    cout << outFileName;
    outFile.open(outFileName);
    outFile << "Writing this to a file.\n";
//    std::vector<std::thread> threads;
//    for (int i=0; i<numThreads; i++) {
//        threads.push_back(std::thread(solveSudokuPuzzles));
//    }
//    for (auto& th : threads) th.join();
    solveSudokuPuzzles();
    inFile.close();
    outFile.close();
    return 0;
}