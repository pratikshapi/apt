#include <iostream>
#include <vector>
using namespace std;

// This custom file is used to calculate unique paths on a maze with nRow rows and nCols columns.
// We use dynamic programming to efficiently solve this problem.

uint64_t numberGridPaths(unsigned int nRows, unsigned int nCols){

    if (nRows == 0 || nCols==0){
        return 0;
    }

    vector< vector<uint64_t> > dp(nRows, vector<uint64_t>(nCols, 1));

    for (int i = 1; i < nRows; i++) {
        for (int j = 1; j < nCols; j++) {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }
    return dp[nRows - 1][nCols - 1];
}