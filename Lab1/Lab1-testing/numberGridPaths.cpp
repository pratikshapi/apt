//
// Created by Pratiksha Pai on 10/1/22.
//

#include <iostream>
using namespace std;

uint64_t gcd(uint64_t a, uint64_t b)
{
    return b == 0 ? a : gcd(b, a % b);
}


uint64_t calculateNcR(int n, int r)
{
    uint64_t p = 1, k = 1;
    if (n - r < r)
        r = n - r;

    if (r != 0) {
        while (r) {
            p *= n;
            k *= r;

            uint64_t m = gcd(p, k);
            p /= m;
            k /= m;

            n--;
            r--;
        }
    }

    else
        p = 1;

    return p;
}

uint64_t numberGridPaths(unsigned int nRows, unsigned int nCols){

    if (nRows == 0 || nCols==0){
        return 0;
    }
    unsigned int temp=0;
    if (nRows<nCols){
        temp=nRows;
    }else {
        temp=nCols;
    }
    return calculateNcR(nRows+nCols-2,temp-1);
}
