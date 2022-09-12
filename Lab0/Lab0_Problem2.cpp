/*
 * Author: Pratiksha Pundalik Pai
 * Class: ECE6122
 * Last Date Modified: 12th September 2022
 * Description:
 * This function implements One more one from projecteuler.net
 * Details about the problem can be found at https://projecteuler.net/problem=672
*/

#include <iostream>
using namespace std;

/*
 * Function implements one more one problem from projecteuler.net
 * Inputs: Takes in "number" as input
 * Outputs: None
 * Prints: The function prints the number of 1's that are being added
*/
void oneMoreOne(int number){

    int count = 0;
    while (number != 1){
        if (number%7 == 0){
            number = number/7;
        } else{
            number += 1;
            count += 1;
        }
    }

    cout << "The sequence had " << count << " instances of the number 1 being added." << endl;
    return;
}


/*
 * This is main function or entrypoint for the code.
 * This function checks for edge cases and runs oneMoreOne function.
*/
int main(){

    cout << "Please enter the starting number n (0 to stop): ";
    int number = 0;
    while (cin >> number){

        // define the edge cases, else run the operation
        if (number == 0) break;
        if (number < 0) cout << "Invalid input!! Please try again." << endl;
        else oneMoreOne(number);
        cout << "Please enter the starting number n (0 to stop): ";
    }
    return 0;
}