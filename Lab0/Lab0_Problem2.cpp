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
 * This function checks for edge cases else runs oneMoreOne
 * Edge Cases include not a number, starts with 0, negative numbers, floating points
*/
int main(){

    std::string number_str;
    string error_statement = "Invalid input!! Please try again.";
    string cout_statement = "Please enter the starting number n: ";

    cout << cout_statement;

    while(std::getline(cin, number_str, '\n')){

        // define the edge cases, else run the operation
        try {

            int number = stoi(number_str);

            // handles string inputs with array of numbers like 1,1
            if (number_str.find(",") != std::string::npos){
                cout << error_statement << endl;
            }
            // handles string floating point numbers
            else if (number_str.find(".") != std::string::npos){
                cout << error_statement << endl;
            }
            // handles strings like -0, -00, 00 etc
            else if (number == 0 and number_str.length()>1) {
                cout << error_statement << endl;
            }
            // handles for the case where number = 0
            else if (number == 0) break;
            // handles numbers that start with 0 like 012
            else if (number_str.rfind("0", 0) == 0) {
                cout << error_statement << endl;
            }
            // handles negative numbers
            else if (number < 0) {
                cout << error_statement << endl;
            }
            // all the edge cases have passed, now we process the number
            else oneMoreOne(number);
        }
        // handles string inputs
        catch(std::invalid_argument& e){
            cout << error_statement << endl;
        }

        cout << cout_statement;
    }
    return 0;
}