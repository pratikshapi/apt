#include <iostream>
using namespace std;

void oneMoreOne(int number){

    int count = 0;
    while (number!=1){
        if (number%7 == 0){
            number=number/7;
        } else{
            number+=1;
            count+=1;
        }
    }

    cout << "The sequence had " << count << " instances of the number 1 being added." << endl;
    return;
}

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