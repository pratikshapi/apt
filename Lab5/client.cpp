/*
Author: Pratiksha Pai
Date last modified: 11/21/2022
Organization: ECE6122

Description:
Simple TCP Clinet implementaion.
*/

#include <iostream>
#include <SFML/Network.hpp>

using namespace std;

///////////////////////////////////////////////////////////////////////
// Initialise some common errors
///////////////////////////////////////////////////////////////////////
char *WRONG_COMMAND_LINE = "Command line arguments should be of the form ./client IP PORT";
char *WRONG_IP = "The IP address should be of the form 255.255.255.255";
char *WRONG_PORT = "The Port should be a numerical value of the range 61000-65535";
char *INVALID_ARGUMENT = "Invalid command line argument detected: %s Please check your values and press any key to end the program!";

///////////////////////////////////////////////////////////////////////
// Check if command line arguments are of the form ./server port
///////////////////////////////////////////////////////////////////////
void checkInputs(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << WRONG_COMMAND_LINE << endl;
        cout << "Invalid command line argument detected.";
        cout << "Please check your values and press any key to end the program!";
        cin.get();
        exit(1);
    }
}

///////////////////////////////////////////////////////////////////////
// Parse the IP address from the command line arguments
// Test for edge cases
///////////////////////////////////////////////////////////////////////
char *checkIP(int argc, char **argv)
{
    char *ip = argv[1];
    char *dummy;

    // check if given input is localhost
    // if its not locahost check if it is of the form 255.255.255.255
    if (strcmp(ip, "localhost") != 0)
    {

        for (char *it = ip; *it; ++it)
        {
            if ((*it != '.') && (!isdigit(*it)))
            {
                cout << WRONG_IP << endl;
                cout << "Invalid command line argument detected: " << ip << endl;
                cout << "Please check your values and press any key to end the program!";
                cin.get();
                exit(1);
            }
        }
    }

    return ip;
}

///////////////////////////////////////////////////////////////////////
// Parse the port number from the command line arguments
// Test for edge cases
///////////////////////////////////////////////////////////////////////
int getPort(int argc, char **argv)
{
    string port_string = argv[2];
    char *dummy;

    try
    {
        int port = stoi(port_string);
        if (port < 61000 || port > 65535)
        {
            cout << WRONG_PORT << endl;
            cout << "Invalid command line argument detected: " << port_string << endl;
            cout << "Please check your values and press any key to end the program!";
            cin.get();
            exit(1);
        }
        return port;
    }
    catch (std::invalid_argument &e)
    {
        cout << WRONG_PORT << endl;
        cout << "Invalid command line argument detected: " << port_string << endl;
        cout << "Please check your values and press any key to end the program!";
        cin.get();
        exit(1);
    }
    return -1;
}

///////////////////////////////////////////////////////////////////////
// Driver function to check to run the server on a given port
///////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

    // get the inputs from command line
    checkInputs(argc, argv);
    string ip = checkIP(argc, argv);
    int port = getPort(argc, argv);
    sf::Time timeout = sf::seconds(2.0f);

    // initialise the listener and Tcp socket selectors
    sf::TcpSocket socket;
    sf::Socket::Status status = socket.connect(ip, port, timeout);

    if (status != sf::Socket::Done)
    {
        cout << "Failed to connect to the server at " << ip << " on " << port << "." << endl;
        cout << "Please check your values and press any key to end program!";
        cin.get();
        exit(1);
    }

    // establish a new connection and send data
    while (status == sf::Socket::Done)
    {
        char data[100];
        cout << "Please enter a message: ";
        cin >> data;
        if (strcmp(data, "quit") == 0)
        {
            if (socket.send(data, 100) == sf::Socket::Done)
            {
                socket.disconnect();
            }

            break;
        }
        else if (socket.send(data, 100) != sf::Socket::Done)
        {
            cout << "data not sent!" << endl;
        }
    }

    return 0;
}
