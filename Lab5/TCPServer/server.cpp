/*
Author: Pratiksha Pai
Date last modified: 11/21/2022
Organization: ECE6122

Description:
Simple TCP Server implementaion that handles multiple clients
*/

#include <iostream>
#include <fstream>
#include <SFML/Network.hpp>

using namespace std;

///////////////////////////////////////////////////////////////////////
// Get TimeStamp for the given connection
///////////////////////////////////////////////////////////////////////
char *getTime()
{
    time_t now = time(0);
    char *dt = ctime(&now);
    dt[strcspn(dt, "\n")] = '\0';
    return dt;
}

///////////////////////////////////////////////////////////////////////
// Initialise some common errors
///////////////////////////////////////////////////////////////////////
char *WRONG_COMMAND_LINE = "Command line arguments should be of the form ./server PORT";
char *WRONG_IP = "The IP address should be of the form 255.255.255.255";
char *WRONG_PORT = "The Port should be a numerical value of the range 61000-65535";

///////////////////////////////////////////////////////////////////////
// Check if command line arguments are of the form ./server port
///////////////////////////////////////////////////////////////////////
void checkInputs(int argc, char **argv)
{

    if (argc != 2)
    {
        cout << WRONG_COMMAND_LINE << endl;
        cout << "Invalid command line argument detected.";
        cout << "Please check your values and press any key to end the program!";
        cin.get();
        exit(1);
    }
}

///////////////////////////////////////////////////////////////////////
// Parse the port number from the command line arguments
// Test for edge cases
///////////////////////////////////////////////////////////////////////
int getPort(int argc, char **argv)
{
    string port_string = argv[1];
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

    // reference : https://www.sfml-dev.org/documentation/2.5.1/classsf_1_1SocketSelector.php

    // get the inputs from command line
    checkInputs(argc, argv);
    int port = getPort(argc, argv);

    // initialise the listener and Tcp socket selectors
    sf::TcpListener listener;
    listener.listen(port);
    std::vector<sf::TcpSocket *> clients;
    sf::SocketSelector selector;
    selector.add(listener);
    bool running = true;
    char *dt;

    // initialise the out file to which logs are redirected
    string outFileName = "server.log";
    std::ofstream outFile;
    outFile.open(outFileName);

    // when running check if any new connections are established
    while (running)
    {
        if (selector.wait())
        {
            if (selector.isReady(listener))
            {

                sf::TcpSocket *client = new sf::TcpSocket;

                if (listener.accept(*client) == sf::Socket::Done)
                {
                    dt = getTime();
                    // date & time :: ip_address of client :: message string
                    outFile << dt << " :: " << client->getRemoteAddress() << " :: Connected" << endl;
                    clients.push_back(client);
                    selector.add(*client);
                }
                else
                {
                    delete client;
                }
            }
            else
            {
                // if new connection is established check if there is any data coming through
                for (std::vector<sf::TcpSocket *>::iterator it = clients.begin(); it != clients.end(); ++it)
                {
                    sf::TcpSocket &client = **it;
                    if (selector.isReady(client))
                    {
                        // if data is coming through redirect it to the server logs
                        char data[100];
                        std::size_t received;
                        if (client.receive(data, 100, received) == sf::Socket::Done)
                        {
                            if (strcmp(data, "quit") == 0)
                            {
                                dt = getTime();
                                // date & time :: ip_address of client :: message string
                                outFile << dt << " :: " << client.getRemoteAddress() << " :: Disconnected" << endl;
                            }
                            else
                            {
                                dt = getTime();
                                // date & time :: ip_address of client :: message string
                                outFile << dt << " :: " << client.getRemoteAddress() << " :: " << data << endl;
                            }
                        }
                    }
                }
            }
        }
    }

    outFile.close();
    return 0;
}