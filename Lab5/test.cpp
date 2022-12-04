#include <iostream>
#include <SFML/Network.hpp>

using namespace std;

char *WRONG_COMMAND_LINE = "Command line arguments should be of the form ./client IP PORT";
char *WRONG_IP = "The IP address should be of the form 255.255.255.255";
char *WRONG_PORT = "The Port should be a numerical value of the range 61000-65535";

void checkInputs(int argc, char **argv)
{

    if (argc != 3)
    {
        cout << WRONG_COMMAND_LINE << endl;
        exit(1);
    }
}

string getIP(int argc, char **argv)
{
    string ip = argv[1];

    // check if given input is localhost
    // if its not locahost check if it is of the form 255.255.255.255
    if (ip != "localhost")
    {
        int cnt = 0;
        // Traverse the string s
        // int n = sizeof(ip)/sizeof(ip[0]);

        for (int i = 0; i < ip.size(); i++)
        {
            cout << ip[i]<<endl;
            if (ip[i] == '.')
            {
                cnt++;
            }
            else
            {
                if (stoi((char*)ip[i]) > 255 || stoi((char*)ip[i]) < 0)
                {
                    cout << WRONG_IP << endl;
                    // exit(1);
                }
            }
        }
        if (cnt != 3)
        {
            cout << WRONG_IP << endl;
            // exit(1);
        }
        else
        {
            return ip;
        }
    }
    else if (ip =="localhost")
    {
        return "127.0.0.1";
    }
    else
    {
        cout << WRONG_IP << endl;
        exit(1);
    }

    return "127.0.0.1";
}
int getPort(int argc, char **argv)
{

    return 0;
}

sf::Socket::Status checkConnection(char *ip, int port)
{

    return sf::Socket::Error;
}

int main(int argc, char **argv)
{

    checkInputs(argc, argv);
    string ip = getIP(argc, argv);
    int port = getPort(argc, argv);
    cout<<ip<<" "<<port<<endl;

    return 0;
}

// sf::Socket::Status status = checkConnection(ip, port);

    // cout << status << endl;
    // string s_port;
    // cin >> s_port;
    // int port = stoi(s_port);

    // sf::Socket::Status status = socket.connect("127.0.0.1", port);

// sf::Socket::Status checkConnection(string ip, int port)
// {


//     // return status;
//     // return sf::Socket::Error;
// }