#include <iostream>
#include <SFML/Network.hpp>

using namespace std;

int main(int argc, char** argv)
{

    sf::TcpSocket socket;
    string s_port;
    cin>>s_port;
    int port = stoi(s_port);
    cout<<port<<endl;
    sf::Socket::Status status = socket.connect("127.0.0.1", port);
    cout<< status<<endl;
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
        else if (socket.send(data, 100) == sf::Socket::Done)
        {
            cout << "data sent!" << endl;
        }
    }

    return 0;
}
