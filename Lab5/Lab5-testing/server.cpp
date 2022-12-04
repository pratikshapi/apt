#include <iostream>
#include <SFML/Network.hpp>

using namespace std;

char *getTime()
{
    time_t now = time(0);
    char *dt = ctime(&now);
    dt[strcspn(dt, "\n")] = '\0';
    return dt;
}

int main(int argc, char **argv)
{

    sf::TcpListener listener;
    listener.listen(53000);
    std::vector<sf::TcpSocket *> clients;
    sf::SocketSelector selector;
    selector.add(listener);
    bool running = true;
    char *dt;
    // reference : https://www.sfml-dev.org/documentation/2.5.1/classsf_1_1SocketSelector.php
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
                    cout << dt << " :: " << client->getRemoteAddress() << " " // << client->getRemotePort() << " " << client->getLocalPort() << " "
                         << " :: Connected" << endl;
                    clients.push_back(client);
                    // cout << "number of clients " << clients.size() << endl;

                    selector.add(*client);
                }
                else
                {
                    delete client;
                }
            }
            else
            {
                for (std::vector<sf::TcpSocket *>::iterator it = clients.begin(); it != clients.end(); ++it)
                {
                    sf::TcpSocket &client = **it;
                    if (selector.isReady(client))
                    {
                        char data[100];
                        std::size_t received;
                        if (client.receive(data, 100, received) == sf::Socket::Done)
                        {
                            // cout << data << endl;
                            if (strcmp(data, "quit") == 0)
                            {
                                dt = getTime();
                                // date & time :: ip_address of client :: message string
                                cout << dt << " :: " << client.getRemoteAddress() << " :: Disconnected" << endl;
                                // clients.erase(it);
                            }
                            else
                            {
                                dt = getTime();
                                // date & time :: ip_address of client :: message string
                                cout << dt << " :: " << client.getRemoteAddress() << " :: " << data << endl;
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}