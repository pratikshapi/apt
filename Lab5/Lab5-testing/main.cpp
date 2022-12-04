
#include <cstdlib>
#include <iostream>


void runTcpServer(unsigned short port);
void runTcpClient(unsigned short port);


int main()
{
    const unsigned short port = 50001;
    char who;
    std::cout << "Do you want to be a server (s) or a client (c)? ";
    std::cin >> who;
    if (who == 's')
        runTcpServer(port);
    else
        runTcpClient(port);

    std::cout << "Press enter to exit..." << std::endl;
    std::cin.ignore(10000, '\n');
    std::cin.ignore(10000, '\n');

    return 0;
}