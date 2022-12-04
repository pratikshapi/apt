#include <iostream>
#include <sstream>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

using namespace sf;
using namespace std;

enum side
{
    LEFT,
    RIGHT,
    UP,
    DOWN,
    None
};

int MAP_WIDTH;
float CELL_SIZE = 20.45f;
float ENTITY_SPEED = 0.09f;
float GHOST_SPEED = 0.05f;
float X_OFFSET = 21.5f;
float Y_OFFSET = 17.0f;
// float X_LEFT = 42.0f;
// float X_RIGHT = 555.0f;
float X_LEFT = 32.0f;
float X_RIGHT = 570.0f;
float Y_UP = 40.0f;
float Y_DOWN = 608.0f;
float INIT_GHOST_Y = 245.f;
float INIT_GHOST_X = 230.f;
int number_of_ghosts_to_test = 4;
float Y_MIDDLE = 270.0f;
sf::Sprite spriteGhost[4];

vector<vector<int>> grid = {
    {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
    {3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3},
    {3, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 3, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3},
    {3, 2, 0, 3, 3, 3, 1, 0, 3, 3, 3, 3, 1, 0, 3, 1, 0, 3, 3, 3, 3, 1, 0, 3, 3, 3, 2, 0, 3},
    {3, 1, 0, 3, 3, 3, 1, 0, 3, 3, 3, 3, 1, 0, 3, 1, 0, 3, 3, 3, 3, 1, 0, 3, 3, 3, 1, 0, 3},
    {3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3},
    {3, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3},
    {3, 1, 0, 3, 3, 3, 1, 0, 3, 1, 0, 3, 3, 3, 3, 3, 3, 3, 1, 0, 3, 1, 0, 3, 3, 3, 1, 0, 3},
    {3, 1, 1, 1, 1, 1, 1, 0, 3, 1, 1, 1, 1, 0, 3, 1, 1, 1, 1, 0, 3, 1, 1, 1, 1, 1, 1, 0, 3},
    {3, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 3},
    {3, 3, 3, 3, 3, 3, 1, 0, 3, 3, 3, 3, 0, 0, 3, 0, 0, 3, 3, 3, 3, 1, 0, 3, 3, 3, 3, 3, 3},
    {3, 3, 3, 3, 3, 3, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 3, 3, 3, 3, 3, 3},
    {3, 3, 3, 3, 3, 3, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 3, 3, 3, 3, 3, 3},
    {3, 3, 3, 3, 3, 3, 1, 0, 3, 0, 0, 3, 3, 0, 0, 0, 3, 3, 0, 0, 3, 1, 0, 3, 3, 3, 3, 3, 3},
    {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
    {3, 3, 3, 3, 3, 3, 1, 0, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3, 1, 0, 3, 3, 3, 3, 3, 3},
    {3, 3, 3, 3, 3, 3, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 3, 3, 3, 3, 3, 3},
    {3, 3, 3, 3, 3, 3, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 3, 3, 3, 3, 3, 3},
    {3, 3, 3, 3, 3, 3, 1, 0, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3, 1, 0, 3, 3, 3, 3, 3, 3},
    {3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3},
    {3, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 3, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3},
    {3, 1, 0, 3, 3, 3, 1, 0, 3, 3, 3, 3, 1, 0, 3, 1, 0, 3, 3, 3, 3, 1, 0, 3, 3, 3, 1, 0, 3},
    {3, 2, 1, 1, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 1, 1, 2, 0, 3},
    {3, 0, 0, 1, 0, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 3, 1, 0, 0, 0, 3},
    {3, 3, 3, 1, 0, 3, 1, 0, 3, 1, 0, 3, 3, 3, 3, 3, 3, 3, 1, 0, 3, 1, 0, 3, 1, 0, 3, 3, 3},
    {3, 1, 1, 1, 1, 1, 1, 0, 3, 1, 1, 1, 1, 0, 3, 1, 1, 1, 1, 0, 3, 1, 1, 1, 1, 1, 1, 0, 3},
    {3, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 3, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1, 0, 3},
    {3, 1, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 3, 1, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 3},
    {3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3},
    {3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3},
    {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}};

int main()
{

    VideoMode vm(641, 728);
    RenderWindow window(vm, "Timber!!!");
    View view(sf::FloatRect(0, 0, 641, 728));
    window.setView(view);

    Clock clock;
    Time dt;
    // Time bar
    RectangleShape timeBar;
    float timeBarStartWidth = 300;
    float timeBarHeight = 40;
    timeBar.setFillColor(Color::Green);
    timeBar.setPosition(450, 680);

    Time gameTimeTotal;
    float timeRemaining = 0.0f;
    float timeBarWidthPerSecond = timeBarStartWidth / timeRemaining;

    Texture textureBackground;
    textureBackground.loadFromFile("graphics/maze.bmp");
    Sprite spriteBackground;
    spriteBackground.setTexture(textureBackground);
    // spriteBackground.setPosition(630, 270);
    spriteBackground.setPosition(0, 0);

    Texture texturePacman;
    texturePacman.loadFromFile("graphics/pacman.bmp");
    Sprite spritePacman;
    spritePacman.setTexture(texturePacman, true);
    spritePacman.setPosition(300, 364);

    // randomly assigned ghost positions
    int temp_x = 0;
    int temp_y = 0;
    int count_of_all_coins = 246; // 116;

    Texture textureOrange;
    textureOrange.loadFromFile("graphics/orange_ghost.bmp");
    Texture texturePink;
    texturePink.loadFromFile("graphics/pink_ghost.bmp");
    Texture textureRed;
    textureRed.loadFromFile("graphics/red_ghosts.bmp");
    Texture textureBlue;
    textureBlue.loadFromFile("graphics/blue_ghost.bmp");

    spriteGhost[0].setTexture(textureOrange);
    spriteGhost[0].setPosition(INIT_GHOST_X, INIT_GHOST_Y);
    spriteGhost[1].setTexture(texturePink);
    spriteGhost[1].setPosition(INIT_GHOST_X, INIT_GHOST_Y);
    spriteGhost[2].setTexture(textureRed);
    spriteGhost[2].setPosition(INIT_GHOST_X, INIT_GHOST_Y);
    spriteGhost[3].setTexture(textureBlue);
    spriteGhost[3].setPosition(INIT_GHOST_X, INIT_GHOST_Y);

    vector<bool> ghostIsDead = {false, false, false, false};
    vector<side> ghostDirection = {static_cast<side>(rand() % side::DOWN), static_cast<side>(rand() % side::DOWN),
                                   static_cast<side>(rand() % side::DOWN), static_cast<side>(rand() % side::DOWN)};

    // Track whether the game is running
    bool paused = true;
    int score = 0;
    int powerUp = 0;
    int powerUpTime = 0;
    bool acceptInput = false;
    side pacmanSide = side::None;
    vector<side> tempGhostDirection;

    sf::Text messageText;
    sf::Text scoreText;
    sf::Font font;

    font.loadFromFile("fonts/KOMIKAP_.ttf");
    messageText.setFont(font);
    scoreText.setFont(font);
    messageText.setString("Press Enter to start!");
    scoreText.setString(" = 0");

    messageText.setCharacterSize(30);
    scoreText.setCharacterSize(100);
    messageText.setColor(Color::White);
    scoreText.setColor(Color::White);

    FloatRect textRect = messageText.getLocalBounds();
    messageText.setOrigin(textRect.left +
                              textRect.width / 2.0f,
                          textRect.top +
                              textRect.height / 2.0f);

    messageText.setPosition(641 / 2.0f, 728 / 2.0f);

    scoreText.setPosition(700, 20);

    // for grid
    sf::Sprite sprite[grid.size()][grid[0].size()];
    Texture tempTexture;
    tempTexture.loadFromFile("graphics/white.bmp");
    CircleShape coins;
    coins.setRadius(CELL_SIZE / 4.0f);
    coins.setTexture(&tempTexture);

    window.clear();

    while (window.isOpen())
    {

        dt = clock.restart();
        Event event;
        while (window.pollEvent(event))
        {
            if (event.type == Event::KeyReleased && !paused)
            {
                // Listen for key presses again

                spriteBackground.setPosition(0, 0);
                acceptInput = true;
            }
        }

        if (Keyboard::isKeyPressed(Keyboard::Escape))
        {
            window.close();
        }

        // Start the game
        if (Keyboard::isKeyPressed(Keyboard::Return))
        {
            spriteGhost[0].setTexture(textureOrange);
            spriteGhost[0].setPosition(INIT_GHOST_X, INIT_GHOST_Y);
            spriteGhost[1].setTexture(texturePink);
            spriteGhost[1].setPosition(INIT_GHOST_X, INIT_GHOST_Y);
            spriteGhost[2].setTexture(textureRed);
            spriteGhost[2].setPosition(INIT_GHOST_X, INIT_GHOST_Y);
            spriteGhost[3].setTexture(textureBlue);
            spriteGhost[3].setPosition(INIT_GHOST_X, INIT_GHOST_Y);
            paused = false;
            score = 0;
            spritePacman.setPosition(300, 364);
            acceptInput = true;
        }

        // movement of the ghost
        for (int idx = 0; idx < number_of_ghosts_to_test; idx++)
        {
            // Ghost is already dead
            if (ghostIsDead[idx])
            {
                continue;
            }
            // Pacman killed the ghost in this frame
            else if (spriteGhost[idx].getGlobalBounds().intersects(spritePacman.getGlobalBounds()) &&
                     timeRemaining > 0)
            {
                ghostIsDead[idx] = true;
                continue;
            }
            // Ghost killed the Pacman in this frame
            else if (spriteGhost[idx].getGlobalBounds().intersects(spritePacman.getGlobalBounds()))
            {
                paused = true;
                acceptInput = false;
                messageText.setString("Game Over!");
                break;
            }
            // no one is killing anyone
            else
            {
                // ghost and pacman are not intersecting
                if (ghostDirection[idx] == side::LEFT)
                {
                    spriteGhost[idx].setPosition(
                        spriteGhost[idx].getPosition().x - GHOST_SPEED,
                        spriteGhost[idx].getPosition().y);

                    if (spriteGhost[idx].getPosition().x < X_LEFT)
                    {
                        spriteGhost[idx].setPosition(
                            X_RIGHT,
                            spriteGhost[idx].getPosition().y);
                    }
                }
                else if (ghostDirection[idx] == side::RIGHT)
                {
                    spriteGhost[idx].setPosition(
                        spriteGhost[idx].getPosition().x + GHOST_SPEED,
                        spriteGhost[idx].getPosition().y);
                    if (spriteGhost[idx].getPosition().x < X_LEFT)
                    {
                        spriteGhost[idx].setPosition(
                            X_RIGHT,
                            spriteGhost[idx].getPosition().y);
                    }
                }
                else if (ghostDirection[idx] == side::UP)
                {
                    spriteGhost[idx].setPosition(
                        spriteGhost[idx].getPosition().x,
                        spriteGhost[idx].getPosition().y - GHOST_SPEED);
                }
                else
                {
                    spriteGhost[idx].setPosition(
                        spriteGhost[idx].getPosition().x,
                        spriteGhost[idx].getPosition().y + GHOST_SPEED);
                }
                // check if ghost is colliding with the walls
                for (int i = 0; i < grid.size(); i++)
                {
                    for (int j = 0; j < grid[0].size(); j++)
                    {
                        if (grid[i][j] == 3 &&
                            spriteGhost[idx].getGlobalBounds().intersects(sprite[i][j].getGlobalBounds()))
                        {

                            if (ghostDirection[idx] == side::LEFT)
                            {
                                spriteGhost[idx].setPosition(
                                    spriteGhost[idx].getPosition().x + GHOST_SPEED,
                                    spriteGhost[idx].getPosition().y);

                                if (spriteGhost[idx].getPosition().x < X_LEFT)
                                {
                                    spriteGhost[idx].setPosition(
                                        X_RIGHT,
                                        spriteGhost[idx].getPosition().y);
                                }
                            }
                            else if (ghostDirection[idx] == side::RIGHT)
                            {
                                spriteGhost[idx].setPosition(
                                    spriteGhost[idx].getPosition().x - GHOST_SPEED,
                                    spriteGhost[idx].getPosition().y);
                                if (spriteGhost[idx].getPosition().x < X_LEFT)
                                {
                                    spriteGhost[idx].setPosition(
                                        X_RIGHT,
                                        spriteGhost[idx].getPosition().y);
                                }
                            }
                            else if (ghostDirection[idx] == side::UP)
                            {
                                spriteGhost[idx].setPosition(
                                    spriteGhost[idx].getPosition().x,
                                    spriteGhost[idx].getPosition().y + GHOST_SPEED);
                            }
                            else
                            {
                                spriteGhost[idx].setPosition(
                                    spriteGhost[idx].getPosition().x,
                                    spriteGhost[idx].getPosition().y - GHOST_SPEED);
                            }
                            // cout << "changing direction for " << i << " " << j << endl;
                            tempGhostDirection = {
                                side::RIGHT, side::DOWN, side::UP, side::LEFT};
                            ghostDirection[idx] = tempGhostDirection[rand() % 4];

                            break;
                        }
                    }
                }
            }
        }

        if (acceptInput)
        {
            if (timeRemaining > 0)
            {
                timeRemaining -= dt.asSeconds();
                // timeBarWidthPerSecond = timeBarStartWidth - timeRemaining;
                timeBar.setSize(Vector2f(timeRemaining * CELL_SIZE, timeBarHeight));
                // cout<<timeRemaining<<endl;
            }

            if (Keyboard::isKeyPressed(Keyboard::Left))
            {
                // Make sure the player is on the right
                pacmanSide = side::LEFT;
                // if (spritePacman.getPosition().x > X_LEFT)
                {
                    spritePacman.setPosition(
                        spritePacman.getPosition().x - ENTITY_SPEED,
                        spritePacman.getPosition().y);
                }
                if (spritePacman.getPosition().x < X_LEFT)
                {
                    spritePacman.setPosition(
                        X_RIGHT,
                        spritePacman.getPosition().y);
                }
            }
            else if (Keyboard::isKeyPressed(Keyboard::Right))
            {
                pacmanSide = side::RIGHT;
                // if (spritePacman.getPosition().x < X_RIGHT)
                {
                    spritePacman.setPosition(
                        spritePacman.getPosition().x + ENTITY_SPEED,
                        spritePacman.getPosition().y);
                }
                if (spritePacman.getPosition().x > X_RIGHT)
                {
                    spritePacman.setPosition(
                        X_LEFT,
                        spritePacman.getPosition().y);
                }
            }
            else if (Keyboard::isKeyPressed(Keyboard::Up))
            {
                pacmanSide = side::UP;
                // if (spritePacman.getPosition().y > Y_UP)
                {
                    spritePacman.setPosition(
                        spritePacman.getPosition().x,
                        spritePacman.getPosition().y - ENTITY_SPEED);
                }
            }
            else if (Keyboard::isKeyPressed(Keyboard::Down))
            {
                pacmanSide = side::DOWN;
                // if (spritePacman.getPosition().y < Y_DOWN)
                {
                    spritePacman.setPosition(
                        spritePacman.getPosition().x,
                        spritePacman.getPosition().y + ENTITY_SPEED);
                }
            }
        }
        for (int i = 0; i < grid.size(); i++) // rows, x
        {
            for (int j = 0; j < grid[0].size(); j++) // cols, y
            {
                if (spritePacman.getGlobalBounds().intersects(sprite[i][j].getGlobalBounds()))
                {
                    switch (grid[i][j])
                    {
                    case 1:
                    {
                        grid[i][j] = 0;
                        score += 1;
                        std::stringstream ss;
                        ss << " = " << score;
                        scoreText.setString(ss.str());
                        scoreText.setCharacterSize(30);
                        scoreText.setColor(Color::White);
                        scoreText.setPosition(130, 680);
                        count_of_all_coins -= 1;
                        break;
                    }
                    case 2:
                    {
                        score += 1;
                        grid[i][j] = 0;
                        timeRemaining += 5.0;
                        dt = clock.restart();
                        count_of_all_coins -= 1;
                        break;
                    }
                    case 3:
                    {
                        if (pacmanSide == side::LEFT)
                        {
                            // if (spritePacman.getPosition().x > X_LEFT)
                            {
                                spritePacman.setPosition(
                                    spritePacman.getPosition().x + ENTITY_SPEED,
                                    spritePacman.getPosition().y);
                            }
                        }
                        else if (pacmanSide == side::RIGHT)
                        {
                            // if (spritePacman.getPosition().x < X_RIGHT)
                            {
                                spritePacman.setPosition(
                                    spritePacman.getPosition().x - ENTITY_SPEED,
                                    spritePacman.getPosition().y);
                            }
                        }
                        else if (pacmanSide == side::UP)
                        {
                            // if (spritePacman.getPosition().y > Y_UP)
                            {
                                spritePacman.setPosition(
                                    spritePacman.getPosition().x,
                                    spritePacman.getPosition().y + ENTITY_SPEED);
                            }
                        }
                        else if (pacmanSide == side::DOWN)
                        {
                            // if (spritePacman.getPosition().y < Y_DOWN)
                            {
                                spritePacman.setPosition(
                                    spritePacman.getPosition().x,
                                    spritePacman.getPosition().y - ENTITY_SPEED);
                            }
                        }

                        break;
                    }
                    default:
                        break;
                    }
                }
            }
        }

        // window.clear();

        if (paused)
        {
            window.draw(spriteBackground);
            window.draw(spritePacman);
            window.draw(messageText);
            window.display();
        }
        else
        {
            window.draw(spriteBackground);
            window.draw(spritePacman);
            for (int i = 0; i < grid.size(); i++) // rows, x
            {
                for (int j = 0; j < grid[0].size(); j++) // cols, y
                {
                    // for each sprite draw stuff
                    sprite[i][j].setPosition(X_OFFSET + CELL_SIZE * j, Y_OFFSET + CELL_SIZE * i);

                    switch (grid[i][j])
                    {
                    // wall
                    case 3:
                    {
                        sprite[i][j].setTexture(tempTexture);
                        sprite[i][j].setColor(sf::Color(0, 0, 0, 128)); // Color(255, 255, 255, 128)
                        sprite[i][j].setTextureRect(sf::IntRect(CELL_SIZE, CELL_SIZE, CELL_SIZE, CELL_SIZE));
                        window.draw(sprite[i][j]);
                        break;
                    }
                    // food
                    case 1:
                    {
                        sprite[i][j].setPosition(X_OFFSET + CELL_SIZE * (j + 1), Y_OFFSET + CELL_SIZE * (i + 1));
                        sprite[i][j].setTexture(tempTexture);
                        sprite[i][j].setColor(sf::Color(255, 255, 255, 255));
                        sprite[i][j].setTextureRect(sf::IntRect(CELL_SIZE / 4, CELL_SIZE / 4, CELL_SIZE / 4, CELL_SIZE / 4));
                        window.draw(sprite[i][j]);
                        break;
                    }
                    // power up
                    case 2:
                    {
                        sprite[i][j].setPosition(X_OFFSET + CELL_SIZE * (j + 1), Y_OFFSET + CELL_SIZE * (i + 1));
                        sprite[i][j].setTexture(tempTexture);
                        sprite[i][j].setColor(sf::Color(255, 255, 255, 255));
                        sprite[i][j].setTextureRect(sf::IntRect(CELL_SIZE / 2, CELL_SIZE / 2, CELL_SIZE / 2, CELL_SIZE / 2));
                        window.draw(sprite[i][j]);
                        break;
                    }
                    }
                }
            }

            for (int i = 0; i < number_of_ghosts_to_test; i++)
            {
                if (ghostIsDead[i] == true)
                {
                    continue;
                }
                window.draw(spriteGhost[i]);
            }
            if (count_of_all_coins == 0)
            {
                paused = true;
                acceptInput = false;
                messageText.setString("Congratulations!");
            }
            window.draw(scoreText);
            window.draw(timeBar);

            window.display();
        }
    }

    return 0;
}