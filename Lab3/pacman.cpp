/*
Author: Pratiksha Pai
Class: ECE6122
Last Date Modified: 10/31/2022
Description:
Pacman! */

#include <iostream>
#include <sstream>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

using namespace sf;
using namespace std;

// Pacman and Ghosts can move these four sides.
enum side
{
    LEFT,
    RIGHT,
    UP,
    DOWN,
    None
};

// variables used throughout the code
int MAP_WIDTH;
float CELL_SIZE = 20.45f;
float ENTITY_SPEED = 0.09f;
float GHOST_SPEED = 0.05f;
float X_OFFSET = 21.5f;
float Y_OFFSET = 17.0f;
float X_LEFT = 32.0f;
float X_RIGHT = 570.0f;
float Y_UP = 40.0f;
float Y_DOWN = 608.0f;
float INIT_GHOST_Y = 245.f;
float INIT_GHOST_X = 230.f;
int number_of_ghosts_to_test = 4;
float Y_MIDDLE = 270.0f;
sf::Sprite spriteGhost[4];

// Grid of the bitmap
// 3 - wall
// 0 - space
// 1 - coin
// 2 - powerup
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

// Main function, this has all the code
int main()
{
    // the rendered window and variables
    VideoMode vm(641, 728);
    RenderWindow window(vm, "Timber!!!");
    View view(sf::FloatRect(0, 0, 641, 728));
    window.setView(view);

    // timebar for the powerup and variables
    Clock clock;
    Time dt;
    RectangleShape timeBar;
    float timeBarStartWidth = 300;
    float timeBarHeight = 40;
    timeBar.setFillColor(Color::Green);
    timeBar.setPosition(450, 680);
    Time gameTimeTotal;
    float timeRemaining = 0.0f;
    float timeBarWidthPerSecond = timeBarStartWidth / timeRemaining;

    // Background Bitmap
    Texture textureBackground;
    textureBackground.loadFromFile("graphics/maze.bmp");
    Sprite spriteBackground;
    spriteBackground.setTexture(textureBackground);
    spriteBackground.setPosition(0, 0);

    // Pacman Bitmap
    Texture texturePacman;
    texturePacman.loadFromFile("graphics/pacman.bmp");
    Sprite spritePacman;
    spritePacman.setTexture(texturePacman, true);
    spritePacman.setPosition(300, 364);

    // keeping track of the total coins on the grid
    int countOfAllCoins = 246;

    // load all the four ghost textures
    Texture textureOrange;
    textureOrange.loadFromFile("graphics/orange_ghost.bmp");
    Texture texturePink;
    texturePink.loadFromFile("graphics/pink_ghost.bmp");
    Texture textureRed;
    textureRed.loadFromFile("graphics/red_ghosts.bmp");
    Texture textureBlue;
    textureBlue.loadFromFile("graphics/blue_ghost.bmp");

    // initialise ghost sprites
    spriteGhost[0].setTexture(textureOrange);
    spriteGhost[0].setPosition(INIT_GHOST_X, INIT_GHOST_Y);
    spriteGhost[1].setTexture(texturePink);
    spriteGhost[1].setPosition(INIT_GHOST_X, INIT_GHOST_Y);
    spriteGhost[2].setTexture(textureRed);
    spriteGhost[2].setPosition(INIT_GHOST_X, INIT_GHOST_Y);
    spriteGhost[3].setTexture(textureBlue);
    spriteGhost[3].setPosition(INIT_GHOST_X, INIT_GHOST_Y);

    // additional variables to keep track of the ghost state
    vector<bool> ghostIsDead = {false, false, false, false};
    vector<side> ghostDirection = {static_cast<side>(rand() % side::DOWN), static_cast<side>(rand() % side::DOWN),
                                   static_cast<side>(rand() % side::DOWN), static_cast<side>(rand() % side::DOWN)};
    vector<side> tempGhostDirection;

    // Track whether the game is running
    bool paused = true;
    int score = 0;
    int powerUp = 0;
    int powerUpTime = 0;
    bool acceptInput = false;
    side pacmanSide = side::None;

    // load messages and scores
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

    // initialise grid of walls, powerups, coins and spaces
    sf::Sprite sprite[grid.size()][grid[0].size()];
    Texture tempTexture;
    tempTexture.loadFromFile("graphics/white.bmp");

    // clear window before rendering pacman bitmaps
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

        // stop the game
        if (Keyboard::isKeyPressed(Keyboard::Escape))
        {
            window.close();
        }

        // Start the game
        if (Keyboard::isKeyPressed(Keyboard::Return))
        {
            // initialise ghost and pacman positions at the start of the game

            spriteGhost[0].setTexture(textureOrange);
            spriteGhost[0].setPosition(INIT_GHOST_X, INIT_GHOST_Y);
            spriteGhost[1].setTexture(texturePink);
            spriteGhost[1].setPosition(INIT_GHOST_X, INIT_GHOST_Y);
            spriteGhost[2].setTexture(textureRed);
            spriteGhost[2].setPosition(INIT_GHOST_X, INIT_GHOST_Y);
            spriteGhost[3].setTexture(textureBlue);
            spriteGhost[3].setPosition(INIT_GHOST_X, INIT_GHOST_Y);
            spritePacman.setPosition(300, 364);

            // start the game, keep track of score, and accept inputs for pacman
            acceptInput = true;
            paused = false;
            score = 0;
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
                // check if ghost is colliding with the walls after the previous movement
                for (int i = 0; i < grid.size(); i++)
                {
                    for (int j = 0; j < grid[0].size(); j++)
                    {
                        if (grid[i][j] == 3 &&
                            spriteGhost[idx].getGlobalBounds().intersects(sprite[i][j].getGlobalBounds()))
                        {
                            // if the last step pushed the ghost into the wall, revert itxs
                            if (ghostDirection[idx] == side::LEFT)
                            {
                                spriteGhost[idx].setPosition(
                                    spriteGhost[idx].getPosition().x + GHOST_SPEED,
                                    spriteGhost[idx].getPosition().y);

                                // make sure we do not let the ghost go out of the bounds of the grid
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

                                // make sure we do not let the ghost go out of the bounds of the grid
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

                            // if ghost collided with the wall in the last step, change the direction randomly
                            tempGhostDirection = {
                                side::RIGHT, side::DOWN, side::UP, side::LEFT};
                            ghostDirection[idx] = tempGhostDirection[rand() % 4];
                            break;
                        }
                    }
                }
            }
        }

        // accept inputs for pacman movement
        if (acceptInput)
        {
            if (timeRemaining > 0)
            {
                // reduce timeRemaining, to update the timebar for powerup
                timeRemaining -= dt.asSeconds();
                timeBar.setSize(Vector2f(timeRemaining * CELL_SIZE, timeBarHeight));
            }

            // take inputs from left, right, up, down keys and add logics
            if (Keyboard::isKeyPressed(Keyboard::Left))
            {
                // Make sure the player is on the right
                pacmanSide = side::LEFT;
                spritePacman.setPosition(
                    spritePacman.getPosition().x - ENTITY_SPEED,
                    spritePacman.getPosition().y);
                // take care of the tunnel on the left side
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

                spritePacman.setPosition(
                    spritePacman.getPosition().x + ENTITY_SPEED,
                    spritePacman.getPosition().y);
                // take care of the tunnel on the right side
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
                spritePacman.setPosition(
                    spritePacman.getPosition().x,
                    spritePacman.getPosition().y - ENTITY_SPEED);
            }
            else if (Keyboard::isKeyPressed(Keyboard::Down))
            {
                pacmanSide = side::DOWN;
                spritePacman.setPosition(
                    spritePacman.getPosition().x,
                    spritePacman.getPosition().y + ENTITY_SPEED);
            }
        }
        for (int i = 0; i < grid.size(); i++) // rows, x
        {
            for (int j = 0; j < grid[0].size(); j++) // cols, y
            {
                if (spritePacman.getGlobalBounds().intersects(sprite[i][j].getGlobalBounds()))
                {
                    // take action according to the grid value.
                    switch (grid[i][j])
                    {
                    // coin
                    case 1:
                    {
                        // coin is eaten -> becomes space, score gets incremented
                        grid[i][j] = 0;
                        score += 1;
                        std::stringstream ss;
                        ss << " = " << score;
                        scoreText.setString(ss.str());
                        scoreText.setCharacterSize(30);
                        scoreText.setColor(Color::White);
                        scoreText.setPosition(130, 680);
                        countOfAllCoins -= 1;
                        break;
                    }
                    // powerup
                    case 2:
                    {
                        // coin is eaten -> becomes space, score gets incremented, timeRemaining is updated 5 seconds
                        score += 1;
                        grid[i][j] = 0;
                        timeRemaining += 5.0;
                        dt = clock.restart();
                        countOfAllCoins -= 1;
                        break;
                    }
                    // wall
                    case 3:
                    {
                        // revert the action of collision of pacman with a wall
                        // by keeping track of direction and reverting that action
                        if (pacmanSide == side::LEFT)
                        {
                            spritePacman.setPosition(
                                spritePacman.getPosition().x + ENTITY_SPEED,
                                spritePacman.getPosition().y);
                        }
                        else if (pacmanSide == side::RIGHT)
                        {
                            spritePacman.setPosition(
                                spritePacman.getPosition().x - ENTITY_SPEED,
                                spritePacman.getPosition().y);
                        }
                        else if (pacmanSide == side::UP)
                        {
                            spritePacman.setPosition(
                                spritePacman.getPosition().x,
                                spritePacman.getPosition().y + ENTITY_SPEED);
                        }
                        else if (pacmanSide == side::DOWN)
                        {
                            spritePacman.setPosition(
                                spritePacman.getPosition().x,
                                spritePacman.getPosition().y - ENTITY_SPEED);
                        }

                        break;
                    }
                    // space
                    default:
                        break;
                    }
                }
            }
        }
        // if the game is over - either win or lose banner shows up
        if (paused)
        {
            window.draw(spriteBackground);
            window.draw(spritePacman);
            window.draw(messageText);
            window.display();
        }
        else
        {
            // draw background and pacman
            window.draw(spriteBackground);
            window.draw(spritePacman);
            // draw the grid of walls, powerups, coins
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
                        // transparent grid, can be used for debugging by setting a color
                        sprite[i][j].setTexture(tempTexture);
                        sprite[i][j].setColor(sf::Color(0, 0, 0, 128)); 
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
            // draw ghosts that are alive
            for (int i = 0; i < number_of_ghosts_to_test; i++)
            {
                if (ghostIsDead[i] == true)
                {
                    continue;
                }
                window.draw(spriteGhost[i]);
            }
            // check if the pacman has eaten all coins, if yes, congratulations banner comes up.
            if (countOfAllCoins == 0)
            {
                paused = true;
                acceptInput = false;
                messageText.setString("Congratulations!");
            }
            // display scores, timebar etc
            window.draw(scoreText);
            window.draw(timeBar);

            window.display();
        }
    }

    return 0;
}