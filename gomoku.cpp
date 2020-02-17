#include <math.h>
#include <iostream>
#include <random>
#include "gomoku.h"

static thread_local std::mt19937 generator;
Gomoku::Gomoku(unsigned int n, unsigned int n_in_row, int first_color)
        : n(n), n_in_row(n_in_row), cur_color(first_color), last_move(-1) {
    this->board = std::vector<std::vector<int>>(n, std::vector<int>(n, 0));
}

std::vector<int> Gomoku::get_legal_moves() {
    auto n = this->n;
    std::vector<int> legal_moves(this->get_action_size(), 0);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            if (this->board[i][j] == 0) {
                legal_moves[i * n + j] = 1;
            }
        }
    }

    return legal_moves;
}

bool Gomoku::has_legal_moves() {
    auto n = this->n;

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            if (this->board[i][j] == 0) {
                return true;
            }
        }
    }
    return false;
}

bool Gomoku::reset_game(){
    std::uniform_int_distribution<int> randint(0,1);
    int rand_out=randint(generator);
    int first_color;
    if (rand_out==0)
        first_color=-1;
    else
        first_color=1;

    this->board = std::vector<std::vector<int>>(n, std::vector<int>(n, 0));


    this->cur_color=first_color;
    this->last_move=-1;
    return true;
}



void Gomoku::execute_move(move_type move) {
    auto i = move / this->n;
    auto j = move % this->n;

    if (!this->board[i][j] == 0) {
        throw std::runtime_error("execute_move borad[i][j] != 0.");
    }

    this->board[i][j] = this->cur_color;
    this->last_move = move;
    // change player
    this->cur_color = -this->cur_color;
}

std::vector<int> Gomoku::get_game_status() {
    // return (is ended, winner)
    auto n = this->n;
    auto n_in_row = this->n_in_row;

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            if (this->board[i][j] == 0) {
                continue;
            }

            if (j <= n - n_in_row) {
                auto sum = 0;
                for (unsigned int k = 0; k < n_in_row; k++) {
                    sum += this->board[i][j + k];
                }
                if (abs(sum) == n_in_row) {
                    return {1, this->board[i][j]};
                }
            }

            if (i <= n - n_in_row) {
                auto sum = 0;
                for (unsigned int k = 0; k < n_in_row; k++) {
                    sum += this->board[i + k][j];
                }
                if (abs(sum) == n_in_row) {
                    return {1, this->board[i][j]};
                }
            }

            if (i <= n - n_in_row && j <= n - n_in_row) {
                auto sum = 0;
                for (unsigned int k = 0; k < n_in_row; k++) {
                    sum += this->board[i + k][j + k];
                }
                if (abs(sum) == n_in_row) {
                    return {1, this->board[i][j]};
                }
            }

            if (i <= n - n_in_row && j >= n_in_row - 1) {
                auto sum = 0;
                for (unsigned int k = 0; k < n_in_row; k++) {
                    sum += this->board[i + k][j - k];
                }
                if (abs(sum) == n_in_row) {
                    return {1, this->board[i][j]};
                }
            }
        }
    }

    if (this->has_legal_moves()) {
        return {0, 0};
    } else {
        return {1, 0};
    }
}

void Gomoku::display() const {
    auto n = this->board.size();

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            std::cout << this->board[i][j] << ", ";
        }
        std::cout << std::endl;

    }
    std::cout<<std::endl;
}
