#pragma once

#include "gomoku.h"
#include "mcts.h"
#include <memory>
using namespace std;

class Learner {
    double dirichlet_alpha=0.5;
    int num_exporle=5;

public:

    bool  self_play(int first_color,shared_ptr<MCTS>pMCTS,shared_ptr<Gomoku> pgame,
                     vector<tuple<board_type,vector<double>,int,int,double>> &train_examples);

    bool  contest(shared_ptr<NeuralNetwork> cur_network,int n,int n_in_row);

};