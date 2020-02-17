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
                    vector<std::tuple<torch::Tensor,torch::Tensor,double>> &train_examples);

    bool convert_to_torch(int winner,vector<tuple<board_type,vector<double>,int,int>> &examples,
                              vector<std::tuple<torch::Tensor,torch::Tensor,double>> &train_examples);

    bool  contest(shared_ptr<NeuralNetwork> cur_network,int n,int n_in_row,bool use_gpu,unsigned int num_mcts_sims,
        double c_puct,
        double c_virtual_loss,
        unsigned int thread_num);

};