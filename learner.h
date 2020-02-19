#pragma once

#include "gomoku.h"
#include "mcts.h"
#include <memory>
using namespace std;

class Learner {
    float dirichlet_alpha=0.5;
    int num_exporle=5;

public:

    bool  self_play(int first_color,shared_ptr<MCTS>pMCTS,shared_ptr<Gomoku> pgame,
                    vector<std::tuple<torch::Tensor,torch::Tensor,float>> &train_examples);

    bool convert_to_torch(int winner,vector<tuple<board_type,vector<float>,int,int>> &examples,
                              vector<std::tuple<torch::Tensor,torch::Tensor,float>> &train_examples);

    bool  contest(shared_ptr<NeuralNetwork> cur_network,int n,int n_in_row,bool use_gpu,unsigned int num_mcts_sims,
        float c_puct,
        float c_virtual_loss,
        unsigned int thread_num);

};