#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include "gomoku.h"
#include "thread_pool.h"
#include "libtorch.h"

class TreeNode {
public:
    // friend class can access private variables
    friend class MCTS;

    TreeNode();
    TreeNode(const TreeNode &node);
    TreeNode(TreeNode *parent, float p_sa, unsigned action_size);

    TreeNode &operator=(const TreeNode &p);

    unsigned int select(float c_puct, float c_virtual_loss);
    void expand(const std::vector<float> &action_priors);
    void backup(float leaf_value);

    float get_value(float c_puct, float c_virtual_loss,
                     unsigned int sum_n_visited) const;
    inline bool get_is_leaf() const { return this->is_leaf; }

private:
    // store tree
    TreeNode *parent;
    std::vector<TreeNode *> children;
    bool is_leaf;
    std::mutex lock;

    std::atomic<unsigned int> n_visited;
    float p_sa;
    float q_sa;
    std::atomic<int> virtual_loss;
};

class MCTS {
public:
    MCTS(shared_ptr<NeuralNetwork> neural_network, unsigned int thread_num, float c_puct,
         unsigned int num_mcts_sims, float c_virtual_loss,
         unsigned int action_size);
    std::vector<float> get_action_probs(shared_ptr<Gomoku> gomoku, float temp = 1e-3);
    void update_with_move(int last_move);

private:
    void simulate(std::shared_ptr<Gomoku> game);
    static void tree_deleter(TreeNode *t);

    // variables
    std::unique_ptr<TreeNode, decltype(MCTS::tree_deleter) *> root;
    std::unique_ptr<ThreadPool> thread_pool;
    shared_ptr<NeuralNetwork> neural_network;

    unsigned int action_size;
    unsigned int num_mcts_sims;
    float c_puct;
    float c_virtual_loss;
};
