
#include "learner.h"
#include "tools.h"
#include <ctime>



bool Learner::self_play(int first_color,shared_ptr<MCTS>pMCTS,shared_ptr<Gomoku> pgame,
                        vector<std::tuple<torch::Tensor,torch::Tensor,double>> &train_examples){

    vector<tuple<board_type,vector<double>,int,int>>examples ;

    unsigned int episode_step=0;
    int palyers[]={-1,1};

    while(true){
        episode_step+=1;
        vector<double> model_probs;
        time_t t1,t2;
        time(&t1);

        if (episode_step<=this->num_exporle)//exporle
            model_probs=pMCTS->get_action_probs(pgame,1);
        else
            model_probs=pMCTS->get_action_probs(pgame,0);
        time(&t2);
        cout<<"cost_time:"<<t2-t1<<endl;



        board_type board=pgame->get_board();
        int last_action=pgame->get_last_move();
        int cur_player=pgame->get_current_color();

        auto example=make_tuple(board,model_probs,last_action,cur_player);
        examples.push_back(example);

        vector<double> probs(model_probs);
        std::vector<int> legal_moves= pgame->get_legal_moves();
        vector<double> ones(legal_moves.size(),1);
        vector<double> noise(legal_moves.size());
        dirichlet_random(noise,ones,legal_moves.size());

        std::for_each(probs.begin(),probs.end(),[](double &val){val*=0.9;});
        std::for_each(noise.begin(),noise.end(),[](double &val){val*=0.1;});
        int j=0;
        for(int i=0;i<probs.size();++i){
            if (legal_moves[i]==1)//need debug
            {
                probs[i] += noise[j];
                j+=1;
            }
        }

        double sum=std::accumulate(probs.begin(),probs.end(),0.0);
        std::for_each(probs.begin(),probs.end(),[sum](double &val){val/=sum;});
        //choose action
        int action=random_choise(probs,probs.size());
        pgame->execute_move(action);
        pMCTS->update_with_move(action);
        vector<int>result= pgame->get_game_status();
        int ended=result[0];
        int winner=result[1];

        if (ended==1){
            //convert to train type
            convert_to_torch(winner,examples,train_examples);

            pMCTS->update_with_move(-1);
            pgame->reset_game();
            cout<<"episode_step"<<episode_step<<endl;
            return true;
        }


    }
}


bool Learner::convert_to_torch(int winner,vector<tuple<board_type,vector<double>,int,int>> &examples,
                               vector<std::tuple<torch::Tensor,torch::Tensor,double>> &train_examples) {
    for (int i = 0; i < examples.size(); i++) {
        board_type board;
        vector<double> prob;
        int last_move;
        int cur_player;
        std::tie(board, prob, last_move, cur_player) = examples[i];
        int n = board.size();
        std::vector<int> board0;
        for (unsigned int i = 0; i < board.size(); i++) {
            board0.insert(board0.end(), board[i].begin(), board[i].end());
        }
        torch::Tensor temp =
                torch::from_blob(board0.data(), {1, 1, n, n}, torch::dtype(torch::kInt32));
        torch::Tensor state0 = (temp == 1).toType(torch::kFloat32);
        torch::Tensor state1 = (temp == -1).toType(torch::kFloat32);
        if (cur_player == -1) {
            std::swap(state0, state1);
        }
        torch::Tensor state2 =
                torch::zeros({1, 1, n, n}, torch::dtype(torch::kFloat32));
        if (last_move != -1) {
            state2[0][0][last_move / n][last_move % n] = 1;
        }
        // torch::Tensor states = torch::cat({state0, state1}, 1);
        torch::Tensor states_torch = torch::cat({state0, state1, state2}, 1);

        torch::Tensor prob_torch = torch::from_blob(prob.data(), {1, n, n}, torch::dtype(torch::kFloat32));
        double value = cur_player * winner;
        for (int k=0;k<4;++k){
            for(int flip=0;flip<2;++flip)
            {
                auto new_states_torch=torch::rot90(states_torch,k,{2,3});
                auto new_prob_torch=torch::rot90(prob_torch,k,{1,2});
                if (flip==1)
                {
                    new_states_torch=torch::flip(new_states_torch,3);
                    new_prob_torch=torch::flip(new_prob_torch,2);
                }

                auto train_example = std::make_tuple(new_states_torch, new_prob_torch.reshape({1,n*n}), value);
                train_examples.push_back(std::move(train_example));

            }
        }

    }
    return true;
}





bool Learner::contest(shared_ptr<NeuralNetwork> cur_network,int n,int n_in_row,bool use_gpu) {//-1 for cur 1 for best
    uniform_int_distribution<int> randint(0,1);
    int rand_out=randint(generator);
    int first_color;
    if (rand_out==0)
        first_color=-1;
    else
        first_color=1;

    shared_ptr<Gomoku> pgame = make_shared<Gomoku>(n, n_in_row, first_color);


    shared_ptr<NeuralNetwork> best_network = make_shared<NeuralNetwork>(n,n_in_row,use_gpu, 4);
    best_network->load();
    shared_ptr<MCTS> cur_MCTS = make_shared<MCTS>(cur_network, 8, 1, 50, 1, 10 * 10);
    shared_ptr<MCTS> best_MCTS = make_shared<MCTS>(best_network, 8, 1, 50, 1, 10 * 10);

    int players[2] = {first_color, -first_color};
    int cur_index = 0;
    int cur_player;
    shared_ptr<MCTS> choiceMCTS;
    while (true) {
        cur_player = players[cur_index];
        if (cur_player == -1)
            choiceMCTS = cur_MCTS;
        else
            choiceMCTS = best_MCTS;


        vector<double> probs = choiceMCTS->get_action_probs(pgame, 1);
        auto max_move = std::max_element(probs.begin(), probs.end());
        int best_move = std::distance(probs.begin(), max_move);
        pgame->execute_move(best_move);
        cur_MCTS->update_with_move(best_move);
        best_MCTS->update_with_move(best_move);
        vector<int> result = pgame->get_game_status();
        int ended = result[0];
        int winner = result[1];

        if (ended == 1) {
            if (winner == -1)
                return true;
            else
                return false;
        }
        ++cur_index;
        cur_index=cur_index%2;


    }
}


