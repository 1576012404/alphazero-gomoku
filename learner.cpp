
#include "learner.h"
#include "tools.h"



bool Learner::self_play(int first_color,shared_ptr<MCTS>pMCTS,shared_ptr<Gomoku> pgame,
                        vector<tuple<board_type,vector<double>,int,int,double>> &train_examples){

    vector<tuple<board_type,vector<double>,int,int>>examples ;

    unsigned int episode_step=0;
    int palyers[]={-1,1};

    while(true){
        episode_step+=1;
        vector<double> model_probs;

        if (episode_step<=this->num_exporle)//exporle
            model_probs=pMCTS->get_action_probs(pgame,1);
        else
            model_probs=pMCTS->get_action_probs(pgame,0);



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


        for(int i=0;i<examples.size();i++){
            board_type board;
            vector<double> probs;
            int last_action;
            std::tie(board,probs,last_action,cur_player)=examples[i];
            auto train_example=make_tuple(board,model_probs,last_action,cur_player,cur_player*winner);
            train_examples.push_back(train_example);
        }


        if (ended==1){
            pMCTS->update_with_move(-1);
            pgame->reset_game();
            cout<<"episode_step"<<episode_step<<endl;
            return true;
        }

    }
}

bool Learner::contest(shared_ptr<NeuralNetwork> cur_network,int n,int n_in_row) {//-1 for cur 1 for best
    uniform_int_distribution<int> randint(0,1);
    int rand_out=randint(generator);
    int first_color;
    if (rand_out==0)
        first_color=-1;
    else
        first_color=1;

    shared_ptr<Gomoku> pgame = make_shared<Gomoku>(10, 4, first_color);

    shared_ptr<NeuralNetwork> best_network = make_shared<NeuralNetwork>(n,n_in_row,false, 4);
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


