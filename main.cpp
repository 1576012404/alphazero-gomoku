#include <iostream>
#include "gomoku.h"
#include "mcts.h"
#include "libtorch.h"
#include "learner.h"
using namespace std;




int train(){
    int iter_num=10;
    int train_batch_size=128;
    unsigned int sim_batch_size=16;
    int contest_num=4;
    int epoch_num=100;
    int check_interval=2;
    int n=10;
    int n_in_row=4;
    //network param

    Learner learner=Learner();
    shared_ptr<Gomoku> pgame=make_shared<Gomoku>(n,n_in_row,1);
    bool use_gpu=torch::cuda::is_available();
    shared_ptr<NeuralNetwork> neural_network =make_shared<NeuralNetwork> (n,n_in_row,use_gpu,sim_batch_size);
    shared_ptr<MCTS> pMCTS=make_shared<MCTS>(neural_network, 2, 1,50, 1,10*10);




    for(int epoch=1;epoch<epoch_num;epoch++){
        cout<<"epoch:"<<epoch<<endl;

        vector<std::tuple<torch::Tensor,torch::Tensor,double>> train_examples;
        for(int i=0;i<=iter_num;++i){
            cout<<"iter_num:"<<i<<endl;
            learner.self_play(1,pMCTS,pgame,train_examples);
        }

        cout<<"start train:"<<epoch<<endl;
        neural_network->train(train_examples,train_batch_size);
        cout<<"end train:"<<epoch<<endl;

        if (epoch==1){
            neural_network->save();
            continue;
        }
        if (epoch%check_interval!=0){
            continue;
        }

        //contest
        bool btest_use_gpu=torch::cuda::is_available();
        vector<future<bool>> vec;
        for (int i=0;i<=contest_num;++i){
            auto run=std::async(std::launch::async,&Learner::contest,learner,neural_network, n, n_in_row,btest_use_gpu);
            vec.push_back(std::move(run));
        }
        int iWin=0;
        for(int i=0;i<=contest_num;++i){
            int ans=vec[i].get();
            if (ans==true) iWin++;
        }
        float win_percent=iWin/contest_num;
//        float win_percent=0.65;


        if (win_percent>0.5)
            neural_network->save();
    }
    std::cout << "Hello, World!" << std::endl;
    return 0;

}


int eval(int first_color=1){//human:1
    Learner learner=Learner();
    int n=10;
    int n_in_row=4;
    shared_ptr<Gomoku> pgame=make_shared<Gomoku>(n,n_in_row,1);
    shared_ptr<NeuralNetwork> best_network = make_shared<NeuralNetwork>(n,n_in_row,false, 4);
//    best_network->load();
    shared_ptr<MCTS> best_MCTS = make_shared<MCTS>(best_network, 8, 1, 50, 1, 10 * 10);

    int players[2] = {first_color, -first_color};
    int cur_index = 0;
    int cur_player;

    while (true) {
        pgame->display();
        cur_player = players[cur_index];
        int next_move;
        if (cur_player == -1){
            vector<double> probs = best_MCTS->get_action_probs(pgame, 1);
            auto max_move = std::max_element(probs.begin(), probs.end());
            next_move = std::distance(probs.begin(), max_move);
        }
        else{
            cin>>next_move;

        }

        pgame->execute_move(next_move);

        best_MCTS->update_with_move(next_move);
        vector<int> result = pgame->get_game_status();
        int ended = result[0];
        int winner = result[1];

        if (ended == 1) {
            if (winner == -1)
            {cout<<"computer win"<<endl;
                return true;
            }
            else
            {
                cout<<"human win"<<endl;
                return false;
            }
        }
        ++cur_index;
        cur_index=cur_index%2;
    }
}



int main() {
//    eval();
    train();

//    shared_ptr<Gomoku> pgame=make_shared<Gomoku>(10,4,1);
//    pgame->execute_move( 1);
//    pgame->execute_move( 12);
//    pgame->execute_move( 2);
//    pgame->execute_move( 13);
//    pgame->execute_move( 3);
//    pgame->display();
//    Gomoku* raw_pgame=pgame.get();
//    shared_ptr<NeuralNetwork> neural_network =make_shared<NeuralNetwork> (false,4);
//    NeuralNetwork *raw_neural_network=neural_network.get();
//    shared_ptr<MCTS> pMCTS=make_shared<MCTS>(raw_neural_network, 8, 1,50, 1,10*10);
//    vector<double> probs=pMCTS->get_action_probs(raw_pgame,1);
//    cout<<"print probs"<<endl;
//    for (double i:probs) cout<<i<<endl;


    return 0;
}
