#include "libtorch.h"

#include <iostream>
#include <tuple>
#include <random>
using namespace std::chrono_literals;
using namespace std;
NeuralNetwork::NeuralNetwork( int n,int n_in_row,bool use_gpu,
                             unsigned int batch_size)
        :n(n),
        n_in_row(n_in_row),
        module(Net()),//num_layers,int num_channels,int n,int action_size 4,256,n,n*n
          opt(module->parameters(), torch::optim::AdamOptions(1e-3)),
          use_gpu(use_gpu),
          batch_size(batch_size),
          running(true),
          loop(nullptr) {

    if (use_gpu) {
        torch::DeviceType device_type=torch::kCUDA;
        torch::Device device(device_type);
        cout<<"use_gpu"<<endl;
        // move to CUDA
        this->module->to(device);
    }
    else{
        cout<<"use cpu"<<endl;
    };

    // run infer thread
    this->loop = std::make_unique<std::thread>([this] {
        while (this->running) {
            this->infer();
        }
    });
}

NeuralNetwork::~NeuralNetwork() {
    this->running = false;
    this->loop->join();
}

std::future<NeuralNetwork::return_type> NeuralNetwork::commit(Gomoku* gomoku) {
    int n = gomoku->get_n();

    // convert data format
    auto board = gomoku->get_board();
    std::vector<int> board0;
    for (unsigned int i = 0; i < board.size(); i++) {
        board0.insert(board0.end(), board[i].begin(), board[i].end());
    }

    torch::Tensor temp =
            torch::from_blob(board0.data(), {1, 1, n, n}, torch::dtype(torch::kInt32));

    torch::Tensor state0=(temp==1).toType(torch::kFloat32);
    torch::Tensor state1=(temp==-1).toType(torch::kFloat32);

    int last_move = gomoku->get_last_move();
    int cur_player = gomoku->get_current_color();

    if (cur_player == -1) {
        std::swap(state0, state1);
    }

    torch::Tensor state2 =
            torch::zeros({1, 1, n, n}, torch::dtype(torch::kFloat32));

    if (last_move != -1) {
        state2[0][0][last_move / n][last_move % n] = 1;
    }

    // torch::Tensor states = torch::cat({state0, state1}, 1);
    torch::Tensor states = torch::cat({state0, state1, state2}, 1);

    // emplace task
    std::promise<return_type> promise;
    auto ret = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(this->lock);
        tasks.emplace(std::make_pair(states, std::move(promise)));
    }

    this->cv.notify_all();

    return ret;
}

void NeuralNetwork::infer() {
    // get inputs
    std::vector<torch::Tensor> states;
    std::vector<std::promise<return_type>> promises;

    bool timeout = false;
    while (states.size() < this->batch_size && !timeout) {
        // pop task
        {
            std::unique_lock<std::mutex> lock(this->lock);
            if (this->cv.wait_for(lock, 1ms,
                                  [this] { return this->tasks.size() > 0; })) {
                auto task = std::move(this->tasks.front());
                states.emplace_back(std::move(task.first));
                promises.emplace_back(std::move(task.second));

                this->tasks.pop();

            } else {
                // timeout
                // std::cout << "timeout" << std::endl;
                timeout = true;
            }
        }
    }

    // inputs empty
    if (states.size() == 0) {
        return;
    }
//    std::cout<<"infer_block:"<<states.size()<<"batch_size:"<<this->batch_size<<std::endl;

    // infer

    torch::Tensor states_batch=torch::cat(states,0);
    if (this->use_gpu){
      cout<<"data to gpu"<<endl;
        torch::DeviceType device_type=torch::kCUDA;
        torch::Device device(device_type);
        states_batch=states_batch.to(device);
    }
    std::tuple<torch::Tensor,torch::Tensor> out=module->forward(states_batch);
  cout<<"after forward"<<endl;
    torch::Tensor p_batch,v_batch;
    std::tie(p_batch,v_batch)=out;
  cout<<"after_tie"<<endl;
    p_batch=p_batch.detach().exp();
    v_batch=v_batch.detach();

    // set promise value
    for (unsigned int i = 0; i < promises.size(); i++) {
//        std::cout<<"set_promise"<<i<<std::endl;
        torch::Tensor p = p_batch[i];
        torch::Tensor v = v_batch[i];

        std::vector<double> prob(static_cast<float*>(p.data_ptr()),
                                 static_cast<float*>(p.data_ptr()) + p.size(0));
        std::vector<double> value{v.item<float>()};
        return_type temp{std::move(prob), std::move(value)};

        promises[i].set_value(std::move(temp));
    }
//    std::cout<<"set promises"<<std::endl;

}



bool NeuralNetwork::save(){
    torch::save(module,"model.pt");
    return true;

}

bool NeuralNetwork::load(){
    torch::load(module,"model.pt");
    return true;

}

void NeuralNetwork:: train(vector<std::tuple<torch::Tensor,torch::Tensor,double>> &train_data,int batch_size){
    static thread_local std::mt19937 generator;
    std::shuffle(std::begin(train_data),std::end(train_data),generator);

    int epoch=train_data.size()/batch_size;
    for (int i=0;i<epoch;++i){
//        cout<<"batch_index:"<<i<<",batch_size"<<batch_size<<endl;
        vector<torch::Tensor> boards;
        vector<torch::Tensor> probs;
        vector<double> values;
        for(int j=0;j<batch_size;++j){
            int index=batch_size*i+j;
            torch::Tensor board;
            torch::Tensor prob;
            double value;
            std::tie(board,prob,value)=train_data[index];

            boards.push_back(std::move(board));
            probs.push_back(std::move(prob));
            values.push_back(value);
        }

        torch::Device device(torch::kCPU);
        if (this->use_gpu){
            device=torch::Device(torch::kCUDA);

        }
        torch::Tensor boards_tor=torch::cat(boards,0).to(device);
        torch::Tensor probs_tor=torch::cat(probs,0).to(device);
        torch::Tensor values_tor=torch::from_blob(values.data(),{batch_size,1}).to(device);
        //forward

        this->opt.zero_grad();
        std::tuple<torch::Tensor,torch::Tensor> out=module->forward(boards_tor);
        torch::Tensor log_ps,v_batch;
        std::tie(log_ps,v_batch)=out;


        torch::Tensor value_loss = torch::mean(torch::pow(v_batch - values_tor, 2));
        torch::Tensor policy_loss = -torch::mean(torch::sum(probs_tor * log_ps, 1));
        torch::Tensor toal_loss=value_loss+policy_loss;
        toal_loss.backward();
        this->opt.step();
    }

}
