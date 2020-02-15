#pragma once

#include <torch/torch.h>  // One-stop header.

#include <future>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <tuple>
using namespace std;

#include "gomoku.h"


struct NetImpl : torch::nn::Module {
    NetImpl()
            : conv1(torch::nn::Conv2dOptions(3, 10, /*kernel_size=*/2)),
              conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/2)),
              fc1(20, 50),
              fc2(50, 10*10),
              fc3(50, 1){
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv2_drop", conv2_drop);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    std::tuple<torch::Tensor,torch::Tensor> forward(torch::Tensor x) {
        cout<<"1"<<endl;
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(
                torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
        x = x.view({-1, 20});
        cout<<"2"<<endl;
        x = torch::relu(fc1->forward(x));
        cout<<"3"<<endl;
        x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
        auto p = fc2->forward(x);
        p=torch::log_softmax(p, /*dim=*/1);
        cout<<"4"<<endl;

        auto v = fc3->forward(x);
        cout<<"5"<<endl;

        return std::make_tuple(p,v);
    }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Dropout2d conv2_drop;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
};






//#define conv3x3(in_channels,out_channels,strid)(torch::nn::Conv2dOptions(in_channels, out_channels,3).stride(strid).padding(1).bias(false))
//
//struct ResidualBlock:torch::nn::Module{
//    ResidualBlock(int in_channels,int out_channels,int stride=1):
//            in_channels(in_channels),out_channels(out_channels),stride(stride),
//            conv1(conv3x3(in_channels,out_channels,stride)),
//            conv2(conv3x3(out_channels,out_channels,stride)),
//            bn1(torch::nn::BatchNorm2d(out_channels)),
//            bn2(torch::nn::BatchNorm2d(out_channels)),
//            downsample_conv(conv3x3(out_channels,out_channels,stride)),
//            downsample_bn(torch::nn::BatchNorm2d(out_channels))
//    {
//        if (in_channels!=out_channels || stride !=1)
//            downsample= true;
//        else
//            downsample=false;
//        register_module("conv1", conv1);
//        register_module("conv2", conv2);
//        register_module("downsample_conv", downsample_conv);
//
//        register_module("bn1", bn1);
//        register_module("bn2", bn2);
//        register_module("downsample_bn", downsample_bn);
//
//    }
//
//    torch::nn::Conv2d conv1,conv2,downsample_conv;
//    torch::nn::BatchNorm2d bn1, bn2,downsample_bn;
//    int in_channels;
//    int out_channels;
//    int stride;
//    bool downsample;
//
//    torch::Tensor forward(torch::Tensor x) {
//        torch::Tensor residual=x;
//        x = torch::relu(bn1(conv1(x)));
//        x = bn2(conv2(x));
//        if (downsample){
//            residual=downsample_bn(downsample_conv(x));
//
//        }
//        x=x+residual;
//        x=torch::relu(x);
//        return x;
//    }
//
//
//};
//
//
//struct NetImpl : torch::nn::Module {
//    NetImpl(int num_layers,int num_channels,int n,int action_size)
//            :p_conv(torch::nn::Conv2dOptions(num_channels, 4,1).padding(0).bias(false)),
//             v_conv(torch::nn::Conv2dOptions(num_channels, 2,1).padding(0).bias(false)),
//             p_bn(torch::nn::BatchNorm2d(4)),
//             v_bn(torch::nn::BatchNorm2d(2)),
//
//             p_fc(torch::nn::Linear(4*pow(n,2),action_size)),
//             v_fc1(torch::nn::Linear(2*pow(n,2),256)),
//             v_fc2(torch::nn::Linear(256,1))
//    {
//        res_list->push_back(ResidualBlock(3,num_channels));
//        for(int i=0;i<num_layers;++i)
//            res_list->push_back(ResidualBlock(num_channels,num_channels));
//
//
//        register_module("p_conv", p_conv);
//        register_module("p_bn", p_bn);
//        register_module("p_fc", p_fc);
//
//        register_module("v_conv", v_conv);
//        register_module("v_bn", v_bn);
//        register_module("v_fc1", v_fc1);
//        register_module("v_fc2", v_fc2);
//    }
//
//    std::tuple<torch::Tensor,torch::Tensor> forward(torch::Tensor x) {
////    torch::Tensor forward(torch::Tensor x) {
//
//        torch::Tensor out = res_list->forward(x);
//        torch::Tensor p=p_conv(out);
//        p=p_bn(p);
//        p=torch::relu(p);
//        p=p_fc(p.view({p.size(0),-1}));
//        p=torch::log_softmax(p,1);
//
//
//        torch::Tensor v=v_conv(out);
//        v=v_bn(v);
//        v=torch::relu(v);
//
//        v=v_fc1(v.view({v.size(0),-1}));
//        v=torch::relu(v);
//        v=v_fc2(v);
//        v=torch::tanh(v);
////        return p;
//
//        return std::make_tuple(p,v);
//    }
//    torch::nn::Sequential res_list;
//    torch::nn::Conv2d p_conv,v_conv;
//    torch::nn::BatchNorm2d p_bn, v_bn;
//    torch::nn::Linear p_fc,v_fc1,v_fc2;
//
//};
//
TORCH_MODULE(Net);


class NeuralNetwork {
public:
    using return_type = std::vector<std::vector<double>>;

    NeuralNetwork( int n,int n_in_row,bool use_gpu, unsigned int batch_size);
    ~NeuralNetwork();

    std::future<return_type> commit(Gomoku* gomoku);  // commit task to queue
    void set_batch_size(unsigned int batch_size) {    // set batch_size
        this->batch_size = batch_size;void train(std::vector<tuple<board_type,std::vector<double>,int,int,int>> train_data,int batch_size);//train
    };

    void train(vector<std::tuple<torch::Tensor,torch::Tensor,double>> &train_examples,int batch_size);//train
    bool save();
    bool load();


private:
    int n;
    int n_in_row;
    using task_type = std::pair<torch::Tensor, std::promise<return_type>>;

    void infer();  // infer



    std::unique_ptr<std::thread> loop;  // call infer in loop
    bool running;                       // is running
    bool use_gpu;
    std::queue<task_type> tasks;  // tasks queue
    std::mutex lock;              // lock for tasks queue
    std::condition_variable cv;   // condition variable for tasks queue

    Net module;  // torch module
    torch::optim::Adam opt;
    unsigned int batch_size;                             // batch size
                                          // use gpu
};
