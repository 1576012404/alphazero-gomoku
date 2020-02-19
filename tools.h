#pragma once
#include <iostream>
#include <torch/torch.h>
#include <tuple>
#include <cfloat>
#include <cmath>
#include <random>
#include <memory>
#include <numeric>
using namespace std;


static thread_local std::mt19937 generator;
void dirichlet_random(vector<float> &p,vector<float> &alpha,int alphalen){
    float sum=0;
    for(int i=0;i<alphalen;i++){
        gamma_distribution<float> d(alpha[i],1.0);
        p[i]=d(generator);
        sum+=p[i];

    }
    for(int i=0;i<alphalen;i++){
        p[i]/=sum;
//        cout<<i<<" "<<p[i]<<endl;
    }
}


int random_choise(vector<float> &a,int len){
    float sum=std::accumulate(a.begin(),a.end(),0.0);
    uniform_real_distribution<float> value{0.0,sum};
    float fchoice{value(generator)};
    float total=0;
    for(int i=0;i<len;++i){
        total+=a[i];
        if (fchoice<total){
            return i;
        }
    }
    throw ("random_choise err");
}


