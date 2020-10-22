/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   ConnectedLayer.h
 * Author: broesel233
 *
 * Created on 6 September 2020, 15:51
 */

#ifndef CONNECTEDLAYER_H
#define CONNECTEDLAYER_H

#include <utility>

#include "layer.h"
#include "lib/network.h"

class ConnectedLayer : public Layer{
public:
    ConnectedLayer(){};
    ConnectedLayer(size_t t_functiontype, std::vector<size_t> t_sizes, bool t_raw_output) : Layer(0,0,0,Layer::types::CONNECTED), m_functiontype(t_functiontype), m_net_in_size(t_sizes[0]){
        std::vector<std::pair<int,int>> net_sizes;
        for(size_t layer_size:t_sizes){
            net_sizes.push_back(std::make_pair(layer_size,t_functiontype));
        }
        if(!t_raw_output) {
            net_sizes.push_back(std::make_pair(t_sizes[t_sizes.size()-1],functiontype::SIGMOID));
        }
        m_net = new Network(net_sizes);
    };
    //ConnectedLayer(const ConnectedLayer& orig) : Layer(orig) {
    //};
    virtual ~ConnectedLayer(){

    };
    std::vector<float> train(std::pair<std::vector<float>,std::vector<float>> &t_training_data,float t_learning_rate,float t_momentum){
        std::vector<float> deltas = m_net->train_once(t_training_data,t_learning_rate,t_momentum);
        return deltas;
    }
    size_t forward(){
        if(m_net == NULL || m_values[0].size() == 0) return 1;
        m_net_output = m_net->predict(m_values[0]);
        return 0;
    };
    std::vector<float> get_net_output(){
        return m_net_output;
    }
    size_t get_type(){
        return Layer::CONNECTED;
    }
    size_t get_in_size(){
        return m_net_in_size;
    }
private:
    size_t m_functiontype;
    size_t m_net_in_size;
    std::vector<float> m_net_output;
    Network* m_net;
};

#endif /* CONNECTEDLAYER_H */

