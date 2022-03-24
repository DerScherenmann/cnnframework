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
#include <iostream>

#include "Layer.h"
#include "lib/network.h"

using namespace layer;

class ConnectedLayer : public Layer{
public:
    ConnectedLayer(array_2f t_values,size_t t_functiontype, std::vector<size_t> t_sizes, bool t_raw_output) : 
        Layer(t_values,Layer::types::CONNECTED),m_functiontype(t_functiontype), m_net_in_size(t_sizes[0]){
        std::vector<std::pair<int,int>> net_sizes;
        for(size_t layer_size:t_sizes){
            net_sizes.push_back(std::make_pair(layer_size,t_functiontype));
        }
        if(!t_raw_output) {
            net_sizes[net_sizes.size()-1] = std::make_pair(t_sizes[t_sizes.size()-1],functiontype::SIGMOID);
        }
        m_net = new Network(net_sizes);
    };
    //ConnectedLayer(const ConnectedLayer& orig) : Layer(orig) {
    //};
    virtual ~ConnectedLayer(){

    };
    std::vector<float> train(std::vector<float> &t_corrected_outputs,float t_learning_rate,float t_momentum){
        std::vector<float> input_data;
        input_data.reserve(m_values[0].size());
        for(float f:m_values[0]){
            input_data.push_back(f);
        }
        std::pair<std::vector<float>,std::vector<float>> t_training_data  = std::make_pair(input_data, t_corrected_outputs);
        std::vector<float> deltas = m_net->train_once(t_training_data,t_learning_rate,t_momentum);
        return deltas;
    }
    size_t forward(){
        if(m_net == NULL || m_values[0].size() == 0) return 1;
        std::vector<float> net_inputs;
        net_inputs.reserve(m_values[0].size());
        for(size_t i = 0;i < m_values[0].size();i++){
            net_inputs.push_back(m_values[0][i]);
        }
        m_net_output = m_net->predict(net_inputs);
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

