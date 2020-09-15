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
#include "knn/network.h"

class ConnectedLayer : public Layer{
public:
    ConnectedLayer(size_t t_functiontype, std::vector<std::pair<int,int>> t_sizes) : Layer(m_width,m_height,m_depth,Layer::types::CONNECTED), m_functiontype(t_functiontype){
        m_net = new Network(t_sizes);
    };
    //ConnectedLayer(const ConnectedLayer& orig) : Layer(orig) {
    //};
    virtual ~ConnectedLayer(){

    };
    size_t forward(std::vector<float> &input){
        if(m_net == NULL) return 1;
        m_net->feedForward(input);
        return 0;
    };
    enum functiontype{
        SWISH = 0,SIGMOID,RELU
    };
private:
    size_t m_functiontype;
    Network* m_net;
};

#endif /* CONNECTEDLAYER_H */

