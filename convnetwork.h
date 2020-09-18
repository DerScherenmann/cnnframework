/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   convnetwork.h
 * Author: broesel233
 *
 * Created on 5 September 2020, 21:02
 */

#ifndef CONVNETWORK_H
#define CONVNETWORK_H

#include <vector>
#include <string>
#include <algorithm>

#include "filter.h"
#include "layer.h"
#include "ConvolutionLayer.h"
#include "PoolLayer.h"
#include "ActivationLayer.h"
#include "ConnectedLayer.h"

class Convolutional{
public:
    /**
     * Constructs a convolutional neural network
     * @param vector that holds width, height, depth of input
     * @param vector that holds width, height of output
     * @param vector that holds architecture of repeated layers
     * @param times that architecture should be repeated
     * @param number of filters in conv layer
     * @param downsizing of layers
     */
    Convolutional(std::vector<size_t> t_input_matrix_size,std::vector<size_t> t_output_matrix_size,std::vector<size_t> t_architecture,size_t t_repeats,size_t t_filters,size_t t_pools_scale) : m_filters(t_filters), m_pools_scale(t_pools_scale) {
        //push back input layers example r,g,b,a
        std::vector<Layer*> input_layers;
        for(size_t i = 0;i < t_input_matrix_size[2];i++){
            Layer* input;
            input_layers.push_back(input);
        }
        m_layers.push_back(input_layers);

        //add conv+pool layers by repeats
        for(size_t repeats = 0;repeats < t_repeats;repeats++){
            for(size_t i = 0;i < m_layers.size();i++){
                std::vector<Layer*> repeated_layers_vertical;
                for(size_t type:t_architecture){
                    //add so much layers needed to process prev slice, for each layer we need one, if conv layer prev slice we need num*filters as pool layers
                    for(size_t j = 0;j < m_layers[i].size();j++){
                        switch(type){
                            case Layer::CONV:
                                ConvolutionLayer* conv;
                                repeated_layers_vertical.push_back(conv);
                                break;
                            case Layer::POOL:
                                //push back as much as we need if prev layer is conv layer
                                if(dynamic_cast<ConvolutionLayer*>(m_layers[i][j])){
                                    for(size_t k = 0;k < m_layers[i].size()*t_filters;k++){
                                        PoolLayer* pool;
                                        repeated_layers_vertical.push_back(pool);
                                    }
                                }else{
                                    PoolLayer* pool;
                                    repeated_layers_vertical.push_back(pool);
                                }
                                break;
                            case Layer::ACT:
                                //see pool case
                                if(dynamic_cast<ConvolutionLayer*>(m_layers[i][j])){
                                    for(size_t k = 0;k < m_layers[i].size()*t_filters;k++){
                                        ActivationLayer* act;
                                        repeated_layers_vertical.push_back(act);
                                    }
                                }else{
                                    ActivationLayer* act;
                                    repeated_layers_vertical.push_back(act);
                                }
                                break;
                        }
                    }
                }
                m_layers.push_back(repeated_layers_vertical);
            }
        }


    };

    size_t train(std::vector<std::pair<std::vector<std::vector<float>>, std::vector<float>>> &t_training_data,float t_learning_rate,float t_momentum, size_t t_epochs);
    size_t run_tests();
private:
    size_t m_num_layers;
    size_t m_filters;
    size_t m_pools_scale;
    float m_learning_rate;
    float m_momentum;
    size_t m_epochs;

    std::vector<float> feed_forward(std::pair<std::vector<std::vector<float>>, std::vector<float>> &t_data);

    std::vector<std::vector<Layer*>> m_layers;
};

#endif /* CONVNETWORK_H */

