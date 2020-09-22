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
#include <iostream>

#include "filter.h"
#include "layer.h"
#include "ConvolutionLayer.h"
#include "PoolLayer.h"
#include "ActivationLayer.h"
#include "ConnectedLayer.h"

class Convolutional{
public:
    /**
     * @brief Constructs a convolutional neural network
     * @param vector that holds width, height, depth of input
     * @param vector that holds width, height of output
     * @param vector that holds architecture of repeated layers
     * @param times that architecture should be repeated
     * @param number of filters in conv layer
     * @param downsizing of layers
     */
    Convolutional(std::vector<size_t> t_input_matrix_size,std::vector<size_t> t_output_matrix_size,std::vector<size_t> t_architecture,size_t t_repeats,size_t t_num_filters,size_t t_filters_size,size_t t_pools_scale,size_t t_zero_padding) :
        m_pools_scale(t_pools_scale), m_num_filters(t_num_filters), m_filters_size(t_filters_size), m_zero_padding(t_zero_padding)
    {

        //push back input layers example r,g,b,a
        std::vector<Layer*> input_layers;
        for(size_t i = 0;i < t_input_matrix_size[2];i++){
            input_layers.push_back(new Layer());
        }
        m_layers.push_back(input_layers);

        //add conv+pool layers by repeats
        std::vector<Layer*> repeated_layers_vertical;
        for(size_t type:t_architecture) {
            for(size_t i = 0; i < m_layers[m_layers.size()-1].size();i++){
                switch(type){
                    case Layer::CONV:{
                        repeated_layers_vertical.push_back(new ConvolutionLayer());
                        break;
                    }
                    case Layer::POOL:{
                        //push back as much as we need if prev layer is conv layer
                        if(m_layers[m_layers.size()-1][0]->get_type() == Layer::CONV){
                            for(size_t k = 0;k < m_layers[i].size()*t_num_filters;k++){
                                repeated_layers_vertical.push_back(new PoolLayer());
                            }
                        }else{
                            repeated_layers_vertical.push_back(new PoolLayer());
                        }
                        break;
                    }
                    case Layer::ACT:{
                        //see pool case
                        if(m_layers[m_layers.size()-1][0]->get_type() == Layer::CONV){
                            for(size_t k = 0;k < m_layers[i].size()*t_num_filters;k++){
                                repeated_layers_vertical.push_back(new ActivationLayer());
                            }
                        }else{
                            repeated_layers_vertical.push_back(new ActivationLayer());
                        }
                        break;
                    }
                }
            }
            m_layers.push_back(repeated_layers_vertical);
            repeated_layers_vertical.clear();
        }


        //add connected layer
        std::vector<Layer*> connected_layers;
        connected_layers.push_back(new ConnectedLayer());
        m_layers.push_back(connected_layers);
    };
    enum function_type{
        SWISH = 0,RELU,SIGMOID
    };

    size_t train(std::vector<std::pair<std::vector<std::vector<float>>, std::vector<float>>> &t_training_data,size_t t_funcion_type,float t_learning_rate,float t_momentum, size_t t_epochs, size_t t_stride_filters,size_t t_stride_pools);
    size_t run_tests();
private:
    size_t m_num_layers;
    size_t m_filters_size;
    size_t m_num_filters;
    size_t m_stride;
    size_t m_pools_scale;
    float m_learning_rate;
    float m_momentum;
    size_t m_epochs;
    size_t m_zero_padding;
    size_t m_function_type;
    bool m_test;
    size_t m_stride_pool;
    size_t m_stride_filters;

    std::vector<float> feed_forward(std::pair<std::vector<std::vector<float>>, std::vector<float>> &t_data);
    std::vector<float> feed_forward_first(std::pair<std::vector<std::vector<float>>, std::vector<float>> &t_data);

    std::vector<std::vector<Layer*>> m_layers;
};

#endif /* CONVNETWORK_H */

