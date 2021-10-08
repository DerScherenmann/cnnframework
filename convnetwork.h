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

#define INPUT_DIMENSIONS 3

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <boost/multi_array.hpp>

#include "Layer.h"
#include "ConvolutionLayer.h"
#include "PoolLayer.h"
#include "ActivationLayer.h"
#include "ConnectedLayer.h"

#include "lib/cifar10_reader.hpp"

using namespace layer;

/**
 * Initialize convolutional network
 * @brief Constructs a convolutional neural network
 * @param vector<size_t> that holds width, height, depth of input
 * @param vector<size_t> that holds settings for connected layer network
 * @param vector<size_t> that holds architecture of repeated layers
 * @param size_t times architecture should be repeated
 * @param size_t of filters in conv layer
 * @param size_t pool size
 * @param size_t zero padding
*/
class Convolutional{

public:
    
    //define multi array type
    typedef boost::multi_array<float, INPUT_DIMENSIONS> array_t;
    typedef boost::multi_array_types::index_range range_t;
    //multi array input shape
    boost::array<array_t::index, INPUT_DIMENSIONS> shape_array_images = {{ INPUT_DIMENSIONS, 32, 32 }};
    //index_gen for array type
    array_t::index_gen indices;
    
    typedef struct struct_training_data {
        array_t image_data;
        size_t image_label;
    } struct_training_data;
    
    Convolutional(std::vector<size_t> t_input_matrix_size,std::vector<size_t> t_connected_matrix_size,std::vector<size_t> t_architecture,size_t t_repeats,size_t t_num_filters,size_t t_filters_size,size_t t_pools_size,size_t t_zero_padding) :
         m_connected_matrix_size(t_connected_matrix_size),m_num_repeats(t_repeats),m_num_filters(t_num_filters),  m_filters_size(t_filters_size), m_pools_size(t_pools_size), m_zero_padding(t_zero_padding)
    {   
        //push back input layers example r,g,b,a
        std::vector<Layer*> input_layers;
        for(size_t i = 0;i < t_input_matrix_size[2];i++){
            input_layers.push_back(new Layer());
        }
        m_layers.push_back(input_layers);

        for(size_t repeats = 0;repeats < m_num_repeats;repeats++){
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
                                for(size_t k = 0;k < t_num_filters;k++){
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
        }
        

        //add connected layer
        std::vector<Layer*> connected_layers;
        connected_layers.push_back(new ConnectedLayer());
        m_layers.push_back(connected_layers);
    };
    enum function_type{
        SWISH = 0,RELU,SIGMOID
    };

    float act(float x){

        if(m_function_type == SWISH){
            x = math.swish(x);
        }
        if(m_function_type == SIGMOID){
            x = math.sigmoid(x);
        }
        return x;
    }

    float actPrime(float x){

        if(m_function_type == SWISH){
            x = math.swishPrime(x);
        }
        if(m_function_type == SIGMOID){
            x = math.sigmoidPrime(x);
        }

        return x;
    }
    /**
     * Train network
     * @param std::vector<struct_training_data> training data
     * @param size_t function type
     * @param float learning rate
     * @param float momentum
     * @param size_t epochs
     * @param size_t filter stride
     * @param size_t pool stride
     * @return size_t on success
     */
    size_t train(std::vector<struct_training_data> &t_training_data,size_t t_funcion_type,float t_learning_rate,float t_momentum, size_t t_epochs, size_t t_stride_filters,size_t t_stride_pools);
    float backprop_momentum(float &deltaCurrent, float &activationBefore, float &oldChange);
    size_t run_tests();
    
private:
    size_t m_num_layers;
    size_t m_filters_size;
    size_t m_num_filters;
    size_t m_stride;
    size_t m_pools_size;
    float m_learning_rate;
    float m_momentum;
    size_t m_epochs;
    size_t m_zero_padding;
    size_t m_function_type;
    bool m_test;
    size_t m_stride_pool;
    size_t m_stride_filters;
    size_t m_num_repeats;
    std::vector<size_t> m_connected_matrix_size;
    
    Math math;
    
    /**
     * Feed forward through network
     * @param struct_training_data training data
     * @return std::vector<float> ouputs
     * @return std::vector<float> deltas
     */
    std::pair<std::vector<float>,std::vector<float>> feed_forward(struct_training_data &t_data);
    /**
     * Initialize network
     * @param struct_training_data training data sample
     * @return std::vector<float> ouputs
     */
    std::vector<float> initialize(struct_training_data &t_data);

    std::vector<std::vector<Layer*>> m_layers;
    
};

#endif /* CONVNETWORK_H */

