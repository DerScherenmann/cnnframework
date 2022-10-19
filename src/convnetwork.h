/*
 * File:   convnetwork.h
 * Author: DerScherenmann
 *
 * Created on 5 September 2020, 21:02
 */

#pragma once

#define INPUT_DIMENSIONS 3

#include <thread>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <boost/multi_array.hpp>
#include <boost/timer/progress_display.hpp>
#include <boost/ptr_container/ptr_container.hpp>

#include "Layer.h"
#include "ConvolutionLayer.h"
#include "PoolLayer.h"
#include "ActivationLayer.h"
#include "ConnectedLayer.h"
#include "kernel.h"

#include "lib/cifar10_reader.hpp"

using namespace layer;

/**
 * @brief Constructs a convolutional neural network
 * @param vector<size_t> that holds settings for connected layer network (size of hidden layers and output layers)
 * @param vector<size_t> that holds architecture of repeated layers (Layer::types)
 * @param size_t of filters in conv layer
 * @param size_t filter size
 * @param size_t pool size
 * @param size_t zero padding
*/
class Convolutional{

public:
    
    //define multi array type
    typedef boost::multi_array<float, INPUT_DIMENSIONS> array_3f;
    typedef boost::multi_array<Filter*, 2> array_2flt;
    typedef boost::multi_array_types::index_range range_t;
    //multi array input shape
    boost::array<array_3f::index, INPUT_DIMENSIONS> shape_array_images = {{ INPUT_DIMENSIONS, 32, 32 }};
    //index_gen for array type
    array_3f::index_gen indices;
    
    typedef std::shared_ptr<Filter> fshared_ptr_t;

    typedef struct struct_training_data {
        array_3f image_data;
        // TODO rename this maybe
        std::vector<float> corrrect_outputs;
    } struct_training_data;
    
    /*
     * TODO compare float and double as soon as the network is complete
     */
    Convolutional(std::vector<size_t> t_connected_matrix_size,std::vector<size_t> t_architecture,std::vector<size_t> t_num_filters,std::vector<size_t> t_filters_size,size_t t_pools_size,size_t t_zero_padding) :
         m_connected_matrix_size(t_connected_matrix_size),m_architecture(t_architecture),m_num_filters(t_num_filters),  m_filters_size(t_filters_size), m_pools_size(t_pools_size), m_zero_padding(t_zero_padding)
    {   
        // TODO implement something that changes filter numbers and filter options...
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
    
    /**
     * @brief Feed forward through network
     * @param struct_training_data training data
     * @return std::pair<std::vector<float>,std::vector<float>> vector of ouputs,deltas
     */
    std::pair<std::vector<float>,std::vector<float>> feed_forward(struct_training_data &t_data);

    [[maybe_unused]] size_t run_tests();

    /**
     * @brief Get error from network
     * @return float network error
     */
    float getError();

private:
    size_t m_num_layers;
    std::vector<size_t> m_filters_size;
    std::vector<size_t> m_num_filters;
    size_t m_stride;
    size_t m_pools_size;
    float m_learning_rate;
    float m_momentum;
    size_t m_epochs;
    size_t m_zero_padding;
    float m_error;
    // TODO make this a pointer and enable custom function submissions
    size_t m_function_type;
    bool m_test;
    size_t m_stride_pool;
    size_t m_stride_filters;
    std::vector<size_t> m_architecture;
    std::vector<size_t> m_connected_matrix_size;
    bool m_initialized = false;
    std::vector<std::vector<fshared_ptr_t>> m_filters;
    
    Math math;
    
    /**
     * @brief Initialize network
     * @param struct_training_data training data sample
     * @return std::vector<float> ouputs
     */
    size_t initialize(struct_training_data &t_data);

    std::vector<std::vector<Layer*>> m_layers;

};

