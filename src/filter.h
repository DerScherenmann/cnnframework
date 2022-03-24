/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   kernel.h
 * Author: broesel233
 *
 * Created on 6 September 2020, 00:35
 */

#ifndef FILTER_H
#define FILTER_H

#include "lib/mathhelper.h"
#include "Layer.h"

#include <iostream>
#include <vector>

using namespace layer;

/**
 * Filters are kernels
 */
class Filter {
public:
    /**
    * Initializes a Kernel/Filter with random values
    * @param width
    * @param height
    */
    Filter(size_t t_width,size_t t_stride,size_t t_function_type) : m_function_type(t_function_type), m_width(t_width), m_height(t_width), m_stride(t_stride){
        Math math;
        m_center = m_width/2 +1;

        m_values.resize(t_width);
        for(size_t i = 0;i<m_width;i++){
            m_values[i].resize(m_width);
            for(size_t j = 0;j < m_height;j++){
                m_values[i][j] = math.rng();
            }
        }

        //zero old changes so first time is always 0
        for(size_t i = 0;i < t_width;i++){
            std::vector<float> column;
            for(size_t j = 0;j < t_width;j++){
                column.push_back(0);
            }
            m_old_changes.push_back(column);
        }
        
        //initialize deltas
        m_deltas.resize(t_width);
        for(size_t i = 0;i<m_width;i++){
            m_deltas[i].resize(m_width);
            for(size_t j = 0;j < m_height;j++){
                m_deltas[i][j] = math.rng();
            }
        }
    }

    virtual Layer::array_2d_t calculate_output(Layer& t_input_layer){
        
        size_t output_width = (t_input_layer.get_width()-m_width+2*t_input_layer.get_padding())/m_stride;
        size_t output_height = (t_input_layer.get_height()-m_height+2*t_input_layer.get_padding())/m_stride;
        
        boost::array<Layer::array_2d_t::index, 2> shape_output = {{ (long) output_width,(long) output_height }};
        Layer::array_2d_t output_values(shape_output);
        
        Layer::array_2d_t input_values = t_input_layer.get_values();
        
        for(size_t width_input = 1;width_input < t_input_layer.get_width()-1;width_input+=m_stride){
            for(size_t height_input = 1;height_input < t_input_layer.get_height()-1;height_input+=m_stride){
                float output = 0;
                for(size_t i = 0;i < m_width;i++){
                    float inner_sum = 0;
                    for(size_t j = 0;j < m_height;j++){
                        inner_sum += input_values[width_input-i+m_width/2][height_input-j+m_width/2] * m_values[i][j];
                    }
                    
//                     if(width_input == t_input_layer.get_width()-start){
//                         for(size_t j = 0;j < m_height;j++){
//                             inner_sum += input_values[width_input-(i+1)][height_input-j] * m_values[i][j];
//                         }
//                     }
//                     else if(width_input == t_input_layer.get_height()-start){
//                         for(size_t j = 0;j < m_height;j++){
//                             inner_sum += input_values[width_input-i][height_input-(j+1)] * m_values[i][j];
//                         }
//                     }
//                     else{
//                         for(size_t j = 0;j < m_height;j++){
//                             inner_sum += input_values[width_input-(i/2)][height_input-(j/2)] * m_values[i][j];
//                         }
//                     }
                    output += inner_sum;
                }
                output_values[width_input][height_input] = act(output);
            }
        }
        
        return output_values;
    };
    float act(float x){

        if(m_function_type == Layer::SWISH){
            x = math.swish(x);
        }
        if(m_function_type == Layer::SIGMOID){
            x = math.sigmoid(x);
        }
        if(m_function_type == Layer::RELU){

        }
        return x;
    }

    float actPrime(float x){

        if(m_function_type == Layer::SWISH){
            x = math.swishPrime(x);
        }
        if(m_function_type == Layer::SIGMOID){
            x = math.sigmoidPrime(x);
        }
        if(m_function_type == Layer::RELU){

        }
        return x;
    }
    virtual size_t get_width(){
        return m_width;
    }
    virtual size_t get_height(){
        return m_height;
    }
    virtual std::vector<std::vector<float>> get_weights(){
        return m_values;
    }
    virtual std::vector<std::vector<float>> get_old_changes(){
        return m_old_changes;
    }
    virtual std::vector<std::vector<float>> get_deltas(){
        return m_deltas;
    }
    Layer* get_calculated_ouput(){
        return m_output_layer;
    }
    size_t set_deltas(std::vector<std::vector<float>> t_deltas){
        m_deltas = t_deltas;
        return 0;
    }
    size_t set_old_changes(std::vector<std::vector<float>> t_old_changes){
        m_deltas = t_old_changes;
        return 0;
    }
    
private:
    size_t m_width;
    size_t m_height;
    size_t m_stride;
    float m_bias;
    size_t m_center;
    size_t m_function_type;
    Math math;
    //essentially the weights (compared to a knn)
    std::vector<std::vector<float>> m_values;
    //sums without activation
    std::vector<std::vector<float>> m_sums;
    //old changes
    std::vector<std::vector<float>> m_old_changes;
    //deltas maybe not needed if we change and calculate in one go
    std::vector<std::vector<float>> m_deltas;
    //last calculated output, shouldn't take much memory
    Layer* m_output_layer;
};


#endif /* FILTER_H */

