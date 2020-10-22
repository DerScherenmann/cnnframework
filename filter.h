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
#include "layer.h"

#include <iostream>
#include <vector>
/**
 * Filters are kernels
 */
class Filter {
public:
    /**
    * Initializes a Kernel with random values
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
    }

    virtual Layer calculate_output(Layer& t_input_layer){

        size_t output_width = (t_input_layer.get_width()-m_width+2*t_input_layer.get_padding())/m_stride+2;
        size_t output_height = output_width;

        size_t start = 1;

        std::vector<std::vector<float>> output_values;
        for(size_t width_input = start;width_input < t_input_layer.get_width();width_input+=m_stride){
            std::vector<float> output_column;
            for(size_t height_input = start;height_input < t_input_layer.get_height();height_input+=m_stride){
                float output = 0;
                for(size_t i = 0;i < m_width;i++){
                    float inner_sum = 0;
                    if(width_input == t_input_layer.get_width()-1){
                        for(size_t j = 0;j < m_height;j++){
                            inner_sum += t_input_layer.get_values()[width_input-i+1][height_input-j] * m_values[i][j];
                        }
                    }
                    if(width_input == t_input_layer.get_height()-1){
                        for(size_t j = 0;j < m_height;j++){
                            inner_sum += t_input_layer.get_values()[width_input-i][height_input-j+1] * m_values[i][j];
                        }
                    }
                    else{
                        for(size_t j = 0;j < m_height;j++){
                            inner_sum += t_input_layer.get_values()[width_input-i/2][height_input-j/2] * m_values[i][j];
                        }
                    }
                    output += inner_sum;
                }
                output_column.push_back(act(output));
            }
            output_values.push_back(output_column);
        }

        Layer output_layer = Layer(output_width,output_height,t_input_layer.get_depth(),Layer::types::OUTPUT);
        output_layer.set_values(output_values);
        return output_layer;
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
    virtual std::vector<std::vector<float>> get_values(){
        return m_values;
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
};


#endif /* FILTER_H */

