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
#pragma once

#include "knn/mathhelper.h"
#include "layer.h"

#include <iostream>
/**
 * Filters are kernels
 */
class Filter{
public:
    /**
    * Initializes a Kernel with random values
    * @param width
    * @param height
    */
   Filter(size_t t_width,size_t t_stride) : m_width(t_width), m_height(t_width), m_stride(t_stride){
       Math math;
       m_center = m_width/2 +1;

       m_values.resize(t_width);
       for(size_t i = 0;i<m_width;i++){
           m_values[i].resize(m_width);
           for(size_t j = 0;j < m_height;j++){
               m_values[i][j] = math.rng();
           }
       }
    }

    Layer calculate_output(Layer& t_input_layer){
        size_t output_width = (t_input_layer.get_width()-m_width+2*t_input_layer.get_padding())/m_stride +2;
        size_t output_height = output_width;

        std::vector<std::vector<float>> output_values;
        for(size_t width = 1;width < t_input_layer.get_width()-1;width+=m_stride){
            std::vector<float> output_column;
            for(size_t height = 1;height < t_input_layer.get_height()-1;height+=m_stride){
                float output = 0;
                for(size_t i = 0;i < m_width;i++){
                    float inner_sum = 0;
                    if(width == t_input_layer.get_width()-2){
                        for(size_t j = 0;j < m_height;j++){
                            inner_sum += t_input_layer.get_values()[width-i+1][height-j] * m_values[i][j];
                        }
                    }
                    if(width == t_input_layer.get_height()-2){
                        for(size_t j = 0;j < m_height;j++){
                            inner_sum += t_input_layer.get_values()[width-i][height-j+1] * m_values[i][j];
                        }
                    }
                    else{
                        for(size_t j = 0;j < m_height;j++){
                            inner_sum += t_input_layer.get_values()[width-i][height-j] * m_values[i][j];
                        }
                    }
                    output += inner_sum;
                }
                output_column.push_back(output);
            }
            output_values.push_back(output_column);
        }

        Layer output_layer = Layer(output_width,output_height,t_input_layer.get_depth(),Layer::types::OUTPUT);
        output_layer.set_values(output_values);
        return output_layer;
    };
    size_t get_width(){
        return m_width;
    }
    size_t get_height(){
        return m_height;
    }
private:
    size_t m_width;
    size_t m_height;
    size_t m_stride;
    float m_bias;
    size_t m_center;
    //essentially the weights (compared to a knn)
    std::vector<std::vector<float>> m_values;
};


#endif /* FILTER_H */

