/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   PoolLayer.h
 * Author: broesel233
 *
 * Created on 6 September 2020, 15:42
 */

#ifndef POOLLAYER_H
#define POOLLAYER_H

#include <algorithm>
#include <tuple>
#include <stdexcept>
#include <iostream>

#include "Layer.h"

using namespace layer;

/**
 * Pooling Layer
 */
class PoolLayer : public Layer {
public:

    PoolLayer(array_2f t_values,size_t t_width,size_t t_height,size_t t_stride,size_t t_pool_op) : Layer(t_values,Layer::POOL), m_width(t_width), m_height(t_height), m_stride(t_stride), m_pool_op(t_pool_op) {};
    //PoolLayer(const PoolLayer& orig) : Layer(orig) {};
    virtual ~PoolLayer(){};
    /**
     * Pool inputs (input must be multiple of width and height)
     * @param array_2d_t input
     * @return size_t on success
     */
    size_t pool(array_2f t_input){
        m_input_values.resize(boost::extents[t_input.size()][t_input[0].size()]);
        m_input_values = t_input;
        if(m_pool_op == MAX){
            pool_max(t_input);
        }
        if(m_pool_op == AVERAGE){
            pool_average(t_input);
        }
        return 0;
    }

    //TODO test reference vs normal
    //TODO padding if input width or height cant be divided by m_stride
    size_t pool_max(array_2f t_input){

        // Add padding if stride > 1
        if(m_stride > 1){
            //Padding can't be greater than 1 (except if stride >= m_width/m_height)
            size_t padding_width = t_input.size() % m_width;
            size_t padding_height = t_input[0].size() % m_height;

            if(padding_width > 0 || padding_height > 0){

                boost::array<array_2f::index, 2> shape_padding_pool = {{ (long) (padding_width+t_input.size()), (long) (padding_height+t_input[0].size())}};
                t_input.resize(boost::extents[(padding_width+t_input.size())][padding_height+t_input[0].size()]);

                for(size_t i = 0;i < padding_width;i++){
                    for(size_t height = 0;height < t_input[0].size();height++){
                        //duplicate outer numbers
                        t_input[t_input.size()-i-1][height] = t_input[t_input.size()-i-2][height];
                    }
                }
                for(size_t i = 0;i < padding_height;i++){
                    for(size_t width = 0;width < t_input[0].size();width++){
                        t_input[width][t_input[0].size()-i-1] = t_input[width][t_input[0].size()-i-2];
                    }
                }
            }
        }

        // Resize m_values to match output
        size_t output_width = (t_input.size()-m_width)/m_stride+1;
        size_t output_height = (t_input[0].size()-m_height)/m_stride+1;

        m_values.resize(boost::extents[output_width][output_height]);
        m_input_indices.resize(boost::extents[t_input.size()][t_input[0].size()]);

        // this could be removed by editing this function and replaced by boost multi array
        std::vector<std::vector<float>> pane;

        // Do this so pane.size()-1 doesn't throw an error
        pane.resize(1);

        for(size_t input_width = 0;input_width < t_input.size();input_width+=m_stride){
            std::vector<float> column;
            std::vector<std::pair<float, std::pair<size_t,size_t>>> index_column;
            // Stop if we reached the end of the input array
            if(input_width+m_height > t_input.size()){
                continue;
            }
            for(size_t input_height = 0;input_height < t_input[0].size();input_height+=m_stride){
                //get highest value and add to outputs
                std::vector<float> values;
                std::vector<std::pair<size_t,size_t>> indices;
                std::pair<size_t,size_t> index;

                if(input_height+m_height > t_input[0].size()){
                    continue;
                }
                   
                for(size_t x = 0;x < m_width;x++){
                    for(size_t y = 0;y < m_height;y++){
                        float value = t_input[input_width+x][input_height+y];
                        values.push_back(value);

                        std::sort(values.begin(),values.end(),std::greater<float>());
                        if(value == values[0] && values.size() == m_width*m_height){
                            m_input_indices[input_width+x][input_height+y] = 1;
                        }
                    }
                }

                column.push_back(values[0]);
                m_values[pane.size()-1][column.size()-1] = values[0];
            }
            pane.push_back(column);
        }
        
        return 0;
    }

    size_t pool_average(array_2f t_input){

       // Add padding if stride > 1
        if(m_stride > 1){
            //Padding can't be greater than 1 (except if stride >= m_width/m_height)
            size_t padding_width = t_input.size() % m_width;
            size_t padding_height = t_input[0].size() % m_height;

            if(padding_width > 0 || padding_height > 0){

                boost::array<array_2f::index, 2> shape_padding_pool = {{ (long) (padding_width+t_input.size()), (long) (padding_height+t_input[0].size())}};
                t_input.resize(boost::extents[(padding_width+t_input.size())][padding_height+t_input[0].size()]);

                for(size_t i = 0;i < padding_width;i++){
                    for(size_t height = 0;height < t_input[0].size();height++){
                        //duplicate outer numbers
                        t_input[t_input.size()-i-1][height] = t_input[t_input.size()-i-2][height];
                    }
                }
                for(size_t i = 0;i < padding_height;i++){
                    for(size_t width = 0;width < t_input[0].size();width++){
                        t_input[width][t_input[0].size()-i-1] = t_input[width][t_input[0].size()-i-2];
                    }
                }
            }
        }

        // Resize m_values to match output
        size_t output_width = (t_input.size()-m_width)/m_stride+1;
        size_t output_height = (t_input[0].size()-m_height)/m_stride+1;

        m_values.resize(boost::extents[output_width][output_height]);
        m_input_indices.resize(boost::extents[t_input.size()][t_input[0].size()]);

        // this could be removed by editing this function and replaced by boost multi array
        std::vector<std::vector<float>> pane;

        // Do this so pane.size()-1 doesn't throw an error
        pane.resize(1);

        for(size_t input_width = 0;input_width < t_input.size();input_width+=m_stride){
            std::vector<float> column;
            std::vector<std::pair<float, std::pair<size_t,size_t>>> index_column;
            // Stop if we reached the end of the input array
            if(input_width+m_height > t_input.size()){
                continue;
            }
            for(size_t input_height = 0;input_height < t_input[0].size();input_height+=m_stride){
                //get highest value and add to outputs
                std::vector<float> values;
                std::vector<std::pair<float, std::pair<size_t,size_t>>> indices;
                std::pair<float, std::pair<size_t,size_t>> index;

                if(input_height+m_height > t_input[0].size()){
                    continue;
                }
                
                for(size_t x = 0;x < m_width;x++){
                    for(size_t y = 0;y < m_height;y++){
                        float value = t_input[input_width+x][input_height+y];
                        values.push_back(value);

                        // Every input affects average pooling
                        m_input_indices[input_width+x][input_height+y] = 1;
                    }
                }
                float average = 0;
                for(float f:values){
                    average += f;
                }
                values[0] = average/values.size();

                column.push_back(values[0]);
                m_values[pane.size()-1][column.size()-1] = values[0];
            }
            pane.push_back(column);
        }
        
        return 0;
    }

    size_t backwards_propagation(array_2f t_previous_layer){
        if(m_pool_op == MAX){
            backwards_propagation_max(t_previous_layer);
        }
        else if(m_pool_op == AVERAGE){
            backwards_propagation_average(t_previous_layer);
        }
        return 0;
    }

    size_t backwards_propagation_max(array_2f t_previous_layer){
        array_2f output_deltas;
        output_deltas.resize(boost::extents[t_previous_layer.size()][t_previous_layer[0].size()]);
        
        for(size_t i = 0;i < m_input_indices.size();i++){
            for(size_t j = 0;j < m_input_indices[i].size();j++){
                // Check if delta > 0 so we don't make unnecessary changes (delta == 0 => change also 0)
                if(m_input_indices[i][j] == 1 && m_deltas[i][j] > 0){
                    output_deltas[i][j] = m_deltas[i][j];
                }
            }
        }
        m_deltas.resize(boost::extents[output_deltas.size()][output_deltas[0].size()]);
        m_deltas = output_deltas;
        return 0;
    }
    size_t backwards_propagation_average(array_2f t_previous_layer){
        array_2f output_deltas;
        output_deltas.resize(boost::extents[t_previous_layer.size()][t_previous_layer[0].size()]);

        // m_input_indices should be as big as t_previous_layer
        for(size_t i = 0;i < m_input_indices.size();i++){
            for(size_t j = 0;j < m_input_indices[i].size();j++){
                // Check if delta > 0 so we don't make unnecessary changes (delta == 0 => change also 0)
                if(m_input_indices[i][j] == 1){
                    // Average pooling: delta = delta * 1/(k*k)
                    output_deltas[i][j] = (1/static_cast<float>(m_width*m_height)) * m_deltas[(i/2)%m_width][(j/2)%m_height];
                }
            }
        }
        m_deltas.resize(boost::extents[output_deltas.size()][output_deltas[0].size()]);
        m_deltas = output_deltas;
        return 0;
    }

    // TODO change this to hold std::pair<std::pair<size_t,size_t>,std::pair<size_t,size_t>>,
    // First pair indices of pool, second pair indices of prev layer
    array_2t get_input_indices(){
        return m_input_indices;
    }
    size_t get_type(){
        return Layer::POOL;
    }
private:
    // Array, that is 1 if value was pooled and 0 if it doesn't affect error function
    array_2t m_input_indices;
    size_t m_stride;
    size_t m_width;
    size_t m_height;
    size_t m_pool_op;
    array_2f m_input_values;
};

#endif /* POOLLAYER_H */
