/*
 * File:   ConvolutionLayer.h
 * Author: DerScherenmann
 *
 * Created on 8 September 2020, 19:51
 */

#pragma once

#include <iostream>

#include "Layer.h"
#include "kernel.h"

using namespace layer;

class ConvolutionLayer : public Layer {
public:

    /**
     *
     */
    ConvolutionLayer(array_2f t_values,size_t t_zero_padding) : Layer(t_values,Layer::types::CONV) {
        m_zero_padding = t_zero_padding;
    };

    //ConvolutionLayer(const ConvolutionLayer& orig) : Layer(orig) {};
    //virtual ~ConvolutionLayer(){};

    /**
     * Make zero padding to lose less data
     */
    size_t make_padding() override {
        //this is not needed here
        m_width = m_values.size()+2*m_zero_padding;
        m_height = m_values[0].size()+2*m_zero_padding;
        
        boost::array<array_2f::index, 2> shape_padding = {{ (long) m_width, (long) m_height }};
        array_2f array_padding(shape_padding);
        
        if(m_zero_padding != 0){
            for(size_t num_padding = 0;num_padding < m_zero_padding;num_padding++){
                for(size_t i = 0;i < m_width;i++){
                    for(size_t j = 0;j < m_height;j++){
                        //padding
                        if(i < m_zero_padding || i+1 > m_width-m_zero_padding || j < m_zero_padding || j+1 > m_height-m_zero_padding){
                            array_padding[i][j] = 0;
                        }else{
                            //add m_values to array
                            array_padding[i][j] = m_values[i-m_zero_padding][j-m_zero_padding];
                        }
                    }
                }
            }
        }
        m_values.resize(boost::extents[m_width][m_height]);
        m_values = array_padding;
        m_has_padding = true;

        return 0;
    }

    size_t backwards_propagation(array_2f t_current_layer) override {
        
        array_2f deltas;
        // TODO operator overloading in Filter class
        deltas.resize(boost::extents[m_filter->get_output_width(m_deltas.size()+2)][m_filter->get_output_height(m_deltas.size()+2)]);
        
        for(size_t i = 0;i < m_filter->size();i++){
            array_2f deltas_kernel = m_filter->get_kernels()[i]->calculate_deltas(m_deltas);
            for(size_t j = 0;j < deltas_kernel.size();j++){
                for(size_t k = 0;k < deltas_kernel[0].size();k++){
                    deltas[j][k] += deltas_kernel[j][k];
                }
            }
        }
        for(size_t i = 0;i < t_current_layer.size();i++){
            for(size_t j = 0;j < t_current_layer[0].size();j++){
                t_current_layer[i][j] *= m_filter->get_kernels()[0]->actPrime(t_current_layer[i][j]);
            }
        }

        m_deltas.resize(boost::extents[deltas.size()][deltas[0].size()]);

        // std::cout << deltas[0].size() << std::endl;
        // std::cout << t_current_layer[0].size() << std::endl;
        // std::cout << m_deltas.size() << std::endl;
        
        return 0;
    }

    bool has_padding() override {
        return m_has_padding;
    }
    size_t get_type () {
        return Layer::CONV;
    }
    size_t set_filter(fshared_ptr_t t_filter) override {
        m_filter = t_filter;
        return 0;
    }
    fshared_ptr_t get_filter() override {
        return m_filter;
    }
private:
    // Holds Filter of following convolutional layer
    fshared_ptr_t m_filter;
    bool m_has_padding;

    typedef boost::multi_array<float, 3> array_3f;

    array_2f flip(array_2f t_input_values){
        array_2f temp;
        temp.resize(boost::extents[t_input_values.size()][t_input_values[0].size()]);
        //flip input
        for(size_t i = 0;i < t_input_values.size();i++){
            for(size_t j = 0;j < t_input_values[0].size();j++){
                float value = t_input_values[i][j];
                temp[t_input_values.size()-1-i][t_input_values[0].size()-1-j] = value;
            }
        }
        return temp;
    }
};
