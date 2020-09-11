/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   ConvolutionLayer.h
 * Author: broesel233
 *
 * Created on 8 September 2020, 19:51
 */

#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H

#include <iostream>

#include "layer.h"
#include "filter.h"

class ConvolutionLayer : public Layer {
public:
    
    /**
     * 
     */
    ConvolutionLayer(size_t t_height,size_t t_width,size_t t_depth,size_t t_zero_padding,size_t t_stride,size_t t_num_filters, size_t t_filter_sizes) :
                        Layer(t_height,t_width,t_depth,Layer::types::CONV), m_num_filters(t_num_filters), m_filter_sizes(t_filter_sizes), m_stride(t_stride) {

        m_zero_padding = t_zero_padding;
        m_filters.reserve(m_num_filters);
        for(size_t i = 0;i < m_num_filters;i++){
            Filter k = Filter(m_filter_sizes,m_stride);
            m_filters.push_back(k);
        }
    };
    
    std::vector<Filter> get_filters(){
        return m_filters;
    }
    
    ConvolutionLayer(const ConvolutionLayer& orig) : Layer(orig) {};
    virtual ~ConvolutionLayer(){};

    /**
     * Forward input volume through convolutional layer
     * @param t_input_volume (depth,width,height)
     * @return
     */
    size_t make_padding(){

        m_width = (m_values.size()-m_filter_sizes+2*m_zero_padding)/m_stride +2;
        m_height = m_width;
        m_depth = m_num_filters;

        if(m_zero_padding != 0){
            for(size_t i = 0;i < m_zero_padding;i++){
                for(size_t width = 0;width < m_values.size();width++){
                    m_values[width].insert(m_values[width].begin(),0);
                    m_values[width].push_back(0);
                }
                std::vector<float> padding_column;
                padding_column.reserve(m_height);
                for(size_t height = 0;height < m_height;height++){
                    padding_column.push_back(0);
                }
                m_values.insert(m_values.begin(),padding_column);
                m_values.push_back(padding_column);
                padding_column.clear();
            }
        }

        return 0;
    }
    
private:
    size_t m_num_filters;
    size_t m_filter_sizes;
    size_t m_stride;
    std::vector<Filter> m_filters;
};

#endif /* CONVOLUTIONLAYER_H */
