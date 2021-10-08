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

#include "Layer.h"
#include "filter.h"

using namespace layer;

class ConvolutionLayer : public Layer {
public:

    /**
     *
     */
    ConvolutionLayer(){};
    ConvolutionLayer(array_2d_t t_values,size_t t_zero_padding,size_t t_stride,size_t t_num_filters, size_t t_filter_sizes,size_t t_function_type) :
                        Layer(t_values,Layer::types::CONV), m_num_filters(t_num_filters), m_filter_sizes(t_filter_sizes), m_stride(t_stride) {

        m_zero_padding = t_zero_padding;
        m_filters.reserve(m_num_filters);
        for(size_t i = 0;i < m_num_filters;i++){
            Filter k = Filter(m_filter_sizes,m_stride,t_function_type);
            m_filters.push_back(k);
        }

//        //fucking retarded weights are in filter
//        //init random weigths
//        for(size_t i = 0;i < t_width;i++){
//            std::vector<float> column;
//            for(size_t j = 0;j < t_height;j++){
//                column.push_back(math.rng());
//            }
//            m_weights.push_back(column);
//        }
//        //zero old changes so first time is always 0
//        for(size_t i = 0;i < t_width;i++){
//            std::vector<float> column;
//            for(size_t j = 0;j < t_height;j++){
//                column.push_back(0);
//            }
//            old_changes.push_back(column);
//        }
    };

    std::vector<Filter> get_filters(){
        return m_filters;
    }

    //ConvolutionLayer(const ConvolutionLayer& orig) : Layer(orig) {};
    //virtual ~ConvolutionLayer(){};

    /**
     * Make zero padding to lose less data
     */
    size_t make_padding(){
        //this is not needed here
        m_width = m_values.size()+2*m_zero_padding;
        m_height = m_values[0].size()+2*m_zero_padding;
        
        boost::array<array_2d_t::index, 2> shape_padding = {{ (long) m_width, (long) m_height }};
        array_2d_t array_padding(shape_padding);
        
        if(m_zero_padding != 0){
            for(size_t num_padding = 0;num_padding < m_zero_padding;num_padding++){
                for(size_t i = 0;i < array_padding.size();i++){
                    for(size_t j = 0;j < array_padding[0].size();j++){
                        //padding
                        if(i < m_zero_padding || i+1 > array_padding.size()-m_zero_padding || j < m_zero_padding || j+1 > array_padding[0].size()-m_zero_padding){
                            array_padding[i][j] = 0;
                        }else{
                            //add m_values to array
                            array_padding[i][j] = m_values[i-m_zero_padding][j-m_zero_padding];
                        }
                    }
                }
            }
        }
        m_values.resize(boost::extents[array_padding.size()][array_padding[0].size()]);
        m_values = array_padding;
        
        return 0;
    }
    
    size_t get_type(){
        return Layer::CONV;
    }
private:
    size_t m_num_filters;
    size_t m_filter_sizes;
    size_t m_stride;
    std::vector<Filter> m_filters;
};

#endif /* CONVOLUTIONLAYER_H */
