/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   Layer.h
 * Author: broesel233
 *
 * Created on 6 September 2020, 02:22
 */
#ifndef LAYER_H
#define LAYER_H

#include "lib/mathhelper.h"
#include <boost/multi_array.hpp>

class Filter;

namespace layer{
    
class Layer
{
public:
    
    //define multi array type
    typedef boost::multi_array<float, 2> array_2d_t;
    typedef boost::multi_array_types::index_range range_t;
    //index_gen for array type
    array_2d_t::index_gen indices;
    
    Layer() {};
    Layer(array_2d_t t_values,size_t t_type) : m_values(t_values), m_width(t_values.size()),m_height(t_values[0].size()) ,m_type(t_type)
    {
        for(size_t i = 0; i<m_width; i++)
        {
            for(size_t j = 0; j < m_height; j++)
            {
                m_values[i][j] = math.rng();
            }
        }
    };
    //Layer(const Layer& orig){};
    virtual ~Layer() {};

    enum types
    {
        INPUT = 0,CONV,POOL,ACT,CONNECTED,OUTPUT
    };
    enum functiontype{
        SWISH = 0,SIGMOID,RELU
    };

    virtual size_t get_width()
    {
        return m_width;
    }
    virtual size_t get_height()
    {
        return m_height;
    }
    virtual size_t get_padding()
    {
        return m_zero_padding;
    }
    virtual size_t get_type()
    {
        return m_type;
    }
    virtual array_2d_t get_values()
    {
        return m_values;
    }
    virtual size_t set_values(array_2d_t t_input_values)
    {
        m_values = t_input_values;
        return 0;
    }
    virtual size_t set_deltas(std::vector<std::vector<float>> &t_deltas){
        m_deltas = t_deltas;
        return 0;
    }
    virtual std::vector<std::vector<float>> get_deltas(){
        return m_deltas;
    }

    //declare some virtual functions for base classes to avoid dynamic casting
    virtual std::vector<float> get_net_output(){};
    virtual std::vector<float> train(std::pair<std::vector<float>,std::vector<float>> &t_training_data,float t_learning_rate,float t_momentum){};
    virtual size_t forward(){};
    virtual std::vector<std::vector<std::pair<float, std::pair<size_t,size_t>>>> get_org_index(){}
    virtual std::vector<Filter> get_filters(){};
    virtual size_t make_padding(){};
    virtual size_t pool(std::vector<std::vector<float>> &t_input){};
    virtual size_t calculate(std::vector<std::vector<float>> &inputValues){};
    virtual size_t get_in_size(){};

protected:
    //workaround because array_view has no constructor; essentially the subarray; is input and output during forward pass;
    array_2d_t m_values;
    //store deltas
    std::vector<std::vector<float>> m_deltas;
    size_t m_width = 0;
    size_t m_height = 0;
    size_t m_zero_padding = 0;
    Math math;
private:
    size_t m_type;
};
}
#endif
