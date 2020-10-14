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

#include "knn/mathhelper.h"
//#include "filter.h"

class Filter;

class Layer
{
public:
    Layer() {};
    Layer(size_t t_width,size_t t_height,size_t t_depth,size_t t_type) : m_width(t_width), m_height(t_height),m_depth(t_depth), m_type(t_type)
    {
        m_values.resize(m_width);
        for(size_t i = 0; i<m_width; i++)
        {
            m_values[i].resize(m_height);
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
    virtual size_t get_depth()
    {
        return m_depth;
    }
    virtual size_t get_padding()
    {
        return m_zero_padding;
    }
    virtual size_t get_type()
    {
        return m_type;
    }
    virtual std::vector<std::vector<float>> get_values()
    {
        return m_values;
    }
//    virtual std::vector<std::vector<float>> get_old_changes()
//    {
//        return m_old_changes;
//    }
    virtual size_t set_values(std::vector<std::vector<float>> &t_input_values)
    {
        m_values = t_input_values;
        return 0;
    }

    //declare some virtual functions for base classes to avoid dynamic casting
    virtual std::vector<float> get_net_output() {};
    virtual std::vector<float> train(std::pair<std::vector<float>,std::vector<float>> &t_training_data,float t_learning_rate,float t_momentum){}
    virtual size_t forward(){};
    virtual std::vector<std::vector<std::pair<float, std::pair<size_t,size_t>>>> get_org_index(){}
    virtual std::vector<Filter> get_filters(){};
    virtual size_t make_padding(){};
    virtual size_t pool(std::vector<std::vector<float>>& t_input){};
    virtual size_t calculate(std::vector<std::vector<float>> &inputValues){};
    virtual size_t get_in_size(){};

protected:
    //outer vector width inner height
    std::vector<std::vector<float>> m_values;
//    //weights outer width inner height
//    std::vector<std::vector<float>> m_weights;
//    //old_changes outer width inner height
//    std::vector<std::vector<float>> m_old_changes;
    size_t m_width;
    size_t m_height;
    size_t m_depth;
    size_t m_zero_padding = 0;
    Math math;
private:
    size_t m_type;
};
#endif
