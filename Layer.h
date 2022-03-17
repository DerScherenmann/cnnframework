/*
 * File:   Layer.h
 * Author: broesel233
 *
 * Created on 6 September 2020, 02:22
 */
#ifndef LAYER_H
#define LAYER_H

class Filter;

#include "lib/mathhelper.h"
#include <boost/multi_array.hpp>
#include <boost/scoped_ptr.hpp>

namespace layer{
    
//define multi array type
typedef boost::multi_array<float, 2> array_2f;
typedef boost::multi_array_types::index_range range_t;
typedef boost::multi_array<size_t, 2> array_2t;

typedef std::shared_ptr<Filter> fshared_ptr_t;
typedef std::unique_ptr<Filter> funique_ptr_t;
typedef boost::scoped_ptr<Filter> fscope_ptr_t;

class Layer
{
public:
    //index_gen for array type
    array_2f::index_gen indices;
    
    Layer(array_2f t_values,size_t t_type) : m_values(t_values), m_width(t_values.size()),m_height(t_values[0].size()) ,m_type(t_type)
    {
        for(size_t i = 0;i < m_width;i++)
        {
            for(size_t j = 0;j < m_height;j++)
            {
                m_values[i][j] = math.rng();
            }
        }
    };
    //Layer(const Layer& orig){};
    virtual ~Layer() {};
    /*
     * Makes layer types acessible
     */
    enum types
    {
        INPUT = 0,CONV,POOL,ACT,CONNECTED,OUTPUT
    };
    /*
     * Determines wich function should be used inside Convolutional and activation layer
     */
    enum functiontype{
        SWISH = 0,SIGMOID,RELU
    };
    /*
     * Determines pooling operation (Pooling layers only)
     */
    enum operation {
        MAX = 0,AVERAGE
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
    virtual array_2f get_values()
    {
        return m_values;
    }
    virtual size_t set_values(array_2f t_input_values)
    {
        m_values.resize(boost::extents[t_input_values.size()][t_input_values[0].size()]);
        m_values = t_input_values;
        return 0;
    }
    virtual size_t set_deltas(array_2f &t_deltas){
        m_deltas.resize(boost::extents[t_deltas.size()][t_deltas[0].size()]);
        m_deltas = t_deltas;
        return 0;
    }
    virtual array_2f get_deltas(){
        return m_deltas;
    }
    virtual size_t set_filter(fshared_ptr_t t_filter) = 0;
    virtual fshared_ptr_t get_filter() = 0;

    //declare some virtual functions for base classes to avoid dynamic casting
    virtual std::vector<float> get_net_output() = 0;
    virtual std::vector<float> train(std::vector<float> &t_training_data,float t_learning_rate,float t_momentum) = 0;
    virtual size_t forward() = 0;
    virtual array_2t get_input_indices() = 0;
    virtual size_t make_padding() = 0;
    virtual bool has_padding() = 0;
    virtual size_t pool(array_2f t_input) = 0;
    virtual size_t calculate(array_2f t_input) = 0;
    virtual size_t get_in_size() = 0;
    virtual size_t backwards_propagation(array_2f t_layer_values) = 0;

protected:
    array_2f m_values;
    //store deltas
    array_2f m_deltas;
    size_t m_width = 0;
    size_t m_height = 0;
    size_t m_zero_padding = 0;
    
    Math math;
private:
    size_t m_type = 0;
};
}
#endif
