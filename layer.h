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

class Layer {
public:
    Layer(size_t t_width,size_t t_height,size_t t_depth,size_t t_type) : m_width(t_width), m_height(t_height),m_depth(t_depth), m_type(t_type) {
        m_values.resize(m_width);
        for(size_t i = 0;i<m_width;i++){
            m_values[i].resize(m_height);
            for(size_t j = 0;j < m_height;j++){
                m_values[i][j] = math.rng();
            }
        }
    };
    //Layer(const Layer& orig){};
    virtual ~Layer(){};

    enum types{
        INPUT = 0,CONV,POOL,ACT,CONNECTED,OUTPUT
    };

    size_t get_width(){
        return m_width;
    }
    size_t get_height(){
        return m_height;
    }
    size_t get_depth(){
        return m_depth;
    }
    std::vector<std::vector<float>> get_values(){
        return m_values;
    }
    size_t set_values(std::vector<std::vector<float>> t_input_values){
        m_values = t_input_values;
        return 0;
    }
    size_t get_padding(){
        return m_zero_padding;
    }
protected:
    //outer vector width inner height
    std::vector<std::vector<float>> m_values;
    size_t m_width;
    size_t m_height;
    size_t m_depth;
    size_t m_zero_padding = 0;
    Math math;
private:
    size_t m_type;
};

#endif /* LAYER_H */

