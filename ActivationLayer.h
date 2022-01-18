/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   ActivationLayer.h
 * Author: broesel233
 *
 * Created on 6 September 2020, 14:53
 */

#ifndef ACTIVATIONLAYER_H
#define ACTIVATIONLAYER_H

#include "Layer.h"

using namespace layer;

class ActivationLayer : public Layer {
public:

    ActivationLayer(array_2f t_values,size_t t_function_t) : Layer(t_values,Layer::ACT), m_function_t(t_function_t){

    };
    //ActivationLayer(const ActivationLayer& orig) : Layer(orig) {};
    virtual ~ActivationLayer(){};

    size_t calculate(array_2f t_input){

        m_width = t_input.size();
        m_height = t_input[0].size();

        m_values.resize(boost::extents[t_input.size()][t_input[0].size()]);
        for(size_t i = 0;i < t_input.size();i++){
            for(size_t j = 0;j < t_input[0].size();j++){
                m_values[i][j] = act(t_input[i][j]);
            }
        }

        return 0;
    }

    enum function_type{
        SWISH = 0,RELU,SIGMOID
    };
    size_t get_type(){
        return Layer::ACT;
    }
private:
    size_t m_function_t;

    float act(float x){

        switch(m_function_t){
            case SWISH:
                x = math.swish(x);
                break;
            case SIGMOID:
                x = math.sigmoid(x);
            case RELU:
                break;
        }
        return x;
    }
};

#endif /* ACTIVATIONLAYER_H */

