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

#include "layer.h"

class ActivationLayer : public Layer {
public:
    size_t typeFunction;

    ActivationLayer(){};
    ActivationLayer(size_t width,size_t height,size_t depth,size_t functiontype) : Layer(width,height,depth,Layer::types::ACT), typeFunction(functiontype){

    };
    //ActivationLayer(const ActivationLayer& orig) : Layer(orig) {};
    virtual ~ActivationLayer(){};

    size_t calculate(std::vector<std::vector<float>> inputValues){

        for(size_t i = 0;i < m_width;i++){
            for(size_t j = 0;j < m_height;j++){
                m_values[i][j] = act(inputValues[i][j]);
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
    float act(float x){

        switch(typeFunction){
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

