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

    ActivationLayer(size_t width,size_t height,size_t depth,size_t functiontype) : Layer(width,height,depth,Layer::types::ACT), typeFunction(functiontype){

    };
    ActivationLayer(const ActivationLayer& orig) : Layer(orig) {};
    virtual ~ActivationLayer(){};

    size_t calculate(std::vector<std::vector<float>> &inputValues);

    enum enumFunctionType{
        SWISH = 1,RELU,SIGMOID
    };
private:
    float act(float x);
};

#endif /* ACTIVATIONLAYER_H */

