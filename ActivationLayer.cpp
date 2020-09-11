/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   ActivationLayer.cpp
 * Author: broesel233
 *
 * Created on 6 September 2020, 14:53
 */

#include "ActivationLayer.h"
size_t ActivationLayer::calculate(std::vector<std::vector<float>> &inputValues){

    for(size_t i = 0;i < m_width;i++){
        for(size_t j = 0;j < m_height;j++){
            m_values[i][j] = act(inputValues[i][j]);
        }
    }
    return 0;
}

float ActivationLayer::act(float x){

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
