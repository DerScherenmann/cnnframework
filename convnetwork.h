/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   convnetwork.h
 * Author: broesel233
 *
 * Created on 5 September 2020, 21:02
 */

#ifndef CONVNETWORK_H
#define CONVNETWORK_H

#include <vector>
#include <string>
#include <algorithm>

#include "filter.h"

class Convultional{
public:
    Convultional(std::vector<size_t> input_matrix_size,size_t num_layers,size_t filters,size_t pools) : num_layers(num_layers), filters(filters), pools(pools){};
    size_t train(std::vector<std::pair<std::vector<float>, std::vector<float>>> &training_data,float learning_rate,float momentum, size_t epochs);
    size_t test();
private:
    size_t num_layers;
    size_t filters;
    size_t pools;
    float learning_rate;
    float momentum;
    size_t epochs;

    std::vector<std::vector<float>> kernels;
    std::vector<std::vector<Layer>> layers;
};

#endif /* CONVNETWORK_H */

