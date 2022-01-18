/*
 * File:   main.cpp
 * Author: broesel233
 *
 * Created on 5 September 2020, 17:10
 */

#define INPUT_DIMENSIONS 3

#include "convnetwork.h"
#include <iostream>

int main(int argc, char** argv) {
    
    size_t num_repeats = 2;
    size_t num_filters = 5;
    size_t num_filter_size = 4;
    size_t num_pool_size = 2;
    size_t num_zero_padding = 1;

    // WARNING knn library still bugged, can only be trained once TODO fix
    Convolutional conv = Convolutional({28,28,3},{10,1},{1,2},num_repeats,num_filters,num_filter_size,num_pool_size,num_zero_padding);
    
    conv.run_tests();

    std::cout << "Exited normally" << std::endl;

    getchar();

    return 0;
}
