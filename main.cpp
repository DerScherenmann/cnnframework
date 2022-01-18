/*
 * File:   main.cpp
 * Author: DerScherenmann
 *
 * Created on 5 September 2020, 17:10
 */

#define INPUT_DIMENSIONS 3

#include "convnetwork.h"

//read https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
int main(int argc, char** argv) {
    
    size_t num_repeats = 1;
    std::vector<size_t> num_filters = {10};
    std::vector<size_t> num_filter_size = {5};
    size_t num_pool_size = 2;
    size_t num_zero_padding = 1;
    
    Convolutional conv = Convolutional({30,10},{1},num_filters,num_filter_size,num_pool_size,num_zero_padding);
    
    boost::array<Convolutional::array_3f::index, 3> shape_array_training_images = {{ 3, 32, 32 }};
    Convolutional::array_3f image_array(shape_array_training_images);
    
    auto dataset = cifar::read_dataset<std::vector,std::vector,uint8_t,size_t>(1,1);
    
    for(size_t channel = 0;channel < 3;channel++){
        for(size_t i = 0;i < 32;i++){
            for(size_t j = 0;j < 32;j++){
                image_array[channel][i][j] = (float) dataset.training_images[0][(i * 32 * 32) + (j * 32) + channel] / 255;
            }
        }
    }
    
    std::vector<Convolutional::struct_training_data> training_data;
    Convolutional::struct_training_data single_data;
    single_data.image_data.resize(boost::extents[3][32][32]);
    single_data.image_data = image_array;

    std::vector<float> labels;
    for(size_t i = 0;i < 10;i++){
        if(i == dataset.training_labels[0]){
            labels.push_back(1);
        }else{
            labels.push_back(0);
        }
    }
    single_data.corrrect_outputs = labels;

    training_data.push_back(single_data);
    
    //sample input
    boost::array<Convolutional::array_3f::index, 3> shape_array_images = {{3,7,7}};
    Convolutional::array_3f inputs(shape_array_images);
    
    //simulate rgb channel
    for(size_t i = 0;i < 3;i++){
        for(size_t j = 0;j < 5;j++){
            std::vector<float> column;
            for(size_t k = 0;k < 5;k++){
                inputs[i][j][k] = (float) k + i + j;
            }
        }
    }
    std::cout << std::endl;
    Convolutional::struct_training_data test;
    test.image_data.resize(boost::extents[3][7][7]);
    test.image_data = inputs;
    test.corrrect_outputs = {1,1,1,1,1,1,1,1,1,0};

    std::vector<Convolutional::struct_training_data> dummy_training;
    dummy_training.push_back(test);

    //train
    // TODO delta and gradient calculation not working
    conv.train(dummy_training,Convolutional::SIGMOID,0.02,0.01,1,1,1);
    while(getchar() != 0){
        conv.train(dummy_training,Convolutional::SIGMOID,0.02,0.01,1,1,1);
    }
    

    return 0;
}
