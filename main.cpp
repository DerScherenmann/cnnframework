/*
 * File:   main.cpp
 * Author: DerScherenmann
 *
 * Created on 5 September 2020, 17:10
 */

#define INPUT_DIMENSIONS 3

#include "src/convnetwork.h"

//read https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
int main(int argc, char** argv) {
    
    /*
     * 
     * Read cifar dataset
     * 
     */
    
    boost::array<Convolutional::array_3f::index, 3> shape_array_training_images = {{ 3, 32, 32 }};
    Convolutional::array_3f image_array(shape_array_training_images);
    
    // Just read one for now
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
    
    /*
     * TODO move this to a test function
     * 
     * Create sample input with pattern to recognize
     * 
     * Generate simple cross consisting of 1's in a background of 0's
     */
    std::vector<Convolutional::struct_training_data> dummy_training;
    boost::array<Convolutional::array_3f::index, 3> shape_array_images = {{3,20,20}};
    Convolutional::array_3f inputs(shape_array_images);
    
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist20(3,17);
    std::uniform_int_distribution<std::mt19937::result_type> dist100(0,99);
    
    for(size_t amount_dummy_training_images = 0;amount_dummy_training_images < 100;amount_dummy_training_images++){
        
        //simulate rgb channel
        for(size_t i = 0;i < 3;i++){
            for(size_t j = 0;j < 20;j++){
                for(size_t k = 0;k < 20;k++){
                    inputs[i][j][k] = 0;
                }
            }
        }
        
        size_t cross_positionx = dist20(rng);
        size_t cross_positiony = dist20(rng);
        
        for(size_t i = 0;i < 3;i++){
            inputs[i][cross_positionx][cross_positiony] = 1;
            inputs[i][cross_positionx+1][cross_positiony] = 1;
            inputs[i][cross_positionx+2][cross_positiony] = 1;
            inputs[i][cross_positionx-1][cross_positiony] = 1;
            inputs[i][cross_positionx-2][cross_positiony] = 1;
            inputs[i][cross_positionx][cross_positiony+1] = 1;
            inputs[i][cross_positionx][cross_positiony+2] = 1;
            inputs[i][cross_positionx][cross_positiony-1] = 1;
            inputs[i][cross_positionx][cross_positiony-2] = 1;
        }
        
        Convolutional::struct_training_data test;
        test.image_data.resize(boost::extents[3][20][20]);
        test.image_data = inputs;
        test.corrrect_outputs = {0,0,0,0};
        
        // Let the Network recognize the corresponding quadrant
        if(cross_positionx > 10){
            if(cross_positiony > 10){
                test.corrrect_outputs[3] = 1;
            }else{
                test.corrrect_outputs[0] = 1;
            }
        }else{
            if(cross_positiony > 10){
                test.corrrect_outputs[2] = 1;
            }else{
                test.corrrect_outputs[1] = 1;
            }
        }
        
        dummy_training.push_back(test);
    }
    
    
    
    /*
     * Create out network
     */
    size_t num_repeats = 1;
    std::vector<size_t> num_filters = {5,5};
    std::vector<size_t> num_filter_size = {5,5};
    size_t num_pool_size = 2;
    size_t num_zero_padding = 1;
    
    Convolutional conv = Convolutional({30,10,4},{Layer::types::CONV,Layer::types::CONV},num_filters,num_filter_size,num_pool_size,num_zero_padding);
    
    //train
    // TODO delta and gradient calculation not working
    conv.train(dummy_training,Convolutional::SIGMOID,0.05,0.1,100,1,1);
    
    while(getchar() != 0){
        size_t test_number = dist100(rng);
        std::pair<std::vector<float>,std::vector<float>> outputs = conv.feed_forward(dummy_training[test_number]);
        std::cout << "Testing: " << test_number << std::endl;
        for(size_t i = 0;i < dummy_training[test_number].corrrect_outputs.size();i++){
            std::cout << dummy_training[test_number].corrrect_outputs[i] << " ";
        }
        std::cout << std::endl << "Network outputs: " << std::endl;
        for(size_t i = 0;i < outputs.first.size();i++){
            std::cout << outputs.first[i] << " ";
        }
    }

    return 0;
}
