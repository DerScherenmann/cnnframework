/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <fstream>

#include "convnetwork.h"
#include "PoolLayer.h"
#include "ConvolutionLayer.h"
#include "ConnectedLayer.h"
#include "filter.h"

typedef unsigned char uchar;
std::vector<std::vector<float>> images;
uchar* labels;
std::vector<std::vector<float>> testimages;
uchar* testlabels;
size_t readStuff(size_t number_of_images,size_t number_of_labels);

size_t Convultional::train(std::vector<std::pair<std::vector<float>, std::vector<float>>> &training_data,float learning_rate,float momentum, size_t epochs){

    for(size_t epoch = 0;epoch < epochs;epoch++){

    }

    return 0;
}

size_t Convultional::test(){

    std::vector<std::vector<ConvolutionLayer>> architecture;
    std::vector<PoolLayer> last_layers;

    /*
    read_stuff(1,1);
    std::vector<std::vector<float>> values;
    values.resize(28);
    for(size_t i = 0;i < 28;i++){
        values[i].resize(28);
       for(size_t j = 0;j < 28;j++){
           values[i][j] = images[0][i*j];
        }
    }
    */
    /*
     * Generate sample input
     */
    size_t input_size = 6;
    size_t depth = 3;
    //simulate rgb channel
    std::vector<std::vector<std::vector<float>>> inputs;
    inputs.resize(depth);
    for(size_t i = 0;i < depth;i++){
        for(size_t j = 0;j<input_size;j++){
            std::vector<float> column;
            for(size_t k = 0;k<input_size;k++){
                column.push_back((float) k + i + j);
                std::cout << " " << k + i + j;
            }
            inputs[i].push_back(column);
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    /*
     * Test for Convolutional Layer
     */
    std::cout << "Test Convolutional Layer / zero padding: " << std::endl << std::endl;

    std::vector<ConvolutionLayer> conv_layers;
    for(size_t i = 0;i < inputs.size();i++){

        ConvolutionLayer conv = ConvolutionLayer(input_size,input_size,1,1,1,5,2);
        conv.set_values(inputs[i]);
        conv.make_padding();
        conv_layers.push_back(conv);
        //print to check
        for(size_t width = 0;width < conv.get_values().size();width++){
            for(size_t height = 0;height < conv.get_values().size();height++){
                std::cout << conv.get_values()[width][height] << " ";
            }
            std::cout << std::endl;
        }
        //add filters

        std::cout << std::endl;
        std::cout << std::endl;
    }
    architecture.push_back(conv_layers);

    std::cout << std::endl << std::endl;

    /*
     * Test for filters
     */
    std::cout << "Test Filters: " << std::endl << std::endl;

    for(size_t i = 0; i < architecture.size();i++){

        for(size_t j = 0;j < architecture[i].size();j++) {
            std::vector<Filter> filters = architecture[i][j].get_filters();
            std::cout << "Filters: " << filters.size() << std::endl;
            if(filters.size() == 0){
                break;
            }
            for(Filter filter:filters){
                Layer output = filter.calculate_output(architecture[i][j]);
                for(size_t width = 0;width < output.get_values().size();width++){
                    for(size_t height = 0;height < output.get_values()[0].size();height++){
                        std::cout << output.get_values()[width][height] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl << std::endl;
    }

    std::cout << std::endl << std::endl;

    /*
     * Test for PoolLayer
     */
    std::cout << "Test Pool Layer: " << std::endl;

    for(size_t i = 0; i < architecture.size();i++){

        for(size_t j = 0;j < architecture[i].size();j++) {
            std::vector<Filter> filters = architecture[i][j].get_filters();
            if(filters.size() == 0){
                break;
            }

            for(Filter filter:filters){
                Layer output = filter.calculate_output(architecture[i][j]);

                PoolLayer pool = PoolLayer(2,2,1,2);
                pool.pool(output.get_values());

                std::cout << std::endl;
                for(size_t j = 0;j<pool.get_values().size();j++){
                    for(size_t k = 0;k < pool.get_values()[0].size();k++){
                        std::cout << " " << pool.get_values()[j][k];
                    }
                    std::cout << std::endl;
                }

                std::cout << std::endl;
                for(size_t j = 0;j < pool.get_org_index().size();j++){
                    for(size_t k = 0;k < pool.get_org_index()[0].size();k++){
                        auto [x,y] = pool.get_org_index()[j][k].second;
                        std::cout << pool.get_org_index()[j][k].first << ": " << x << "," << y << " ";
                    }
                    std::cout << std::endl;
                }

                last_layers.push_back(pool);
            }
        }
        std::cout << std::endl << std::endl;
    }

    std::cout << std::endl << std::endl;

    /*
     * Test for connected layer
     */
    std::cout << "Test connected layer: " << std::endl << std::endl;

    std::vector<float> connected_inputs;

    for(PoolLayer pool:last_layers){

        //make all pool outputs to a 1d vector
        for(size_t i = 0;i<pool.get_values().size();i++){
            for(size_t j = 0;j < pool.get_values()[0].size();j++){
                connected_inputs.push_back(pool.get_values()[i][j]);
            }
        }
    }
    std::vector<std::vector<float>> dim;
    dim.push_back(connected_inputs);

    std::vector<std::pair<int,int>> sizes;
    ConnectedLayer connected = ConnectedLayer(ConnectedLayer::functiontype::SWISH,sizes);
    connected.set_values(dim);

    std::cout << std::endl << std::endl;

    return 0;
}

size_t read_stuff(size_t number_of_images,size_t number_of_labels) {

    Math math;

    /*
    * Open training images
    */
    std::ifstream file("train-images.idx3-ubyte", std::ios::binary);
    if (file.is_open())
    {
        std::cout << "Opening images..." << std::endl;
        size_t magic_number = 0;
        size_t number_of_images = 0;
        size_t n_rows = 0;
        size_t n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = math.reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = math.reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = math.reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = math.reverseInt(n_cols);
        //std::thread imageThread = std::thread([number_of_images,n_rows,n_cols,file,images] {
        std::vector<float> test;
        images.reserve(number_of_images);
        for (size_t i = 0; i < number_of_images; ++i)
        {
            test.reserve(784);
            for (size_t r = 0; r < n_rows; ++r)
            {
                for (size_t c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    test.push_back(temp);
                }
            }
            images.push_back(test);
            test.clear();
        }
        //});
        file.close();
    }
    /*
    * Open training labels
    */
    file.open("train-labels.idx1-ubyte", std::ios::binary);
    if (file.is_open()) {

        std::cout << "Opening labels..." << std::endl;
        size_t magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = math.reverseInt(magic_number);

        if (magic_number != 2049) return 1;

        file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = math.reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for (size_t i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        labels = _dataset;
        file.close();
    }
    /*
    * Open Test Images
    */
    file.open("t10k-images.idx3-ubyte", std::ios::binary);
    if (file.is_open())
    {
        std::cout << "Opening testing images..." << std::endl;
        size_t magic_number = 0;
        size_t number_of_images = 0;
        size_t n_rows = 0;
        size_t n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = math.reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = math.reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = math.reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = math.reverseInt(n_cols);
        testimages.reserve(10000);
        //std::thread imageThread = std::thread([number_of_images,n_rows,n_cols,file,images] {
        std::vector<float> test;
        for (size_t i = 0; i < number_of_images; ++i)
        {
            for (size_t r = 0; r < n_rows; ++r)
            {
                for (size_t c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    test.push_back(temp);
                }
            }
            testimages.push_back(test);
            test.clear();
        }
        //});
        file.close();
    }
    /*
    * Open Test Labels
    */
    file.open("t10k-labels.idx1-ubyte", std::ios::binary);
    if (file.is_open()) {

        std::cout << "Opening test labels..." << std::endl;
        size_t magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = math.reverseInt(magic_number);

        if (magic_number != 2049) return 1;

        file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = math.reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for (size_t i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        testlabels = _dataset;
    }

    return 0;
}

