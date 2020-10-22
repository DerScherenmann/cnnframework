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
#include <thread>

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
size_t read_stuff(size_t number_of_images_open,size_t number_of_labels_open);

size_t Convolutional::train(std::vector<std::pair<std::vector<std::vector<float>>, std::vector<float>>> &t_training_data,size_t t_funcion_type,float t_learning_rate,float t_momentum, size_t t_epochs, size_t t_stride_filters,size_t t_stride_pools) {
    m_stride_pool = t_stride_pools;
    m_stride_filters = t_stride_filters;
    m_epochs = t_epochs;
    m_learning_rate = t_learning_rate;
    m_momentum = t_momentum;
    m_function_type = t_funcion_type;

    //initialize layers
    feed_forward_first(t_training_data[0]);

    std::vector<std::vector<float>> all_outputs;

    std::cout << "Starting training..." << std::endl;

    //train for epoch amount
    for(size_t epoch = 0;epoch < t_epochs;epoch++){
        std::cout << "Epoch: " << epoch << std::endl;
        //go through batch
        for(std::pair<std::vector<std::vector<float>>, std::vector<float>> &data:t_training_data){
            //feed forward
            //TODO other feed_forward function
            std::vector<float> outputs_layers = feed_forward(data);

            //deltas filter->width->height
            std::vector<std::vector<std::vector<float>>> deltas_before;
            std::vector<std::vector<std::pair<float, std::pair<size_t,size_t>>>> org_index;
            size_t prev_layer_type = 0;
            for(size_t i = m_layers.size()-1;i > 0;i--){
                //only for adding all outputs together
                std::vector<float> outputs_connected_layers;

                for(size_t j = 0;j < m_layers[i].size();j++){

                    //last layer(s) is/are always connected layer(s)
                    if(i == m_layers.size()-1){
                        std::vector<std::vector<float>> input_net;
                        input_net.push_back(outputs_layers);

                        m_layers[i][j]->set_values(input_net);
                        m_layers[i][j]->forward();
                        std::vector<float> net_output = m_layers[i][j]->get_net_output();
                        outputs_connected_layers.insert(outputs_connected_layers.begin(),net_output.begin(),net_output.end());

                        //make training pair
                        std::pair<std::vector<float>,std::vector<float>> training_pair;

                        std::vector<float> training_vector;
                        for(size_t num_layers = 0;num_layers < m_layers[i-1].size();num_layers++){
                            for(size_t width = 0;width < m_layers[i-1][num_layers]->get_height();width++){
                                for(size_t height = 0;height < m_layers[i-1][num_layers]->get_width();height++){
                                    training_vector.push_back(m_layers[i-1][num_layers]->get_values()[width][height]);
                                }
                            }
                        }

                        training_pair = std::make_pair(training_vector,data.second);

                        //get deltas and make to 3d format
                        std::vector<float> deltas_connected = m_layers[i][j]->train(training_pair,m_learning_rate,m_momentum);
                        for(size_t num_layers = 0;num_layers < m_layers[i-1].size();num_layers++){
                            std::vector<std::vector<float>> deltas_filter;
                            for(size_t width = 0;width < m_layers[i-1][num_layers]->get_height();width++){
                                std::vector<float> column;
                                for(size_t height = 0;height < m_layers[i-1][num_layers]->get_width();height++){
                                    column.push_back(deltas_connected[width*height]);
                                }
                                deltas_filter.push_back(column);
                            }
                            deltas_before.push_back(deltas_filter);
                        }

                        prev_layer_type = Layer::CONNECTED;
                    }else{
                        //other layers
                        //we have to do different things for different layers, so 'ol switch case is pretty handy
                        size_t layer_type = m_layers[i][j]->get_type();
                        switch(layer_type){
                            case Layer::CONV:{
                                if(prev_layer_type == Layer::POOL){
                                    size_t filter_num = 0;
                                    for(Filter &filter:m_layers[i][j]->get_filters()){
                                        //backpropagate filter
                                        //delta = f'(sum)*sum(prevDeltas*weights)
                                        float sum = 0;
                                        for(size_t width = 0;width < filter.get_values().size();width++){
                                            for(size_t height = 0;height < filter.get_values()[width].size();height++){
                                                //check if relevant and not pooled
                                                auto [x,y] = org_index[width][height].second;
                                                if(x != width && y != height){
                                                    //sum += filter.get_weights()[width][height] * deltas_before[filter_num][width][height];
                                                }
                                            }
                                        }
                                        filter_num++;
                                    }
                                }
                                break;
                            }
                            case Layer::POOL:{
                                //we essentially dont have to do anything because we dont calculate anything for pool layers
                                //only pass on indici
                                org_index = m_layers[i][j]->get_org_index();
                                break;
                            }
                            //TODO make class first
                            case Layer::ACT:{

                                break;
                            }
                        }
                        prev_layer_type = layer_type;
                    }
                }
                //only add to all outputs if in last layers
                if(i == m_layers.size()-1){
                    all_outputs.push_back(outputs_connected_layers);
                }
            }

        }
    }

    std::cout << "Finished training" << std::endl;

    return 0;
}

std::vector<float> Convolutional::feed_forward_first(std::pair<std::vector<std::vector<float>>, std::vector<float>> &t_data){
    std::vector<float> outputs;
    std::vector<std::thread> forward_threads;

    //TODO only initialize layers the first feed forward iteration
    //go through each "slice" of the cnn example: input, conv conv , pool pool , output
    for(size_t i = 0;i < m_layers.size();i++){
        //connected layer vals
        std::vector<std::vector<float>> conn_values;
        conn_values.push_back(std::vector<float>());
        //TODO multithreading till connected layer?
        for(size_t j = 0;j < m_layers[i].size();j++){
            if(i == 0){
                //set input values
                Layer* layer = new Layer(t_data.first.size(),t_data.first[0].size(),1,Layer::INPUT);
                layer->set_values(t_data.first);
                m_layers[i][j] = layer;
            }else{
                //get previous type, index 0 bc all vertical layer should be the same
                size_t layer_prev_type = m_layers[i-1][0]->get_type();
                //get current type
                size_t layer_type = m_layers[i][j]->get_type();

                //forward values
                std::vector<std::vector<float>> values;

                if(m_layers[i-1][0]->get_type() == Layer::CONV){
                    values = m_layers[i-1][j/m_num_filters]->get_values();
                }else {
                    values = m_layers[i-1][j]->get_values();
                }

//                if(m_test){
//                    std::cout << "Size: i = " << i << ": " << m_layers[i].size() << std::endl;
//                    std::cout << "Layertype: " << m_layers[i][j]->get_type() << std::endl;
//                    std::cout << "Layertype before: " << m_layers[i-1][0]->get_type() << std::endl;
//                    std::cout << "Size Values before: " << values.size() << "x" << values[0].size() << std::endl;
//                }

                switch(layer_type){
                    case Layer::CONV:{
                        ConvolutionLayer* conv_layer = new ConvolutionLayer(t_data.first.size(),t_data.first[0].size(),1,m_zero_padding,m_stride_filters,m_num_filters,m_filters_size,m_function_type);
                        conv_layer->set_values(t_data.first);
                        conv_layer->make_padding();
                        m_layers[i][j] = conv_layer;
                        break;
                    }
                    case Layer::POOL:{
                        //push back as much as we need if prev layer is conv layer

                        PoolLayer* pool_layer = new PoolLayer(2,2,1,m_stride_pool);
                        pool_layer->pool(values);
                        m_layers[i][j] = pool_layer;
                        break;
                    }
                    case Layer::ACT:{
                        //see pool case
                        ActivationLayer* act_layer = new ActivationLayer(values.size(),values[0].size(),1,m_function_type);
                        act_layer->calculate(values);
                        m_layers[i][j] = act_layer;
                        break;
                    }
                    //this does not work here because the connected layer train function need to be called up there
                    //just init this layer here
                    case Layer::CONNECTED:{
                        std::vector<float> layer_1d;
                        for(size_t num_layers = 0;num_layers < m_layers[i-1].size();num_layers++){
                            for(size_t width = 0;width < m_layers[i-1][num_layers]->get_height();width++){
                                for(size_t height = 0;height < m_layers[i-1][num_layers]->get_width();height++){
                                    values = m_layers[i-1][num_layers]->get_values();
                                    //TODO why is this always 0?
                                    float value = values[width][height];
                                    layer_1d.push_back(value);
                                }
                            }
                        }
                        //values should be empty if connected layer
                        conn_values.insert(conn_values.begin(),layer_1d);
                        ConnectedLayer* conn_layer = new ConnectedLayer(m_function_type,{values[0].size(),30,10},true);
                        conn_layer->set_values(values);
//                        std::pair<std::vector<float>,std::vector<float>> conn_training_data = std::make_pair(values[0],t_data.second);
//                        //has to be called before get_net_output
//                        conn_layer->forward();
                        m_layers[i][j] = conn_layer;
                        //insert outputs from prev layers into outputs, connected layer is trained in Convolutional::train();
                        outputs.insert(outputs.end(),values[0].begin(),values[0].end());
                    }
                }
            }
            //if last layer is reached
//            if(i == m_layers.size()-1){
//                //moved up to layer::connected
//            }
        }
    }

    return outputs;
}

std::vector<float> Convolutional::feed_forward(std::pair<std::vector<std::vector<float>>, std::vector<float>> &t_data){

    std::vector<float> outputs;
    std::vector<std::thread> forward_threads;

    //TODO only initialize layers the first feed forward iteration
    //go through each "slice" of the cnn example: input, conv conv , pool pool , output
    for(size_t i = 0;i < m_layers.size();i++){
        //TODO multithreading till connected layer?
        for(size_t j = 0;j < m_layers[i].size();j++){
            if(i == 0){
                //set input values
                m_layers[i][j]->set_values(t_data.first);
            }else{
                //get previous type, index 0 bc all vertical layer should be the same
                size_t layer_prev_type = m_layers[i-1][0]->get_type();
                //get current type
                size_t layer_type = m_layers[i][j]->get_type();

                //forward values
                std::vector<std::vector<float>> values;

                if(m_layers[i-1][0]->get_type() == Layer::CONV){
                    values = m_layers[i-1][j/m_num_filters]->get_values();
                }else{
                    values = m_layers[i-1][j]->get_values();
                }

//                if(m_test){
//                    std::cout << "Size: i = " << i << ": " << m_layers[i].size() << std::endl;
//                    std::cout << "Layertype: " << m_layers[i][j]->get_type() << std::endl;
//                    std::cout << "Layertype before: " << m_layers[i-1][0]->get_type() << std::endl;
//                    std::cout << "Size Values before: " << values.size() << "x" << values[0].size() << std::endl;
//                }

                switch(layer_type){
                    case Layer::CONV:{
                        m_layers[i][j]->set_values(t_data.first);
                        m_layers[i][j]->make_padding();
                        break;
                    }
                    case Layer::POOL:{
                        //push back as much as we need if prev layer is conv layer
                        m_layers[i][j]->pool(values);
                        break;
                    }
                    case Layer::ACT:{
                        //see pool case
                        m_layers[i][j]->calculate(values);
                        break;
                    }
                    //this does not work here because the connected layer train function need to be called up there
                    //just init this layer here
                    case Layer::CONNECTED:{
                        for(size_t layer = 0;layer < m_layers[i-1].size();layer++){
                            std::vector<float> layer_1d;
                            for(size_t width = 0;width < m_layers[i-1][layer]->get_values().size();width++){
                                for(size_t height = 0;height < m_layers[i-1][layer]->get_values()[width].size();height++){
                                    layer_1d.push_back(m_layers[i-1][layer]->get_values()[width][height]);
                                }
                            }
                            //insert all prev values to connected layer
                            values[0].insert(values[0].begin(),layer_1d.begin(),layer_1d.end());
                        }
                        m_layers[i][j]->set_values(values);
//                        std::pair<std::vector<float>,std::vector<float>> conn_training_data = std::make_pair(values[0],t_data.second);
//                        //has to be called before get_net_output
//                        conn_layer->forward();
                        //insert outputs from prev layers into outputs, connected layer is trained in Convolutional::train();
                        outputs.insert(outputs.end(),values[0].begin(),values[0].end());
                    }
                }
            }
            //if last layer is reached
//            if(i == m_layers.size()-1){
//                //moved up to layer::connected
//            }
        }
    }

    return outputs;
}


//backpropagation with momentum
float Convolutional::backprop_momentum(float &delta_current, float &activation_before, float &old_change) {

	//j -> i
	//learningrate * deltai * activationj
	float weight_change = (1-m_momentum) * m_learning_rate * delta_current * activation_before + m_momentum * old_change;

	return weight_change;
}

size_t Convolutional::run_tests(){

    m_test = true;

    std::vector<std::vector<ConvolutionLayer>> architecture;
    std::vector<PoolLayer> last_layers;

    /*
     * Generate sample input
     */
    size_t input_size = 4;
    size_t depth = 1;

    size_t filter_size = 2;
    size_t filter_stride = 1;
    size_t num_filters = 2;

    size_t pool_size = 2;
    size_t pool_stride = 2;

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

        ConvolutionLayer conv = ConvolutionLayer(input_size,input_size,1,1,filter_stride,num_filters,filter_size,m_function_type);
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
            std::cout << "Filters: " << filters.size() << std::endl << std::endl;
            if(filters.size() == 0){
                break;
            }
            for(Filter filter:filters){
                Layer output = filter.calculate_output(architecture[i][j]);
                std::cout << output.get_values().size() << "x" << output.get_values()[0].size() << std::endl;
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
                std::vector<std::vector<float>> output_values = output.get_values();

                PoolLayer pool = PoolLayer(pool_size,pool_size,1,pool_stride);
                pool.pool(output_values);

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
    //superclass layer takes 2d inputs, we only need 1d input
    std::vector<std::vector<float>> dim;
    dim.push_back(connected_inputs);

    //add neurons manually
    std::vector<size_t> sizes;
    sizes.push_back(dim.size());
    sizes.push_back(30);
    sizes.push_back(20);
    sizes.push_back(10);

    //test raw and normalized output
    for(size_t i = 0;i < 2;i++){
        if(i == 0){
            std::cout << "Raw Output: " << std::endl;
        }else{
            std::cout << "Normalized Output: " << std::endl;
        }
        ConnectedLayer connected = ConnectedLayer(ConnectedLayer::functiontype::SWISH,sizes,i);
        connected.set_values(dim);
        connected.forward();
        std::vector<float> outputs = connected.get_net_output();
        for(size_t j = 0;j < outputs.size();j++){
            std::cout << j << ":" << outputs[j] << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << std::endl;

    /*
     * Test training
     */
    std::cout << "Testing training: " << std::endl;
    read_stuff(10,10);
    std::vector<std::pair<std::vector<std::vector<float>>, std::vector<float>>> test_data;
    for(int i = 0;i < images.size();i++){
        //make image vectors
        std::vector<std::vector<float>> values;
        values.resize(28);
        for(size_t j = 0;j < 28;j++){
            values[j].resize(28);
            for(size_t k = 0;k < 28;k++){
                values[j][k] = images[i][j*k] / 255;
            }
        }

        //make label vectors
        std::vector<float> temp;
        temp.resize(10);
        for (int j = 0; j < 10; j++) {
            if (labels[i] == j) {
                temp[j] = 1;
            }
            else {
                temp[j] = 0;
            }
        }
        test_data.push_back(std::make_pair(values,temp));
    }
    std::cout << "Architecture: " << std::endl;
    for(size_t i = 0;i < m_layers.size();i++){
        for(size_t j = 0;j < m_layers[i].size();j++){
            std::cout << "Type: " << m_layers[i][j]->get_type() << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    this->train(test_data,SWISH,0.02,0.05,100,1,3);

    m_test = false;

    std::cout << "Finished tests" << std::endl;

    return 0;
}





size_t read_stuff(size_t number_of_images_open,size_t number_of_labels_open) {

    Math math;

    /*
    * Open training images
    */
    std::ifstream file("train-images.idx3-ubyte", std::ios::binary);
    if (file.is_open())
    {
        std::cout << "Opening images..." << std::endl;
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
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
        images.reserve(number_of_images_open);
        for (size_t i = 0; i < number_of_images_open; ++i)
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
        int magic_number = 0;
        int number_of_labels;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = math.reverseInt(magic_number);

        if (magic_number != 2049) return 1;

        file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = math.reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels_open];
        for (size_t i = 0; i < number_of_labels_open; i++) {
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
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
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
        int magic_number = 0;
        int number_of_labels;
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
    std::cout << std::endl;

    return 0;
}

