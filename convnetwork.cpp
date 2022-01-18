#include "convnetwork.h"
#include "Layer.h"
#include "PoolLayer.h"
#include "kernel.h"
#include <memory>
#include <utility>

typedef unsigned char uchar;
std::vector<std::vector<float>> images;
uchar* labels;
std::vector<std::vector<float>> testimages;
uchar* testlabels;
size_t read_stuff(size_t number_of_images_open,size_t number_of_labels_open);

using namespace layer;

size_t Convolutional::train(std::vector<struct_training_data> &t_training_data,size_t t_funcion_type,float t_learning_rate,float t_momentum, size_t t_epochs, size_t t_stride_filters,size_t t_stride_pools) {
    // TODO move to initializer list but i am too lazy
    m_stride_pool = t_stride_pools;
    m_stride_filters = t_stride_filters;
    m_epochs = t_epochs;
    m_learning_rate = t_learning_rate;
    m_momentum = t_momentum;
    m_function_type = t_funcion_type;
    
    // initialize network on first training method call
    if(!m_initialized) this->initialize(t_training_data[0]);

    std::cout << "Starting training..." << std::endl;

    //train for epoch amount
    for(size_t epoch = 0;epoch < t_epochs;epoch++){
        std::cout << "Epoch: " << epoch << std::endl;
        float batch_error = 0;
        //go through batch
        for(struct_training_data &data:t_training_data){
            // feed forward
            // this returns deltas of connected layer 
            std::pair<std::vector<float>,std::vector<float>> return_values = feed_forward(data);
            
            // Problem: if there are negative values and zero padding is used with max pooling, net output will be 0
            std::vector<float> outputs = return_values.first;
            std::vector<float> deltas = return_values.second;

            // Error
            float data_error = 0;
            for(size_t i = 0;i < outputs.size();i++){
                data_error = pow(data.corrrect_outputs[i] - outputs[i],2);
            }
            data_error *= 0.5;
            batch_error += data_error;

            // Start calculating backwards, connected layer already handled in knn lib
            std::cout << deltas.size() << std::endl;

            // for(size_t i = 0;i < m_layers.size();i++){
            //     std::cout << m_layers[i].size() << "x" << m_layers[i][0]->get_type() << std::endl;
            // }
            // for(size_t i = 0;i < m_filters.size();i++){
            //     std::cout << m_filters[i].size() << "x" << m_filters[i][0]->size() << std::endl;
            // }

            // Split deltas to corresponding layers
            for(size_t i = 0;i < m_layers[m_layers.size()-2].size();i++){
                array_2f layer_deltas;
                layer_deltas.resize(boost::extents[m_layers[m_layers.size()-2][i]->get_values().size()][m_layers[m_layers.size()-2][i]->get_values()[0].size()]);
                for(size_t j = 0;j < m_layers[m_layers.size()-2][i]->get_values().size();j++){
                    for(size_t k = 0;k < m_layers[m_layers.size()-2][i]->get_values()[0].size();k++){
                        layer_deltas[j][k] = deltas[i* (j*m_layers[m_layers.size()-2][i]->get_values().size()+k)];
                    }
                }
                m_layers[m_layers.size()-2][i]->set_deltas(layer_deltas);
            }

            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

            // Backpropagate through network; Exclude input layer
            for(size_t i = m_layers.size()-2;i > 0;i--){
                size_t layer_type = m_layers[i][0]->get_type();
                for(size_t num_layer = 0;num_layer < m_layers[i].size();num_layer++){
                    switch (layer_type){
                        case(Layer::CONV):{
                            // Get index for Filters that should be used
                            size_t num_conv_layer_incl = 0;
                            for(size_t j = 0;j < m_layers.size();j++){
                                if(m_layers[j][0]->get_type() == Layer::CONV){
                                    num_conv_layer_incl++;
                                }
                                if(j == i){
                                    break;
                                }
                            }
                            size_t index_filter = i - ((i+1)-num_conv_layer_incl);
                            
                            // If last layer before connected layer, don't get filter because deltas are obtained from mlp
                            if(i != m_layers.size()-2){
                                // layer::fshared_ptr_t f = m_filters[index_filter-1][num_layer];
                                // m_layers[i][num_layer]->set_filter(f);
                                
                                array_2f deltas = m_layers[i+1][num_layer]->get_deltas();
                                m_layers[i][num_layer]->set_deltas(deltas);
                                m_layers[i][num_layer]->backwards_propagation(m_layers[i][num_layer]->get_values());
                            }else{
                                // Deltas are already set
                                m_layers[i][num_layer]->backwards_propagation(m_layers[i][num_layer]->get_values());
                            }
                            break;
                        }
                        case(Layer::POOL):{
                            // Deltas for last layer before connected layer where set above
                            array_2f deltas = (i == (m_layers.size()-2)) ? m_layers[i][num_layer]->get_deltas() : m_layers[i+1][num_layer]->get_deltas();
                            m_layers[i][num_layer]->set_deltas(deltas);
                            m_layers[i][num_layer]->backwards_propagation(m_layers[i-1][num_layer]->get_values());
                            break;
                        }
                        // Just get deltas from previous layer and apply them
                        case(Layer::ACT):{
                            array_2f deltas = (i == (m_layers.size()-2)) ? m_layers[i][num_layer]->get_deltas() : m_layers[i+1][num_layer]->get_deltas();
                            m_layers[i][num_layer]->set_deltas(deltas);
                            break;
                        }
                    }
                }
            }
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double,std::milli> time_span = std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(t2 - t1);
            std::cout << "Time for backprop: " << time_span.count() << "ms" << std::endl;

            // Change Weights; Exclude input and connected layer
            for(size_t i = 1;i < m_layers.size()-1;i++){
                size_t layer_type = m_layers[i][0]->get_type();
                if(layer_type == Layer::CONV){

                    // Get index for Filters that should be used
                    size_t num_conv_layer_incl = 0;
                    for(size_t j = 0;j < m_layers.size();j++){
                        if(m_layers[j][0]->get_type() == Layer::CONV){
                            num_conv_layer_incl++;
                        }
                        if(j == i){
                            break;
                        }
                    }
                    size_t index_filter = i - ((i+1)-num_conv_layer_incl) -1;
                    
                    for(size_t j = 0;j < m_layers[i-1].size();j++){
                        for(size_t filter_num = 0;filter_num < m_filters[index_filter].size();filter_num++){
                            fshared_ptr_t filter = m_filters[index_filter][filter_num];
                            filter->get_kernels()[j]->calculate_gradient(m_layers[i-1][j]->get_values());
                        }        
                    }
                }
            }


        }
        std::cout << "Error: " << batch_error/t_training_data.size() << std::endl;
    }

    std::cout << "Finished training" << std::endl;

    return 0;
}

std::pair<std::vector<float>,std::vector<float>> Convolutional::feed_forward(struct_training_data &t_data){
    // for(size_t i = 0;i < m_layers.size();i++){
    //     std::cout << m_layers[i].size() << "x" << m_layers[i][0]->get_type() << std::endl;
    // }
    // for(size_t i = 0;i < m_filters.size();i++){
    //     std::cout << m_filters[i].size() << "x" << m_filters[i][0]->size() << std::endl;
    // }
    std::pair<std::vector<float>,std::vector<float>> outputs;

    // this iterates through all layers
    for(size_t i = 0;i < m_layers.size();i++){

        array_3f prev_values3f;
        if(i > 0){
            for(size_t j = 0;j < m_layers[i-1].size();j++){
                if(j > 0){
                    prev_values3f.resize(boost::extents[m_layers[i-1].size()][m_layers[i-1][j-1]->get_values().size()][m_layers[i-1][j-1]->get_values()[0].size()]);
                }else{
                    prev_values3f.resize(boost::extents[m_layers[i-1].size()][m_layers[i-1][0]->get_values().size()][m_layers[i-1][0]->get_values()[0].size()]);
                }      
                prev_values3f[j] = m_layers[i-1][j]->get_values();
            }
        }

        for(size_t num_layer = 0;num_layer < m_layers[i].size();num_layer++) {

            Layer* layer = m_layers[i][num_layer];
            size_t type_layer = (i==0) ? Layer::INPUT : layer->get_type();

            // We can use a 0 here because all previous layers have the same type
            size_t type_prev_layer = (i==0) ? Layer::INPUT : m_layers[i-1][0]->get_type();
            switch(type_layer){
                // Set input values to network
                case Layer::INPUT: {
                    // Still convolutional layer
                    layer->set_values(t_data.image_data[num_layer]);
                    layer->make_padding();
                    break;
                }
                // Use filters to calculate convolution
                case Layer::CONV: {
                    // Get index for Filters that should be used
                    size_t num_conv_layer_incl = 0;
                    for(size_t j = 0;j < m_layers.size();j++){
                        if(m_layers[j][0]->get_type() == Layer::CONV){
                            num_conv_layer_incl++;
                        }
                        if(j == i){
                            break;
                        }
                    }
                    size_t index_filter = i - ((i+1)-num_conv_layer_incl);

                    layer::fshared_ptr_t f = m_filters[index_filter-1][num_layer];
                    array_2f forwarded_values = f->forward(prev_values3f);

                    m_layers[i][num_layer]->set_values(forwarded_values);

                    // Make padding if next layer is not connected layer TODO make this different (different kind, only if wanted ...)?
                    // Pool layer with padding?
                    if(m_layers[i+1][0]->get_type() != Layer::CONNECTED && m_layers[i+1][0]->get_type() != Layer::POOL && m_layers[i+1][0]->get_type() != Layer::ACT){
                        m_layers[i][num_layer]->make_padding();
                    }
                    // Set correct filter; TODO in forward propagation
                    m_layers[i][num_layer]->set_filter(f);
                    
                    break;
                }
                case Layer::POOL: {
                    array_2f values = m_layers[i-1][num_layer]->get_values();
                    m_layers[i][num_layer]->pool(values);
                    break;
                }
                case Layer::ACT: {
                    array_2f values = m_layers[i-1][num_layer]->get_values();
                    m_layers[i][num_layer]->calculate(values);
                    break;
                }
                case Layer::CONNECTED: {
                    
                    array_3f input_values;
                    // TODO test this vs inserting values into vector
                    // All output sizes should be the same, otherwise something is very wrong
                    input_values.resize(boost::extents[m_layers[i-1].size()][m_layers[i-1][0]->get_values().size()][m_layers[i-1][0]->get_values()[0].size()]);
                    for(size_t j = 0;j < m_layers[i-1].size();j++){
                        input_values[j] = m_layers[i-1][j]->get_values();
                    }

                    boost::array<array_3f::index, INPUT_DIMENSIONS> new_shape = {{1, 1,  static_cast<long>(m_layers[i-1].size()*m_layers[i-1][0]->get_values().size()*m_layers[i-1][0]->get_values()[0].size())}}; 
                    input_values.reshape(new_shape);
                    
                    m_layers[i][num_layer]->set_values(input_values[0]);
                    m_layers[i][num_layer]->forward();

                    std::vector<float> net_outputs = m_layers[i][num_layer]->get_net_output();
                    std::vector<float> deltas = m_layers[i][num_layer]->train(t_data.corrrect_outputs, m_learning_rate, m_momentum);
                    
                    outputs = std::make_pair(net_outputs, deltas);

                    break;
                }
            }
        }
    }
    return outputs;
}

/*
 * Essentially feeds the values once through the network and initializes all sizes
*/
size_t Convolutional::initialize(struct_training_data &t_data){
    
    std::cout << "Initializing Network" << std::endl;

    for(size_t i = 0;i < m_architecture.size();i++){
        size_t layer_type = m_architecture[i];
        if(layer_type == Layer::CONV){
            m_architecture.insert(m_architecture.begin()+i+1,Layer::ACT);
        }
    }

    boost::progress_display layers_progress(m_architecture.size()+1);

    // Initialize input layers with convolutional layer so we can add zero padding
    std::vector<Layer*> input_layers;
    for(size_t i = 0;i < t_data.image_data.size();i++){
        ConvolutionLayer* input_layer = new ConvolutionLayer(t_data.image_data[i],m_zero_padding);
        input_layer->make_padding();
        input_layers.push_back(input_layer);
    }
    m_layers.push_back(input_layers);

    ++layers_progress;

    // Resize filters
    size_t num_conv_layers = std::count(m_architecture.begin(),m_architecture.end(),Layer::CONV);
    m_filters.resize(num_conv_layers);

    // Add connected layer to m_architecture
    m_architecture.push_back(Layer::CONNECTED);

    // Add additional Layers
    for(size_t num_layer = 0;num_layer < m_architecture.size();num_layer++){
        size_t num_layers_size = m_layers.size();
        // get layertype of currently initializing layer
        size_t layertype = m_architecture[num_layer];
        switch (layertype) {
            // Initialize convolutional layers
            case Layer::CONV:{
                
                size_t num_conv_layer_incl = 0;
                for(size_t i = 0;i < m_architecture.size();i++){
                    if(m_architecture[i] == Layer::CONV){
                        num_conv_layer_incl++;
                    }
                    if(i == num_layer){
                        break;
                    }
                }

                // Resize filter array
                size_t index = num_layer - ((num_layer+1)-num_conv_layer_incl);

                // Can't be used because in 2d arrays all rows must have the same size
                //m_filters.resize(boost::extents[num_conv_layers][m_num_filters[index]]);
                m_filters[index] = std::vector<fshared_ptr_t>();
                m_filters[index].resize(m_num_filters[index]);
                
                // Get depth of last layer slice
                for(size_t i = 0;i < m_filters[index].size();i++){
                    size_t depth = m_layers[num_layers_size-1].size();
                    size_t width = m_filters_size[index];
                    m_filters[index][i] = std::make_shared<Filter>(depth,width,m_stride_filters,m_function_type);
                }

                // Add previous values to 3d layer
                array_3f values_forward;
                for(size_t i = 0;i < m_layers[num_layers_size-1].size();i++){
                    array_2f values = m_layers[num_layers_size-1][i]->get_values();
                    values_forward.resize(boost::extents[m_layers[num_layers_size-1].size()][values.size()][values[0].size()]);
                    values_forward[i] = values;
                }

                // Forward values
                array_3f forwarded_values;
                for(size_t i = 0;i < m_filters[index].size();i++){
                    forwarded_values.resize(boost::extents[m_filters[index].size()][m_filters[index][i]->get_output_width(values_forward[0].size())][m_filters[index][i]->get_output_height(values_forward[0][0].size())]);
                    forwarded_values[i] = m_filters[index][i]->forward(values_forward);
                }

                // Finally add convolutional layers
                std::vector<Layer*> conv_layers;
                for(size_t i = 0;i < forwarded_values.size();i++){
                    ConvolutionLayer* layer = new ConvolutionLayer(forwarded_values[i],m_zero_padding);
                    layer->set_values(forwarded_values[i]);
                    // Make padding if not last layer
                    if(m_architecture[num_layer+1] != Layer::CONNECTED && m_architecture[num_layer+1] != Layer::POOL && m_architecture[num_layer+1] != Layer::ACT){
                        layer->make_padding();
                    }
                    // layer->set_kernels(&m_filters[index]);
                    conv_layers.push_back(layer);
                }
                m_layers.push_back(conv_layers);

                break;
            }
            case Layer::POOL:{
                // Add pool layers
                std::vector<Layer*> pool_layers;
                for(size_t i = 0;i < m_layers[num_layers_size-1].size();i++){
                    PoolLayer* layer = new PoolLayer(m_layers[num_layers_size-1][i]->get_values(),m_pools_size,m_pools_size,m_stride_pool,Layer::AVERAGE);
                    layer->pool(m_layers[num_layers_size-1][i]->get_values());
                    pool_layers.push_back(layer);
                }
                m_layers.push_back(pool_layers);

                break;
            }
            case Layer::ACT:{
                // Add activation layers
                std::vector<Layer*> activation_layers;
                for(size_t i = 0;i < m_layers[num_layers_size-1].size();i++){
                    ActivationLayer* layer = new ActivationLayer(m_layers[num_layers_size-1][i]->get_values(),m_function_type);
                    layer->calculate(m_layers[num_layers_size-1][i]->get_values());
                    activation_layers.push_back(layer);
                }
                m_layers.push_back(activation_layers);

                break;
            }
            case Layer::CONNECTED:{
                // Add connected layer
                std::vector<Layer*> connected_layers;
                array_2f init_values;
                init_values.resize(boost::extents[1][1]);

                array_3f input_values;
                // All output sizes should be the same, otherwise something is very wrong
                input_values.resize(boost::extents[m_layers[m_layers.size()-1].size()][m_layers[m_layers.size()-1][0]->get_values().size()][m_layers[m_layers.size()-1][0]->get_values()[0].size()]);
                for(size_t j = 0;j < m_layers[m_layers.size()-1].size();j++){
                    input_values[j] = m_layers[m_layers.size()-1][j]->get_values();
                }

                boost::array<array_3f::index, INPUT_DIMENSIONS> new_shape = {{1, 1,  static_cast<long>(m_layers[m_layers.size()-1].size()*m_layers[m_layers.size()-1][0]->get_values().size()*m_layers[m_layers.size()-1][0]->get_values()[0].size())}}; 
                input_values.reshape(new_shape);
                //if necessary print size of connected layer inputs
                //std::cout << input_values[0][0].size() << std::endl;
                m_connected_matrix_size.insert(m_connected_matrix_size.begin(),input_values[0][0].size());
                ConnectedLayer* connected_layer = new ConnectedLayer(init_values,m_function_type,m_connected_matrix_size,false);

                connected_layers.push_back(connected_layer);
                m_layers.push_back(connected_layers);
                break;
            }
        }
        ++layers_progress;
    }

    m_initialized = true;
    return 0;
}

size_t Convolutional::run_tests(){

    m_test = true;

    std::vector<std::vector<Layer>> architecture;
    std::vector<Layer> last_layers;

    /*
     * Generate sample input
     */
    long input_size = 32;
    long depth = 3;
    const size_t dimensions = 3;

    size_t filter_size = 5;
    size_t filter_stride = 1;
    size_t num_filters = 1;

    size_t pool_size = 2;
    size_t pool_stride = 1;

    size_t zero_padding = 1;

    boost::array<array_3f::index, dimensions> shape_array_images = {{ depth,input_size, input_size }};
    array_3f inputs(shape_array_images);
    
    //simulate rgb channel
    std::cout << "Running tests..." << std::endl << std::endl;
    std::cout << "Inputs: " << std::endl << std::endl;
    for(size_t i = 0;i < depth;i++){
        for(size_t j = 0;j < input_size;j++){
            std::vector<float> column;
            for(size_t k = 0;k < input_size;k++){
                inputs[i][j][k] = (float) k + i + j;
                std::cout << " " << (float) k + i + j;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    /*
     * Test for convolutional layer
     */
    std::cout << "Test Convolutional Layer / zero padding: " << std::endl << std::endl;

    std::vector<Layer> conv_layers;
    for(size_t i = 0;i < inputs.size();i++){
        ConvolutionLayer conv = ConvolutionLayer(inputs[i],zero_padding);
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
     * Test for activation layer
     */
    std::cout << "Test activation layer: " << std::endl << std::endl;

    std::vector<Layer> act_layers;
    for(size_t i = 0;i < architecture[architecture.size()-1].size();i++){
        array_2f values = architecture[architecture.size()-1][i].get_values();
        ActivationLayer act = ActivationLayer(values,SIGMOID);
        act.calculate(values);
        act_layers.push_back(act);

        for(size_t width = 0;width < act.get_values().size();width++){
            for(size_t height = 0;height < act.get_values().size();height++){
                std::cout << act.get_values()[width][height] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }
    architecture.push_back(act_layers);
    std::cout << std::endl << std::endl;

    /*
     * Test for PoolLayer
     */
    std::cout << "Test Pool Layer: " << std::endl;

    std::cout << std::endl << "Max Pooling:" << std::endl;
    //Filters were replaced by standartized input values, leave this, to generate more inputs for connected layer
    for(size_t i = 0; i < architecture.size();i++){

        for(size_t n = 0;n < architecture[i].size();n++) {
            
            PoolLayer pool = PoolLayer(inputs[0],pool_size,pool_size,pool_stride,Layer::MAX);

            pool.pool_max(inputs[0]);
            std::cout << std::endl;
            for(size_t j = 0;j < pool.get_values().size();j++){
                for(size_t k = 0;k < pool.get_values()[0].size();k++){
                    std::cout << " " << pool.get_values()[j][k];
                }
                std::cout << std::endl;
            }
            
            std::cout << std::endl;
            for(size_t j = 0;j < pool.get_input_indices().size();j++){
                for(size_t k = 0;k < pool.get_input_indices()[0].size();k++){
                    if(pool.get_input_indices()[j][k] == 1){
                        std::cout << j << ":" << k << " ";
                    }
                }
                std::cout << std::endl;
            }
            last_layers.push_back(pool);

        }
        std::cout << std::endl << std::endl;
    }

    std::cout << "Average Pooling:" << std::endl;

    for(size_t i = 0; i < architecture.size();i++){

        for(size_t n = 0;n < architecture[i].size();n++) {
            
            PoolLayer pool = PoolLayer(inputs[0],pool_size,pool_size,pool_stride,Layer::AVERAGE);

            pool.pool_average(inputs[0]);
            std::cout << std::endl;
            for(size_t j = 0;j < pool.get_values().size();j++){
                for(size_t k = 0;k < pool.get_values()[0].size();k++){
                    std::cout << " " << pool.get_values()[j][k];
                }
                std::cout << std::endl;
            }
            
            std::cout << std::endl;
            for(size_t j = 0;j < pool.get_input_indices().size();j++){
                for(size_t k = 0;k < pool.get_input_indices()[0].size();k++){
                    if(pool.get_input_indices()[j][k] == 1){
                        std::cout << j << ":" << k << " ";
                    }
                }
                std::cout << std::endl;
            }
            last_layers.push_back(pool);

        }
        std::cout << std::endl << std::endl;
    }
    std::cout << std::endl << std::endl;

    /*
     * Test for filters
     */
    std::cout << "Test Filters: " << std::endl << std::endl;

    //compare cuda gcc vs normal gcc
    // double compare_gcc = 0.0;
    // for(size_t runs = 0;runs < 100;runs++){
    //     std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    //     //make some filters
    //     array_2flt filters;
    //     filters.resize(boost::extents[1][5]);
    //     for(size_t k = 0;k < filters[0].size();k++){
    //         filters[0][k] = new Filter(depth,filter_size,filter_stride,SIGMOID);
    //     }

    //     for(size_t i = 0; i < architecture.size();i++){
    //         array_3f values_3f;
    //         for(size_t j = 0;j < architecture[i].size();j++) {
    //             array_2f values = architecture[i][j].get_values();
    //             values_3f.resize(boost::extents[i+1][values.size()][values[0].size()]);
    //             values_3f[i] = values;
    //         }
    //         for(size_t j = 0;j < filters[0].size();j++){
    //             array_2f output = filters[0][j]->forward(values_3f);
    //         }
    //     }
    //     std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double,std::milli> time_span = duration_cast<std::chrono::duration<double,std::milli>>(t2 - t1);
    //     compare_gcc += time_span.count();
    //     for(size_t i = 0;i < filters.size();i++){
    //         for(size_t j = 0;j < filters[i].size();j++){
    //             delete(filters[i][j]);
    //         }
    //     }
    // }
    // std::cout << "100 runs: " << compare_gcc << " milliseconds" << std::endl << std::endl;


    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    //make some filters
    array_2flt filters;
    filters.resize(boost::extents[1][5]);
    for(size_t k = 0;k < filters[0].size();k++){
        filters[0][k] = new Filter(depth,filter_size,filter_stride,1);
    }
    std::cout << "Filters: " << filters.size()*filters[0].size() << std::endl << std::endl;

    for(size_t i = 0; i < architecture.size();i++){
        array_3f values_3f;
        std::cout << "Input values: " << std::endl;
        for(size_t j = 0;j < architecture[i].size();j++) {
            array_2f values = architecture[i][j].get_values();
            values_3f.resize(boost::extents[i+1][values.size()][values[0].size()]);
            values_3f[i] = values;
        }
        for(size_t j = 0;j < filters[0].size();j++){
            
            array_2f output = filters[0][j]->forward(values_3f);
            for(size_t m = 0;m < output.size();m++){
                for(size_t n = 0;n < output[0].size();n++){
                    std::cout << output[m][n] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }  
    }
    
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> time_span = std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(t2 - t1);
    std::cout << "Testing filters took me " << time_span.count() << " milliseconds.";
    std::cout << std::endl;

    // for(size_t i = 0; i < architecture.size();i++){

    //     for(size_t j = 0;j < architecture[i].size();j++) {
            
            
    //         for(size_t k = 0;k < m_filters.size();k++){
    //             Filter f = m_filters[i][j];
    //         }
    //         for(Kernel filter:m_filters){
    //             array_2f output = filter.forward(architecture[i][j]);
    //             std::cout << output.size() << "x" << output[0].size() << std::endl;
    //             for(size_t width = 0;width < output.size();width++){
    //                 for(size_t height = 0;height < output[0].size();height++){
    //                     std::cout << output[width][height] << " ";
    //                 }
    //                 std::cout << std::endl;
    //             }
    //             std::cout << std::endl;
    //         }
    //     }
    //     std::cout << std::endl << std::endl;
    // }
    // std::cout << std::endl;

    /*
     * Test training
     */
    std::cout << "Testing training: " << std::endl;
    
    boost::array<array_3f::index, 3> shape_array_training_images = {{ 3, 32, 32 }};
    array_3f image_array(shape_array_training_images);
    
    auto dataset = cifar::read_dataset<std::vector,std::vector,uint8_t,size_t>(1,1);
    
    for(size_t channel = 0;channel < 3;channel++){
        for(size_t i = 0;i < 32;i++){
            for(size_t j = 0;j < 32;j++){
                image_array[channel][i][j] = (float) dataset.training_images[0][(i * 32 * 32) + (j * 32) + channel] / 255;
            }
        }
    }
    
    std::vector<struct_training_data> training_data;
    
    struct_training_data single_data;
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
    struct_training_data test;
    test.image_data.resize(boost::extents[depth][input_size][input_size]);
    test.image_data = inputs;
    test.corrrect_outputs = {1,1,1,1,1,1,1,1,1,0};

    std::vector<struct_training_data> dummy_training;
    dummy_training.push_back(test);
    
    //train
    this->train(dummy_training,SIGMOID,0.02,0.05,1,1,1);
    
    m_test = false;
    
    std::cout << "Finished tests" << std::endl;

    return 0;
}


size_t read_stuff(size_t number_of_images_open,size_t number_of_labels_open) {

    Math math;

    /*
    * Open training images
    */
    std::ifstream file("../train-images.idx3-ubyte", std::ios::binary);
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
    }else{
        std::cout << "File not found!" << std::endl;
    }
    /*
    * Open training labels
    */
    file.open("../train-labels.idx1-ubyte", std::ios::binary);
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
    }else{
        std::cout << "File not found!" << std::endl;
    }
    /*
    * Open Test Images
    */
    file.open("../t10k-images.idx3-ubyte", std::ios::binary);
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
    }else{
        std::cout << "File not found!" << std::endl;
    }
    /*
    * Open Test Labels
    */
    file.open("../t10k-labels.idx1-ubyte", std::ios::binary);
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
    }else{
        std::cout << "File not found!" << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
