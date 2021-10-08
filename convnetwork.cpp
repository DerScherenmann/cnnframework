#include "convnetwork.h"

typedef unsigned char uchar;
std::vector<std::vector<float>> images;
uchar* labels;
std::vector<std::vector<float>> testimages;
uchar* testlabels;
size_t read_stuff(size_t number_of_images_open,size_t number_of_labels_open);

using namespace layer;

// size_t Convolutional::train(std::vector<struct_training_data> &t_training_data,size_t t_funcion_type,float t_learning_rate,float t_momentum, size_t t_epochs, size_t t_stride_filters,size_t t_stride_pools) {
//     m_stride_pool = t_stride_pools;
//     m_stride_filters = t_stride_filters;
//     m_epochs = t_epochs;
//     m_learning_rate = t_learning_rate;
//     m_momentum = t_momentum;
//     m_function_type = t_funcion_type;
//     
//     TODO move this out of train, otherwise training cant be resumed and network get reinitialized every time we train
//     this->initialize(t_training_data[0]);
//     
//     std::cout << "Starting training..." << std::endl;
// 
//     train for epoch amount
//     for(size_t epoch = 0;epoch < t_epochs;epoch++){
//         std::cout << "Epoch: " << epoch << std::endl;
//         go through batch
//         for(struct_training_data &data:t_training_data){
//             feed forward
//             this returns deltas of connected layer 
//             std::pair<std::vector<float>,std::vector<float>> vec_connected_outputs = feed_forward(data);
//             
//             size_t prev_layer_type = 0;
//             for(size_t i = m_layers.size()-1;i > 0;i--){
//                 for(size_t j = 0;j < m_layers[i].size();j++){
//                     convert 1d deltas to 2d deltas for pool layers
//                     if(i == m_layers.size()-1){
//                         size_t iteration = 0;
//                         std::vector<float> vec_deltas = vec_connected_outputs.second;
//                         for(Layer* &layer:m_layers[i-1]){
//                             std::vector<std::vector<float>> layer_deltas;
//                             layer_deltas.resize(layer->get_values().size());
//                             for(size_t num_width_layer = 0;num_width_layer < layer->get_values().size();num_width_layer++){
//                                 layer_deltas[num_width_layer].resize(layer->get_values()[num_width_layer].size());
//                                 for(size_t num_height_layer = 0;num_height_layer < layer->get_values()[num_width_layer].size();num_height_layer++){
//                                     layer_deltas[num_width_layer][num_height_layer] = vec_deltas[iteration];
//                                     iteration ++;
//                                 }
//                             }
//                             layer->set_deltas(layer_deltas);
//                         }
//                         VS
//                         for(Layer layer:m_layers[i]){
//                             
//                         }
//                         continue;
//                     }else{
//                         other layers
//                         we have to do different things for different layers, so 'ol switch case is pretty handy
//                         size_t layer_type = m_layers[i][j]->get_type();
//                         
//                         switch(layer_type){
//                             case Layer::CONV:{
//                                 size_t filter_num = 0;
//                                 for(Filter &filter:m_layers[i][j]->get_filters()){
//                                     get last calculated output
//                                     Layer* layer = filter.get_calculated_ouput();
//                                    
//                                     get filter weights
//                                     std::vector<std::vector<float>> filter_weights = filter.get_weights();
//                                     
//                                     get filter deltas
//                                     std::vector<std::vector<float>> deltas_filter = filter.get_deltas();
//                                     
//                                     deltas width->height
//                                     std::vector<std::vector<float>> deltas_before;
//                                     
//                                     backpropagation if prev layer is pooling layer
//                                     if(prev_layer_type == Layer::POOL){
//                                         original index of float
//                                         std::vector<std::vector<std::pair<float, std::pair<size_t,size_t>>>> org_index;
//                                         get matching pool layer
//                                         for(size_t num_pool_layer = 0;num_pool_layer < m_layers[i+1].size();num_pool_layer++){
//                                             if((num_pool_layer % m_num_filters) == filter_num){
//                                                 org_index = m_layers[i+1][num_pool_layer]->get_org_index();
//                                                 get previous deltas
//                                                 deltas_before = m_layers[i+1][num_pool_layer]->get_deltas();
//                                             }
//                                         }
//                                         std::vector<float> deltas;
//                                         for(size_t width_index_pane = 0;width_index_pane < org_index.size();width_index_pane++){
//                                             for(size_t height_index_pane = 0;height_index_pane < org_index[width_index_pane].size();height_index_pane++){
//                                                 get pooled index values
//                                                 auto [x,y] = org_index[width_index_pane][height_index_pane].second;
//                                                 std::cout << x << "," << y << std::endl;
//                                                 backpropagate filter
//                                                 delta = f'(sum)*sum(weights*prevDeltas)
//                                                 float sum = 0;
//                                                 sum weights*prevDeltas
//                                                 for(size_t width = 0;width < deltas_before.size();width++){
//                                                     for(size_t height = 0;height < deltas_before[width].size();height++){
//                                                         check if relevant and not pooled
//                                                         if(x == width && y == height){
//                                                             sum += filter_weights[x][y] * deltas_before[width][height];
//                                                         }
//                                                     }
//                                                 }
//                                                 float delta = actPrime(sum) * sum;
//                                                 deltas.push_back(delta);
//                                                 std::cout << delta << std::endl;
//                                             }
//                                         }
//                                         filter_num++;
//                                     }
//                                 }
//                                 TODO implement ActivationLayer
//                                 break;
//                             }
//                             case Layer::POOL:{
//                                 we essentially dont have to do anything because we dont calculate anything for pool layers
//                                 break;
//                             }
//                             TODO make class first
//                             case Layer::ACT:{
//                                 break;
//                             }
//                         }
//                         prev_layer_type = layer_type;
//                     }
//                 }
//             }
//         }
//     }
// 
//     std::cout << "Finished training" << std::endl;
// 
//     return 0;
// }
// 
// std::pair<std::vector<float>,std::vector<float>> Convolutional::feed_forward(struct_training_data &t_data){
//     
//     std::vector<std::thread> forward_threads;
//     
//     go through each "slice" of the cnn example: input, conv conv , pool pool , output
//     for(size_t i = 0;i < m_layers.size();i++){
//         TODO multithreading until connected layer?
//         
//         for(size_t j = 0;j < m_layers[i].size();j++){
//             if(i == 0){
//                 set input values
//                 m_layers[i][j]->set_values(t_data.image_data[j]);
//             }else{
//                 get previous type, index 0 bc all vertical layers should be the same type
//                 size_t layer_prev_type = m_layers[i-1][0]->get_type();
//                 get current layer type
//                 size_t layer_type = m_layers[i][j]->get_type();
// 
//                 forward values
//                 std::vector<std::vector<float>> values;
//                 if(layer_prev_type == Layer::CONV){
//                     calculate values of filters so we can pool them
//                     FIXME static 0 for now
//                     
//                     we know how much layers are in current slice
//                     size_t num_current_slice_size = m_layers[i].size();
//                     divide by num_filters -1 because vectors start at 0
//                     size_t num_prev_layer = num_current_slice_size/m_num_filters -1;
//                     filters of one layer
//                     std::vector<Filter> filters = m_layers[i-1][num_prev_layer]->get_filters();
//                     get corresponding filter to current layer
//                     Layer output = filters[j%m_num_filters].calculate_output(*m_layers[i-1][0]);
//                     
//                     values = output.get_values();
//                 }else{
//                     values = m_layers[i-1][j]->get_values();
//                 }
//                 
//                 if(m_test){
//                     std::cout << "Size: i = " << i << " j = " << j << ": " << m_layers[i].size() << std::endl;
//                     std::cout << "Layertype before: " << m_layers[i-1][0]->get_type() << std::endl;
//                     std::cout << "Layertype current: " << m_layers[i][j]->get_type() << std::endl;
//                     
//                     for(int n = 0;n < m_layers[i][j]->get_values().size();n++){
//                         for(int m = 0;m < m_layers[i][j]->get_values()[0].size();m++){
//                             std::cout << m_layers[i][j]->get_values()[n][m];
//                         }
//                         std::cout << std::endl;
//                     }
//                     std::cout << "Amount of layers before: " << m_layers[i-1].size() << std::endl;
//                     std::cout << "Layertype: " << m_layers[i][j]->get_type() << std::endl;
//                     std::cout << "Size Values before: " << values.size() << "x" << values[0].size() << std::endl;
//                     std::cout << std::endl;
//                 }
//                 
//                 switch(layer_type){
//                     case Layer::CONV:{
//                         
//                         m_layers[i][j]->set_values(values);
//                         m_layers[i][j]->make_padding();
//                         
//                         break;
//                     }
//                     case Layer::POOL:{
//                         
//                      TODO add padding if input width or height cant be divided by m_stride
//         while(t_input.size() % m_stride != 0){
//             std::cout << "Adding zero padding, consider changing stride or other hyperparameters" << std::endl;
//             std::vector<float> padding;
//             for(size_t i = 0;i < t_input[0].size();i++){
//                 padding.push_back(0);
//             }
//             t_input.push_back(padding);
//         }
//         while(t_input[0].size() % m_stride != 0){
//             std::cout << "Adding zero padding, consider changing stride or other hyperparameters" << std::endl;
//             for(size_t i = 0;i < t_input[0].size();i++){
//                 t_input[i].push_back(0);
//             }
//         }
//                         
//                         m_layers[i][j]->pool(values);
//                         
//                         break;
//                     }
//                     case Layer::ACT:{
//                         see pool case
//                         m_layers[i][j]->calculate(values);
//                         break;
//                     }
//                     case Layer::CONNECTED:{
//                         std::vector<std::vector<float>> vec_connected_inputs;
//                         vec_connected_inputs.push_back(*new std::vector<float>);
//                         
//                         add all outputs to 1d vector
//                         for(size_t layer = 0;layer < m_layers[i-1].size();layer++){
//                             std::vector<float> layer_1d;
//                             for(size_t width = 0;width < m_layers[i-1][layer]->get_values().size();width++){
//                                 for(size_t height = 0;height < m_layers[i-1][layer]->get_values()[width].size();height++){
//                                     float value = m_layers[i-1][layer]->get_values()[width][height];
//                                     layer_1d.push_back(value);
//                                 }
//                             }
//                             insert all prev values to connected layer
//                             vec_connected_inputs[0].insert(vec_connected_inputs[0].begin(),layer_1d.begin(),layer_1d.end());
//                         }
//                         
//                         m_layers[i][j]->set_values(vec_connected_inputs);
//                         m_layers[i][j]->forward();
//                         
//                         make training pair with 1d vector and labels
//                         std::pair<std::vector<float>,std::vector<float>> training_pair;
// 
//                         std::vector<float> training_vector;
//                         for(size_t num_layers = 0;num_layers < m_layers[i-1].size();num_layers++){
//                             for(size_t width = 0;width < m_layers[i-1][num_layers]->get_height();width++){
//                                 for(size_t height = 0;height < m_layers[i-1][num_layers]->get_width();height++){
//                                     training_vector.push_back(m_layers[i-1][num_layers]->get_values()[width][height]);
//                                 }
//                             }
//                         }
// 
//                         training_pair = std::make_pair(training_vector,t_data.second);
//                         std::vector<float> deltas_connected = m_layers[i][j]->train(training_pair,m_learning_rate,m_momentum);
//                         
//                         output deltas and output values
//                         return std::make_pair(m_layers[i][j]->get_net_output(),deltas_connected);
//                     }
//                 }
//             }
//         }
//     }
//     return std::make_pair(std::vector<float>(),std::vector<float>());
// }
// 
// std::vector<float> Convolutional::initialize(struct_training_data &t_data){
//     std::vector<float> outputs;
//     std::vector<std::thread> forward_threads;
// 
//     TODO only initialize layers the first feed forward iteration
//     go through each "slice" of the cnn example: input, conv conv , pool pool , output
//     for(size_t i = 0;i < m_layers.size();i++){
//         connected layer vals
//         std::vector<std::vector<float>> conn_values;
//         conn_values.push_back(std::vector<float>());
//         for(size_t j = 0;j < m_layers[i].size();j++){
//             
//             initialize array view because it has no constructor :(
//             get height and width of input
//             size_t num_input_width = t_data.image_data[j].size();
//             size_t num_input_height = t_data.image_data[j][0].size();
//             array_t::array_view<2>::type values = t_data.image_data[indices[j][range_t(0,num_input_width)][range_t(0,num_input_height)]];
//             
//             if(i == 0){
//                 get channel of input
//                 array_t::array_view<2>::type input_pane = t_data.image_data[indices[j][range_t(0,num_input_width)][range_t(0,num_input_height)]];
//                 
//                 set input values
//                 Layer* layer = new Layer(num_input_width,num_input_height,1,Layer::INPUT);
//                 layer->set_values(input_pane);
//                 m_layers[i][j] = layer;
//             }else{
//                 get previous type, index 0 bc all vertical layer should be the same
//                 size_t layer_prev_type = m_layers[i-1][0]->get_type();
//                 get current type
//                 size_t layer_type = m_layers[i][j]->get_type();
// 
//                 if(m_layers[i-1][0]->get_type() == Layer::CONV){
//                     values = m_layers[i-1][0]->get_values();
//                 }else {
//                     values = m_layers[i-1][j]->get_values();
//                 }
// 
//                if(m_test){
//                    std::cout << "Size: i = " << i << ": " << m_layers[i].size() << std::endl;
//                    std::cout << "Layertype: " << m_layers[i][j]->get_type() << std::endl;
//                    std::cout << "Layertype before: " << m_layers[i-1][0]->get_type() << std::endl;
//                    std::cout << "Size Values before: " << values.size() << "x" << values[0].size() << std::endl;
//                }
// 
//                 switch(layer_type){
//                     case Layer::CONV:{
//                         ConvolutionLayer* conv_layer = new ConvolutionLayer(values.size(),values[0].size(),1,m_zero_padding,m_stride_filters,m_num_filters,m_filters_size,m_function_type);
//                         conv_layer->set_values(values);
//                         conv_layer->make_padding();
//                         m_layers[i][j] = conv_layer;
//                         break;
//                     }
//                     case Layer::POOL:{
//                         push back as much as we need if prev layer is conv layer
//                         
//                         PoolLayer* pool_layer = new PoolLayer(m_pools_size,m_pools_size,1,m_stride_pool);
//                         pool_layer->pool(values);
//                         m_layers[i][j] = pool_layer;
//                         break;
//                     }
//                     case Layer::ACT:{
//                         see pool case
//                         ActivationLayer* act_layer = new ActivationLayer(values.size(),values[0].size(),1,m_function_type);
//                         act_layer->calculate(values);
//                         m_layers[i][j] = act_layer;
//                         break;
//                     }
//                     this does not work here because the connected layer train function need to be called up there
//                     just init this layer here
//                     case Layer::CONNECTED:{
//                         std::vector<float> layer_1d;
//                         for(size_t num_layers = 0;num_layers < m_layers[i-1].size();num_layers++){
//                             for(size_t width = 0;width < m_layers[i-1][num_layers]->get_height();width++){
//                                 for(size_t height = 0;height < m_layers[i-1][num_layers]->get_width();height++){
//                                     values = m_layers[i-1][num_layers]->get_values();
//                                     float value = values[width][height];
//                                     layer_1d.push_back(value);
//                                 }
//                             }
//                         }
//                         values should be empty if connected layer
//                         conn_values.insert(conn_values.begin(),layer_1d);
//                         ConnectedLayer* conn_layer = new ConnectedLayer(m_function_type,{values[0].size(),30,10},true);
//                         conn_layer->set_values(values);
//                        std::pair<std::vector<float>,std::vector<float>> conn_training_data = std::make_pair(values[0],t_data.second);
//                        has to be called before get_net_output
//                        conn_layer->forward();
//                         m_layers[i][j] = conn_layer;
//                         insert outputs from prev layers into outputs, connected layer is trained in Convolutional::train();
//                         outputs.insert(outputs.end(),values[0].begin(),values[0].end());
//                     }
//                 }
//             }
//             if last layer is reached
//            if(i == m_layers.size()-1){
//                moved up to layer::connected
//            }
//         }
//     }
// 
//     return outputs;
// }

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
    long input_size = 9;
    size_t depth = 1;

    size_t filter_size = 2;
    size_t filter_stride = 1;
    size_t num_filters = 2;

    size_t pool_size = 3;
    size_t pool_stride = 1;

    boost::array<array_t::index, 3> shape_array_images = {{ 3, input_size, input_size }};
    array_t inputs(shape_array_images);
    
    //simulate rgb channel
    for(size_t i = 0;i < depth;i++){
        for(size_t j = 0;j<input_size;j++){
            std::vector<float> column;
            for(size_t k = 0;k<input_size;k++){
                inputs[i][j][k] = (float) k + i + j;
                std::cout << " " << k + i + j;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    array_t::array_view<2>::type input_slice = inputs[indices[0][range_t(0,input_size)][range_t(0,input_size)]];

    /*
     * Test for Convolutional Layer
     */
    std::cout << "Test Convolutional Layer / zero padding: " << std::endl << std::endl;

    std::vector<ConvolutionLayer> conv_layers;
    for(size_t i = 0;i < inputs.size();i++){

        ConvolutionLayer conv = ConvolutionLayer(inputs[i],1,filter_stride,num_filters,filter_size,m_function_type);
        conv.set_values(inputs[0]);
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
                Layer::array_2d_t output = filter.calculate_output(architecture[i][j]);
                std::cout << output.size() << "x" << output[0].size() << std::endl;
                for(size_t width = 0;width < output.size();width++){
                    for(size_t height = 0;height < output[0].size();height++){
                        std::cout << output[width][height] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl << std::endl;
    }

    std::cout << std::endl;

    /*
     * Test for PoolLayer
     */
    std::cout << "Test Pool Layer: " << std::endl;

    //Filters were replaced by standartized input values, leave this, to generate more inputs for connected layer
    for(size_t i = 0; i < architecture.size();i++){

        for(size_t n = 0;n < architecture[i].size();n++) {
            std::vector<Filter> filters = architecture[i][n].get_filters();
            if(filters.size() == 0){
                break;
            }
            
            for(Filter filter:filters){
                
                PoolLayer pool = PoolLayer(inputs[0],pool_stride);
                
                std::cout << std::endl;
                for(size_t j = 0;j<inputs[0].size();j++){
                    for(size_t k = 0;k < inputs[0][0].size();k++){
                        std::cout << " " << inputs[0][j][k];
                    }
                    std::cout << std::endl;
                }
                
                pool.pool(inputs[0]);

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
//     std::cout << "Test connected layer: " << std::endl << std::endl;
// 
//     std::vector<float> connected_inputs;
// 
//     for(PoolLayer pool:last_layers){
//         make all pool outputs to a 1d vector
//         for(size_t i = 0;i<pool.get_values().size();i++){
//             for(size_t j = 0;j < pool.get_values()[0].size();j++){
//                 connected_inputs.push_back(pool.get_values()[i][j]);
//             }
//         }
//     }
//     superclass layer takes 2d inputs, we only need 1d input
//     std::vector<std::vector<float>> dim;
//     dim.push_back(connected_inputs);
//     
//     std::cout << "Inputs: " << connected_inputs.size() << std::endl;
//     add neurons manually
//     std::vector<size_t> sizes;
//     sizes.push_back(connected_inputs.size());
//     sizes.push_back(30);
//     sizes.push_back(20);
//     sizes.push_back(10);
// 
//     test raw and normalized output
//     for(size_t i = 0;i < 2;i++){
//         if(i == 0){
//             std::cout << "Raw Output: " << std::endl;
//         }else{
//             std::cout << "Normalized Output: " << std::endl;
//         }
//         ConnectedLayer connected = ConnectedLayer(ConnectedLayer::functiontype::SWISH,sizes,i);
//         connected.set_values(dim);
//         connected.forward();
//         std::vector<float> outputs = connected.get_net_output();
//         for(size_t j = 0;j < outputs.size();j++){
//             std::cout << j << ":" << outputs[j] << std::endl;
//         }
//         std::cout << std::endl;
//     }
// 
//     std::cout << std::endl << std::endl;

    /*
     * Test training
     */
    std::cout << "Testing training: " << std::endl;
    
    boost::array<array_t::index, 3> shape_array_training_images = {{ 3, 32, 32 }};
    array_t image_array(shape_array_training_images);
    
    auto dataset = cifar::read_dataset<std::vector,std::vector,uint8_t,size_t>(2,1);
    
    for(size_t channel = 0;channel < 3;channel++){
        for(size_t i = 0;i < 32;i++){
            for(size_t j = 0;j < 32;j++){
                image_array[channel][i][j] = (float) dataset.training_images[0][(i * 32 * 32) + (j * 32) + channel] / 255;
            }
        }
    }
    
    std::vector<struct_training_data> test_data;
    
    struct_training_data test;
    test.image_data.resize(boost::extents[3][32][32]);
    test.image_data = image_array;
    test.image_label = dataset.training_labels[0];
    
    test_data.push_back(test);
    
    std::cout << "Architecture: " << std::endl;
    for(size_t i = 0;i < m_layers.size();i++){
        std::cout << m_layers[i].size() << "* Type: " << m_layers[i][0]->get_type() << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    //initialize with sample of training data
    
    //train
    //this->train(test_data,SWISH,0.02,0.05,1,1,2);
    
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

