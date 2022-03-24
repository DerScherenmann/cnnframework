/*
 * File:   kernel.h
 * Author: DerScherenmann
 *
 * Created on 6 September 2020, 00:35
 */

#ifndef KERNEL_H
#define KERNEL_H

#include "lib/mathhelper.h"
#include "Layer.h"

#include <iostream>
#include <vector>

using namespace layer;

/**
 * Filters are stacked kernels but this class was repurposed because i am ...
 * https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
 */
class Kernel {
public:
    /**
    * @brief Initializes a Kernel with random values
    * @param width
    * @param height
    */
    Kernel(size_t t_width,size_t t_stride,size_t t_function_type) : m_function_type(t_function_type), m_width(t_width), m_height(t_width), m_stride(t_stride){
        Math math;
        m_center = (m_width+1)/2;

        m_values.resize(boost::extents[m_width][m_height]);
        for(size_t i = 0;i<m_width;i++){
            for(size_t j = 0;j < m_height;j++){
                m_values[i][j] = math.rng();
            }
        }

        m_old_changes.resize(boost::extents[m_width][m_height]);
        //zero old changes so first time is always 0
        for(size_t i = 0;i < t_width;i++){
            for(size_t j = 0;j < t_width;j++){
                m_old_changes[i][j] = 0;
            }
        }
        
        //initialize deltas
        m_deltas.resize(boost::extents[m_width][m_height]);
        for(size_t i = 0;i<m_width;i++){
            for(size_t j = 0;j < m_height;j++){
                m_deltas[i][j] = math.rng();
            }
        }
    }

    array_2f forward(array_2f t_input_values){

        size_t output_width = (t_input_values.size()-m_width)/m_stride +1;
        size_t output_height = (t_input_values[0].size()-m_height)/m_stride +1;

        boost::array<array_2f::index, 2> shape_output = {{ static_cast<long>(output_width),static_cast<long>(output_height) }};
        array_2f output_values(shape_output);
        
        for(size_t i = 0;i < output_width;i++){
            for(size_t j = 0;j < output_height;j++){
                float outer_sum = 0;
                for(size_t m = 0;m < m_width;m++){
                    float inner_sum = 0;
                    for(size_t n = 0;n < m_height;n++){
                        inner_sum += m_values[m][n] * t_input_values[i+m][j+n];
                    }
                    outer_sum += inner_sum;
                }
                // Moved to activation layer
                //output_values[i][j] = act(outer_sum);
                output_values[i][j] = outer_sum;
            }
        }
        


        return output_values;
    };

    /**
     * @brief Calculate deltas
     * @param array_2f array to calculate deltas for
     * @return array_2f deltas on success
     */
    array_2f calculate_deltas(array_2f t_previous_deltas){

        array_2f flipped_weights = flip(m_values);

        // Make padding for weights
        array_2f deltas_padding;
        deltas_padding.resize(boost::extents[t_previous_deltas.size()+2][t_previous_deltas[0].size()+2]);
        for(size_t i = 0;i < t_previous_deltas.size();i++){
            for(size_t j = 0;j < t_previous_deltas[0].size();j++){
                deltas_padding[i+1][j+1] = t_previous_deltas[i][j];
            }
        }

        // std::cout << "Prev_deltas: " << t_previous_deltas.size() << std::endl;
        // std::cout << "Deltas_padding: " << deltas_padding.size() << std::endl;

        // Essentially do a convolution operation; TODO find out if stride matters, we should use stride of 1 for now
        size_t deltas_width = (deltas_padding.size()-flipped_weights.size())/m_stride +1;
        size_t deltas_height = (deltas_padding[0].size()-flipped_weights[0].size())/m_stride +1;

        boost::array<array_2f::index, 2> shape_deltas = {{ (long) deltas_width,(long) deltas_height }};
        array_2f deltas_conv(shape_deltas);
        
        // Convolve deltas over weights; Add stride?
        for(size_t i = 0;i < deltas_width;i++){
            for(size_t j = 0;j < deltas_height;j++){
                float outer_sum = 0;
                for(size_t m = 0;m < flipped_weights.size();m++){
                    float inner_sum = 0;
                    for(size_t n = 0;n < flipped_weights[0].size();n++){
                        inner_sum += flipped_weights[m][n] * deltas_padding[i+m][j+n];
                    }
                    outer_sum += inner_sum;
                }
                // Moved to activation layer
                //output_values[i][j] = act(outer_sum);
                deltas_conv[i][j] = outer_sum;
            }
        }

        array_2f deltas;
        deltas.resize(boost::extents[deltas_conv.size()-2][deltas_conv[0].size()-2]);
        
        for(size_t i = 1;i < deltas_conv.size()-1;i++){
            for(size_t j = 1;j < deltas_conv[0].size()-1;j++){
                deltas[i-1][j-1] = deltas_conv[i][j];
                // std::cout << deltas_conv[i][j];
            }
            // std::cout << std::endl;
        }
        
        set_deltas(deltas);

        return deltas;
    }

    /**
     * @brief Calculate gradients
     * @param array_2f activation of layer before (values of activation layer)
     * @return gradients on success
     */
    array_2f calculate_gradient(array_2f t_layer_values){
        // std::cout << t_layer_values.size() << std::endl;
        // std::cout << m_deltas.size() << std::endl;
        array_2f flipped_deltas = flip(m_deltas);

        // Essentially do a convolution operation; TODO find out if stride matters, we should use stride of 1 for now
        size_t gradient_width = (t_layer_values.size()-flipped_deltas.size())/m_stride +1;
        size_t gradient_height = (t_layer_values[0].size()-flipped_deltas[0].size())/m_stride +1;

        boost::array<array_2f::index, 2> shape_gradients = {{ (long) gradient_width,(long) gradient_height }};
        array_2f gradients(shape_gradients);
        
        // Convolve deltas over weights; Add stride?
        for(size_t i = 0;i < gradient_width;i++){
            for(size_t j = 0;j < gradient_height;j++){
                float outer_sum = 0;
                for(size_t m = 0;m < flipped_deltas.size();m++){
                    float inner_sum = 0;
                    for(size_t n = 0;n < flipped_deltas[0].size();n++){
                        inner_sum += flipped_deltas[m][n] * t_layer_values[i+m][j+n];
                    }
                    outer_sum += inner_sum;
                }
                // Moved to activation layer
                //output_values[i][j] = act(outer_sum);
                gradients[i][j] = outer_sum;
            }
        }

        return gradients;
    }

    // // Change Weights
    // size_t change_weights(array_2f gradient_kernel,array_2f t_previous_layer){

    // }
    // float change_weight(float &deltaCurrent,float &activationBefore) {
	//     float weightChange =  -1* learningRate * deltaCurrent * activationBefore;

	//     return weightChange;
    // }
    // //backpropagation with momentum
    // float change_weight_momentum(float &deltaCurrent, float &activationBefore, float &oldChange) {
    // 	float weightChange = (1-momentum) * learningRate * deltaCurrent * activationBefore + momentum * oldChange;

    // 	return weightChange;
    // }
    float act(float x){
        if(m_function_type == Layer::SWISH){
            x = math.swish(x);
        }
        if(m_function_type == Layer::SIGMOID){
            x = math.sigmoid(x);
        }
        if(m_function_type == Layer::RELU){

        }
        return x;
    }
    float actPrime(float x){
        if(m_function_type == Layer::SWISH){
            x = math.swishPrime(x);
        }
        if(m_function_type == Layer::SIGMOID){
            x = math.sigmoidPrime(x);
        }
        if(m_function_type == Layer::RELU){

        }
        return x;
    }
    array_2f flip(array_2f t_input_values){
        array_2f temp;
        temp.resize(boost::extents[t_input_values.size()][t_input_values[0].size()]);
        //flip input
        for(size_t i = 0;i < t_input_values.size();i++){
            for(size_t j = 0;j < t_input_values[0].size();j++){
                float value = t_input_values[i][j];
                temp[t_input_values.size()-1-i][t_input_values[0].size()-1-j] = value;
            }
        }
        return temp;
    }
    size_t get_width(){
        return m_width;
    }
    size_t get_height(){
        return m_height;
    }
    size_t get_stride(){
        return m_stride;
    }
    array_2f get_weights(){
        return m_values;
    }
    array_2f get_old_changes(){
        return m_old_changes;
    }
    Layer* get_calculated_ouput(){
        return m_output_layer;
    }
    size_t get_output_width(size_t t_input_width){
        return (t_input_width-m_width)/m_stride +1;
    }
    size_t get_output_height(size_t t_input_height){
        return (t_input_height-m_width)/m_stride +1;
    }
    size_t set_old_changes(array_2f t_old_changes){
        m_old_changes.resize(boost::extents[t_old_changes.size()][t_old_changes[0].size()]);
        m_old_changes = t_old_changes;
        return 0;
    }
    array_2f get_deltas(){
        return m_deltas;
    }
    size_t set_deltas(array_2f t_deltas){
        m_deltas.resize(boost::extents[t_deltas.size()][t_deltas[0].size()]);
        m_deltas = t_deltas;
        return 0;
    }
    
private:
    size_t m_width;
    size_t m_height;
    size_t m_stride;
    size_t m_center;
    size_t m_function_type;
    Math math;
    //essentially the weights (compared to a knn)
    array_2f m_values;
    //sums without activation
    array_2f m_sums;
    //old changes
    array_2f m_old_changes;
    //deltas maybe not needed if we change and calculate in one go
    array_2f m_deltas;
    //last calculated output, shouldn't take much memory
    Layer* m_output_layer;
};

class Filter {
    public:
        typedef boost::multi_array<float, 3> array_3f;
        Filter(){};
        Filter(size_t t_depth,size_t t_width,size_t t_stride,size_t t_function_type) : m_depth(t_depth){
            for(size_t i = 0;i < m_depth;i++){
                m_kernels.push_back(new Kernel(t_width,t_stride,t_function_type));
            }
            m_bias = math.rand_bias(0,1);
        }

        // takes 3d volume, forwards and adds outputs together
        array_2f forward(array_3f t_inputs){
            array_2f output_total;

            // All Kernels should be the same
            size_t width_k = m_kernels[0]->get_width();
            size_t height_k = m_kernels[0]->get_height();
            size_t stride_k = m_kernels[0]->get_stride();

            size_t output_width = (t_inputs[0].size()-width_k)/stride_k +1;
            size_t output_height = (t_inputs[0][0].size()-height_k)/stride_k +1;
            
            output_total.resize(boost::extents[output_width][output_height]);

            // Switched m_kernels.size to t_inputs.size to ensure that m is always in array
            for(size_t m = 0;m < t_inputs.size();m++){
                Kernel* k = m_kernels[m];
                array_2f output_kernel = k->forward(t_inputs[m]);
                for(size_t i = 0;i < output_kernel.size();i++){
                    for(size_t j = 0;j < output_kernel[0].size();j++){
                        output_total[i][j] += output_kernel[i][j];
                    }
                }
            }
            // Offset by bias
            for(size_t i = 0;i < output_total.size();i++){
                for(size_t j = 0; j < output_total[0].size();j++){
                    output_total[i][j] += m_bias;
                }
            }
            return output_total;
        }
        // This doesn't work
        // Kernel* operator[] (int idx) const {
        //   return m_kernels[idx];
        // }
        size_t size(){
            return m_kernels.size();
        }
        size_t set_kernels(std::vector<Kernel*> &t_kernels){
            m_kernels = t_kernels;
            return 0;
        }
        std::vector<Kernel*> get_kernels(){
            return m_kernels;
        }
        size_t get_output_width(size_t t_input_width){
            return m_kernels[0]->get_output_width(t_input_width);
        }
        size_t get_output_height(size_t t_input_height){
            return m_kernels[0]->get_output_height(t_input_height);
        }
        size_t set_deltas(array_2f t_deltas){
            m_deltas.resize(boost::extents[t_deltas.size()][t_deltas[0].size()]);
            m_deltas = t_deltas;
            return 0;
        }
        array_2f get_deltas(){
            return m_deltas;
        }
        size_t get_depth(){
            return m_depth;
        }
        size_t get_width(){
            return m_kernels[0]->get_width();
        }
        size_t get_height(){
            return m_kernels[0]->get_height();
        }

    private:
        std::vector<Kernel*> m_kernels;
        size_t m_depth;
        size_t m_bias;
        Math math;
        array_2f m_deltas;
};

#endif /* KERNEL_H */
