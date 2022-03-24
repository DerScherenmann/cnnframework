#ifndef NEURON_H
#define NEURON_H
#pragma once
#include <vector>
#include "mathhelper.h"
#include <iostream>

class Neuron
{
	public:
		
		//activation is a number held by the Neuron
		Neuron(float activation,int type,int mode) {
			this->activation = activation;
			this->type = type;
			this->mode = mode;
			//set activation to 1 if bias
			if (type == BIAS) activation = 1;
		}

		//neurons holds neurons of the layer before and therefore their activation, weights are the weights to this neuron
		void calculateActivation(std::vector<Neuron> &neurons, std::vector<float> &weights) {
		    //skip this if neuron is bias

		    if (type == NEURON) {
		        activation = 0;
		        for (int i = 0; i < neurons.size(); i++) {
		            activation += neurons[i].activation * weights[i];
		        }
		        this->sum = activation;
		        this->activation = act(activation);
		        //std::cout << "Activation: " << activation << std::endl;
		    }
		}

		float act(float x){

			if(mode == SWISH){
				x = math.swish(x);
			}
			if(mode == SIGMOID){
				x = math.sigmoid(x);
			}
			return x;
		}

		float actPrime(float x){

			if(mode == SWISH){
				x = math.swishPrime(x);
			}
			if(mode == SIGMOID){
				x = math.sigmoidPrime(x);
			}

			return x;
		}
		
		float activation = 0;
		float sum = 0;
		float delta = 0;
		int type = 1;
		int bias = 1;
		enum types
		{
			BIAS = 0,NEURON
		};
		enum modes{
			SWISH = 0,SIGMOID,RELU
		};
		
		int mode = 0;
    private:
        Math math;
};
#endif
