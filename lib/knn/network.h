#ifndef NETWORK_H
#define NETWORK_H
#pragma once
#include <vector>
#include <map>
#include <string>
#include "mathhelper.h"
#include "mathhelper.cpp"
#include "neuron.h"
#include "neuron.cpp"
class Network
{
public:

    /*
     * Constructor
     */
    Network(std::vector<std::pair<int,int>> sizes);

    /*
     *  Functions
     */
    int train(std::vector<std::pair<std::vector<float>, std::vector<float>>> &trainingData, float learningRate, float momentum, int epochs);
    std::vector<float> predict(std::vector<float>& testData);
    int highestPred(std::vector<float> &outputNeurons);
    std::vector<Neuron> feedForward(std::vector<float>& testData);
    int save(std::string filename);
    int load(std::string filename);

    /*
     *  Variables
     */
    int numLayers;
    float learningRate;
    float momentum;
    //Network sizes
    std::vector<std::pair<int,int>> sizes;

private:
    /*
     *  Functions
     */
    float backProp(float &delta, float &activation);
    float backPropMomentum(float &deltaCurrent, float &activationBefore, float &oldChange);
    float calcMSE(std::vector<std::pair<std::vector<float>, std::vector<float>>> &trainingData, std::vector<std::vector<Neuron>> &outputNeurons);
    float calcRMSE(std::vector<std::pair<std::vector<float>, std::vector<float>>> &trainingData, std::vector<std::vector<Neuron>> &outputNeurons);

    /*
     *  Variables
     */
    //Layer -> Neuron -> Weights (holds weights pointing into neuron)
    std::vector<std::vector<std::vector<float>>> weights;
    //Layer -> Neuron -> change
    std::vector<std::vector<std::vector<float>>> oldchange;
    //Layer -> Neuron -> class values
    std::vector<std::vector<Neuron>> neuronLayers;
};
#endif
