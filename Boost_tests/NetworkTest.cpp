//
// Created by DerScherenemann on 03.05.22.
//
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "convnetwork.h"

struct ConvNetworkFixture {

    ConvNetworkFixture()  {
        /*
         * Create our network
         */
        size_t num_repeats = 1;
        std::vector<size_t> num_filters = {5,5};
        std::vector<size_t> num_filter_size = {5,5};
        size_t num_pool_size = 2;
        size_t num_zero_padding = 1;

        convNetwork = new Convolutional({30,10,4},{Layer::types::CONV,Layer::types::CONV},num_filters,num_filter_size,num_pool_size,num_zero_padding);
    }

    ~ConvNetworkFixture() {
        delete convNetwork;
    }

    Convolutional* convNetwork;

};

BOOST_FIXTURE_TEST_SUITE(NetworkSuite,ConvNetworkFixture)

    BOOST_AUTO_TEST_CASE(Training)
    {
        /*
         * Create sample input with pattern to recognize
         *
         * Generate simple cross consisting of 1's in a background of 0's
         */
        std::vector<Convolutional::struct_training_data> dummy_training;
        boost::array<Convolutional::array_3f::index, 3> shape_array_images = {{3, 20, 20}};
        Convolutional::array_3f inputs(shape_array_images);

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist20(3, 17);
        std::uniform_int_distribution<std::mt19937::result_type> dist100(0, 99);

        for (size_t amount_dummy_training_images = 0;amount_dummy_training_images < 1000; amount_dummy_training_images++) {

            // Simulate rgb channels
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 20; j++) {
                    for (size_t k = 0; k < 20; k++) {
                        inputs[i][j][k] = 0;
                    }
                }
            }

            size_t cross_positionx = dist20(rng);
            size_t cross_positiony = dist20(rng);

            for (size_t i = 0; i < 3; i++) {
                inputs[i][cross_positionx][cross_positiony] = 1;
                inputs[i][cross_positionx + 1][cross_positiony] = 1;
                inputs[i][cross_positionx + 2][cross_positiony] = 1;
                inputs[i][cross_positionx - 1][cross_positiony] = 1;
                inputs[i][cross_positionx - 2][cross_positiony] = 1;
                inputs[i][cross_positionx][cross_positiony + 1] = 1;
                inputs[i][cross_positionx][cross_positiony + 2] = 1;
                inputs[i][cross_positionx][cross_positiony - 1] = 1;
                inputs[i][cross_positionx][cross_positiony - 2] = 1;
            }

            Convolutional::struct_training_data test;
            test.image_data.resize(boost::extents[3][20][20]);
            test.image_data = inputs;
            test.corrrect_outputs = {0, 0, 0, 0};

            // Let the Network recognize the corresponding quadrant
            if (cross_positionx > 10) {
                if (cross_positiony > 10) {
                    test.corrrect_outputs[3] = 1;
                } else {
                    test.corrrect_outputs[0] = 1;
                }
            } else {
                if (cross_positiony > 10) {
                    test.corrrect_outputs[2] = 1;
                } else {
                    test.corrrect_outputs[1] = 1;
                }
            }

            dummy_training.push_back(test);
        }

        //train
        // TODO delta and gradient calculation not working
        convNetwork->train(dummy_training, Convolutional::SIGMOID, 0.05, 0.1, 100, 1, 1);

        BOOST_CHECK_GE(0.5, convNetwork->getError());
    }

BOOST_AUTO_TEST_SUITE_END()