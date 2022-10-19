//
// Created by DerScherenmann on 03.05.22.
//
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include "PoolLayer.h"

namespace utf = boost::unit_test;

struct PoolLayerFixture {

    array_2f inputs;
    size_t width = 10;
    size_t height = 10;

    PoolLayerFixture()  {
        inputs.resize(boost::extents[width][height]);
        // Sample data
        for(size_t j = 0;j < width;j++){
            std::vector<float> column;
            for(size_t k = 0;k < height;k++){
                inputs[j][k] = (float) k + j;
            }
        }

        poolLayerMax = new PoolLayer(inputs,2,2,2,Layer::operation::MAX);
        poolLayerAverage = new PoolLayer(inputs,2,2,2,Layer::operation::AVERAGE);
    }

    ~PoolLayerFixture() {
        delete poolLayerMax;
        delete poolLayerAverage;
    }

    PoolLayer * poolLayerMax;
    PoolLayer * poolLayerAverage;

};

BOOST_FIXTURE_TEST_SUITE(PoolLayerSuite,PoolLayerFixture)

    BOOST_AUTO_TEST_CASE(PoolMax, *utf::description("Test max pooling"))
    {
        BOOST_TEST_MESSAGE("Testing max pooling");

        poolLayerMax->pool(inputs);

        size_t output_width = (inputs.shape()[0]-2)/2+1;
        size_t output_height = (inputs.shape()[1]-2)/2+1;

        BOOST_REQUIRE_EQUAL(output_width,poolLayerMax->get_values().shape()[0]);
        BOOST_REQUIRE_EQUAL(output_height,poolLayerMax->get_values().shape()[1]);
    }

    BOOST_AUTO_TEST_CASE(PoolAverage, *utf::description("Test average pooling"))
    {
        BOOST_TEST_MESSAGE("Testing average pooling");

        poolLayerAverage->pool(inputs);

        size_t output_width = (inputs.shape()[0]-2)/2+1;
        size_t output_height = (inputs.shape()[1]-2)/2+1;

        BOOST_REQUIRE_EQUAL(output_width,poolLayerAverage->get_values().shape()[0]);
        BOOST_REQUIRE_EQUAL(output_height,poolLayerAverage->get_values().shape()[1]);
    }
BOOST_AUTO_TEST_SUITE_END()
