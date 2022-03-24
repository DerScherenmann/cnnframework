#define BOOST_TEST_MODULE LayerTests
#include <boost/test/unit_test.hpp>

int add(int i, int j) { return i+j; }

BOOST_AUTO_TEST_CASE(TestConvolutionalLayer)
{
    BOOST_REQUIRE( add( 3,2 ) == 4 );      // #2 throws on error
}
