set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package (Boost REQUIRED COMPONENTS unit_test_framework)
# TODO ../ could cause problems
include_directories (../src ${Boost_INCLUDE_DIRS})

add_executable (Boost_Tests_run PoolLayerTests.cpp NetworkTest.cpp)
target_link_libraries (Boost_Tests_run ${Boost_LIBRARIES})
target_link_libraries (Boost_Tests_run knn cnn)