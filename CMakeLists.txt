cmake_minimum_required(VERSION 3.0)

project(convnetwork)

add_subdirectory(src)

enable_testing()
add_subdirectory(Boost_tests)

add_executable( ${PROJECT_NAME} main.cpp)
#links lib
target_link_libraries( ${PROJECT_NAME} knn cnn)
