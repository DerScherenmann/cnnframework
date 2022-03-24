cmake_minimum_required(VERSION 3.0)

add_subdirectory(lib)
add_library(cnn convnetwork.cpp convnetwork.h)

target_link_libraries(cnn PRIVATE knn)
