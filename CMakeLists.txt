cmake_minimum_required(VERSION 3.0)

project(cnnkdev)

add_library(network_lib STATIC lib/knn/network.cpp)
add_executable(cnnkdev main.cpp)

target_link_libraries(cnnkdev network_lib)

install(TARGETS cnnkdev RUNTIME DESTINATION bin)
