cmake_minimum_required(VERSION 3.0)

project(cnn)

set(SOURCE_FILES main.cpp convnetwork.cpp ActivationLayer.cpp ConnectedLayer.cpp layer.cpp filter.cpp PoolLayer.cpp ConvolutionLayer.cpp)

include_directories(lib)

link_directories(lib)

add_subdirectory(lib)

add_executable( ${PROJECT_NAME} ${SOURCE_FILES} )

#links lib
target_link_libraries( ${PROJECT_NAME} knn)

install(TARGETS cnn RUNTIME DESTINATION bin)
