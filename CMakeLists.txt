cmake_minimum_required(VERSION 3.0)

project(convnetwork)

set(SOURCE_FILES main.cpp convnetwork.cpp)

add_subdirectory(lib)

add_executable( ${PROJECT_NAME} ${SOURCE_FILES} )

#links lib
target_link_libraries( ${PROJECT_NAME} knn)
#RUNTIME DESTINATION bin
#install(TARGETS convnetwork)
