cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(training)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(SOURCE_FILES training.cpp
                 mydataset.h
                 mydataset.cpp
                 mylogging.h
                 mylogging.cpp
                 transform.h
                 transform.cpp
                 vgg.h
                 vgg.cpp
                 vgg_help.h
                 vgg_help.cpp
                 )

add_executable("${CMAKE_PROJECT_NAME}" ${SOURCE_FILES})
target_link_libraries("${CMAKE_PROJECT_NAME}" ${TORCH_LIBRARIES})
#add_executable(split split.cpp)
#target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 14)