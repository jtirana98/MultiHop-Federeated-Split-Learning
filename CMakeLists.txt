cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(datasets/)
include_directories(utils/)
include_directories(models/vgg/)
include_directories(models/resnet/)
#set (ARCHS armv7 armv7s)
#set(VALID_ARCHS armv6 armv7 armv7s arm64)
set(SOURCE_FILES main.cpp
                datasets/mydataset.h
                datasets/mydataset.cpp
                utils/mylogging.h
                utils/mylogging.cpp
                utils/split_training.h
                utils/split_training.cpp
                datasets/transform.h
                datasets/transform.cpp
                models/vgg/vgg.cpp
                models/vgg/vgg.h
                models/vgg/vgg_help.h
                models/vgg/vgg_help.cpp
                models/vgg/vgg_train.h
                models/vgg/vgg_train.cpp
                models/resnet/resBlock.h
                models/resnet/resBlock.cpp
                models/resnet/resBottleneckBlock.h
                models/resnet/resBottleneckBlock.cpp
                models/resnet/resnet.h
                models/resnet/resnet_split.h
                models/resnet/resnet_split.cpp
                models/resnet/resnet_train.h
                models/resnet/resnet_train.cpp
                )

add_executable("${CMAKE_PROJECT_NAME}" ${SOURCE_FILES})
target_link_libraries("${CMAKE_PROJECT_NAME}" ${TORCH_LIBRARIES})

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 14)