cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread -pedantic -pthread")

include_directories(datasets/)
include_directories(utils/)
include_directories(models/)
include_directories(models/vgg/)
include_directories(models/resnet/)
include_directories(models/lenet5/)
include_directories(pipeline_simulation/)
include_directories(pipeline_simulation/profiling/)
include_directories(third_party/argparse/include/)


set(SOURCE_FILES 
                datasets/mydataset.h
                datasets/mydataset.cpp
                utils/mylogging.h
                utils/mylogging.cpp
                utils/pipeline_logging.h
                utils/pipeline_logging.cpp
                utils/split_training.h
                utils/split_training.cpp
                datasets/transform.h
                datasets/transform.cpp
                models/models.h
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
                models/lenet5/lenet.cpp
                models/lenet5/lenet.h
                models/lenet5/lenet_help.h
                models/lenet5/lenet_help.cpp
                models/lenet5/lenet_train.h
                models/lenet5/lenet_train.cpp
                pipeline_simulation/network_layer.h
                pipeline_simulation/network_layer.cpp
                pipeline_simulation/Task.h
                pipeline_simulation/State.h
                pipeline_simulation/systemAPI.h
                pipeline_simulation/systemAPI.cpp
                pipeline_simulation/profiling/rpi_stats.h
)

add_executable("${CMAKE_PROJECT_NAME}"  ${SOURCE_FILES} main.cpp)
#add_executable(compute_node  ${SOURCE_FILES} pipeline_simulation/compute_node.cpp)
#add_executable(data_owner  ${SOURCE_FILES} pipeline_simulation/data_owner.cpp)
#add_executable(profile_data_owner  ${SOURCE_FILES} pipeline_simulation/profiling/data_owner_saturation.cpp)
#add_executable(simulated_data_owner  ${SOURCE_FILES} pipeline_simulation/profiling/data_owner_simulated.cpp)
#add_executable(aggregator  ${SOURCE_FILES} pipeline_simulation/aggregator.cpp)
#add_library(MyLib SHARED ${SOURCE_FILES})

target_link_libraries("${CMAKE_PROJECT_NAME}" ${TORCH_LIBRARIES})
#target_link_libraries(compute_node ${TORCH_LIBRARIES})
#target_link_libraries(data_owner ${TORCH_LIBRARIES})
#target_link_libraries(profile_data_owner ${TORCH_LIBRARIES})
#target_link_libraries(simulated_data_owner ${TORCH_LIBRARIES})
#target_link_libraries(aggregator ${TORCH_LIBRARIES})

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)
#set_property(TARGET compute_node PROPERTY CXX_STANDARD 17)
#set_property(TARGET data_owner PROPERTY CXX_STANDARD 17)
#set_property(TARGET profile_data_owner PROPERTY CXX_STANDARD 17)
#set_property(TARGET simulated_data_owner PROPERTY CXX_STANDARD 17)
#set_property(TARGET aggregator PROPERTY CXX_STANDARD 17)
