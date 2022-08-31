#include "systemAPI.h"
#include <type_traits>
#include <stdlib.h>
#include <thread>
#include <argparse/argparse.hpp> //https://github.com/p-ranav/argparse

int main(int argc, char **argv) {
    argparse::ArgumentParser program("compute_node");

    program.add_argument("-i", "--id")
        .help("The node's id")
        .required()
        .nargs(1)
        .scan<'i', int>();

    program.add_argument("-l", "--log_directory")
        .help("The directory name to store looging info")
        .default_value(std::string("main_experiment"))
        .nargs(1);

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto myID = program.get<int>("-i");
    auto log_dir = program.get<std::string>("-l");

    systemAPI sys_(false, myID, log_dir);
    Task next_task;
    int next_node;
    torch::Tensor tmp;

    // POINT 10 Initialization phase: do/cn waiting for refactor message
    sys_.my_network_layer.newPoint(INIT_WAIT_FOR_REFACTOR);
    
    // wait for init refactoring
    auto refactor_message = sys_.my_network_layer.check_new_refactor_task();
    
    // POINT 11 Initialization phase: do/cn end waiting for refactor message
    sys_.my_network_layer.newPoint(INIT_END_W_REFACTOR);

    sys_.refactor(refactor_message);
    

    // ----------- TEST ------------
    //sys_.init_state(vgg, vgg_model::v11, 10, 1, 3);
    //sys_.init_state_vector(resnet, resnet_model::resnet18, 10, 9, 13);
    /*
    auto layers = sys_.clients_state.find(0)->second.layers;
    std::cout << sys_.clients_state.find(1)->second.client_id << std::endl;
    for (int i = 0; i< layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< layers[i] << std::endl;
    }
    */
    // -------- TEST ------------

    // POINT 12 Initialization phase: completed
    sys_.my_network_layer.newPoint(INIT_END_INIT);

    while (true) {
        // POINT 13 Execution phase: CN waits for new task
        sys_.my_network_layer.newPoint(CN_START_WAIT);
        next_task = sys_.my_network_layer.check_new_task();
        // POINT 14 Execution phase: CN starts executing task
        auto point1 = sys_.my_network_layer.newPoint(CN_START_EXEC);

        auto task = sys_.exec(next_task, tmp); 

        if (task.type != noOp) {
            bool keep_connection = true;
            next_node = (next_task.type == forward_) ? sys_.inference_path[0] : sys_.inference_path[1];
            if (next_node == -1) {
                keep_connection = false;
                next_node = task.client_id;
            }
            sys_.my_network_layer.new_message(task, next_node, keep_connection);
        }

        // POINT 15 Execution phase: CN completed a task
        auto point2 = sys_.my_network_layer.newPoint(CN_END_EXEC);
        // 14 - 15 interval

        interval_type type_;
        switch (next_task.type) {
        case forward_:
            type_ = fwd_only;
            break;
        case backward_:
            type_ = bwd_only;
            break;
        case optimize_:
            type_ = opz_only;
            break;
        default:
            break;
        }
        //std::cout << "ok!" << std::endl;
        sys_.my_network_layer.mylogger.add_interval(point1, point2, type_);
    }
}