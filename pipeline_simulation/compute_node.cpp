#include "systemAPI.h"
#include <type_traits>
#include <stdlib.h>


int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cerr << "Wrong number of arguments" << std::endl;
        return 1;
    }
    int myID= atoi(argv[1]), next_node;
    systemAPI sys_(false, myID);
    Task next_task;
    torch::Tensor tmp;

    // wait for init refactoring
    auto refactor_message = sys_.my_network_layer.check_new_refactor_task();
    std::cout << "here" << std::endl;
    sys_.refactor(refactor_message);

    // ----------- DRAFT ------------
    //sys_.init_state(vgg, vgg_model::v11, 10, 1, 3);
    //sys_.init_state_vector(resnet, resnet_model::resnet18, 10, 9, 13);
    //auto layers = sys_.clients_state.find(1)->second.layers;
    //std::cout << sys_.clients_state.find(1)->second.client_id << std::endl;
    //for (int i = 0; i< layers.size(); i++) {
    //    std::cout << "new layer: "<< i+1 << " "<< layers[i] << std::endl;
    //}
    // -------- DRAFT ------------
    
    while (true) {
        next_task = sys_.my_network_layer.check_new_task();

        // check if it is a refactoring task
        // else ...

        auto task = sys_.exec(next_task, tmp);
        if (task.type != noOp) {
            next_node = (next_task.type == forward_) ? sys_.inference_path[0] : sys_.inference_path[1];
            sys_.my_network_layer.new_message(task, next_node);
        }
    }
    
}