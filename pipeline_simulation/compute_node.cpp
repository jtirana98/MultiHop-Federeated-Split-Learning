#include "systemAPI.h"
#include <type_traits>
#include <stdlib.h>
#include <thread>
#include <argparse/argparse.hpp> //https://github.com/p-ranav/argparse

class ComputeNode {
 public:
    int myID;
    systemAPI &sys_;

    ComputeNode(int myID, systemAPI &sys_) :myID(myID), sys_(sys_) {}
    void task_operator(/*operation op*/); // this will be the thread
};

void ComputeNode::task_operator(/*operation op*/) {
    Task next_task;
    int next_node;
    torch::Tensor tmp;

    std::string my_operation;
    /*switch (op) {
    case forward_:
        my_operation = "forward";
        break;
    case backward_:
        my_operation = "backward";
        break;
    default:
        break;  
    }*/

    //std::cout << "The " << my_operation << " thread started!" << std::endl;

    int g_c = 0;
    while (true) {
        auto timestamp1 = std::chrono::steady_clock::now();
        next_task = sys_.my_network_layer.check_new_task(/*op==operation::backward_*/); // TODO: CHANGE THAT
        auto timestamp2 = std::chrono::steady_clock::now();
        auto _time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (timestamp2 - timestamp1).count();
        std::cout << "Waiting... " << _time << std::endl;

        interval_type type_;
        std::string operation_;
        switch (next_task.type) {
        case forward_:
            type_ = fwd_only;
            operation_ = "forward";
            break;
        case backward_:
            type_ = bwd_only;
            operation_ = "backward";
            break;
        case optimize_:
            type_ = opz_only;
            operation_ = "optimization";
            break;
        default:
            break;
        }

        //std::cout << operation << std::endl;
        auto timestamp1_ = std::chrono::steady_clock::now();
        auto task = sys_.exec(next_task, tmp);
        auto timestamp2_ = std::chrono::steady_clock::now();
        auto __time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (timestamp2_ - timestamp1_).count();
        //if(g_c % 50 == 0)
        std::cout << "Exec "  << operation_ << " " << __time  << " client " << next_task.client_id << std::endl;

        if (task.type != noOp) {
            bool keep_connection = true;
            next_node = (next_task.type == forward_) ? sys_.inference_path[0] : sys_.inference_path[1];
            if (next_node == -1) {
                keep_connection = false;
                next_node = task.client_id;
            }
            sys_.my_network_layer.new_message(task, next_node, keep_connection); // TODO: CHECK
        }
        g_c++;
    }

} 

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
    std::cout << "myid " << myID << std::endl;
    auto log_dir = program.get<std::string>("-l");
    
    systemAPI sys_(false, myID, log_dir);
    sys_.my_network_layer.findInit();
    auto refactor_message = sys_.my_network_layer.check_new_refactor_task();
    sys_.refactor(refactor_message);
    ComputeNode the_cnode = ComputeNode(myID, sys_);

    // ----------- TEST ------------
    //sys_.init_state(vgg, vgg_model::v11, 10, 1, 3);
    //sys_.init_state_vector(resnet, resnet_model::resnet18, 10, 9, 13);
    
    /*auto layers = sys_.clients_state.find(0)->second.layers;
    std::cout << sys_.clients_state.find(1)->second.client_id << std::endl;
    for (int i = 0; i< layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< layers[i] << std::endl;
    }*/
    
    // -------- TEST ------------

    the_cnode.task_operator();
    

}