#include "systemAPI.h"
#include <type_traits>
#include <stdlib.h>
#include <thread>
//#include <argparse/argparse.hpp> //https://github.com/p-ranav/argparse


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
        //auto timestamp1_ = std::chrono::steady_clock::now();
        next_task = sys_.my_network_layer.check_new_task(/*op==operation::backward_*/); // TODO: CHANGE THAT
        //auto timestamp2_ = std::chrono::steady_clock::now();
        /*auto __time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (timestamp2_ - timestamp1_).count();*/
        //std::cout << "Waiting... " << __time << std::endl;

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
        if(g_c % 50 == 0)
            std::cout << "Exec "  << operation_ << " " << __time << std::endl;

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
    char *p;
    long conv = strtol(argv[1], &p, 10);

    int myID = conv;
    std::string log_dir = "main_experiment";


    systemAPI sys_(false, myID, log_dir);
    
    // wait for init refactoring
    //sys_.my_network_layer.findInit();
    auto refactor_message = sys_.my_network_layer.check_new_refactor_task();
    sys_.refactor(refactor_message);
    ComputeNode the_cnode = ComputeNode(myID, sys_);
    //std::cout << "refactored ok" << std::endl;
    //std::thread forward_operations(&ComputeNode::task_operator, &the_cnode, operation::forward_);
    //std::thread backprop_operations(&ComputeNode::task_operator, &the_cnode, operation::backward_);


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
    //forward_operations.join();
    //backprop_operations.join();


    
    //while (true) {
        
        /* CODE FOR MEASURING RTT - COMMENT */
        /*for(int i = 0; i < 15 ; i ++ ){
            auto task_ = sys_.my_network_layer.check_new_task();
            sys_.my_network_layer.new_message(task_, 0);
        }
        /* CODE FOR MEASURING RTT - COMMENT */

        /*auto timestamp1_ = std::chrono::steady_clock::now();
        next_task = sys_.my_network_layer.check_new_task();
        auto timestamp2_ = std::chrono::steady_clock::now();
        auto __time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (timestamp2_ - timestamp1_).count();
        //std::cout << "Waiting... " << __time << std::endl;

        interval_type type_;
        std::string operation;
        switch (next_task.type) {
        case forward_:
            type_ = fwd_only;
            operation = "forward";
            break;
        case backward_:
            type_ = bwd_only;
            operation = "backward";
            break;
        case optimize_:
            type_ = opz_only;
            operation = "optimization";
            break;
        default:
            break;
        }
        //std::cout << operation << std::endl;
        // POINT 14 Execution phase: CN starts executing task
        auto point1 = sys_.my_network_layer.newPoint(CN_START_EXEC, next_task.client_id, operation);
        auto timestamp1 = std::chrono::steady_clock::now();
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

        auto timestamp2 = std::chrono::steady_clock::now();
        auto _time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (timestamp2 - timestamp1).count();
        //std::cout << "Computing: " << operation << ": " << _time << std::endl;
        
        // POINT 15 Execution phase: CN completed a task
        auto point2 = sys_.my_network_layer.newPoint(CN_END_EXEC, next_task.client_id, operation);
        // 14 - 15 interval
        sys_.my_network_layer.mylogger.add_interval(point1, point2, type_);
    }*/
}