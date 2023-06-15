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

class Job {
    public:
      int client;
      int task;
      Task::operation type; //

      Job(int client, int task, Task::operatio type) : client(client), task(task), type(type) {}
};

void ComputeNode::task_operator(/*operation op*/) {
    Task next_task;
    int next_node;
    torch::Tensor tmp;

    std::map<int, Task> ready_tasks;
    int scheduler_slot = 0;

    // add here a function that constructs the planner
    std::vector<Job> planner{Job(0,1, forward_), Job(0,2,forward_), Job(1,1,forward_), 
                            Job(0,3,forward_), Job(2,1,forward_), Job(2,2,forward_), 
                            Job(2,3,forward_), Job(1,3,forward_), Job(0,1, backward_), 
                            Job(0,2,backward_), Job(1,1,backward_), Job(0,3,backward_), 
                            Job(2,1,backward_), Job(2,2,backward_), 
                            Job(2,3,backward_), Job(1,3,backward_)};
    
    while (true) {
        auto planned_task = planner[scheduler_slot];

        if (planned_task.task == 1) {
            next_task = sys_.my_network_layer.check_new_task(); // wait until task arrives
        }
        else{
            next_task = ready_tasks[planned_task.client];
        }

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

        std::cout << "Exec "  << operation_ << " " << __time  << 
                    " client " << next_task.client_id  << "part" << planned_task.task << std::endl;

        int end = 0;
        switch (next_task.type) {
            case forward_:
                end = sys_.max_tasks_fwd;
                break;
            case backward_:
                end = sys_.max_tasks_back;
                break;
            case optimize_:
                end = sys_.max_tasks_back;
                break;
                break;
            default:
                break;
        }

        if (planned_task.task == end+1) { // end of job
            bool keep_connection = false;
            next_node = task.client_id;

            sys_.my_network_layer.new_message(task, next_node, keep_connection);
        }
        else{
            ready_tasks[planned_task.client] = task;
        }

        scheduler_slot = (sched_get_priority_max+1)%planner.size()
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