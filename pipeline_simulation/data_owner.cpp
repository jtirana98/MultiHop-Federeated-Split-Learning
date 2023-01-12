#include <type_traits>
#include <stdlib.h>
#include <thread>
//#include <argparse/argparse.hpp> //https://github.com/p-ranav/argparse

#include "systemAPI.h"
#include "mydataset.h"
#include "transform.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

int main(int argc, char **argv) {
    /*argparse::ArgumentParser program("data_owner");

    program.add_argument("-i", "--id")
        .help("The node's id")
        .required()
        .nargs(1)
        .scan<'i', int>();

    program.add_argument("-l", "--log_directory")
        .help("The directory name to store looging info")
        .default_value(std::string("main_experiment"))
        .nargs(1);

    program.add_argument("-s", "--splits")
        .help("The splits")
        .default_value(std::string("10,32"))
        .nargs(1);

    program.add_argument("-d", "--data_owners")
        .help("The list of data owners")
        .default_value(std::string("0,2"))
        .nargs(1);

    program.add_argument("-c", "--compute_nodes")
        .help("The list of compute nodes")
        .default_value(std::string("1"))
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
    */

    char *p;
    long conv = strtol(argv[1], &p, 10);

    int myID = conv;
    std::string log_dir = "main_experiment";


    systemAPI sys_(true, myID, log_dir);
    refactoring_data client_message;
    // check if you are the init
    if (myID == 0) {
        // POINT 5 Initialization phase: init node starts preperation
        sys_.my_network_layer.newPoint(INIT_START_MSG_PREP);

        /*auto cut_layers_ = program.get<std::string>("-s");
        auto data_owners_ = program.get<std::string>("-d");
        auto compute_nodes_ = program.get<std::string>("-c");*/

        auto cut_layers_ = "10,32";
        auto data_owners_ = "0,2";
        auto compute_nodes_ = "1";


        const char separator = ',';
        std::string val;
        std::vector<int>data_owners, compute_nodes, cut_layers;

        std::stringstream streamData(cut_layers_);
        while (std::getline(streamData, val, separator)) {
            if (val != "") {
                cut_layers.push_back(stoi(val));
            }
        }

        streamData = std::stringstream(data_owners_);
        while (std::getline(streamData, val, separator)) {
            if (val != "") {
                data_owners.push_back(stoi(val));
            }
        }

        streamData  = std::stringstream(compute_nodes_);
        while (std::getline(streamData, val, separator)) {
            if (val != "") {
                compute_nodes.push_back(stoi(val));
            }
        } 

        int num_parts = compute_nodes.size() + 2;

        /*sys_.my_network_layer.findPeers(data_owners.size() 
                                            + compute_nodes.size() - 1);*/
        std::cout << "found them" << std::endl; 
        sleep(2);

        int data_onwer_end = 2;
        int data_owner_beg = 8;

        int model_name = 2;
        int model_type = 3;
        
        client_message.dataset = CIFAR_10;
        client_message.model_name_ = model_name::resnet;
        client_message.model_type_ = resnet_model::resnet101;
        client_message.end = cut_layers[0];
        client_message.start = cut_layers[cut_layers.size() - 1] + 1;
        client_message.next = compute_nodes[0];
        client_message.prev = compute_nodes[compute_nodes.size() -1];
        client_message.num_class = 10;
        
        // POINT 6 Initialization phase: init node completes preperation - start bcast to dataowners
        sys_.my_network_layer.newPoint(INIT_END_PREP_START_DO_BCAST);

        for (int i=1; i<data_owners.size(); i++) {
           sys_.my_network_layer.new_message(client_message, data_owners[i], false, true);

        }

        // POINT 7 Initialization phase: bcast to do completed - start refactoring
        sys_.my_network_layer.newPoint(INIT_END_BCAST_START_REFACTOR);
        sys_.refactor(client_message);
        
        // POINT 8 Initialization phase: completes refactoring - start bcast to cn
        sys_.my_network_layer.newPoint(INIT_END_REFACTOR_START_CN_BCAST);
        client_message.to_data_onwer = false;
        client_message.data_owners = data_owners;
        
        for (int i=0; i<compute_nodes.size(); i++) {
            
            client_message.start = cut_layers[i] + 1;
            client_message.end = cut_layers[i+1];

            if (i == compute_nodes.size() - 1) 
                client_message.next = -1;
            else
                client_message.next = compute_nodes[i+1];

            if (i == 0) 
                client_message.prev = -1;
            else
                client_message.prev = compute_nodes[i-1];

           sys_.my_network_layer.new_message(client_message, compute_nodes[i], false, true);
        }

        // POINT 9 Initialization phase: bcast to cn completed
        sys_.my_network_layer.newPoint(INIT_END_BCAST_CN);
    }
    else { // if not wait for init refactoring
        // POINT 10 Initialization phase: do/cn waiting for refactor message
        sys_.my_network_layer.newPoint(INIT_WAIT_FOR_REFACTOR);
        sys_.my_network_layer.findInit();
        client_message = sys_.my_network_layer.check_new_refactor_task();
        // POINT 11 Initialization phase: do/cn end waiting for refactor message
        sys_.my_network_layer.newPoint(INIT_END_W_REFACTOR);
        sys_.refactor(client_message);
       
    }

    // find aggregator
    //sys_.my_network_layer.findInit(true);

    std::cout << "loading data..." << std::endl;
    // load dataset
    int type = client_message.dataset;
    auto path_selection = (type == CIFAR_10)? CIFAR10_data_path : CIFAR100_data_path;
    auto datasets = data_owners_data(path_selection, 1, type, false);
    auto train_dataset = datasets[0]
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                    .map(ConstantPad(4))
                                    .map(RandomHorizontalFlip())
                                    .map(RandomCrop({32, 32}))
                                    .map(torch::data::transforms::Stack<>());
    auto num_train_samples = train_dataset.size().value();

    auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), sys_.batch_size);

    int num_classes = (type == CIFAR_10)? 10 : 100;
    /*
    std::cout << sys_.parts[0].layers.size() << std::endl;
    std::cout << sys_.parts[1].layers.size() << std::endl;
    for (int i = 0; i< sys_.parts[0].layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< sys_.parts[0].layers[i] << std::endl;
    }
   
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    
    for (int i = 0; i< sys_.parts[1].layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< sys_.parts[1].layers[i] << std::endl;
    }
    
    // send aggregation task:
    //auto newAggTask = Task(myID, operation::aggregation_, -1);
    //newAggTask.model_part = 1;

    for (int i = 0; i < 15; i++) {
        auto timestamp1_ = std::chrono::steady_clock::now();
        // send aggregation task
        newAggTask.model_part_=sys_.parts[0].layers[0];
        sys_.my_network_layer.new_message(newAggTask,-1);
        // wait for updated model

        auto next_task = sys_.my_network_layer.check_new_task();

        auto timestamp2_ = std::chrono::steady_clock::now();
        auto __time = std::chrono::duration_cast<std::chrono::milliseconds>
                            (timestamp2_ - timestamp1_).count();
            std::cout << "RTT: " << __time << std::endl;
    }
    */
    // POINT 12 Initialization phase: completed
    sys_.my_network_layer.newPoint(INIT_END_INIT);
    
    for (size_t round = 0; round != sys_.rounds; ++round) {
        int batch_index = 0;
        sys_.zero_metrics();
        int total_num = 0;
        for (auto& batch : *train_dataloader) {
            // create task with new batch

            // POINT 16 Execution phase: DO starts new batch
            auto point16 = sys_.my_network_layer.newPoint(DO_START_BATCH);

            auto init_batch = std::chrono::steady_clock::now();
            Task task(sys_.myid, forward_, -1);
            task.size_ = batch.data.size(0);
            task.values = batch.data;
            task = sys_.exec(task, batch.target);
            total_num += task.size_; 

            // send task to next node
            sys_.my_network_layer.new_message(task, sys_.inference_path[0]);

            auto timestamp2 = std::chrono::steady_clock::now();

            auto _time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (timestamp2 - init_batch).count();
            std::cout << "Forward Model Part 1: " << _time << std::endl;
            // POINT 17 Execution phase: DO produced activations from first part
            auto point17 = sys_.my_network_layer.newPoint(DO_FRWD_FIRST_PART);
           
            // 16 - 17 interval forward_part
            sys_.my_network_layer.mylogger.add_interval(point16, point17, fwd_only);

            // wait for next forward task
            task = sys_.my_network_layer.check_new_task();
            // POINT 18 Execution phase: DO received activations from CN
            auto point18 = sys_.my_network_layer.newPoint(DO_END_WAIT);

            auto timestamp1 = std::chrono::steady_clock::now();
            task = sys_.exec(task, batch.target); // forward and backward
            // send task - backward
            sys_.my_network_layer.new_message(task, sys_.inference_path[1]);
            //optimize task
            auto task1 = sys_.my_network_layer.check_new_task();
            task1 = sys_.exec(task1, batch.target); // optimize

            // POINT 19 Execution phase: DO completed training of last part
            auto point19 = sys_.my_network_layer.newPoint(DO_FRWD_BCKWD_SECOND_PART);
            // 18 - 19 interval forward - backward - optimize
            timestamp2 = std::chrono::steady_clock::now();
            _time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (timestamp2 - timestamp1).count();
            std::cout << "Forward and BackProp Model Part 2: " << _time << std::endl;
            sys_.my_network_layer.mylogger.add_interval(point18, point19, fwd_bwd_opz);

            // wait for next backward task
            task = sys_.my_network_layer.check_new_task();
            
            // POINT 20 Execution phase: DO received gradients
            auto point20 = sys_.my_network_layer.newPoint(DO_END_WAIT2);
            timestamp1 = std::chrono::steady_clock::now();
            task = sys_.exec(task, batch.target); //backward and optimize
            timestamp2 = std::chrono::steady_clock::now();
            _time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (timestamp2 - timestamp1).count();
            std::cout << "BackProp Model Part 1: " << _time << std::endl;

            _time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (timestamp2 - init_batch).count();
            std::cout << "One batch " << _time << std::endl;
            // end of batch
            batch_index++;
            // POINT 21 Execution phase: DO completed training for first part
            auto point21 = sys_.my_network_layer.newPoint(DO_END_BATCH);

            // 20 - 21 interval backward and optimize
            sys_.my_network_layer.mylogger.add_interval(point20, point21, bwd_opz);
        }
        
        auto sample_mean_loss = sys_.running_loss / batch_index;
        auto accuracy = sys_.num_correct / total_num;

            
        std::cout << "Epoch [" << (round + 1) << "/" << sys_.rounds << "], Trainset - Loss: "
                << sample_mean_loss << ", Accuracy: " << accuracy << " " << sys_.num_correct << std::endl;

    }
}
