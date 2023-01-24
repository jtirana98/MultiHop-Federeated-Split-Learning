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

        auto cut_layers_ = "2,35";
        auto data_owners_ = argv[2];  // CHANGE
        std::cout << data_owners_ << std::endl;
        auto compute_nodes_ = "1"; // CHANGE


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

        std::cout << "found them" << std::endl; 
        //sleep(2);

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

        auto my_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch());
        std::cout << my_time.count() << std::endl;
        
        
        for (int i=1; i<data_owners.size(); i++) {
            std::cout << data_owners[i] << std::endl;
            if(data_owners[i] > 5) {
                std::pair<std::string, int> my_addr = sys_.my_network_layer.rooting_table.find(5)->second;
                int my_port = my_addr.second;
                my_port = my_port + (data_owners[i] - 5);
                sys_.my_network_layer.rooting_table.insert({data_owners[i], std::pair<std::string, int>(my_addr.first, my_port)});
            }

            sys_.my_network_layer.new_message(client_message, data_owners[i], false, true);
        }
        sys_.refactor(client_message);
        
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
    }
    else { // if not wait for init refactoring
        client_message = sys_.my_network_layer.check_new_refactor_task();
        sys_.refactor(client_message);
    }

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
    */

    auto send_activations = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    auto send_gradients = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

    for (size_t round = 0; round != sys_.rounds; ++round) {
        int batch_index = 0;
        sys_.zero_metrics();
        int total_num = 0;

        send_activations = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        
        for (auto& batch : *train_dataloader) {
            // create task with new batch
            auto init_batch = std::chrono::steady_clock::now();
            
            Task task(sys_.myid, forward_, -1);
            task.size_ = batch.data.size(0);
            task.values = batch.data;
            task = sys_.exec(task, batch.target);
            task.t_start = send_activations.count();
            //std::cout << task.t_start << std::endl;
            total_num += task.size_; 
            task.batch0 = batch_index;
            // send task to next node
            sys_.my_network_layer.new_message(task, sys_.inference_path[0]);
            

            // wait for next forward task
            task = sys_.my_network_layer.check_new_task();

            send_gradients = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            task = sys_.exec(task, batch.target); // forward and backward
            // send task - backward
            
            task.t_start = send_gradients.count();
            sys_.my_network_layer.new_message(task, sys_.inference_path[1]);
            //optimize task
            
            auto task1 = sys_.my_network_layer.check_new_task();

            task1 = sys_.exec(task1, batch.target); // optimize

            // wait for next backward task
            task = sys_.my_network_layer.check_new_task();
            task = sys_.exec(task, batch.target); //backward and optimize


            send_activations = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            auto end_batch = std::chrono::steady_clock::now();
            auto _time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (end_batch - init_batch).count();
            std::cout << "One batch " << _time << std::endl;
            
            // end of batch
            batch_index++;

            if (batch_index > 100)
                break;
        }

        // new epoch
        
        auto sample_mean_loss = sys_.running_loss / batch_index;
        auto accuracy = sys_.num_correct / total_num;

            
        std::cout << "Epoch [" << (round + 1) << "/" << sys_.rounds << "], Trainset - Loss: "
                << sample_mean_loss << ", Accuracy: " << accuracy << " " << sys_.num_correct << std::endl;

    }
}
