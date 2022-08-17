#include <type_traits>
#include <stdlib.h>
#include "systemAPI.h"
#include "mylogging.h"
#include "mydataset.h"
#include "transform.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cerr << "Wrong number of arguments" << std::endl;
        return 1;
    }

    int myID=atoi(argv[1]); // parameter
    systemAPI sys_(true, myID);
    refactoring_data client_message;
    // check if you are the init
    if (myID == 0) {
        
        // POINT 5
        std::vector<int>data_owners{0, 2};
        std::vector<int>compute_nodes = {1, 3};

        int num_parts = compute_nodes.size() + 2;

        // offline decission -- from profiling (?)
        std::vector<int>cut_layers{5, 7, 10};

        int data_onwer_end = 2;
        int data_owner_beg = 8;

        int model_name = 2;
        int model_type = 3;
        
        client_message.dataset = CIFAR_10;
        client_message.model_name_ = model_name::resnet;
        client_message.model_type_ = resnet_model::resnet18;
        client_message.end = cut_layers[0];
        client_message.start = cut_layers[cut_layers.size() - 1] + 1;
        client_message.next = compute_nodes[0];
        client_message.prev = compute_nodes[compute_nodes.size() -1];
        client_message.num_class = 10;
        
        // POINT 6
        for (int i=1; i<data_owners.size(); i++) {
           sys_.my_network_layer.new_message(client_message, data_owners[i]);
        }

        // POINT 7
        sys_.refactor(client_message);
        
        // POINT 8
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

           sys_.my_network_layer.new_message(client_message, compute_nodes[i]);
        }

        // POINT 9
        
    }
    else { // if not wait for init refactoring
        // POINT 10
        client_message = sys_.my_network_layer.check_new_refactor_task();
        sys_.refactor(client_message);
        // POINT 11
    }
      
    // load dataset
    int type = client_message.dataset;
    auto path_selection = (type == CIFAR_10)? CIFAR10_data_path : CIFAR100_data_path;
    
    auto train_dataset = CIFAR(path_selection, type)
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
    for (int i = 0; i< sys_.parts[0].layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< sys_.parts[0].layers[i] << std::endl;
    }

    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

    for (int i = 0; i< sys_.parts[1].layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< sys_.parts[1].layers[i] << std::endl;
    }
    */

    // POINT 12
    for (size_t round = 0; round != sys_.rounds; ++round) {
        int batch_index = 0;
        sys_.zero_metrics();
        int total_num = 0;
        for (auto& batch : *train_dataloader) {
            // create task with new batch

            // POINT 16
            Task task(sys_.myid, forward_, -1);
            task.size_ = batch.data.size(0);
            task.values = batch.data;
            task = sys_.exec(task, batch.target);
            total_num += task.size_; 
            
            // send task to next node
            sys_.my_network_layer.new_message(task, sys_.inference_path[0]);
            
            // POINT 17 - wait for next forward task
            task = sys_.my_network_layer.check_new_task();

            // check if is refactor ...
            // else:

            // POINT 18
            task = sys_.exec(task, batch.target);
            // send task - backward
            sys_.my_network_layer.new_message(task, sys_.inference_path[1]);
            //optimize task
            auto task1 = sys_.my_network_layer.check_new_task();
            task1 = sys_.exec(task1, batch.target);
            // POINT 19

            // wait for next backward task
            task = sys_.my_network_layer.check_new_task();
            // POINT 20 
            task = sys_.exec(task, batch.target);
            
            // end of batch
            batch_index++;
            // POINT 21
        }
        
        auto sample_mean_loss = sys_.running_loss / batch_index;
        auto accuracy = sys_.num_correct / total_num;

            
        std::cout << "Epoch [" << (round + 1) << "/" << sys_.rounds << "], Trainset - Loss: "
                << sample_mean_loss << ", Accuracy: " << accuracy << " " << sys_.num_correct << std::endl;

    }
    
}