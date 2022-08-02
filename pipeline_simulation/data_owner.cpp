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
        std::cout << "here1" << std::endl;
        std::vector<int>data_owners{0};
        std::vector<int>compute_nodes = {1};

        int num_parts = compute_nodes.size() + 2;

        // offline decission -- from profiling (?)
        std::vector<int>cut_layers{2, 4, 8};

        int data_onwer_end = 2;
        int data_owner_beg = 8;

        int model_name = 2;
        int model_type = 3;

        client_message.dataset = CIFAR_10;
        client_message.model_name_ = model_name::resnet;
        client_message.model_type_ = resnet_model::resnet18;
        client_message.start = cut_layers[0];
        client_message.end = cut_layers[cut_layers.size() - 1] + 1;
        client_message.next = compute_nodes[0];
        client_message.prev = compute_nodes[1];

        for (int i=1; i<data_owners.size(); i++) {
            sys_.my_network_layer.new_message(client_message, data_owners[i]);
        }

        for (int i=0; i<compute_nodes.size(); i++) {
            refactoring_data compute_node_message;

            compute_node_message.to_data_onwer = false;
            compute_node_message.model_name_ = model_name::resnet;
            compute_node_message.model_type_ = resnet_model::resnet18;
            compute_node_message.start = cut_layers[i+1];
            compute_node_message.end = cut_layers[i+1] + 1;

            if (i == compute_nodes.size() - 1) 
                compute_node_message.next = -1;
            else
                compute_node_message.next = compute_nodes[i+1];

            if (i == 0) 
                compute_node_message.prev = -1;
            else
                compute_node_message.next = compute_nodes[i-1];

           compute_node_message.data_owners = data_owners;

           sys_.my_network_layer.new_message(compute_node_message, compute_nodes[i]);
        }

        
    }
    else { // if not wait for init refactoring
        client_message = sys_.my_network_layer.check_new_refactor_task();
    }
    std::cout << "here" << std::endl;
    sys_.refactor(client_message);
    
    while(true) {

    }
    
    // load dataset
    int type = client_message.dataset;
    auto path_selection = (type == 1)? CIFAR10_data_path : CIFAR100_data_path;
    
    auto train_dataset = CIFAR(path_selection, type)
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                    .map(ConstantPad(4))
                                    .map(RandomHorizontalFlip())
                                    .map(RandomCrop({32, 32}))
                                    .map(torch::data::transforms::Stack<>());
    auto num_train_samples = train_dataset.size().value();

    auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), sys_.batch_size);

    int num_classes = (type == 1)? 10 : 100;
    //sys_.init_model_sate(resnet, resnet_model::resnet18, num_classes, 4, 3);
    //sys_.init_model_sate(vgg, vgg_model::v11, num_classes, 4, 3);
    
    
    for (int i = 0; i< sys_.parts[0].layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< sys_.parts[0].layers[i] << std::endl;
    }

    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

    for (int i = 0; i< sys_.parts[1].layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< sys_.parts[1].layers[i] << std::endl;
    }
    
    
    for (size_t round = 0; round != sys_.rounds; ++round) {
        int batch_index = 0;
        sys_.zero_metrics();
        int total_num = 0;
        for (auto& batch : *train_dataloader) {
            // create task with new batch
            Task task(sys_.myid, forward_, -1);
            task.size_ = batch.data.size(0);
            task.values = batch.data;
            task = sys_.exec(task, batch.target);
            total_num += task.size_; 
            // send task to next node
            //sys_.my_network_layer.new_message(task, sys_.inference_path[0]);
            
            // wait for next forward task
            //task = sys_.my_network_layer.check_new_task();
            // check if is refactor ...
            // else: 

            task = sys_.exec(task, batch.target);
            // send task - backward
            //sys_.my_network_layer.new_message(task, sys_.inference_path[1]);
        
            //optimize task
            auto task1 = sys_.my_network_layer.check_new_task();
            task1 = sys_.exec(task1, batch.target);

            // wait for next backward task
            //task = sys_.my_network_layer.check_new_task();
            task = sys_.exec(task, batch.target);
            
            // end of batch
            batch_index++;

        }

        auto sample_mean_loss = sys_.running_loss / batch_index;
        auto accuracy = sys_.num_correct / total_num;

            
        std::cout << "Epoch [" << (round + 1) << "/" << sys_.rounds << "], Trainset - Loss: "
                << sample_mean_loss << ", Accuracy: " << accuracy << " " << sys_.num_correct << std::endl;
        
    }
    
}