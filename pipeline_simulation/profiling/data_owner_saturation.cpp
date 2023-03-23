#include <type_traits>
#include <stdlib.h>
#include <thread>
<<<<<<< HEAD
//#include <argparse/argparse.hpp> //https://github.com/p-ranav/argparse
=======
#include <argparse/argparse.hpp> //https://github.com/p-ranav/argparse
>>>>>>> main

#include "systemAPI.h"
#include "mydataset.h"
#include "transform.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

int main(int argc, char **argv) {
<<<<<<< HEAD
    pid_t pid = getpid();

    printf("pid: %lu\n", pid);

    char *p;
    long conv = strtol(argv[1], &p, 10);

    int myID = conv;
    std::string log_dir = "main_experiment";
=======
    argparse::ArgumentParser program("data_owner");
    pid_t pid = getpid();

    printf("pid: %lu\n", pid);
    
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
        .default_value(std::string("2,35"))
        .nargs(1);

    program.add_argument("-d", "--data_owners")
        .help("The list of data owners")
        .default_value(std::string("0"))
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
>>>>>>> main

    systemAPI sys_(true, myID, log_dir);
    refactoring_data client_message;

<<<<<<< HEAD
    auto cut_layers_ = "2,35";
    auto data_owners_ = "0";
    auto compute_nodes_ = "1";
=======
    auto cut_layers_ = program.get<std::string>("-s");
    auto data_owners_ = program.get<std::string>("-d");
    auto compute_nodes_ = program.get<std::string>("-c");
>>>>>>> main

    const char separator = ',';
    std::string val;
    std::vector<int>data_owners, compute_nodes, cut_layers;

    std::stringstream streamData(cut_layers_);
    while (std::getline(streamData, val, separator)) {
        if (val != "") {
            cut_layers.push_back(stoi(val));
        }
    }

<<<<<<< HEAD
    /*streamData = std::stringstream(data_owners_);
=======
    streamData = std::stringstream(data_owners_);
>>>>>>> main
    while (std::getline(streamData, val, separator)) {
        if (val != "") {
            data_owners.push_back(stoi(val));
        }
<<<<<<< HEAD
    }*/
    data_owners.push_back(myID);
=======
    }
>>>>>>> main

    streamData  = std::stringstream(compute_nodes_);
    while (std::getline(streamData, val, separator)) {
        if (val != "") {
            compute_nodes.push_back(stoi(val));
        }
    } 

    int num_parts = compute_nodes.size() + 2;

    sleep(2);
    client_message.dataset = CIFAR_10;
    client_message.model_name_ = model_name::resnet;
    client_message.model_type_ = resnet_model::resnet101;
    client_message.end = cut_layers[0];
    client_message.start = cut_layers[cut_layers.size() - 1] + 1;
    client_message.next = compute_nodes[0];
    client_message.prev = compute_nodes[compute_nodes.size() -1];
    client_message.num_class = 10;
    

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
   
    }

<<<<<<< HEAD
    std::cout << sys_.parts[0].layers.size() << std::endl;
    std::cout << sys_.parts[1].layers.size() << std::endl;
    for (int i = 0; i< sys_.parts[0].layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< sys_.parts[0].layers[i] << std::endl;
    }
   
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    
    for (int i = 0; i< sys_.parts[1].layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< sys_.parts[1].layers[i] << std::endl;
    }


=======
>>>>>>> main
    // change to compute node
    sys_.is_data_owner = false;
    sys_.myid = 1;

    sys_.refactor(client_message);

    // back to data owner
    sys_.is_data_owner = true;
    sys_.myid = myID;

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
    
<<<<<<< HEAD
    
    
    std::vector<int> f1, fb2, b1;
=======
>>>>>>> main
    Task task_forw_c, task_back_c, task_1, task_2;
    bool first = true;
    for (size_t round = 0; round != sys_.rounds; ++round) {
        int batch_index = 0;
        sys_.zero_metrics();
        int total_num = 0;
        for (auto& batch : *train_dataloader) {
<<<<<<< HEAD

            // create task with new batch
            while(batch_index < 150) {
                //auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch());
                //std::cout << t1.count() << std::endl;

=======
            // create task with new batch
            while(batch_index < 150) {
>>>>>>> main
                auto init_batch = std::chrono::steady_clock::now();
                Task task(sys_.myid, forward_, -1);
                task.size_ = batch.data.size(0);
                if(task.size_ < sys_.batch_size) {
                    continue;
                }

                task.values = batch.data;
                
                
                task = sys_.exec(task, batch.target);
                total_num += task.size_; 

                // send task to next node
                //sys_.my_network_layer.new_message(task, sys_.inference_path[0]);

                auto timestamp2 = std::chrono::steady_clock::now();

                auto _time = std::chrono::duration_cast<std::chrono::milliseconds>
                            (timestamp2 - init_batch).count();
<<<<<<< HEAD
                //std::cout << "Forward Model Part 1: " << _time << std::endl;
                f1.push_back(_time);
=======
                std::cout << "Forward Model Part 1: " << _time << std::endl;
                
>>>>>>> main

                // ---------------- wait for next forward task
                // change to compute node
                sys_.is_data_owner = false;
                sys_.myid = 1;
                
                if(first)
                    task_forw_c = sys_.exec(task, batch.target);
                
                // back to data owner
                sys_.is_data_owner = true;
                sys_.myid = myID;
                // -----------------------------


                auto timestamp1 = std::chrono::steady_clock::now();
                task = sys_.exec(task_forw_c, batch.target);
<<<<<<< HEAD
                
                timestamp2 = std::chrono::steady_clock::now();
                _time = std::chrono::duration_cast<std::chrono::milliseconds>
                            (timestamp2 - timestamp1).count();
                //std::cout << "Forward and BackProp Model Part 2: " << _time << std::endl;
                fb2.push_back(_time);

=======
>>>>>>> main
                //std::cout << "do optimize" << std::endl;
                // optimize task
                auto task1 = sys_.my_network_layer.check_new_task();
                task1 = sys_.exec(task1, batch.target); // optimize

<<<<<<< HEAD
=======
                timestamp2 = std::chrono::steady_clock::now();
                _time = std::chrono::duration_cast<std::chrono::milliseconds>
                            (timestamp2 - timestamp1).count();
                std::cout << "Forward and BackProp Model Part 2: " << _time << std::endl;

>>>>>>> main
                // -------------------- wait for next backward task
                
                // change to compute node
                sys_.is_data_owner = false;
                sys_.myid = 1;

                if(first)
                    task_back_c = sys_.exec(task, batch.target);
                
                // back to data owner
                sys_.is_data_owner = true;
                sys_.myid = myID;


                timestamp1 = std::chrono::steady_clock::now();
                task = sys_.exec(task_back_c, batch.target); //backward and optimize
                timestamp2 = std::chrono::steady_clock::now();
                _time = std::chrono::duration_cast<std::chrono::milliseconds>
                            (timestamp2 - timestamp1).count();
<<<<<<< HEAD
                //std::cout << "BackProp Model Part 1: " << _time << std::endl;
                b1.push_back(_time);
=======
                std::cout << "BackProp Model Part 1: " << _time << std::endl;

>>>>>>> main
                _time = std::chrono::duration_cast<std::chrono::milliseconds>
                            (timestamp2 - init_batch).count();
                std::cout << "One batch " << _time << std::endl;
                // end of batch
                batch_index++;
                first = 0;
<<<<<<< HEAD

                //auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch());
                //std::cout << "T2:" << t2.count() << std::endl;

                //std::cout << "diafora:" << t2.count()-t1.count() << std::endl;
                
            }
            break;
        }
        std::cout << "Forward model part 1" << std::endl;
        for (int i=0; i < f1.size(); i++) {
            std::cout << f1[i] << std::endl;
        }

        std::cout << "Forward and Backpro model part p" << std::endl;
        for (int i=0; i < fb2.size(); i++) {
            std::cout << fb2[i] << std::endl;
        }

        std::cout << "Backpro model part 1" << std::endl;
        for (int i=0; i < b1.size(); i++) {
            std::cout << b1[i] << std::endl;
        }

            
        //std::cout << "Epoch [" << (round + 1) << "/" << sys_.rounds << "], Trainset - Loss: "
        //        << sample_mean_loss << ", Accuracy: " << accuracy << " " << sys_.num_correct << std::endl;
=======
            }
            break;
        }
        break;
        auto sample_mean_loss = sys_.running_loss / batch_index;
        auto accuracy = sys_.num_correct / total_num;

            
        std::cout << "Epoch [" << (round + 1) << "/" << sys_.rounds << "], Trainset - Loss: "
                << sample_mean_loss << ", Accuracy: " << accuracy << " " << sys_.num_correct << std::endl;
>>>>>>> main

        break;
    }
}
