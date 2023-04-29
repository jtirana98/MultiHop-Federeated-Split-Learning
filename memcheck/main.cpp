#include <type_traits>
#include <stdlib.h>
#include <thread>
#include <argparse/argparse.hpp> //https://github.com/p-ranav/argparse

#include "systemAPI.h"
#include "mydataset.h"
#include "transform.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

int main(int argc, char **argv) {
    argparse::ArgumentParser program("aggregator");

    
    program.add_argument("-s", "--splits")
        .help("The splits")
        .default_value(std::string("-1,-1"))
        .nargs(1);

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    systemAPI sys_(true, 0, "main_experiment");
    refactoring_data client_message;
    // check if you are the init
    auto cut_layers_ = program.get<std::string>("-s");
    
    const char separator = ',';
    std::string val;
    std::vector<int>data_owners, compute_nodes, cut_layers;

    std::stringstream streamData(cut_layers_);
    while (std::getline(streamData, val, separator)) {
        if (val != "") {
            cut_layers.push_back(stoi(val));
        }
    }

    client_message.dataset = CIFAR_10;
    client_message.model_name_ = model_name::resnet;
    client_message.model_type_ = resnet_model::resnet101;
    client_message.end = -1;
    client_message.start = -1;
    client_message.next = 0;
    client_message.prev = 0;
    client_message.num_class = 10;
    
    sys_.refactor(client_message);

    // ----------- TEST ------------
    auto layers = sys_.parts[0].layers;
    
    for (int i = 0; i< layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< layers[i] << std::endl;
    }
    
    // -------- TEST ------------
    

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

    for (auto& batch : *train_dataloader) {          

        auto init_batch = std::chrono::steady_clock::now();
        Task task(sys_.myid, forward_, -2);
        task.size_ = batch.data.size(0);
        task.values = batch.data;
        
        task = sys_.exec(task, batch.target); // forward-back as last part
        task = sys_.my_network_layer.check_new_task(); // get optimize step
        task = sys_.exec(task, batch.target);

        auto timestamp2 = std::chrono::steady_clock::now();

        auto _time = std::chrono::duration_cast<std::chrono::milliseconds>
                    (timestamp2 - init_batch).count();

        std::cout << "Batch time: " << _time << std::endl;
    }
}