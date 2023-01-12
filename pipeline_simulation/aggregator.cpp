#include "systemAPI.h"
#include <type_traits>
#include <stdlib.h>
#include <thread>

#include <argparse/argparse.hpp> //https://github.com/p-ranav/argparse
#include "systemAPI.h"

int main(int argc, char **argv) {
    argparse::ArgumentParser program("aggregator");

    /*program.add_argument("-d", "--data_owners")
        .help("The number of data owners")
        .required()
        .nargs(1)
        .scan<'d', int>();

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto num_clients = program.get<int>("-d");

    systemAPI sys_(false, -1, "main_experiment");
    sys_.my_network_layer.findPeers(num_clients, true);

    std::cout << "found them" << std::endl;    
    sleep(2);

    while (true) {
        auto next_task = sys_.my_network_layer.check_new_task();
        std::cout << next_task.model_part_->size() << std::endl;
        auto client = next_task.client_id;
        next_task.check_ = true;
        sys_.my_network_layer.new_message(next_task,client);
        
    }*/

    refactoring_data client_message;
    systemAPI sys_(true, -1, "main_experiment");
    int kTrainSize_10 = 50000;
    int train_samples = 100;
    client_message.dataset = CIFAR_10;
    client_message.model_name_ = model_name::resnet;
    client_message.model_type_ = resnet_model::resnet101;
    client_message.end = 10;
    client_message.start = 33;
    client_message.next = 0;
    client_message.prev = 0;
    client_message.num_class = 10;
    sys_.refactor(client_message);
    for (int i = 0; i < 15; i++) {
        auto timestamp1_ = std::chrono::steady_clock::now();
        torch::autograd::GradMode::set_enabled(false); 
        
        for (int k = 0; k<sys_.parts[0].layers.size(); k++) {
            auto model = sys_.parts[0].layers[k];
            auto params = model->named_parameters(true /*recurse*/);
            auto buffers = model->named_buffers(true /*recurse*/);
            for (int j = 0; j < model->named_parameters().size(); j++) {
                auto p_g = model->named_parameters()[j];
                auto p_ = model->named_parameters()[j];
                //std::cout << p_g.value()[0][0] << " !!!!!!!!!!!!!!!! " << std::endl;
                if (i == 0) {
                     p_g.value() = p_.value();
                }
                else {
                    p_g.value() = (p_g.value()+p_.value());
                }
                p_g.value() = torch::div(p_g.value(), (kTrainSize_10/train_samples));
                
                auto name = p_g.key();
                auto* t = params.find(name);
                if (t != nullptr) {
                    t->copy_(p_g.value());
                } else {
                    t = buffers.find(name);
                    if (t != nullptr) {
                        t->copy_(p_g.value());
                    }
                }
            }
        }
        torch::autograd::GradMode::set_enabled(true);

        auto timestamp2_ = std::chrono::steady_clock::now();

        auto __time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (timestamp2_ - timestamp1_).count();
                std::cout << "aggregation time: " << __time << std::endl;
    }

}