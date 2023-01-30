#include "systemAPI.h"
#include <type_traits>
#include <stdlib.h>
#include <thread>

#include <argparse/argparse.hpp> //https://github.com/p-ranav/argparse
#include "systemAPI.h"

int main(int argc, char **argv) {
    argparse::ArgumentParser program("aggregator");

    refactoring_data client_message;
    systemAPI sys_(true, -1, "main_experiment");
    int kTrainSize_10 = 1000;
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

    int num_data_owners = atoi(argv[1]);

    while(true) {
        int received = 0;

        //model part 1
        while (received <= num_data_owners) {
            // wait for the task
            torch::autograd::GradMode::set_enabled(false);
            auto model = sys_.parts[0].layers[0];
            auto params = model->named_parameters(true /*recurse*/);
            auto buffers = model->named_buffers(true /*recurse*/);
            for (int j = 0; j < model->named_parameters().size(); j++) {
                auto p_g = model->named_parameters()[j];
                auto p_ = model->named_parameters()[j]; //THIS SHOULD BE THE RECEIVED
                p_g.value() = (p_g.value()+p_.value());
                p_g.value() = torch::div(p_g.value(), kTrainSize_10);
                
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
            
            torch::autograd::GradMode::set_enabled(true);

            received++;
        }

        // send to 0
        for (int i = 2; i < num_data_owners-1; i++) {
            // send the new model
        }
        /*
        // the same for last model part

        int received = 0;

        //model part 1
        for (int k = 0; k<sys_.parts[1].layers.size(); k++) {
            while (received <= num_data_owners) {
                // wait for the task
                torch::autograd::GradMode::set_enabled(false);
        
                auto model = sys_.parts[1].layers[k];
                auto params = model->named_parameters(true );
                auto buffers = model->named_buffers(true);
                for (int j = 0; j < model->named_parameters().size(); j++) {
                    auto p_g = model->named_parameters()[j];
                    auto p_ = model->named_parameters()[j]; //THIS SHOULD BE THE RECEIVED
                    p_g.value() = (p_g.value()+p_.value());
                    p_g.value() = torch::div(p_g.value(), kTrainSize_10);
                    
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
                received++;
            }
            torch::autograd::GradMode::set_enabled(true);

            
        }
        
        // send to 0
        for (int i = 2; i < num_data_owners-1; i++) {
            // send the new model
        }
        */
    }
}