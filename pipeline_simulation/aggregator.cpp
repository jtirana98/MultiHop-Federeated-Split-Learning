#include "systemAPI.h"
#include <type_traits>
#include <stdlib.h>
#include <thread>

//#include <argparse/argparse.hpp> //https://github.com/p-ranav/argparse
#include "systemAPI.h"

int main(int argc, char **argv) {
    refactoring_data client_message;
    systemAPI sys_(true, -1, "main_experiment");
    int kTrainSize_10 = 1000;
    int train_samples = 100;
    
    client_message = sys_.my_network_layer.check_new_refactor_task();
    sys_.refactor(client_message);

    std::cout << "Refactor ok" << std::endl;

    int num_data_owners = atoi(argv[1]);
    while(true) {
        int received = 0;

        //model part 1
        while (received < num_data_owners) {
            auto model_task = sys_.my_network_layer.check_new_task();

            std::cout << "model part 1: received from " << model_task.client_id << std::endl;
            std::stringstream ss(std::string(model_task.model_parts.begin(), model_task.model_parts.end()));
            torch::load(sys_.parts_[0].layers[0]/*task.model_part_*/, ss);
            // wait for the task
            torch::autograd::GradMode::set_enabled(false);
            auto model = sys_.parts[0].layers[0];
            auto model_2 = sys_.parts_[0].layers[0];
            auto params = model->named_parameters(true /*recurse*/);
            auto buffers = model->named_buffers(true /*recurse*/);
            //std::cout << "size " << model->named_parameters().size() << " " << model_2->named_parameters().size() << std::endl;
            for (int j = 0; j < model->named_parameters().size(); j++) {
                auto p_g = model->named_parameters()[j];
                auto p_ = model_2->named_parameters()[j]; //model->named_parameters()[j]; //THIS SHOULD BE THE RECEIVED
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
        auto newAggTask = Task(-1, operation::aggregation_, -1);
        newAggTask.model_part = 1;

        // send aggregation task
        newAggTask.model_part_=sys_.parts[0].layers[0];
        newAggTask.t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        sys_.my_network_layer.new_message(newAggTask,0);
        for (int i = 2; i <= num_data_owners; i++) {
            newAggTask.t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            sys_.my_network_layer.new_message(newAggTask,i);
        }

        // the same for last model part
        int received1 = 0, received2=0;
        int sum = num_data_owners*sys_.parts[1].layers.size();

        //model part 1
        while (received1 + received2 < sum) {
            auto model_task = sys_.my_network_layer.check_new_task();

            std::cout << "model part "<< model_task.model_part - 1 << " : received from " << model_task.client_id << std::endl;
            std::stringstream ss(std::string(model_task.model_parts.begin(), model_task.model_parts.end()));
            torch::load(sys_.parts[1].layers[model_task.model_part-2], ss);
            // wait for the task
            torch::autograd::GradMode::set_enabled(false);
            auto model = sys_.parts[1].layers[model_task.model_part-2];
            auto model_2 = sys_.parts_[1].layers[model_task.model_part-2];
            auto params = model->named_parameters(true /*recurse*/);
            auto buffers = model->named_buffers(true /*recurse*/);
            //std::cout << "size " << model->named_parameters().size() << " " << model_2->named_parameters().size() << std::endl;
            for (int j = 0; j < model->named_parameters().size(); j++) {
                auto p_g = model->named_parameters()[j];
                auto p_ = model_2->named_parameters()[j]; //model->named_parameters()[j]; //THIS SHOULD BE THE RECEIVED
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

            if(model_task.model_part == 2) 
                received1++;
            else
                received2++;
        }

        // send to 0
        newAggTask = Task(-1, operation::aggregation_, -1);
        newAggTask.model_part = 2;
        for (int i = 0; i < sys_.parts[1].layers.size(); i++) {
            newAggTask.model_part_=sys_.parts[1].layers[newAggTask.model_part-2];
            newAggTask.t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            sys_.my_network_layer.new_message(newAggTask,0);

            for (int i = 2; i <= num_data_owners; i++) {
                newAggTask.t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                sys_.my_network_layer.new_message(newAggTask,i);
            }

            newAggTask.model_part++;
        }
        
    }
}