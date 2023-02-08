#include "systemAPI.h"

void systemAPI::init_state_vector(model_name name, int model_, int num_class, int start, int end) {
    for (int i=0; i<clients.size() ; i++) {
        ModelPart part(name, model_, start, end, num_class);
        //std::vector<torch::optim::Adam *> optimizers(part.layers.size(), nullptr);
        std::vector<torch::optim::SGD *> optimizers(part.layers.size(), nullptr);
         
        for (int i = 0; i < part.layers.size(); i++) {
            optimizers[i] = new torch::optim::SGD(part.layers[i]->parameters(), torch::optim::SGDOptions(learning_rate).momentum(0.9).weight_decay(0.0001));
        }
        State client(clients[i], part.layers, optimizers);
        clients_state.insert(std::pair<int, State>(clients[i], client));
    }
}

void systemAPI::init_model_sate(model_name name, int model_, int num_class, int start, int end) {
    ModelPart first_(name, model_, 1, end, num_class);
    ModelPart last_(name, model_, start, -1, num_class);
    std::vector<torch::optim::SGD *> optimizers_first(first_.layers.size(), nullptr);
    std::vector<torch::optim::SGD *> optimizers_last(last_.layers.size(), nullptr);

    for (int i = 0; i < first_.layers.size(); i++) {
        optimizers_first[i] = new torch::optim::SGD(first_.layers[i]->parameters(), torch::optim::SGDOptions(learning_rate).momentum(0.9).weight_decay(0.0001));
    }
    for (int i = 0; i < last_.layers.size(); i++) {
        optimizers_last[i] = new torch::optim::SGD(last_.layers[i]->parameters(), torch::optim::SGDOptions(learning_rate).momentum(0.9).weight_decay(0.0001));
    }
    State part_first(myid, first_.layers, optimizers_first);
    parts.push_back(part_first);
    State part_last(myid, last_.layers, optimizers_last);
    parts.push_back(part_last);

    if(myid == -1) {
        parts_.push_back(part_first);
        parts_.push_back(part_last);
    }
}

Task systemAPI::exec(Task task, torch::Tensor& target) {
    int client_id, prev_node;
    operation type;
    torch::Tensor values, output;
    Task nextTask;
    operation nextOp;

    client_id = task.client_id;
    prev_node = myid;
    int size_ = task.size_;
    switch (task.type) {
    case forward_:
        nextOp = forward_;
        if (is_data_owner) {
            if (task.prev_node == -1) { // first part
                batch_index += 1;
                values = task.values;   
                parts[0].received_activation = values;  
                parts[0].activations.clear();
                parts[0].detached_activations.clear();
                for (int i=0; i<parts[0].layers.size(); i++) {
                    if (i >= 1) {
                        values = values.view({task.size_, -1});
                    }

                    parts[0].optimizers[i]->zero_grad();

                    output = parts[0].layers[i]->forward(values);
                    parts[0].activations.push_back(output);

                    values = output.clone().detach().requires_grad_(true);
                    parts[0].detached_activations.push_back(values);
                }
            }
            else {
                values = task.values;
                parts[1].received_activation = values;
                parts[1].activations.clear();
                parts[1].detached_activations.clear();
                for (int i=0; i<parts[1].layers.size(); i++) {
                    parts[1].optimizers[i]->zero_grad();
                    output = parts[1].layers[i]->forward(values);

                    if (i != parts[1].layers.size() - 1) {
                        output = output.view({task.size_, -1});
                        parts[1].activations.push_back(output);
                        values = output.clone().detach().requires_grad_(true);
                        parts[1].detached_activations.push_back(values);
                    }
                        
                }

                nextOp = backward_;
                // compute loss 
                torch::Tensor loss =
                   torch::nn::functional::cross_entropy(output, target);

                running_loss += loss.item<double>();
                auto prediction = output.argmax(1);
                auto corr = prediction.eq(target).sum().item<int64_t>();
                auto corr_ = static_cast<double>(corr)/size_;
                num_correct += corr;
                
                loss.backward();

                if (parts[1].activations.size() >= 1) {
                    auto detached_grad = parts[1].detached_activations[0].grad().clone().detach();
                    parts[1].activations[0].backward(detached_grad);
                }

                values = parts[1].received_activation.grad().clone().detach();
                Task opt(client_id, optimize_, prev_node);
                my_network_layer.put_internal_task(opt);
            }
        }
        else { // compute node
            values = task.values;
            auto client_state = &(clients_state.find(client_id)->second);
            client_state->received_activation = values;
            client_state->activations.clear();
            client_state->detached_activations.clear();
            for (int i=0; i<client_state->layers.size(); i++) {
                if (i >= 1) {
                    values = values.view({task.size_, -1});
                }
                client_state->optimizers[i]->zero_grad();
                output = client_state->layers[i]->forward(values);
                client_state->activations.push_back(output);
                values = output.clone().detach().requires_grad_(true);
                client_state->detached_activations.push_back(values);
            }
        }
        nextTask = Task(client_id, nextOp, prev_node);
        nextTask.values = values;
        nextTask.size_ = size_;
        break;
    case backward_:
        nextOp = backward_;
        if (is_data_owner) {
            values = task.values;
            for (int i=parts[0].layers.size()-1; i>=0; i--) {
                parts[0].activations[i].backward(values);
                parts[0].optimizers[i]->step();
                if (i != 0)
                    values = parts[0].detached_activations[i-1].grad().clone().detach();
            }
            nextOp = noOp; // end of batch
            
        }
        else { // compute node
            values = task.values;
            auto client_state = &clients_state.find(client_id)->second;
            for (int i=client_state->layers.size()-1; i>=0; i--) {
                client_state->activations[i].backward(values);
                
                if (i != 0)
                    values = client_state->detached_activations[i-1].grad().clone().detach();
                else
                    values = client_state->received_activation.grad().clone().detach();

            }
            // add optimization task to list
            Task opt(client_id, optimize_, prev_node);
            my_network_layer.put_internal_task(opt);
        }

        nextTask = Task(client_id, nextOp, prev_node);
        nextTask.values = values;
        nextTask.size_ = size_;
        break;
    case optimize_:
        nextOp = noOp;
        if (is_data_owner) {
            for (int i=0; i<parts[1].layers.size(); i++) {
                parts[1].optimizers[i]->step();
            }
        }
        else {
            auto client_state = &clients_state.find(client_id)->second;
            for (int i=0; i<client_state->layers.size(); i++) {
                client_state->optimizers[i]->step();
            }
        }

        nextTask = Task(client_id, nextOp, prev_node);
    default:
        break;
    }

    return nextTask;
}

void systemAPI::refactor(refactoring_data refactor_message) {
    inference_path.clear();
    inference_path.push_back(refactor_message.next);
    
    if(refactor_message.next == -1) {
        my_network_layer.sim_back = true;
    }

    inference_path.push_back(refactor_message.prev);

    if(refactor_message.prev == -1) {
        my_network_layer.sim_forw = true;
    }

    model_name name = (model_name) refactor_message.model_name_;
    int model_ = refactor_message.model_type_;
    int num_class = refactor_message.num_class;
    int start = refactor_message.start;
    int end = refactor_message.end;
    
    //std::cout << "TABLE: " << refactor_message.rooting_table.size() << std::endl;
    if (refactor_message.rooting_table.size() > 0) {
        for (int i=0; i < refactor_message.rooting_table.size(); i++) {
            std::pair<int, std::string> addr = refactor_message.rooting_table[i];
            
            // In VM version we do not have node search
            /*if(addr.first == 0) {
                continue;
            }*/

            if((addr.first > 3) && (addr.first < 23)) {
                std::pair<std::string, int> my_addr = my_network_layer.rooting_table.find(0)->second;
                int my_port = my_addr.second;
                my_port = my_port + (addr.first + 3);
                my_network_layer.rooting_table.insert({addr.first, std::pair<std::string, int>(addr.second, my_port)});
            }
            else if (addr.first > 23) {
                std::pair<std::string, int> my_addr = my_network_layer.rooting_table.find(23)->second;
                int my_port = my_addr.second;
                my_port = my_port + (addr.first-23);
                my_network_layer.rooting_table.insert({addr.first, std::pair<std::string, int>(addr.second, my_port)});
            }
            else {
                my_network_layer.rooting_table[addr.first].first = addr.second; 
            }
        }
    }

    if (is_data_owner) 
        init_model_sate(name, model_, num_class, start, end);
    else {
        clients_state.clear();
        clients.clear();
        clients = refactor_message.data_owners;
        init_state_vector(name, model_, num_class, start, end);
    }

}