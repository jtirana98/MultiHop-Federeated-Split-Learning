#include "systemAPI.h"

//template <typename T>
void systemAPI::init_state_vector(model_name name, int model_, int num_class, int start, int end) {
    ModelPart part(name, model_, start, end, num_class);
    for (int i=0; i<clients.size() ; i++) {
        std::vector<torch::optim::Adam *> optimizers(part.layers.size(), nullptr);
        for (int i = 0; i < part.layers.size(); i++) {
            optimizers[i] = new torch::optim::Adam(part.layers[i]->parameters(), torch::optim::AdamOptions(learning_rate));
        }
        State client(clients[i], part.layers, optimizers);
        clients_state.insert(std::pair<int, State>(clients[i], client));
    }
}

void systemAPI::init_model_sate(model_name name, int model_, int num_class, int start, int end) {
    ModelPart first_(name, model_, 1, end, num_class);
    ModelPart last_(name, model_, start, -1, num_class);
    std::vector<torch::optim::Adam *> optimizers_first(first_.layers.size(), nullptr);
    std::vector<torch::optim::Adam *> optimizers_last(last_.layers.size(), nullptr);

    for (int i = 0; i < first_.layers.size(); i++) {
        optimizers_first[i] = new torch::optim::Adam(first_.layers[i]->parameters(), torch::optim::AdamOptions(learning_rate));
    }

    for (int i = 0; i < last_.layers.size(); i++) {
        optimizers_last[i] = new torch::optim::Adam(last_.layers[i]->parameters(), torch::optim::AdamOptions(learning_rate));
    }
    
    State part_first(myid, first_.layers, optimizers_first);
    //part_first.init_layers(first_.layers);
    parts.push_back(part_first);
    State part_last(myid, last_.layers, optimizers_last);
    //part_last.init_layers(last_.layers);
    parts.push_back(part_last);
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
            std::cout << ">" << task.prev_node << std::endl;
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
                    if (i >= 1) {
                        values = values.view({task.size_, -1});
                    }
                    parts[1].optimizers[i]->zero_grad();
                    output = parts[1].layers[i]->forward(values);
                    parts[1].activations.push_back(output);
                    values = output.clone().detach().requires_grad_(true);
                    parts[1].detached_activations.push_back(values);
                }

                std::cout << output << std::endl;
                std::cout << "<<<<<<<<<<<<<<<<<<<<" << std::endl;
                std::cout << target << std::endl;
                nextOp = backward_;
                // compute loss 
                torch::Tensor loss =
                    torch::nn::functional::cross_entropy(output, target);
                std::cout << "ok7 " << std::endl;
                running_loss += loss.item<double>();
                auto prediction = output.argmax(1);
                auto corr = prediction.eq(target).sum().item<int64_t>();
                auto corr_ = static_cast<double>(corr)/size_;
                num_correct += corr;
                
                loss.backward();

                if (parts[1].activations.size() > 1) {
                    auto detached_grad = parts[1].detached_activations[0].grad().clone().detach();
                    parts[1].activations[0].backward(detached_grad);
                }

                // DRAFT 
                if (batch_index % 15 == 0) {
                    std::cout << /*"Epoch: " << epoch << */" | Batch: " << batch_index
                        << " | Loss: " << loss.item<float>() << "| Acc: " << corr_ << std::endl;
                }
                // DRAFT 

                values = parts[1].received_activation.grad().clone().detach();
                Task opt(client_id, optimize_, prev_node);
                my_network_layer.put_internal_task(opt);
            }
        }
        else {
            values = task.values;
            auto client_state = clients_state.find(client_id)->second;
            client_state.received_activation = values;
            client_state.activations.clear();
            client_state.detached_activations.clear();
            for (int i=0; i<client_state.layers.size(); i++) {
                if (i >= 1) {
                        values = values.view({task.size_, -1});
                }

                client_state.optimizers[i]->zero_grad();
                
                output = client_state.layers[i]->forward(values);
                client_state.activations.push_back(output);

                values = output.clone().detach().requires_grad_(true);
                client_state.detached_activations.push_back(values);
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
            for (int i=parts[0].layers.size()-1; i>0; i--) {
                parts[0].activations[i].backward(values);
                parts[0].optimizers[i]->step();
                if (i != 0)
                    values = parts[0].detached_activations[i-1].grad().clone().detach();
                //else
                //    values = parts[0].received_activation.grad().clone().detach()

            }
            nextOp = noOp; // end of batch
            
        }
        else {
            values = task.values;
            auto client_state = clients_state.find(client_id)->second;
            for (int i=client_state.layers.size()-1; i>0; i--) {
                client_state.activations[i].backward(values);
                
                if (i != 0)
                    values = client_state.detached_activations[i-1].grad().clone().detach();
                else
                    values = client_state.received_activation.grad().clone().detach();

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
            auto client_state = clients_state.find(client_id)->second;
            for (int i=0; i<client_state.layers.size(); i++) {
                client_state.optimizers[i]->step();
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
    inference_path.push_back(refactor_message.prev);
    model_name name = (model_name) refactor_message.model_name_;
    int model_ = refactor_message.model_type_;
    int num_class = refactor_message.num_class;
    int start = refactor_message.start;
    int end = refactor_message.end;

    if (is_data_owner) 
        init_model_sate(name, model_, num_class, start, end);
    else {
        clients_state.clear();
        clients.clear();
        clients = refactor_message.data_owners;

        init_state_vector(name, model_, num_class, start, end);
    }

}