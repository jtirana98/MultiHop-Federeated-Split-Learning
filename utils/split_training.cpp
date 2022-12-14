#include "split_training.h"

void split_cifar(std::vector<torch::nn::Sequential> layers, int type, int batch_size, int avg_point_, double learning_rate, int num_epochs) {
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
            std::move(train_dataset), batch_size);
    
    int num_classes = (type == CIFAR_10)? 10 : 100;

    std::vector<torch::optim::SGD> optimizers;
    
    for (int i=0; i<layers.size(); i++) {
        optimizers.push_back(torch::optim::SGD(layers[i]->parameters(), torch::optim::SGDOptions(learning_rate).momentum(0.9).weight_decay(0.0001)));
        std::cout << "new layer: "<< i+1 << " "<< layers[i] << std::endl;
    }

    gatherd_data data_loads;
    Total totaltimes = Total();

    int avg_point;
    int reverse_count = 0;
    for (int i=layers.size()-1; i>=0; i--) {
        reverse_count = reverse_count + layers[i]->size();
        if (reverse_count > avg_point_) {
            avg_point = i;
            break;
        }
    }
    //std::cout << avg_point << " " <<  avg_point_ << " !!" << std::endl;
    for (size_t epoch = 0; epoch != 1; ++epoch) {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        int batch_index = 0;
        for (auto& batch : *train_dataloader) {
            for (int i=0; i<optimizers.size(); i++) {
                optimizers[i].zero_grad();
            }
            
            std::vector<torch::Tensor> outputs, detached_outputs;

            // Transfer images and target labels to device
            auto data = batch.data;
            auto target = batch.target;

            torch::Tensor prev_out = data;
            int k = 0;
            
            for (torch::nn::Sequential layer : layers) {
                //std::cout << "at:" << k << std::endl;
                if (batch_index != 0)
                    totaltimes.addEvent(Event(measure_type::forward, "", k));

                if (batch_index == 0) {
                    std::cout << prev_out.sizes() << "\t";
                    std::stringstream data_load;
                    torch::save(prev_out, data_load);
                    data_loads.activations.push_back(
                        dataload{k, measure_type::activations_load, data_load.tellp()});
                }

                auto output = layer->forward(prev_out);
                if (batch_index != 0)
                    totaltimes.addEvent(Event(measure_type::forward, "", k));
                if (k == avg_point) {
                    output = output.view({data.size(0), -1});
                }
                if (k != layers.size()-1) {
                    auto output_detached = output.clone().detach().requires_grad_(true);
                    detached_outputs.push_back(output_detached);
                    prev_out = output_detached;
                    outputs.push_back(output);
                }
                else {
                    prev_out = output;
                }

                k += 1;
            }
            //totaltimes.addEvent(Event(measure_type::forward, "", -1));
            if (batch_index == 0) {
                std::stringstream data_load;
                torch::save(prev_out, data_load);
                data_loads.activations.push_back(
                    dataload{k, measure_type::activations_load, data_load.tellp()});
            }
            
            if (batch_index != 0)
                totaltimes.addEvent(Event(backprop, "", layers.size()-1));
            
            torch::Tensor loss =
                   torch::nn::functional::cross_entropy(prev_out, batch.target);
            loss.backward();
            if (batch_index != 0)
                totaltimes.addEvent(Event(backprop, "", layers.size()-1));

            running_loss += loss.item<double>() * data.size(0);

            auto prediction = prev_out.argmax(1);
            num_correct += prediction.eq(target).sum().item<int64_t>();
            
            if (batch_index != 0)
                totaltimes.addEvent(Event(optimize, "", layers.size()-1));
            optimizers[optimizers.size()-1].step();
            if (batch_index != 0)
                totaltimes.addEvent(Event(optimize, "", layers.size()-1));
            
            for (int i = 0; i< detached_outputs.size(); i++) {
                auto prev_grad = detached_outputs[detached_outputs.size()-1-i].grad().clone().detach();
                
                if (batch_index != 0)
                    totaltimes.addEvent(Event(backprop, "", layers.size()-i-2));
                if (batch_index == 0) {
                    std::stringstream data_load;
                    torch::save(prev_grad, data_load);
                    data_loads.gradients.push_back(
                        dataload{k, measure_type::gradients_load, data_load.tellp()});
                }
            

                outputs[outputs.size() - 1 - i].backward(prev_grad);
                if (batch_index != 0)
                    totaltimes.addEvent(Event(backprop, "", layers.size()-i-2));

                if (batch_index != 0)
                    totaltimes.addEvent(Event(optimize, "", layers.size()-i-2));

                optimizers[optimizers.size() - 2 - i].step();

                if (batch_index != 0)
                    totaltimes.addEvent(Event(optimize, "", layers.size()-i-2));
            }

            if (batch_index != 0) {
                totaltimes.addEvent(Event(backprop, "end", -1));
                totaltimes.computeIntervals();
            }

            if (++batch_index % 16 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
                
                if (epoch == 0) {
                    totaltimes.printRes_intervals();
                    break;
                }
            }
        }

    
        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }


    std::cout << "activations load" << std::endl;
    for (int i = 0; i<data_loads.activations.size(); i++) {
        std::cout << data_loads.activations[i].data_load << "\t";
    }

    std::cout << std::endl << "gradients load" << std::endl;
    for (int i = data_loads.gradients.size() - 1; i>0; i--) {
        std::cout << data_loads.gradients[i].data_load << "\t";
    }

}
