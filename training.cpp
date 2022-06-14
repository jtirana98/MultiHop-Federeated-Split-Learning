#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>

#include "mylogging.h"
#include "mydataset.h"
#include "transform.h"
#include "vgg.h"
#include "vgg_help.h"

//#define MNIST_
#define CIFAR_10_
//#define CIFAR_100_

//#define COMMENT
//#define COMMENT_model
#define COMMENT_interval
//#define DATALOAD

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

enum dataset{
    MNIST=1,
    CIFAR_10=2,
    CIFAR_100=3,
    Imagenet=4
};


int main(int argc, char **argv){
    // Hyperparameters
    const int64_t input_size = 784;
    const int64_t hidden_size = 500;
    const int64_t num_classes = 10;
    const int64_t batch_size = 128;
    const double learning_rate = 0.001;
    const size_t num_epochs = 1;

    std::vector<gatherd_data> all_measures;
    dataset data_type = MNIST;

    static std::unordered_map<std::string,dataset> const table = { {"MNIST",dataset::MNIST}, {"CIFAR_10",dataset::CIFAR_10},
                                                                    {"Imagenet",dataset::Imagenet}, {"CIFAR_100",dataset::CIFAR_100} };
    /*
    if (argc >= 2) {
        std::cout << argv[1] << std::endl;
        std::string arg1(argv[1]);
        auto it = table.find(arg1);
        if (it!= table.end())
            data_type = it->second;
        else {
            std::cout << "Wrong arg" << std::endl;
            return 1;
        }
    }
    */
   
    #ifdef MNIST_
        auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
                            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                            .map(torch::data::transforms::Stack<>());
        auto num_train_samples = train_dataset.size().value();
        auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                            .map(torch::data::transforms::Stack<>());

        auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                std::move(train_dataset), batch_size);

        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_dataset), batch_size);

     #endif
     #ifdef CIFAR_10_
        auto train_dataset = CIFAR(CIFAR10_data_path, 1)
                                    .map(ConstantPad(4))
                                    .map(RandomHorizontalFlip())
                                    .map(RandomCrop({32, 32}))
                                    .map(torch::data::transforms::Stack<>());
        
        auto num_train_samples = train_dataset.size().value();
        auto test_dataset =  CIFAR(CIFAR10_data_path, 1, CIFAR::Mode::kTest)
                                 .map(torch::data::transforms::Stack<>());

        auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                std::move(train_dataset), batch_size);

        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_dataset), batch_size);
    
    #endif

    #ifdef CIFAR_100_
        auto train_dataset = CIFAR(CIFAR100_data_path, 0)
                                    .map(ConstantPad(4))
                                    .map(RandomHorizontalFlip())
                                    .map(RandomCrop({32, 32}))
                                    .map(torch::data::transforms::Stack<>());
        auto num_train_samples = train_dataset.size().value();
        auto test_dataset =  CIFAR(CIFAR100_data_path, 0, CIFAR::Mode::kTest)
                                 .map(torch::data::transforms::Stack<>());

        auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                std::move(train_dataset), batch_size);

        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                std::move(test_dataset), batch_size);
    #endif

    // Initialize model with no split
    #ifdef COMMENT_model
    auto model = vgg19(10);
    //auto model = *model;
    #endif
    
    #ifdef COMMENT
    for (const auto& p : model->parameters()) {
        //std::cout << "new layer: " << p.dim() << std::endl;
        int dims = p.dim();
        std::cout << "=";
        for (int i=0; i < dims; i++) {
            //std::cout << i << ": " << p.size(i) << "    ";
            std::cout << p.size(i);
            if (i!=dims-1) 
                std::cout << "*";
        }

        std::cout << "\t";
        
    }
    std::cout << std::endl;
    #endif

    #ifdef COMMENT_model
    // Initilize optimizer
    double weight_decay = 0.0001;  // regularization parameter
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
    #endif

    #ifdef COMMENT_interval
    // Initialize model with splits
    auto layers = vgg19_split(10);
    std::vector<torch::optim::Adam> optimizers;
    
    for (int i=0; i<layers.size(); i++) {
        optimizers.push_back(torch::optim::Adam(layers[i]->parameters(), torch::optim::AdamOptions(learning_rate)));
        #ifdef COMMENT
        std::cout << layers[i] << std::endl;
        #endif
    }

    #endif

    Total totaltimes = Total();
    #ifdef COMMENT_model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        int batch_index = 0;
    
        // Get Total Time in Average
        for (auto& batch : *train_loader) {
            optimizer.zero_grad();

            // Transfer images and target labels to device
            auto data = batch.data;
            auto target = batch.target;

            // Forward
            Event start_forward(forward, "", -1);
            torch::Tensor output = model->forward(batch.data);
            //std::cout << "output " << output.dim() << std::endl;
            //std::cout << "output " << output.size(1) << std::endl;
            
            torch::Tensor loss =
                   torch::nn::functional::cross_entropy(output, batch.target);

            running_loss += loss.item<double>() * data.size(0);

            auto prediction = output.argmax(1);
            num_correct += prediction.eq(target).sum().item<int64_t>();
            Event start_backprop(backprop, "", -1);
            // Compute gradients of the loss and parameters of our model
            loss.backward();

            // Update the parameters based on the calculated gradients.
            optimizer.step();
            Event end_batch(backprop, "", -1);
            // Output the loss every 10 batches.

            totaltimes.addNew(start_forward, start_backprop, end_batch);

            if (++batch_index % 15 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
                
                if (epoch == 0) {
                    totaltimes.printRes();
                    break;
                }
            }
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';

    }
    #endif

    #ifdef DATALOAD
    gatherd_data data_loads;
    #endif

    #ifdef COMMENT_interval
    // Get Interval Times
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        int batch_index = 0;
        for (auto& batch : *train_loader) {
            for (int i=0; i<optimizers.size(); i++) {
                optimizers[i].zero_grad();
            }
            
            std::vector<torch::Tensor> outputs, detached_outputs;

            // Transfer images and target labels to device
            auto data = batch.data;
            auto target = batch.target;

            torch::Tensor prev_out = data;
            // Forward
            int k = 0;
            
            for (torch::nn::Sequential layer : layers) {
                totaltimes.addEvent(Event(measure_type::forward, "", k));
                // get data load:
                #ifdef DATALOAD
                
                if (batch_index == 0) {
                    std::cout << prev_out.sizes() << "\t";
                    std::stringstream data_load;
                    torch::save(prev_out, data_load);
                    //std::cout << data_load.tellp() << "\t";
                    data_loads.activations.push_back(
                        dataload{k, measure_type::activations_load, data_load.tellp()});
                }
                #endif

                auto output = layer->forward(prev_out);
                //std::cout << k << " : " << layer << std::endl;

                if (k == layers.size()-4) {
                    #ifdef DATALOAD
                    if (batch_index == 0)
                        std::cout << "-"<< output.sizes() << "\t";
                    #endif
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

            #ifdef DATALOAD
                
            if (batch_index == 0) {
                std::cout << prev_out.sizes() << "\t";
                std::stringstream data_load;
                torch::save(prev_out, data_load);
                //std::cout << data_load.tellp() << "\t";
                data_loads.activations.push_back(
                    dataload{k, measure_type::activations_load, data_load.tellp()});
                std::cout << std::endl;
            }
            
            #endif

            totaltimes.addEvent(Event(backprop, "", layers.size()-1));
            torch::Tensor loss =
                   torch::nn::functional::cross_entropy(prev_out, batch.target);

            running_loss += loss.item<double>() * data.size(0);

            auto prediction = prev_out.argmax(1);
            num_correct += prediction.eq(target).sum().item<int64_t>();
            loss.backward();
            optimizers[optimizers.size()-1].step();
            
            
            for (int i = 0; i< detached_outputs.size(); i++) {
                //std::cout << "ok" << i << std::endl;
                totaltimes.addEvent(Event(backprop, "", i));
                auto prev_grad = detached_outputs[detached_outputs.size()-1-i].grad().clone().detach();
                
                #ifdef DATALOAD
                if (batch_index == 0) {
                    std::stringstream data_load;
                    torch::save(prev_grad, data_load);
                    data_loads.gradients.push_back(
                        dataload{k, measure_type::gradients_load, data_load.tellp()});
                }
                #endif
                
                outputs[outputs.size() - 1 - i].backward(prev_grad);
                optimizers[optimizers.size() - 2 - i].step();
            }

            totaltimes.addEvent(Event(backprop, "end", -1));
            totaltimes.computeIntervals();
            if (++batch_index % 15 == 0) {
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

    #ifdef DATALOAD
    std::cout << "activations load" << std::endl;
    for (int i = 0; i<data_loads.activations.size(); i++) {
        std::cout << data_loads.activations[i].data_load << "\t";
    }

    std::cout << std::endl << "gradients load" << std::endl;
    for (int i = data_loads.gradients.size() - 1; i>0; i--) {
        std::cout << data_loads.gradients[i].data_load << "\t";
    }
    #endif

    #endif
}