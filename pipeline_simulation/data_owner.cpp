#include <type_traits>
#include "systemAPI.h"
#include "mylogging.h"
#include "mydataset.h"
#include "transform.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

int main(int argc, char **argv) {
    int myID=0; // parameter

    // check if you are the init

    // if not wait for init refactoring
    
    systemAPI sys_(true, myID);

    // load dataset
    int type = 1;
    auto path_selection = (type == 1)? CIFAR10_data_path : CIFAR100_data_path;
    
    auto train_dataset = CIFAR(path_selection, type)
                                    .map(ConstantPad(4))
                                    .map(RandomHorizontalFlip())
                                    .map(RandomCrop({32, 32}))
                                    .map(torch::data::transforms::Stack<>());
    auto num_train_samples = train_dataset.size().value();

    auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), sys_.batch_size);

    int num_classes = (type == 1)? 10 : 100;
    
    //sys_.init_model_sate(resnet, resnet_model::resnet18, num_classes, 4, 3);
    sys_.init_model_sate(vgg, vgg_model::v11, num_classes, 4, 3);
    
    for (int i = 0; i< sys_.parts[0].layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< sys_.parts[0].layers[i] << std::endl;
    }

    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

    for (int i = 0; i< sys_.parts[1].layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< sys_.parts[1].layers[i] << std::endl;
    }
    
    for (size_t round = 0; round != sys_.rounds; ++round) {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        int batch_index = 0;
        std::cout << "here00" << std::endl;
        for (auto& batch : *train_dataloader) {
            // create task with new batch
            std::cout << "here0" << std::endl;
            Task task(sys_.myid, forward_, -1);
            task.size_ = batch.data.size(0);
            task.values = batch.data;
            std::cout << "here1" << std::endl;
            // call task function
            task = sys_.exec(task, batch.target);
            std::cout << "here2" << std::endl;
            // send task to next node
            // ...
            // wait for next forward task


            // send task
            task = sys_.exec(task, batch.target);
            // wait for next backward task
            std::cout << "here3" << std::endl;
            // exec task
            task = sys_.exec(task, batch.target);
            std::cout << "here4" << std::endl;

        }
        
    }
}