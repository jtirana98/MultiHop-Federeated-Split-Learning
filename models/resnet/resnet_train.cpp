#include "resenet_train.h"

auto getModel(resenet_model model_option, int num_classes) {
    switch (model_option) {
    case resnet18:
        std::array<int64_t, 4> layers{2, 2, 2, 2};
        ResNet<ResidualBlock> model(layers, num_classes);
        break;
    case resnet34:
        std::array<int64_t, 4> layers{3, 4, 6, 3};
        ResNet<ResidualBlock> model(layers, num_classes);
        break;
    case resnet50:
        std::array<int64_t, 4> layers{3, 4, 6, 3};
        ResNet<ResBottleneckBlock> model(layers, num_classes, true);
        break;
    case resnet101:
        std::array<int64_t, 4> layers{3, 4, 23, 3};
        ResNet<ResBottleneckBlock> model(layers, num_classes, true);
        break;
    case resenet152:
        std::array<int64_t, 4> layers{3, 8, 36, 3};
        ResNet<ResBottleneckBlock> model(layers, num_classes, true);
        break;
    }
}

// CIFAR
void resnet_cifar(vgg_model model_option, int type, int batch_size, bool test) {
    std::vector<gatherd_data> all_measures;

    auto path_selection = (type == 1)? CIFAR10_data_path : CIFAR100_data_path;

    auto train_dataset = CIFAR(path_selection, type)
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                    .map(torch::data::transforms::Stack<>());

    auto num_train_samples = train_dataset.size().value();

    auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), batch_size);

    int num_classes = (type == 1)? 10 : 100;
    auto model = getModel(model_option, num_classes);

    printModelsParameters(model);
    
    // Initilize optimizer
    double weight_decay = 0.0001;  // regularization parameter
    //torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate).momentum(0.9).weight_decay(weight_decay));
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.001).momentum(0.9));

    Total totaltimes = Total();
    int batch_index = 0;
    Event start_forward, start_backprop, start_optim, end_batch;

    int stop_epochs = 1;
    if (test)
        stop_epochs = num_epochs;


    for (size_t epoch = 0; epoch != stop_epochs; ++epoch) {
        // Initialize running metrics
        double running_loss = 0.0;
        double num_correct = 0;

        batch_index = 0;
        for (auto& batch : *train_dataloader) {
            optimizer.zero_grad();

            // Transfer images and target labels to device
            auto data = batch.data;
            auto target = batch.target;

            // Forward
            if (!test)
                start_forward = Event(forward, "", -1);
            
            torch::Tensor output = model->forward(batch.data);
            
            /*
            if (!test ){
                std::cout << output << std::endl;
                std::cout << std::endl;
                std::cout << target << std::endl;
                std::cout << std::endl;
                std::cout << prediction << std::endl;
            }
            */

            torch::Tensor loss =
                   torch::nn::functional::cross_entropy(output, target);

            running_loss += loss.item<double>();
            auto prediction = output.argmax(1);
            auto corr = prediction.eq(target).sum().item<int64_t>();
            auto corr_ = static_cast<double>(corr)/data.size(0);
            num_correct += corr_;

            if (!test)
                start_backprop = Event(backprop, "", -1);

            loss.backward();

            if (!test)
                start_optim = Event(backprop, "", -1);
            optimizer.step();
            if (!test)
                end_batch = Event(backprop, "", -1);

            if (!test)
                totaltimes.addNew(start_forward, start_backprop, start_optim, end_batch);

            batch_index = batch_index + 1;
            if (!test && (batch_index % 15 == 0)) {
                    totaltimes.printRes();
                    break;
            }
            
            if (batch_index % 15 == 0) {
               std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                        << " | Loss: " << loss.item<float>() << "| Acc: " << corr_ << std::endl;
                
                if (!test)
                    break;
                
            }
            
        }

        if (test){
            auto sample_mean_loss = running_loss / batch_index;
            auto accuracy = num_correct / batch_index;

            
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
                << sample_mean_loss << ", Accuracy: " << accuracy << " " << num_correct << std::endl;
        }    

    }

    if (test) {
        auto test_dataset =  CIFAR(path_selection, type, CIFAR::Mode::kTest)
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                    .map(torch::data::transforms::Stack<>());
        
        auto num_test_samples = test_dataset.size().value();

        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                            std::move(test_dataset), batch_size);



        std::cout << "Training finished!\n\n";
        std::cout << "Testing...\n";

        // Test the model
        model->eval();
        torch::NoGradGuard no_grad;

        double running_loss = 0.0;
        size_t num_correct = 0;

        for (const auto& batch : *test_loader) {
            auto data = batch.data;
            auto target = batch.target;

            auto output = model->forward(data);

            auto loss = torch::nn::functional::cross_entropy(output, target);
            running_loss += loss.item<double>() * data.size(0);

            auto prediction = output.argmax(1);
            num_correct += prediction.eq(target).sum().item<int64_t>();
        }

        std::cout << "Testing finished!\n";

        auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
        auto test_sample_mean_loss = running_loss / num_test_samples;

        std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
    }

}

void train_resnet(dataset dataset_option, resenet_model model_option, bool split, int batch_size, const std::vector<int>& split_points, bool test) {
    if (split) {
        switch (dataset_option) {

    }
    else {
        switch (dataset_option) {
        case MNIST:
            //vgg_mnist(model_option, batch_size, test);
            break;
        case CIFAR_10:
            resnet_cifar(model_option, 1, batch_size, test);
            break;
        case CIFAR_100:
            resnet_cifar(model_option, 0, batch_size, test);
            break;
        default:
            break;
        }

    }
}