#include "resnet_train.h"

template <typename Block>
void printModelsParameters(ResNet<Block>& model) {
    for (const auto& p : model.parameters()) {
        int dims = p.dim();
        std::cout << "=";
        for (int i=0; i < dims; i++) {
            std::cout << p.size(i);
            if (i!=dims-1) 
                std::cout << "*";
        }

        std::cout << "\t";
        
    }
    std::cout << std::endl;
}

// CIFAR
template <typename Block>
void resnet_cifar(resnet_model model_option, int type, int batch_size, bool test) {
    std::vector<gatherd_data> all_measures;
    std::cout << "Start training" << std::endl;
    auto path_selection = (type == CIFAR_10)? CIFAR10_data_path : CIFAR100_data_path;
    std::string model_path = "model.pt";
    torch::serialize::OutputArchive output_archive;
    auto train_dataset = CIFAR(path_selection, type)
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                    .map(torch::data::transforms::Stack<>());

    auto num_train_samples = train_dataset.size().value();

    auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), batch_size);

    int num_classes = (type == CIFAR_10)? 10 : 100;
    auto layers = getLayers(model_option);
    
      
    bool usebottleneck = (model_option <=2) ? false : true;
    ResNet<Block> model(layers, num_classes, usebottleneck);
    
    if(!test)
        printModelsParameters<Block>(model);
    
    // Initilize optimizer
    double weight_decay = 0.001;  // regularization parameter
    //torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(r_learning_rate));
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(weight_decay).momentum(0.9));

    Total totaltimes = Total();
    int batch_index = 0;
    Event start_forward, start_backprop, start_optim, end_batch;

    int stop_epochs = 1;
    if (test)
        stop_epochs = r_num_epochs;

    double best_acc = -100;
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
            
            torch::Tensor output = model.forward(batch.data);
            
            if (!test)
                start_backprop = Event(backprop, "", -1);

            torch::Tensor loss =
                   torch::nn::functional::cross_entropy(output, target);

            running_loss += loss.template item<double>();
            auto prediction = output.argmax(1);
            auto corr = prediction.eq(target).sum().item<int64_t>();
            auto corr_ = static_cast<double>(corr)/data.size(0);
            num_correct += corr_;


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
            
            if (batch_index % 50 == 0) {
               std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                        << " | Loss: " << loss.item<float>() << "| Acc: " << corr_ << std::endl;
                
                if (!test)
                    break;
                
            }
        }

        if (test){
            auto sample_mean_loss = running_loss / batch_index;
            auto accuracy = num_correct / batch_index;
            /*
            if (accuracy > best_acc) {
                model.save(output_archive);
                output_archive.save_to(model_path);
            }
            */
            std::cout << "Epoch [" << (epoch + 1) << "/" << r_num_epochs << "], Trainset - Loss: "
                << sample_mean_loss << ", Accuracy: " << accuracy << " " << num_correct << std::endl;
        }    

    }

    if (test) {
        auto test_dataset =  CIFAR(path_selection, type, CIFAR::Mode::kTest)
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                    .map(ConstantPad(4))
                                    .map(RandomHorizontalFlip())
                                    .map(RandomCrop({32, 32}))
                                    .map(torch::data::transforms::Stack<>());
        
        auto num_test_samples = test_dataset.size().value();

        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                            std::move(test_dataset), batch_size);



        std::cout << "Training finished!\n\n";
        std::cout << "Testing...\n";

        // Test the model
        model.eval();
        torch::NoGradGuard no_grad;

        double running_loss = 0.0;
        size_t num_correct = 0;

        for (const auto& batch : *test_loader) {
            auto data = batch.data;
            auto target = batch.target;

            auto output = model.forward(data);

            auto loss = torch::nn::functional::cross_entropy(output, target);
            running_loss += loss.template item<double>() * data.size(0);

            auto prediction = output.argmax(1);
            num_correct += prediction.eq(target).sum().template item<int64_t>();
        }

        std::cout << "Testing finished!\n";

        auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
        auto test_sample_mean_loss = running_loss / num_test_samples;

        std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
    }
    
}

template <typename Block>
void resnet_split_cifar(resnet_model model_option, int type, int batch_size, const std::vector<int>& split_points) { 
    auto layers_ = getLayers(model_option);
    bool usebottleneck = (model_option <=2) ? false : true;
    int num_classes = (type == CIFAR_10)? 10 : 100;
    auto layers =  resnet_split(layers_, num_classes, usebottleneck, split_points);

    split_cifar(layers, type, batch_size, 1, r_learning_rate, r_num_epochs);
}

void train_resnet(dataset dataset_option, resnet_model model_option, bool split, int batch_size, const std::vector<int>& split_points, bool test) {
    if (split) {
        switch (dataset_option) {
            case MNIST:
            //vgg_mnist(model_option, batch_size, test);
            break;
            case CIFAR_10:
                if (model_option <= 2)
                    resnet_split_cifar<ResidualBlock>(model_option, 1, batch_size, split_points);
                else
                    resnet_split_cifar<ResidualBottleneckBlock>(model_option, 1, batch_size, split_points);
                break;
            case CIFAR_100:
                if (model_option <= 2)
                    resnet_split_cifar<ResidualBlock>(model_option, 0, batch_size, split_points);
                else
                    resnet_split_cifar<ResidualBottleneckBlock>(model_option, 0, batch_size, split_points);
                break;
            default:
                break;
            }
        }
    else {
        switch (dataset_option) {
        case MNIST:
            //vgg_mnist(model_option, batch_size, test);
            break;
        case CIFAR_10:
            if (model_option <= 2)
                resnet_cifar<ResidualBlock>(model_option, CIFAR_10, batch_size, test);
            else
                resnet_cifar<ResidualBottleneckBlock>(model_option, CIFAR_10, batch_size, test);
            break;
        case CIFAR_100:
            if (model_option <= 2)
                resnet_cifar<ResidualBlock>(model_option, CIFAR_100, batch_size, test);
            else
                resnet_cifar<ResidualBottleneckBlock>(model_option, CIFAR_100, batch_size, test);
            break;
        default:
            break;
        }
    }
}