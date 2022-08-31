#include "lenet_train.h"

void printModelsParameters(LeNet5& model) {
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

void lenet_cifar() {
    std::vector<gatherd_data> all_measures;

    auto path_selection = (type == 1)? CIFAR10_data_path : CIFAR100_data_path;

    auto train_dataset = CIFAR(path_selection, type)
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                    .map(torch::data::transforms::Stack<>());

    auto num_train_samples = train_dataset.size().value();

    auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), batch_size);

    int num_classes = (type == 1)? 10 : 100;
 
    LeNet5 model(num_classes);
    
    printModelsParameters(model);
    
    // Initilize optimizer
    double weight_decay = 0.0001;  // regularization parameter
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(l_learning_rate));
    //torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.001).momentum(0.9));

    Total totaltimes = Total();
    int batch_index = 0;
    Event start_forward, start_backprop, start_optim, end_batch;

    int stop_epochs = 1;
    if (test)
        stop_epochs = l_num_epochs;


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

            
            std::cout << "Epoch [" << (epoch + 1) << "/" << l_num_epochs << "], Trainset - Loss: "
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

void lenet_split_cifar(int type, int batch_size, bool test) {
    int num_classes = (type == 1)? 10 : 100;
    auto layers =  lenet_split(num_classes, split_points);

    split_cifar(layers, type, batch_size, 2, l_learning_rate, l_num_epochs);

}

void train_lenet(dataset dataset_option, 
                bool split, int batch_size = g_batch_size, 
                const std::vector<int>& split_points = std::vector<int>(), 
                bool test = false) {

    if (split) {
        switch (dataset_option) {
            case MNIST:
            //vgg_mnist(model_option, batch_size, test);
            break;
            case CIFAR_10:
                lenet_split_cifar(1, batch_size, split_points);
                break;
            case CIFAR_100:
                lenet_split_cifar(0, batch_size, split_points);
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
            lenet_cifar(1, batch_size, test);
            break;
        case CIFAR_100:
            lenet_cifar(0, batch_size, test);
            break;
        default:
            break;
        }
    }

}