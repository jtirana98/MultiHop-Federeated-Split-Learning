#include "resnet_train.h"
#include <cstdlib>
#include <set>
#include <memory>

/*template <typename Block>*/
void printModelsParameters(ResNet/*<Block>*/& model) {
    for (const auto& p : model->parameters()) {
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
/*template <typename Block>*/
void resnet_cifar(resnet_model model_option, int type, int batch_size, bool test) {
    std::vector<gatherd_data> all_measures;
    std::cout << "Start training" << std::endl;
    auto path_selection = (type == CIFAR_10)? CIFAR10_data_path : CIFAR100_data_path;
    std::string model_path = "model.pt";
    torch::serialize::OutputArchive output_archive;
    
    int num_samples, val_samples, train_samples;
    std::set<int> validation;
    srand((unsigned) time(NULL));
    int sum = 0;
    int kTrainSize_10 = 50000;
    int kTrainSize_100 = 50000;
    if (type == CIFAR_10) {
        num_samples = kTrainSize_10;
    }
    else {
        num_samples = kTrainSize_100;
    }

    val_samples = num_samples*(10.0/100);
    train_samples = num_samples - val_samples;
    
    int random = rand() % num_samples;
    validation.insert(random);
    while(validation.size() < val_samples) {
        random = rand() % num_samples;
        validation.insert(random);
        //std::cout << random << std::endl;
    }
    
    
    auto train_dataset = CIFAR(path_selection, type, false, validation)
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                    .map(torch::data::transforms::Stack<>());

    
    auto validation_dataset = CIFAR(path_selection, type, true, validation)
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                    .map(torch::data::transforms::Stack<>());
     

    auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), batch_size);
    
    auto validation_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(validation_dataset), batch_size);

    auto num_train_samples = train_dataset.size().value();
    auto num_valid_samples = validation_dataset.size().value();

    //std::cout << "train: " << num_train_samples /*<< " val: " << num_valid_samples*/ << std::endl;
    
    //#ifdef COMMENT
    int num_classes = (type == CIFAR_10)? 10 : 100;
    auto layers = getLayers(model_option);
    
      
    bool usebottleneck = (model_option <=2) ? false : true;
    ResNet/*<Block>*/ model(layers, num_classes, usebottleneck);
    
    if(!test)
        printModelsParameters/*<Block>*/(model);
    
    // Initilize optimizer
    double weight_decay = 0.0001;  // regularization parameter

    //torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(r_learning_rate));
    torch::optim::SGD optimizer(model->parameters(), 
                torch::optim::SGDOptions(r_learning_rate).momentum(0.9).weight_decay(weight_decay));

    Total totaltimes = Total();
    int batch_index = 0;
    Event start_forward, start_backprop, start_optim, end_batch;

    int stop_epochs = 1;
    if (test)
        stop_epochs = r_num_epochs;
    
    double best_loss = 100;
    for (size_t epoch = 0; epoch != 0; ++epoch) {
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
            if (!test)
                start_backprop = Event(backprop, "", -1);

            torch::Tensor loss =
                   torch::nn::functional::cross_entropy(output, target);

            running_loss += loss.template item<double>()* data.size(0);;
            auto prediction = output.argmax(1);
            auto corr = prediction.eq(target).sum().item<int64_t>()/(data.size(0)*1.0);
            num_correct += prediction.eq(target).sum().template item<int64_t>();


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
               std::cout << "Epoch: " << (epoch + 1) << " | Batch: " << batch_index
                        << " | Loss: " << loss.item<float>() << "| Acc: " << corr << std::endl;
                
                if (!test)
                    break;
                
            }
            //break;
        }
        if (test){ // END OF EPOCH
            
            auto sample_mean_loss = running_loss / train_samples;
            auto accuracy = num_correct / train_samples;
           
            std::cout << "Epoch [" << (epoch + 1) << "/" << r_num_epochs << "], Trainset - Loss: "
                << sample_mean_loss << ", Accuracy: " << accuracy << " " << num_correct << std::endl;

            
            { // validation set
            running_loss = 0.0;
            num_correct = 0;
            torch::NoGradGuard no_grad;
            batch_index = 0;
            for (const auto& batch : *validation_dataloader) {
                auto data = batch.data;
                auto target = batch.target;

                auto output = model->forward(data);

                auto loss = torch::nn::functional::cross_entropy(output, target);
                running_loss += loss.template item<double>() * data.size(0);

                auto prediction = output.argmax(1);
                num_correct += prediction.eq(target).sum().template item<int64_t>();
                batch_index = batch_index + 1;
            }

            auto test_accuracy = static_cast<double>(num_correct) / val_samples;
            auto test_sample_mean_loss = running_loss / val_samples;

            std::cout << "Epoch [" << (epoch + 1) << " Validation - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';

            
            if (test_sample_mean_loss < best_loss) {
                torch::save(model, model_path);
                //model->save(output_archive);
                //output_archive.save_to(model_path);
            }

            }
        }
    }

    if (test) {
        std::cout << "Training finished!\n\n";
        std::cout << "Testing...\n";

        auto test_dataset =  CIFAR(path_selection, type, false, validation, CIFAR::Mode::kTest)
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                    .map(torch::data::transforms::Stack<>());
        auto num_test_samples = 10000;
        std::cout << test_dataset.size().value() <<std::endl;
        //test_dataset.size().value();
        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                            std::move(test_dataset), batch_size);



        std::cout << "Loaded\n";

        // Test the model

        
        //try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
        torch::load(model, model_path);
       /* }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
            return ;
        }
        */
        torch::NoGradGuard no_grad;
        model->eval();

        double running_loss = 0.0;
        size_t num_correct = 0;
        
        for (const auto& batch : *test_loader) {
            auto data = batch.data;
            auto target = batch.target;

            auto output = model->forward(data);

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

    //#endif 
    
}

/*template <typename Block>*/
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
                    resnet_split_cifar/*<ResidualBlock>*/(model_option, 1, batch_size, split_points);
                else
                    resnet_split_cifar/*<ResidualBottleneckBlock>*/(model_option, 1, batch_size, split_points);
                break;
            case CIFAR_100:
                if (model_option <= 2)
                    resnet_split_cifar/*<ResidualBlock>*/(model_option, 0, batch_size, split_points);
                else
                    resnet_split_cifar/*<ResidualBottleneckBlock>*/(model_option, 0, batch_size, split_points);
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
                resnet_cifar/*<ResidualBlock>*/(model_option, CIFAR_10, batch_size, test);
            else
                resnet_cifar/*<ResidualBottleneckBlock>*/(model_option, CIFAR_10, batch_size, test);
            break;
        case CIFAR_100:
            if (model_option <= 2)
                resnet_cifar/*<ResidualBlock>*/(model_option, CIFAR_100, batch_size, test);
            else
                resnet_cifar/*<ResidualBottleneckBlock>*/(model_option, CIFAR_100, batch_size, test);
            break;
        default:
            break;
        }
    }
}