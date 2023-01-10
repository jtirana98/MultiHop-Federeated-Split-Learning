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
    std::string model_path = "model_g.pt";
    std::string model_path_g = "model_gg.pt";
    torch::serialize::OutputArchive output_archive;
    int kTrainSize_10 = 50000;
    int kTrainSize_100 = 50000;
    int sum = 0, num_samples = kTrainSize_100, val_samples;

    if (type == CIFAR_10) {
        num_samples = kTrainSize_10;
    }
    
    val_samples = num_samples*(10.0/100);

    auto datasets = data_owners_data(path_selection, 1, type, false);
    
    /*auto validation_dataset = datasets[0]
                                .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                .map(torch::data::transforms::Stack<>());
    auto validation_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(validation_dataset), batch_size);
    */
    auto train_dataset = datasets[0]
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                    .map(ConstantPad(4))
                                    .map(RandomHorizontalFlip())
                                    .map(RandomCrop({32, 32}))
                                    .map(torch::data::transforms::Stack<>());
    auto train_samples = train_dataset.size().value();
    
    auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), batch_size);
    
    //#ifdef COMMENT
    int num_classes = (type == CIFAR_10)? 10 : 100;
    auto layers = getLayers(model_option);
    
      
    bool usebottleneck = (model_option <=2) ? false : true;
    usebottleneck = false;
    ResNet/*<Block>*/ model(layers, num_classes, usebottleneck);
    
    if(!test)
        printModelsParameters/*<Block>*/(model);
    
    // Initilize optimizer
    double weight_decay = 0.0001;  // regularization parameter

    const size_t learning_rate_decay_frequency = 8;  // number of epochs after which to decay the learning rate
    const double learning_rate_decay_factor = 10;/*1.0 / 3.0;*/
    //torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(r_learning_rate));
    torch::optim::SGD optimizer(model->parameters(), 
                torch::optim::SGDOptions(r_learning_rate).momentum(0.9).weight_decay(weight_decay));
    //torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(r_learning_rate));
    
    Total totaltimes = Total();
    int batch_index = 0;
    Event start_forward, start_backprop, start_optim, end_batch;
    Event start_batch, end_batch_;
    int stop_epochs = 1;
    if (test)
        stop_epochs = r_num_epochs;
    
    double best_loss = 100;
    int itera = 1;
    auto current_learning_rate = r_learning_rate;
    for (size_t epoch = 0; epoch != r_num_epochs; ++epoch) {
        // Initialize running metrics
        double running_loss = 0.0;
        double num_correct = 0;
        batch_index = 0;
        for (auto& batch : *train_dataloader) {
            start_batch = Event(measure_type::start_batch, "", -1);
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
            
            end_batch_ = Event(measure_type::end_batch, "", -1);
            totaltimes.addNew(start_batch, end_batch_);
            batch_index = batch_index + 1;
            if (batch_index % 15 == 0) {
                    totaltimes.printRes(-1);
                    totaltimes = Total();
                    break;
            }
            /*
            if (batch_index % 50 == 0) {
               std::cout << "Epoch: " << (epoch + 1) << " | Batch: " << batch_index
                        << " | Loss: " << loss.item<float>() << "| Acc: " << corr << std::endl;
                
                if (!test)
                    break;
                
            }
            */
            //break;
            if (itera == 32000 || itera == 48000) {
                current_learning_rate = current_learning_rate/learning_rate_decay_factor;
                static_cast<torch::optim::SGDOptions&>(optimizer.param_groups().front()
                    .options()).lr(current_learning_rate);
            }
            itera++;
        }
        if (test){ // END OF EPOCH
            /*
            if ((epoch + 1) % learning_rate_decay_frequency == 0) {
                current_learning_rate *= learning_rate_decay_factor;
                static_cast<torch::optim::AdamOptions&>(optimizer.param_groups().front()
                    .options()).lr(current_learning_rate);
            }
            */
            auto sample_mean_loss = running_loss / train_samples;
            auto accuracy = num_correct / train_samples;
           
            std::cout << "Epoch [" << (epoch + 1) << "/" << r_num_epochs << "], Trainset - Loss: "
                << sample_mean_loss << ", Accuracy: " << accuracy << " " << num_correct << std::endl;

            
            /*{ // validation set
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
                best_loss = test_sample_mean_loss;
            }
            }
            */
        }
    }

    if (test) {
        std::cout << "Training finished!\n\n";
        std::cout << "Testing...\n";

        auto test_dataset =  CIFAR(path_selection, type, false, std::set<int>(), CIFAR::Mode::kTest)
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                    .map(torch::data::transforms::Stack<>());
        auto num_test_samples = 10000;
        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                            std::move(test_dataset), batch_size);



        std::cout << "Loaded\n";
        torch::load(model, model_path);
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

//    #endif 
    
}
#ifdef COMMENT
void resnet_cifar_FL(resnet_model model_option, int type, int batch_size, int data_owners) {
    std::vector<gatherd_data> all_measures;
    std::cout << "Start training" << std::endl;
    auto path_selection = (type == CIFAR_10)? CIFAR10_data_path : CIFAR100_data_path;
    std::string model_path = "model_g.pt";
    std::string model_path_g = "model_gg.pt";
    torch::serialize::OutputArchive output_archive;
    int kTrainSize_10 = 50000;
    int kTrainSize_100 = 50000;
    int sum = 0, num_samples = kTrainSize_100, val_samples;

    if (type == CIFAR_10) {
        num_samples = kTrainSize_10;
    }

    val_samples = num_samples*(10.0/100);
    //#ifdef COMMENT
    auto datasets = data_owners_data(path_selection, data_owners, type);
    std::vector<int> train_samples(data_owners);
    
    auto validation_dataset = datasets[0]
                                .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                .map(torch::data::transforms::Stack<>());
    auto validation_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(validation_dataset), batch_size);

    std::vector<data_loaders_struct> dataloaders_(data_owners);
    
    for (int i = 0; i < data_owners; i++) {
        
        auto train_dataset = datasets[i+1]
                                    .map(ConstantPad(4))
                                    .map(RandomHorizontalFlip())
                                    .map(RandomCrop({32, 32}))
                                    .map(torch::data::transforms::Stack<>());
        dataloaders_[i].my_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), batch_size);
        
        // = &train_dataloader;

        train_samples[i] = datasets[i+1].size().value();
    }
    //#endif
    int num_classes = (type == CIFAR_10)? 10 : 100;
    auto layers = getLayers(model_option);
    
    bool usebottleneck = false;
    
    ResNet/*<Block>*/ model(layers, num_classes, usebottleneck); // This is global
    //#ifdef COMMENT
    torch::save(model, model_path);

    // store the init weights
    std::vector<ResNet> models;
    for(int i = 0; i < data_owners; i++) {
        ResNet model_(layers, num_classes, usebottleneck);
        torch::load(model_, model_path);
        models.push_back(model_);
    }
    
    // Initilize optimizer
    double weight_decay = 0.0001;  // regularization parameter
    const size_t learning_rate_decay_frequency = 8;  // number of epochs after which to decay the learning rate
    const double learning_rate_decay_factor = 10;
    torch::optim::SGD optimizer(model->parameters(), 
                torch::optim::SGDOptions(r_learning_rate).momentum(0.9).weight_decay(weight_decay));
    std::vector<torch::optim::SGD *> optimizers(data_owners, nullptr);
    for (int i = 0; i < data_owners; i++) {
        optimizers[i] = new torch::optim::SGD(models[i]->parameters(), 
                torch::optim::SGDOptions(r_learning_rate).momentum(0.9).weight_decay(weight_decay));
    }
    int frequency = 5;   
    double best_loss = 100;

    std::vector<double> running_loss(data_owners), current_learning_rate(data_owners);
    std::vector<double> num_correct(data_owners), itera(data_owners);
    
    fill(current_learning_rate.begin(), current_learning_rate.end(), r_learning_rate);
    fill(itera.begin(), itera.end(), 1);
    for (size_t epoch = 0; epoch != r_num_epochs; ++epoch) {
        // Initialize running metrics
        for(int i = 0; i < data_owners; i++) {
            for(int r=0; r < frequency; r++) {
                
                fill(running_loss.begin(), running_loss.end(), 0.0);
                fill(num_correct.begin(), num_correct.end(), 0);
                for (auto& batch : *dataloaders_[i].my_dataloader) {
                    optimizers[i]->zero_grad();

                    // Transfer images and target labels to device
                    auto data = batch.data;
                    auto target = batch.target;
                    torch::Tensor output = models[i]->forward(batch.data);

                    torch::Tensor loss =
                        torch::nn::functional::cross_entropy(output, target);

                    running_loss[i] += loss.template item<double>()* data.size(0);;
                    auto prediction = output.argmax(1);
                    auto corr = prediction.eq(target).sum().item<int64_t>()/(data.size(0)*1.0);
                    num_correct[i] += prediction.eq(target).sum().template item<int64_t>();


                    loss.backward();
                    optimizers[i]->step();
                
                    if (itera[i] == 32000 || itera[i] == 48000) {
                        current_learning_rate[i] = current_learning_rate[i]/learning_rate_decay_factor;
                        static_cast<torch::optim::SGDOptions&>(optimizers[i]->param_groups().front()
                            .options()).lr(current_learning_rate[i]);
                    }
                    itera[i]++;
                } // END OF data owner's round

                // computing training loss for data owner
                auto sample_mean_loss = running_loss[i] / train_samples[i];
                auto accuracy = num_correct[i] / train_samples[i];
            
                std::cout << "D: " << i << ": Epoch [" << (epoch + 1) << "/" << r_num_epochs << ", round: " << r << "]" << ", Trainset - Loss: "
                << sample_mean_loss << ", Accuracy: " << accuracy << std::endl;

            }
        }
        
        // fucking aggregation:  
        // iterating model from: https://stackoverflow.com/questions/54317378/set-neural-network-initial-weight-values-in-c-torch
        
        
       /*for (auto &p : model->named_parameters()) {
            std::cout << p.value()[0][0] << std::endl;
            break;
        }*/
        
        for (int i = 0; i < data_owners; i++) {
            torch::autograd::GradMode::set_enabled(false); 
            auto params = model->named_parameters(true /*recurse*/);
            auto buffers = model->named_buffers(true /*recurse*/);
            for (int j = 0; j < model->named_parameters().size(); j++) {
                //auto &[p_g, p_] : zip(model->named_parameters(), models[i]->named_parameters())
                //std::string y = p.key();
                auto p_g = model->named_parameters()[j];
                auto p_ = models[i]->named_parameters()[j];
                //std::cout << p_g.value()[0][0] << " !!!!!!!!!!!!!!!! " << p_.value()[0][0] << std::endl;
                if (i == 0) {
                     p_g.value() = p_.value();
                }
                else {
                    p_g.value() = (p_g.value()+p_.value());
                }
                p_g.value() = torch::div(p_g.value(), (kTrainSize_10/train_samples[i]));
                
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
                /*
                torch::OrderedDict<std::string, at::Tensor> to_insr("ekstra");
                to_insr.insert(p_g.key(),p_g.value());
                model->named_parameters().erase(p_g.key());
                */
                //model->named_parameters().insert(p_g.key(),p_g.value());
                //std::cout << "---------------" << std::endl;
                //std::cout << p_g.value()[0][0] << std::endl;
            }
            torch::autograd::GradMode::set_enabled(true);;
        }


        /*for (auto &p : model->named_parameters()) {
            std::cout << p.value()[0][0] << std::endl;
            break;
        }*/
        

        torch::save(model, model_path_g);

        for(int i = 0; i < data_owners; i++) {
            torch::load(models[i], model_path_g);
        } 

        {   // validation set
        double running_loss_ = 0.0;
        double num_correct_ = 0;
        for (const auto& batch : *test_loader) {
            auto data = batch.data;
            auto target = batch.target;
            auto output = model->forward(data);

            auto loss = torch::nn::functional::cross_entropy(output, target);
            running_loss_ += loss.template item<double>() * data.size(0);

            auto prediction = output.argmax(1);
            num_correct_ += prediction.eq(target).sum().template item<int64_t>();
        }
        
        if (test_sample_mean_loss < best_loss) {
            torch::save(model, model_path);
            best_loss = test_sample_mean_loss;
        }

        }
    }
    //#endif
    std::cout << "Training finished!\n\n";
    std::cout << "Testing...\n";

    auto test_dataset =  CIFAR(path_selection, type, false, std::set<int>(), CIFAR::Mode::kTest)
                                .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                                .map(torch::data::transforms::Stack<>());
    auto num_test_samples = 10000;
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                        std::move(test_dataset), batch_size);



    std::cout << "Loaded\n";
    torch::load(model, model_path);
    torch::NoGradGuard no_grad;
    model->eval();

    double running_loss_ = 0.0;
    double num_correct_ = 0;
    for (const auto& batch : *test_loader) {
        auto data = batch.data;
        auto target = batch.target;
        auto output = model->forward(data);

        auto test_accuracy = static_cast<double>(num_correct_) / num_test_samples;
        auto test_sample_mean_loss = running_loss_ / num_test_samples;

        std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
    }
    //#endif 
    
}
#endif 
/*template <typename Block>*/
void resnet_split_cifar(resnet_model model_option, int type, int batch_size, const std::vector<int>& split_points) { 
    auto layers_ = getLayers(model_option);
    bool usebottleneck = (model_option <=2) ? false : true;
    usebottleneck = false;
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
                    resnet_split_cifar/*<ResidualBlock>*/(model_option, CIFAR_10, batch_size, split_points);
                else
                    resnet_split_cifar/*<ResidualBottleneckBlock>*/(model_option, CIFAR_10, batch_size, split_points);
                break;
            case CIFAR_100:
                if (model_option <= 2)
                    resnet_split_cifar/*<ResidualBlock>*/(model_option, CIFAR_100, batch_size, split_points);
                else
                    resnet_split_cifar/*<ResidualBottleneckBlock>*/(model_option, CIFAR_100, batch_size, split_points);
                break;
            default:
                break;
            }
        }
    else {
        //if (fl) {
        //    resnet_cifar_FL/*<ResidualBlock>*/(model_option, CIFAR_10, batch_size, 5);
        //}
        //else {
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
        //}
    }
}