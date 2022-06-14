#pragma once
#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <string>

// root files

const std::string MNIST_data_path = "/Users/joanatirana/Documents/coding/datasets/MNIST/";
const std::string CIFAR10_data_path = "/Users/joanatirana/Documents/coding/datasets/cifar/cifar-10-batches-bin/";
const std::string CIFAR100_data_path = "/Users/joanatirana/Documents/coding/datasets/cifar/cifar-100-binary/";

// MNISR
//torch::data::datasets::MNIST getMNIST_train();
//torch::data::datasets::MNIST getMNIST_test();


// CIFAR
class CIFAR : public torch::data::datasets::Dataset<CIFAR> {
 public:
    enum Mode { kTrain, kTest };
    explicit CIFAR(const std::string& root, int type=1, Mode mode = Mode::kTrain); //binary version
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
    bool is_train() const noexcept;
    const torch::Tensor& images() const;
    const torch::Tensor& targets() const;

 private:
    torch::Tensor images_;
    torch::Tensor targets_;
    Mode mode_;
    int type; // 1 for CIFAR-10 else for CIFAR-100
};

//CIFAR getCIFAR_train(int type);
//CIFAR getCIFAR_test(int type);

// ImageNet
