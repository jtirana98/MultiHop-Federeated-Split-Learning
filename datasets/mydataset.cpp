#include "mydataset.h"
#include "transform.h"

#include <tuple>
#include <cstdlib>

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

namespace {
// CIFAR10 dataset description can be found at https://www.cs.toronto.edu/~kriz/cifar.html.
constexpr uint32_t kTrainSize = 50000;
constexpr uint32_t kTestSize = 10000;
constexpr uint32_t kSizePerBatch = 10000;
constexpr uint32_t kImageRows = 32;
constexpr uint32_t kImageColumns = 32;
constexpr uint32_t kBytesPerRow = 3073;
constexpr uint32_t kBytesPerChannelPerRow = 1024;
constexpr uint32_t kBytesPerBatchFile = kBytesPerRow * kSizePerBatch;

std::vector<std::string> kTrainDataBatchFiles_10 = {
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
};

const std::vector<std::string> kTrainDataBatchFiles_100 = {
    "train.bin"
};

const std::vector<std::string> kTestDataBatchFiles_10 = {
    "test_batch.bin"
};

const std::vector<std::string> kTestDataBatchFiles_100 = {
    "test.bin"
};

// Source: https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/data/datasets/mnist.cpp.
std::string join_paths(std::string head, const std::string& tail) {
    if (head.back() != '/') {
        head.push_back('/');
    }
    head += tail;
    return head;
}
// Partially based on https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/data/datasets/mnist.cpp.
std::pair<torch::Tensor, torch::Tensor> read_data(const std::string& root, bool train, int type, bool val, std::set<int> &validation) {
    const auto& files1 = train ? kTrainDataBatchFiles_10 : kTestDataBatchFiles_10;
    const auto& files2 = train ? kTrainDataBatchFiles_100 : kTestDataBatchFiles_100;
    uint32_t num_samples;
    const auto& files = (type==CIFAR_10) ? files1 : files2;

    num_samples = train ? kTrainSize : kTestSize;
    
    std::vector<char> data_buffer;
    data_buffer.reserve(files.size() * kBytesPerBatchFile);

    for (const auto& file : files) {
        const auto path = join_paths(root, file);
        std::ifstream data(path, std::ios::binary);
        TORCH_CHECK(data, "Error opening data file at", path);

        data_buffer.insert(data_buffer.end(), std::istreambuf_iterator<char>(data), {});
    }

    TORCH_CHECK(data_buffer.size() == files.size() * kBytesPerBatchFile, "Unexpected file sizes");


    int num_samples_;
    if(train) {
            if(val) {
                num_samples_ = validation.size();
            }
            else {
                num_samples_ = num_samples - validation.size();
            }
    }
    else {
        num_samples_ = num_samples;
    }


    auto targets = torch::empty(num_samples_, torch::kByte);
    auto images = torch::empty({num_samples_, 3, kImageRows, kImageColumns}, torch::kByte);

    //const auto bytes_row = (type == CIFAR_10) ? kBytesPerRow_10 : kBytesPerRow_100;
    //std::cout << "samples: " << num_samples << " validation: " << validation.size() << std::endl;
    int j = 0;
    for (uint32_t i = 0; i != num_samples; ++i) {
        bool con = false;
        if(train) {
            if(val) {
                if (validation.find(i) == validation.end()) {
                    con = true;
                }
            }
            else {
                if (validation.find(i) != validation.end()) {
                    con = true;
                }
            }
        }
        // The first byte of each row is the target class index.
        uint32_t start_index = i * kBytesPerRow;
        if(con)
            auto ignore = data_buffer[start_index];
        else {
            targets[j] = data_buffer[start_index];
        }

        // The next bytes correspond to the rgb channel values in the following order:
        // red (32 *32 = 1024 bytes) | green (1024 bytes) | blue (1024 bytes)
        uint32_t image_start = start_index + 1;
        uint32_t image_end = image_start + 3 * kBytesPerChannelPerRow;
        if (con) {
            auto ignore_ = torch::empty({1, 3, kImageRows, kImageColumns}, torch::kByte); 
            std::copy(data_buffer.begin() + image_start, data_buffer.begin() + image_end,
                reinterpret_cast<char*>(ignore_.data_ptr()));
        }        
        else {
            std::copy(data_buffer.begin() + image_start, data_buffer.begin() + image_end,
                reinterpret_cast<char*>(images[j].data_ptr()));
            j++;
        }
    }

    return {images.to(torch::kFloat32).div_(255), targets.to(torch::kInt64)};
}
}  // namespace

CIFAR::CIFAR(const std::string& root, int type, bool val, std::set<int> validation, Mode mode) : mode_(mode), type(type){
    auto data = read_data(root, mode == Mode::kTrain, type, val, validation);

    //std::cout << data.first.size(0) << std::endl;
    images_ = std::move(data.first);
    targets_ = std::move(data.second);
}

torch::data::Example<> CIFAR::get(size_t index) {
    return {images_[index], targets_[index]};
}

torch::optional<size_t> CIFAR::size() const {
    return images_.size(0);
}

bool CIFAR::is_train() const noexcept {
    return mode_ == Mode::kTrain;
}

const torch::Tensor& CIFAR::images() const {
    return images_;
}
