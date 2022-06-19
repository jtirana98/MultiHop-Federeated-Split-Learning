#include "mydataset.h"
#include "transform.h"

#include <tuple>

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

namespace {
// CIFAR10 dataset description can be found at https://www.cs.toronto.edu/~kriz/cifar.html.
constexpr uint32_t kTrainSize_10 = 50000;
constexpr uint32_t kTrainSize_100 = 50000;
constexpr uint32_t kTestSize_10 = 10000;
constexpr uint32_t kTestSize_100 = 100;
constexpr uint32_t kSizePerBatch = 10000;
constexpr uint32_t kImageRows = 32;
constexpr uint32_t kImageColumns = 32;
constexpr uint32_t kBytesPerRow_10 = 3073;
constexpr uint32_t kBytesPerRow_100 = 3074;
constexpr uint32_t kBytesPerChannelPerRow = 1024;
constexpr uint32_t kBytesPerBatchFile_10 = kBytesPerRow_10 * kSizePerBatch;
constexpr uint32_t kBytesPerBatchFile_train_100 = kBytesPerRow_100 * kTrainSize_100;
constexpr uint32_t kBytesPerBatchFile_test_100 = kBytesPerRow_100 * kTestSize_100;

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
std::pair<torch::Tensor, torch::Tensor> read_data(const std::string& root, bool train, int type) {
    const auto& files1 = train ? kTrainDataBatchFiles_10 : kTestDataBatchFiles_10;
    const auto& files2 = train ? kTrainDataBatchFiles_100 : kTestDataBatchFiles_100;
    uint32_t num_samples, size_;
    if (type == 1) {
        num_samples = train ? kTrainSize_10 : kTestSize_10;
    }
    else {
        num_samples = train ? kTrainSize_100 : kTestSize_100;
    }
    const auto& files = (type==1) ? files1 : files2;
    std::vector<char> data_buffer;
    if (type == 1) {
        data_buffer.reserve(files.size() * kBytesPerBatchFile_10);
        size_ = kBytesPerBatchFile_10;
    }
    else {
        size_  = train ? kBytesPerBatchFile_train_100 : kBytesPerBatchFile_test_100; 
        data_buffer.reserve(files.size() * size_);
    }

    for (const auto& file : files) {
        const auto path = join_paths(root, file);
        std::ifstream data(path, std::ios::binary);
        TORCH_CHECK(data, "Error opening data file at", path);

        data_buffer.insert(data_buffer.end(), std::istreambuf_iterator<char>(data), {});
    }

    TORCH_CHECK(data_buffer.size() == files.size() * size_, "Unexpected file sizes");

    auto targets = torch::empty(num_samples, torch::kByte);
    auto images = torch::empty({num_samples, 3, kImageRows, kImageColumns}, torch::kByte);

    const auto bytes_row = (type == 1) ? kBytesPerRow_10 : kBytesPerRow_100;

    for (uint32_t i = 0; i != num_samples; ++i) {
        // The first byte of each row is the target class index.
        uint32_t start_index = i * bytes_row;
        targets[i] = data_buffer[start_index];

        // The next bytes correspond to the rgb channel values in the following order:
        // red (32 *32 = 1024 bytes) | green (1024 bytes) | blue (1024 bytes)
        uint32_t image_start = start_index + 1;
        uint32_t image_end = image_start + 3 * kBytesPerChannelPerRow;
        std::copy(data_buffer.begin() + image_start, data_buffer.begin() + image_end,
            reinterpret_cast<char*>(images[i].data_ptr()));
    }

    return {images.to(torch::kFloat32).div_(255), targets.to(torch::kInt64)};
}
}  // namespace

CIFAR::CIFAR(const std::string& root, int type, Mode mode) : mode_(mode), type(type) {
    auto data = read_data(root, mode == Mode::kTrain, type);

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
