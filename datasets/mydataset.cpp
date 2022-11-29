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
std::pair<torch::Tensor, torch::Tensor> read_data(const std::string& root, bool train, int type) {
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

    auto targets = torch::empty(num_samples, torch::kByte);
    auto images = torch::empty({num_samples, 3, kImageRows, kImageColumns}, torch::kByte);

    //const auto bytes_row = (type == CIFAR_10) ? kBytesPerRow_10 : kBytesPerRow_100;
    //std::cout << "samples: " << num_samples << " validation: " << validation.size() << std::endl;
    for (uint32_t i = 0; i != num_samples; ++i) {
        // The first byte of each row is the target class index.
        uint32_t start_index = i * kBytesPerRow;
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

CIFAR::CIFAR(std::pair<torch::Tensor, torch::Tensor>data, int type, Mode mode) : mode_(mode), type(type) {
    images_ = std::move(data.first);
    targets_ = std::move(data.second);
}

//#ifdef COMMENT

CIFAR::CIFAR(const std::string& root, int type, bool val, std::set<int> validation, Mode mode) : mode_(mode), type(type){
    auto data = read_data(root, mode == Mode::kTrain, type/*, val, validation*/);

    //std::cout << data.first.size(0) << std::endl;
    images_ = std::move(data.first);
    targets_ = std::move(data.second);
}

//#endif

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


std::vector<CIFAR> data_owners_data(const std::string& root, int data_owners, int type) {
    std::cout << type << std::endl;

    auto data = read_data(root,/*mode=*/ true, type);
    std::cout << data << std::endl;
    auto images_ = std::move(data.first);
    auto targets_ = std::move(data.second);
    std::vector<CIFAR> datasets;

    int val_samples = kTrainSize*(10.0/100);
    int train_samples = (kTrainSize - val_samples);
    
    std::set<int> validation;
    srand((unsigned) time(NULL));
    std::cout << val_samples << std::endl;
    int random = rand() % kTrainSize;
    validation.insert(random);
    while(validation.size() < val_samples) {
        random = rand() % kTrainSize;
        validation.insert(random);
        //std::cout << random << std::endl;
    }
    // for validation set:

    auto targets = torch::empty(val_samples, torch::kByte);
    auto images = torch::empty({val_samples, 3, kImageRows, kImageColumns}, torch::kByte);

   std::set<int>::iterator it;
   int i = 0;
    for(it = validation.begin(); it!=validation.end(); ++it){
        int ans = *it;
        images[i] = images_[ans];
        targets[i] = targets_[ans];
        i++;
    }
    std::cout << train_samples
     << std::endl;
    
    datasets.push_back(CIFAR(std::pair<torch::Tensor, torch::Tensor>{images, targets}, type));
    /*
    std::vector<std::pair<torch::Tensor, torch::Tensor>>data_owners_;
    for (i = 0; i < data_owners; i++ ) {

        targets = torch::empty(train_samples, torch::kByte);
        images = torch::empty({train_samples, 3, kImageRows, kImageColumns}, torch::kByte);

        data_owners_.push_back(std::pair<torch::Tensor, torch::Tensor>{images, targets});

    }
    */
    targets = torch::empty(train_samples, torch::kByte);
    images = torch::empty({train_samples, 3, kImageRows, kImageColumns}, torch::kByte);
    int j =0;
    for (i = 0; i < kTrainSize; i++) {
        if (validation.find(i) == validation.end()) {
            continue;
        }

        images[j] = images_[i];
        targets[j] = targets_[i];
        j++;
    }

    datasets.push_back(CIFAR(std::pair<torch::Tensor, torch::Tensor>{images, targets}, type));

    return(datasets);
}
