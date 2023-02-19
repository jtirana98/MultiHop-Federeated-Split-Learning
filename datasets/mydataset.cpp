#include "mydataset.h"
#include "transform.h"

#include <tuple>
#include <cstdlib>

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

namespace {
// CIFAR10 dataset description can be found at https://www.cs.toronto.edu/~kriz/cifar.html.
constexpr uint32_t kTrainSize = 50000;
constexpr uint32_t k_temp = 500;
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
    num_samples = k_temp;
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
    std::cout << data.second.sizes() << std::endl;
    images_ = std::move(data.first);
    targets_ = std::move(data.second);
}

//#ifdef COMMENT

CIFAR::CIFAR(const std::string& root, int type, bool val, std::set<int> validation, Mode mode) : mode_(mode), type(type){
    auto data = read_data(root, mode == Mode::kTrain, type/*, val, validation*/);

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


std::vector<CIFAR> data_owners_data(const std::string& root, int data_owners, int type, bool val) {
    auto data = read_data(root,/*mode=*/ true, type);
    auto images_ = std::move(data.first);
    auto targets_ = std::move(data.second);
    std::vector<CIFAR> datasets;

    int val_samples = kTrainSize*(10.0/100);
    int train_samples = (kTrainSize - val_samples)/data_owners;
    train_samples = k_temp;
    if(val == false) {
        datasets.push_back(CIFAR(std::pair<torch::Tensor, torch::Tensor>
                        {images_, targets_}, type));
        return(datasets);
    }
    
    std::set<int> validation;
    srand((unsigned) time(NULL));
    int random = rand() % kTrainSize;
    validation.insert(random);
    
    while(validation.size() < val_samples) {
        random = rand() % kTrainSize;
        validation.insert(random);
    }

    // for validation set:
    auto targets = torch::empty(1, torch::kByte);
    auto images = torch::empty({1, 3, kImageRows, kImageColumns}, torch::kByte);

    std::set<int>::iterator it;
    int i = 0;
    for(it = validation.begin(); it!=validation.end(); ++it){
        int ans = *it;
        images = torch::cat({images, images_[ans].unsqueeze(0)});
        //targets[i] = targets_[ans].item<int64_t>();
        targets = torch::cat({targets, targets_[ans].unsqueeze(0)});
        i++;
    }
    
    datasets.push_back(CIFAR(std::pair<torch::Tensor, torch::Tensor>
                        {images.index({Slice(1, None), None, None, None})
                        .squeeze(1).squeeze(1).squeeze(1),
                        targets.index({Slice(1, None)})}, type));
    
    std::vector<std::pair<torch::Tensor, torch::Tensor>>data_owners_;
    for (i = 0; i < data_owners; i++ ) {
        targets = torch::empty(1, torch::kByte);
        images = torch::empty({1, 3, kImageRows, kImageColumns}, torch::kByte);

        data_owners_.push_back(std::pair<torch::Tensor, torch::Tensor>{images, targets});
    }

    std::vector<int> vect_seq(data_owners); //number of elements in vector
    int stop=1000;
    fill(vect_seq.begin(), vect_seq.end(), 0);
    
    int j =0, who;
    std::cout << "---------------" << std::endl;

    
    for (i = 0; i < kTrainSize; i++) {
        if (validation.find(i) != validation.end()) {
            continue;
        }
        who = rand() % data_owners;
        if ((data_owners_[who].first.sizes()[0] >= train_samples) || (vect_seq[who] >= stop)) { // search for other who
            who = (who+1)%data_owners;

        }
        for (int k = 0; k < data_owners; k++) {
            if (k == who)
                vect_seq[k] += 1;
            else
                vect_seq[k] = 0;
        }

        data_owners_[who].first = torch::cat({data_owners_[who].first, images_[i].unsqueeze(0)});
        data_owners_[who].second = torch::cat({data_owners_[who].second, targets_[i].unsqueeze(0)});
        /*
        if (j >= train_samples) {
            data_owners_[who].second = torch::cat({data_owners_[who].second, targets_[i].unsqueeze(0)});
            std::cout << "-------here----------" << std::endl;
            std::cout << data_owners_[who].second.sizes() << std::endl;
            std::cout << "-----------------" << std::endl;
            std::cout << data_owners_[who].second << std::endl;
            std::cout << targets_[i].unsqueeze(0) << std::endl;
        }
        else {
            data_owners_[who].second[j] = targets_[i].item<int64_t>();
        }
        */
        j++;
    }
    
    for (i = 0; i < data_owners; i++) {
        std::cout << i << ": "<< data_owners_[i].first.sizes() << std::endl;
        datasets.push_back(CIFAR(std::pair<torch::Tensor, torch::Tensor>
                        {data_owners_[i].first.index({Slice(1, None), None, None, None})
                        .squeeze(1).squeeze(1).squeeze(1),
                        data_owners_[i].second.index({Slice(1, None)})}, type));
    }
    
    return(datasets);
}
