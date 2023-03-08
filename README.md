# Pipelines federated split learning with multiple hops

This repository contains:
- *`main.cpp`*: Main code that implements testing environments. It contains two modes: 

    *(i)* conventional trainining: in which we run the model as one,

    *(ii)* Split Learning: the model is split into parts and is trained in a way to simulate a split learning environmet.
    
- directory *`datasets/`* :
    Contains source code to load a dataset into a dataloader.

    **NOTE1**: The pytorch C++ Frontend at this point does not support an API for dataset downloda like the torchvision.dataset does. So, make sure you have downloaded the datasets locally and provide the corresponding directories before start training.

    **NOTE2**: Inside the file `datasets\mydataset.h` we define the file path that lead to the dataset's files. Please change this path accordintly to where you aim to store the datasets on your machine.

    we declare the enum `mydataset::dataset` to use one of the options below:
    - 'MNIST': you can use this script to [download MNIST dataset](https://gist.github.com/goldsborough/6dd52a5e01ed73a642c1e772084bcd03)
    - 'CIFAR-10' or 'CIFAR-100': [source files](http://www.cs.toronto.edu/~kriz/cifar.html)

- directory *`models/`* : implementation of state-of-the-art neural networks in pytorch C++ Frontend.
    - VGG models (VGG11, VG13, VGG16, VGG19) following the implementation from: [PyTorch: Source Code For torchvision.models.VGG
](https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html).
         - to select the VGG model for profiling, use function:
        `train_vgg(dataset dataset_option, vgg_model model_option, bool split, vector<int> split_points)`
            - `dataset_option`: select dataset
            - `model_option`: value from `vgg::vgg_model` enum to select the vgg model: v11, v11_bn, v13, v13_bn, v16, v16_bn, v19, v19_bn.
            - `split`: *true* if you want to profile upon the split mode.
            - `split_points`: optional parameter (no impact if `split == `*`false`*). It is the vector with cut layers.
    - ResNet models (resnet18, resnet34, resnet50, resnet101, resenet152) following the implementation from: [Github: ResNet_PyTorch.ipynb
](https://github.com/liao2000/ML-Notebook/blob/main/ResNet/ResNet_PyTorch.ipynb).
         - to select the ResNet model for profiling, use function:
        `train_resnet(dataset dataset_option, resnet_model model_option, bool split, int batch_size = 64, const std::vector<int>& split_points)`
            - `dataset_option`: select dataset
            - `model_option`: value from `resnet_model` enum to select the ResNet model: resnet18, resnet34, resnet50, resnet101, resenet152.
            - `split`: *true* if you want to profile upon the split mode.
            - `split_points`: optional parameter (no impact if `split == `*`false`*). It is the vector with cut layers.
How to run program:

   - Requirments:
    
        - Create python environment for package managment with python version $\geq 3.7$
            - For example you can use [Python Virtual Environments](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/)
        - Install pytorch from [Pytorch website](https://pytorch.org/get-started/locally/).

  - To compile run the commands below:
        
        > source /filepath/to/python/environment %Activate python environment.
        > mkdir build 
        > cd build
        > cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
        > cmake --build . --config Release

What we measure:

