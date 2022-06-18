# Profiling Environment for Deep Neural Networks using pytorch C++ Frontend

This repository contains:
- `training.cpp`: Main code that implements testing environments. It contains two modes: 

    *(i)* conventional trainining: in which we run the model as one,

    *(ii)* Split Learning: the model is split into parts and is trained in a way to simulate a split learning environmet.

    This modes can be activated by commenting in/out the definintions of the following Directives in the code:
    
    *(i)* COMMENT_model

    *(ii)* COMMENT_interval

    Additionally, by commenting out the respective directive user selects a dataset.

- directory *models/* : implementation of state-of-the-art neural networks in pytorch C++ Frontend.
    - VGG models (VGG11, VG13, VGG16, VGG19) following the implementation from: [PyTorch: Source Code For torchvision.models.VGG
](https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html). Contains also a helper function `make_layers` that returns a vector of `torch::nn::Sequential` object, in which each entry is a layer from the corresponding vgg network. This helper function can be used for a split learning training.

- directory *datasets/* :

    NOTE: The pytorch C++ Frontend at this point does not support an API for dataset downloda like the torchvision.dataset does. So, make sure you have downloaded the datasets locally and provide the corresponding directories before start training.
    - MNIST: you can use this script to [download MNIST dataset](https://gist.github.com/goldsborough/6dd52a5e01ed73a642c1e772084bcd03)
    - CIFAR-10 & CIFAR-100: [source files](http://www.cs.toronto.edu/~kriz/cifar.html)

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

