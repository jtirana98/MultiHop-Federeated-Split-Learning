# testing_environment

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
    - MNIST: 
    - CIFAR-10 & CIFAR-100:

How to run program:

   - Requirments:
    
        - Create conda environment with python version $\geq 3.7$
        - Install pytorch from [Pytorch website](https://pytorch.org/get-started/locally/).

  - How to compile:
        
        > Activate python environment.
        > Create folder `\build`.
        > Run command: ``cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..`` to create makefiles.
        > Run command: `cmake --build . --config Release`.

What we measure:

