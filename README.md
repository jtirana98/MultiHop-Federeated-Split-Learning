# Pipelined federated split learning with multiple hops

The repository structure:

For a more detailed discription of the documentation follow this [link](https://docs.google.com/document/d/1DaWOX27c4_4_VUT-l_UrgUV-zFa8UsIZ5zUv06pgc0s/edit?usp=sharing)    
- directory *datasets/* :
    Contains source code to load a dataset into a dataloader.

    **NOTE1**: The pytorch C++ Frontend at this point does not support an API for dataset downloda like the torchvision.dataset does. So, make sure you have downloaded the datasets locally and provide the corresponding directories before start training.

    **NOTE2**: Inside the file `datasets\mydataset.h` we define the file path that lead to the dataset's files. Please change this path accordintly to where you aim to store the datasets on your machine.

    we declare the enum mydataset::dataset to use one of the options below:
    - 'MNIST': you can use this script to [download MNIST dataset](https://gist.github.com/goldsborough/6dd52a5e01ed73a642c1e772084bcd03)
    - 'CIFAR-10' or 'CIFAR-100': [source files](http://www.cs.toronto.edu/~kriz/cifar.html)

- directory *models/* : implementation of state-of-the-art neural networks in pytorch C++ Frontend.

- *main.cpp*
- directory *utils/*: Contains the libraries for logging and split learning training.

- directory *pipeline_simulation/*: Here we implement SplitPipe's compoments.
    - module: Split Learning engine consist of:
        -   State.h
        -   Task.h
        -   systemAPI.cpp
        -   systemAPI.h: contains the API of the module.
    -   module Task delivery: 
        - Message.h
        - network_layer.cpp
        - network_layer.h: contains the API of the module.
    - Main Entities:
        - compute_node.cpp
        - data_owner.cpp
        - aggregator.cpp
    - directory *profiling/*: Is used to emulate data owners and run experiments with a large number of data owners. 


How to run program and connect Libtorch:

   - Requirments:
    
        - Create python environment for package managment with python version $\geq 3.7$
            - For example you can use [Python Virtual Environments](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/)
        - Install pytorch from [Pytorch website](https://pytorch.org/get-started/locally/).

  - In order to compile the framework just use the Makefile, and follow the steps below:
        
        $ source /filepath/to/python/environment %Activate python environment.
        $ mkdir build 
        $ cd build
        $ cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
        $ cmake --build . --config Release


Running SplitPipe in a distributed manner:

- configuring root-table
    - enable mulit-task (if applicable)
- parameters for each entity.
- include a figure of the structure.
- emulated version.
