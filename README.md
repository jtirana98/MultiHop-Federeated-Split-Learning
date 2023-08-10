<p align="center">
  <img src="./logo.png" alt="Sublime's custom image"/>
</p>

--------------------------------------------------------------------------------

For a more detailed discription of the documentation follow this [link](https://docs.google.com/document/d/1DaWOX27c4_4_VUT-l_UrgUV-zFa8UsIZ5zUv06pgc0s/edit?usp=sharing)  or check the wiki.


Repository structure:

  
- directory *datasets/* :
    Contains source code to load a dataset into a dataloader.

    **NOTE1**: The pytorch C++ Frontend at this point does not support an API for dataset downloda like the torchvision.dataset does. So, make sure you have downloaded the datasets locally and provide the corresponding directories before start training.

    **NOTE2**: Inside the file `datasets\mydataset.h` we define the file path that lead to the dataset's files. Please change this path accordintly to where you aim to store the datasets on your machine.

    we declare the enum mydataset::dataset to use one of the options below:
    - 'MNIST': you can use this script to [download MNIST dataset](https://gist.github.com/goldsborough/6dd52a5e01ed73a642c1e772084bcd03)
    - 'CIFAR-10' or 'CIFAR-100': [source files](http://www.cs.toronto.edu/~kriz/cifar.html)

- directory *models/* : implementation of state-of-the-art neural networks in pytorch C++ Frontend.

- *main.cpp* : Main code that is used to profile or train the models.
- directory *utils/*: Contains the libraries for logging and split learning training.

- directory *pipeline_simulation/*: Here we implement SplitFL's compoments.
    - module: *Split Learning engine*:
        -   State.h
        -   Task.h
        -   systemAPI.cpp
        -   systemAPI.h: contains the API of the module.
    -   module *Task delivery*: 
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
        - cmake > 3.
        - c++17 and gcc > 8.
        - Create a directory *third_party/* and git clone the code for [argparse](https://github.com/p-ranav/argparsehttps://github.com/p-ranav/argparse).

  - In order to compile the framework just use the Makefile, and follow the steps below:
        
        $ source /filepath/to/python/environment %Activate python environment.
        $ mkdir build 
        $ cd build
        $ cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
        $ cmake --build . --config Release


Running SplitFL in a distributed manner:

*Case 0: Model profiling*
An example code is in main, you can either get the delay for each batch or get the per-layer delay.

*Case 1: Real system*

In this case you will run the data owners as real devices. You can run all entities in one machine or use different devices (within the same network)

- If you cannot use multicast:
    - comment the following parts in the code:
        - in data_owner.cpp: Comment the findPeers() call and the findInit()
        - in compute_node.cpp: Comment the findInit()
        - in aggregator.cpp: Comment the findInit()
        - in network_layer.cpp: comment the line 506 of versio 1.0.0
    - update the rooting table in pipeline_simulation/network_layer.h

For each data owner device call: 

    $ ./data_owner -i id -d <number-of-data-owners> -c <number-of-compute-nodes> -s <split-rule>

If not an init data owner you just give the node's id

For each compute node device call:

    $ ./compute_node -i id

or use script run_cn.sh in pipeline_simulation/profiling

For the aggregator:

    $ ./aggregator -i id -d <number-of-data-owners> -c <number-of-compute-nodes>

or use script run_aggr.sh in pipeline_simulation/profiling

NOTE: There is support for logging and checkpoining but this feature is deactivated for this version. You can use the utils/pipeline_logging.sh to do so.

*Case 3: Emulated environment*

In this case the data owners are running in an emulated environmet. Note that this version does not supprt multicast. 
You can add in the pipeline_simulation/profiling/rpi_stat.h the device characteristics and use the script run_data_owners_init.sh and run_data_owners_worker.sh in 
pipeline_simulation/profiling. 
The results are stored to logging files as are indicated in the script files (change them accordingly)

The code for the emulated data owner is in pipeline_simulation/profiling/data_owner_simulatede.cpp
