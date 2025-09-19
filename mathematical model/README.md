- horizontal.py: used for experiments: "cost and training time of SplitPipe vs. horizontally scaled Parallel SL". It simulates the horizontally scaled parallalel SL. 
- ../optimize_split/split_model.py: used for the above experiments but for the SplitPipe configuration
        - the splits are optimized with the solver

- fully_utilized_resnet and fully_utilized_vgg have the latest code for the cost model.

-resenet_heter_net.py: script for resnet for the comparison of splitpipe and splitNN in the heterogeneous context (for resnet)
    - I used the resnet_1_d2.dat file and the splits: [2,30],[2,21,36],[2,17,26,36]


-vgg_heter_net.py: script for resnet for the comparison of splitpipe and splitNN in the heterogeneous context (for vgg)
    - I used the vgg_iot2.dat file and the splits: [12,20],[4,12,20],[4,11,15,23]

