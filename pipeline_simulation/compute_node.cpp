#include "systemAPI.h"
#include <type_traits>


int main(int argc, char **argv) {
    int new_task;
    network_layer my_network_layer;
    systemAPI sys_;
    
    //std::thread rcv(&network_layer::receiver, &my_network_layer);
    //std::thread snd(&network_layer::receiver, &my_network_layer);

    // wait for init refactoring
    sys_.clients.push_back(1);
    //sys_.init_state(vgg, vgg_model::v11, 10, 1, 3);
    sys_.init_state(resnet, resnet_model::resnet18, 10, 9, 13);
    auto layers = sys_.clients_state.find(1)->second.layers;

    //std::cout << sys_.clients_state.find(1)->second.client_id << std::endl;
    for (int i = 0; i< layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< layers[i] << std::endl;
    }
    
    /*
    while (true) {
        new_task = my_network_layer.check_new_task();
        
    }

    rcv.join();
    snd.join();
    */
}