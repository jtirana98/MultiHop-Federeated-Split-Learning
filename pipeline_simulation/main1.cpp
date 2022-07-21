//#include "utils.h"
#include "network_layer.h"
#include "vgg.h"

int main(int argc, char **argv) {
    int new_task;
    network_layer my_network_layer;
    
    std::thread rcvr(&network_layer::receiver, &my_network_layer);


    while (true) {
        new_task = my_network_layer.check_new_task();
        std::cout << "new task " << new_task << std::endl;
    }

    rcvr.join();

}