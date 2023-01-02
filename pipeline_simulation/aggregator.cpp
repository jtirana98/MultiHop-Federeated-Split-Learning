#include "systemAPI.h"
#include <type_traits>
#include <stdlib.h>
#include <thread>

#include <argparse/argparse.hpp> //https://github.com/p-ranav/argparse

int main(int argc, char **argv) {
    argparse::ArgumentParser program("aggregator");

    program.add_argument("-d", "--data_owners")
        .help("The number of data owners")
        .required()
        .nargs(1)
        .scan<'d', int>();

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto num_clients = program.get<int>("-d");

    systemAPI sys_(false, -1, "main_experiment");
    sys_.my_network_layer.findPeers(num_clients, true);

    std::cout << "found them" << std::endl;    
    sleep(2);

    while (true) {
        auto next_task = sys_.my_network_layer.check_new_task();
        auto client = next_task.client_id;
        next_task.check_ = true;
        sys_.my_network_layer.new_message(next_task,client);
        
    }

}