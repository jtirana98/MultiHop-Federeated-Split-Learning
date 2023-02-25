#include <type_traits>
#include <stdlib.h>
#include <thread>
//#include <argparse/argparse.hpp> //https://github.com/p-ranav/argparse

#include "systemAPI.h"
#include "mydataset.h"
#include "transform.h"
#include "rpi_stats.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

int main(int argc, char **argv) {
    char *p;
    long conv = strtol(argv[1], &p, 10);

    int myID = conv;
    std::string log_dir = "main_experiment";
    rpi_stats my_rpi(1);
    if(myID <= 22)
        my_rpi = rpi_stats(1);
    else
        my_rpi = rpi_stats(2);

    systemAPI sys_(true, myID, log_dir);
    refactoring_data client_message;
    // check if you are the init
    if (myID == 0) {
        // POINT 5 Initialization phase: init node starts preperation
        sys_.my_network_layer.newPoint(INIT_START_MSG_PREP);

        auto cut_layers_ = "10,19";
        //auto data_owners_ = argv[2];  // CHANGE
        int num_data_owners = atoi(argv[2]);
        //std::cout << data_owners_ << std::endl;
        //auto compute_nodes_ = atoi(argv[3]);
        int num_compute_nodes = 1;

        if(argc >= 4)
            num_compute_nodes = atoi(argv[3]);

        if (num_compute_nodes == 2)
            cut_layers_ = "3,13,19";
        if (num_compute_nodes == 3)
            cut_layers_ = "2,15,25,35";//"3,8,14,19";
        //if (num_compute_nodes == 4)

        const char separator = ',';
        std::string val;
        std::vector<int>data_owners, compute_nodes, cut_layers;

        std::stringstream streamData(cut_layers_);
        while (std::getline(streamData, val, separator)) {
            if (val != "") {
                cut_layers.push_back(stoi(val));
            }
        }

        /*streamData = std::stringstream(data_owners_);
        while (std::getline(streamData, val, separator)) {
            if (val != "") {
                data_owners.push_back(stoi(val));
            }
        }*/ 

        data_owners.push_back(0);
        for (int i = 0; i < num_data_owners-1; i++) {
            data_owners.push_back(i+3 +1);
            //std::cout << "d: " << i+num_compute_nodes +1 << std::endl;
        }


        for (int i = 1; i <= num_compute_nodes; i++) {
            compute_nodes.push_back(i);
        }

        int num_parts = compute_nodes.size() + 2;

        //std::cout << "found them" << std::endl; 
        //sleep(2);

        int data_onwer_end = 2;
        int data_owner_beg = 8;

        int model_name = 2;
        int model_type = 3;
        
        client_message.dataset = CIFAR_10;
        client_message.model_name_ = model_name::resnet;//model_name::vgg;
        client_message.model_type_ =resnet_model::resnet101;//vgg_model::v19;
        client_message.end = cut_layers[0];
        client_message.start = cut_layers[cut_layers.size() - 1] + 1;
        client_message.next = compute_nodes[0];
        client_message.prev = compute_nodes[compute_nodes.size() -1];
        client_message.num_class = 10;

        auto my_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch());
        std::cout << my_time.count() << std::endl;
        
        int size_table = (int)sys_.my_network_layer.rooting_table.size();
        for (int i=1; i<data_owners.size(); i++) {

            //add data owner to rooting table
            if((data_owners[i] > 3) && (data_owners[i] < 18)) {
                std::pair<std::string, int> my_addr = sys_.my_network_layer.rooting_table.find(0)->second;
                int my_port = my_addr.second;
                my_port = my_port + (data_owners[i] +3);
                sys_.my_network_layer.rooting_table.insert({data_owners[i], std::pair<std::string, int>(my_addr.first, my_port)});
            }
            /*else if(data_owners[i] > 13 && data_owners[i] < 18) {
                std::pair<std::string, int> my_addr = sys_.my_network_layer.rooting_table.find(13)->second;
                int my_port = my_addr.second;
                my_port = my_port + (data_owners[i] - 13);
                sys_.my_network_layer.rooting_table.insert({data_owners[i], std::pair<std::string, int>(my_addr.first, my_port)});
            }
            else if (data_owners[i] > 18 && data_owners[i] < 33){
                std::pair<std::string, int> my_addr = sys_.my_network_layer.rooting_table.find(18)->second;
                int my_port = my_addr.second;
                my_port = my_port + (data_owners[i] - 18);
                sys_.my_network_layer.rooting_table.insert({data_owners[i], std::pair<std::string, int>(my_addr.first, my_port)});
            }
            else if (data_owners[i] > 33 && data_owners[i] < 43) {
                std::pair<std::string, int> my_addr = sys_.my_network_layer.rooting_table.find(33)->second;
                int my_port = my_addr.second;
                my_port = my_port + (data_owners[i] - 33);
                sys_.my_network_layer.rooting_table.insert({data_owners[i], std::pair<std::string, int>(my_addr.first, my_port)});
            }*/
            else if (data_owners[i] >= 18) {
                std::pair<std::string, int> my_addr = sys_.my_network_layer.rooting_table.find(18)->second;
                int my_port = my_addr.second;
                my_port = my_port + (data_owners[i] - 18);
                sys_.my_network_layer.rooting_table.insert({data_owners[i], std::pair<std::string, int>(my_addr.first, my_port)});
            }

            sys_.my_network_layer.new_message(client_message, data_owners[i], false, true);
        }

        // send to aggregator:
        sys_.my_network_layer.new_message(client_message, -1, false, true);

        sys_.refactor(client_message);
        
        client_message.to_data_onwer = false;
        client_message.data_owners = data_owners;
        
        for (int i=0; i<compute_nodes.size(); i++) {
            
            client_message.start = cut_layers[i] + 1;
            client_message.end = cut_layers[i+1];

            if (i == compute_nodes.size() - 1) 
                client_message.next = -1;
            else
                client_message.next = compute_nodes[i+1];

            if (i == 0) 
                client_message.prev = -1;
            else
                client_message.prev = compute_nodes[i-1];

           sys_.my_network_layer.new_message(client_message, compute_nodes[i], false, true);
        }
    }
    else { // if not wait for init refactoring
        client_message = sys_.my_network_layer.check_new_refactor_task();
        sys_.refactor(client_message);
    }

    std::cout << "loading data..." << std::endl;
    // load dataset
    int type = client_message.dataset;
    auto path_selection = (type == CIFAR_10)? CIFAR10_data_path : CIFAR100_data_path;
    auto datasets = data_owners_data(path_selection, 1, type, false);
    auto train_dataset = datasets[0]
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2018, 0.1994, 0.2010}))
                                    .map(ConstantPad(4))
                                    .map(RandomHorizontalFlip())
                                    .map(RandomCrop({32, 32}))
                                    .map(torch::data::transforms::Stack<>());
    auto num_train_samples = train_dataset.size().value();
    //std::cout << train_dataset.size().value() << std::endl;
    auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), sys_.batch_size);

    int num_classes = (type == CIFAR_10)? 10 : 100;
    
    std::cout << sys_.parts[0].layers.size() << std::endl;
    std::cout << sys_.parts[1].layers.size() << std::endl;
    for (int i = 0; i< sys_.parts[0].layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< sys_.parts[0].layers[i] << std::endl;
    }
   
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    
    for (int i = 0; i< sys_.parts[1].layers.size(); i++) {
        std::cout << "new layer: "<< i+1 << " "<< sys_.parts[1].layers[i] << std::endl;
    }
    
    auto init_epoch = std::chrono::steady_clock::now();
    auto send_activations = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    auto send_gradients = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    int epoch_count = 0, g_epoch_count = 0; // communication round
    bool new_r = true;
    for (size_t round = 0; round != sys_.rounds; ++round) {
        int batch_index = 0;
        sys_.zero_metrics();
        int total_num = 0;
        if (new_r) {
            //std::cout << "New round " << std::endl;
            init_epoch = std::chrono::steady_clock::now();
            new_r = false;
        }

        send_activations = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        //long c = send_activations.count();
        //std::cout << c << std::endl;
        //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << std::endl;
        int g_i = 0;
        for (int inter_batch = 0; inter_batch < 4; inter_batch++ ) {    
            for (auto& batch : *train_dataloader) {
                // create task with new batch
                auto init_batch = std::chrono::steady_clock::now();
                
                Task task(sys_.myid, forward_, -1);
                task.size_ = batch.data.size(0);
                task.values = batch.data;
                task = sys_.exec(task, batch.target);
                //task.t_start = send_activations.count();
                //std::cout << task.t_start << std::endl;
                total_num += task.size_; 
                task.batch0 = batch_index;
                auto end_f1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                
                long real_duration = 0;
                real_duration = my_rpi.rpi_fm1;
                if (batch_index != 0) {
                    real_duration = real_duration + my_rpi.rpi_bm1;
                }

                if (end_f1-send_activations.count() > real_duration) {
                    std::cout << "Model part 1: Cannot Simulate " << (end_f1-send_activations.count() - real_duration) << std::endl;
                } 
                else{
                    //std::cout << "go to sleep " << real_duration-(end_f1-send_activations.count()) << std::endl;
                    
                    usleep(real_duration-(end_f1-send_activations.count()));
                }
                
                //std::cout << "f1-end: " << end_f1-send_activations.count() << std::endl;
                // send task to next node
                task.t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                //usleep(myID*200);
                std::cout << "Send forward task to C1 " << end_f1-send_activations.count() << std::endl;
                sys_.my_network_layer.new_message(task, sys_.inference_path[0]);
                

                // wait for next forward task
                task = sys_.my_network_layer.check_new_task();

                send_gradients = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
                task = sys_.exec(task, batch.target); // forward and backward
                // send task - backward
                auto end_m2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                //std::cout << "m2-end: " << end_m2-send_gradients.count() << std::endl;
                
                real_duration = 0;
                real_duration = my_rpi.rpi_fbm2;
                
                if (end_m2-send_gradients.count() > real_duration) {
                    std::cout << "Model part last: Cannot Simulate " << (end_m2-send_gradients.count() - real_duration)<< std::endl;
                }
                else{
                    //std::cout << "go to sleep " << real_duration-(end_m2-send_gradients.count()) << std::endl;
                    usleep(real_duration-(end_m2-send_gradients.count()));
                }
                
                //task.t_start = send_gradients.count();
                task.t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                //usleep(myID*110);
                std::cout << "Send backprop task to C1 " << end_f1-send_activations.count() << std::endl;
                sys_.my_network_layer.new_message(task, sys_.inference_path[1]);
                //optimize task
                
                auto task1 = sys_.my_network_layer.check_new_task();

                task1 = sys_.exec(task1, batch.target); // optimize

                // wait for next backward task
                task = sys_.my_network_layer.check_new_task();
                task = sys_.exec(task, batch.target); //backward and optimize


                send_activations = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
                auto end_batch = std::chrono::steady_clock::now();
                auto _time = std::chrono::duration_cast<std::chrono::milliseconds>
                            (end_batch - init_batch).count();
                
                //if (g_i % 50 == 0)
                    std::cout << "One batch: global epoch " << g_epoch_count+1 << " local epoch: " << epoch_count+1 <<" b: " << batch_index+1  << " is " << _time << std::endl;
                
                // end of batch
                batch_index++;
                g_i++;
            }
        }
        epoch_count++;

        if(epoch_count <  2)
            continue;
        
        epoch_count = 0;
        g_epoch_count++;
        new_r = true;
        // stdout end of round
        auto aggr_beg = std::chrono::steady_clock::now();
        // new epoch
        std::cout << "sending to the aggegator" << std::endl;
        auto newAggTask = Task(myID, operation::aggregation_, -1);
        newAggTask.model_part = 1;

        // send aggregation task
        newAggTask.model_part_=sys_.parts[0].layers[0];
        newAggTask.t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        sys_.my_network_layer.new_message(newAggTask,-1);
        auto next_task = sys_.my_network_layer.check_new_task();
        std::stringstream ss(std::string(next_task.model_parts.begin(), next_task.model_parts.end()));
        torch::load(sys_.parts[0].layers[0], ss);

        std::cout << "received global firt part model" << std::endl;

        newAggTask.model_part = 2;
        for (int i = 0; i < sys_.parts[1].layers.size(); i++) {
            // << newAggTask.model_part << std::endl;
            newAggTask.model_part_=sys_.parts[1].layers[i];
            newAggTask.t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            sys_.my_network_layer.new_message(newAggTask,-1);

            newAggTask.model_part++;
        }

        for (int i = 0; i < sys_.parts[1].layers.size(); i++) {
            //std::cout << "wait " << i << " " << sys_.parts[1].layers.size() << std::endl;
            next_task = sys_.my_network_layer.check_new_task();
            std::stringstream sss(std::string(next_task.model_parts.begin(), next_task.model_parts.end()));
            torch::load(sys_.parts[1].layers[next_task.model_part-2], sss);
        }

        auto end_epoch = std::chrono::steady_clock::now();
        auto _time = std::chrono::duration_cast<std::chrono::milliseconds>
                    (end_epoch - init_epoch).count();

        auto _time2 = std::chrono::duration_cast<std::chrono::milliseconds>
                    (end_epoch - aggr_beg).count();
        std::cout << "aggregation: " << _time2 << std::endl;
        std::cout << "One epoch e: " << g_epoch_count << " took: " << _time << std::endl;
        
    }
}
