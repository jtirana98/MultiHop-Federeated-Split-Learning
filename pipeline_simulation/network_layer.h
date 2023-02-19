#ifndef _NETWORK_LAYER_H_
#define _NETWORK_LAYER_H_

#include <iostream>
#include <tuple>
#include <iterator>
#include <condition_variable>
#include <map>
#include <string>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <sys/time.h>
#include <string.h>
#include <netdb.h>
#include <sstream>
#include <vector>


#include "Task.h"
#include "pipeline_logging.h"
#include "rpi_stats.h"
//#include "Message.h"

class network_layer {
 public:
    int myid;
    bool is_data_owner;
    std::map<int, std::pair<std::string, int>> rooting_table; // (node_id, (ip, port))
    std::mutex m_mutex_new_message;
    std::condition_variable m_cv_new_message;
    bool sim_forw = false;
    bool sim_back = false;
    rpi_stats my_rpi;
    int num_data_owners;
    int history = 0;
    bool forward_step = true;

    // pending tasks for the APP
    //std::queue<Task> pending_tasks;
    std::vector<std::pair<long, Task>> pending_tasks;
    std::mutex m_mutex_new_task;
    std::condition_variable m_cv_new_task;

    // pending tasks for the APP - compute node 
    std::queue<Task> pending_tasks_;
    std::mutex m_mutex_new_task_;
    std::condition_variable m_cv_new_task_;

    // pending refactor messages for the APP
    std::queue<refactoring_data> pending_refactor_tasks;
    std::mutex m_mutex_new_refactor_task;
    std::condition_variable m_cv_new_refactor_task;

    logger mylogger;
    std::thread logger_thread;

    network_layer(int myid, std::string log_dir, bool is_data_owner) : myid(myid), 
    is_data_owner(is_data_owner),
    mylogger(myid, log_dir),
    logger_thread(&logger::logger_, &mylogger) ,
    my_rpi(1)
    {   
        rooting_table.insert({-2, std::pair<std::string, int>("10.96.12.132", 8079)});
        rooting_table.insert({-1, std::pair<std::string, int>("10.96.12.136", 8080)});
        rooting_table.insert({0, std::pair<std::string, int>("10.96.12.138", 8081)});
        rooting_table.insert({1, std::pair<std::string, int>("10.96.12.130", 8082)}); //cn1
        rooting_table.insert({2, std::pair<std::string, int>("10.96.12.131", 8083)}); //cn2
        rooting_table.insert({3, std::pair<std::string, int>("10.96.12.132", 8083)}); //cn3
        rooting_table.insert({28, std::pair<std::string, int>("10.96.12.139", 8081)});
        //rooting_table.insert({13, std::pair<std::string, int>("10.96.12.139", 8081)});
        //rooting_table.insert({23, std::pair<std::string, int>("10.96.12.133", 8081)});
        //rooting_table.insert({33, std::pair<std::string, int>("10.96.12.132", 8081)});
        //rooting_table.insert({43, std::pair<std::string, int>("10.96.12.131", 8081)});
    }

    void findPeers(int num, bool aggr = false);
    void findInit(bool aggr = false);

    void new_message(Task task, int send_to, bool compute_to_compute=false); // produce -- new message
    void new_message(refactoring_data task, int send_to, bool compute_to_compute=false, bool rooting_table_=false); // produce -- new message
    
    void put_internal_task(Task task, long timestamp=-1, bool back=false);
    void put_internal_task(refactoring_data task);

    //Task check_new_task(); //consumer - new task
    Task check_new_task(bool back=false); //consumer - new task
    refactoring_data check_new_refactor_task(); //consumer - new task

    //threads
    void receiver(); // producer -- new task
    void sender(); // consumer -- new message

    Point newPoint(int point_code, int id=-1, std::string op="n/a") {
        Point point(myid, point_code);
        this->mylogger.add_point(point);

        return point;
    }
    
    void terminate() {
        logger_thread.join();
    }

};

#endif
