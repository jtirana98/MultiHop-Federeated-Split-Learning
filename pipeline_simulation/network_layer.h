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
//#include "Message.h"

class network_layer {
 public:
    int myid;
    bool is_data_owner;
    std::map<int, std::pair<std::string, int>> rooting_table; // (node_id, (ip, port))
    std::mutex m_mutex_new_message;
    std::condition_variable m_cv_new_message;

    // pending tasks for the APP
    std::queue<Task> pending_tasks;
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
    logger_thread(&logger::logger_, &mylogger) 
    {
        rooting_table.insert({0, std::pair<std::string, int>("localhost", 8081)});
        rooting_table.insert({1, std::pair<std::string, int>("localhost", 8082)});
        rooting_table.insert({2, std::pair<std::string, int>("localhost", 8083)});
        rooting_table.insert({3, std::pair<std::string, int>("localhost", 8084)});
    }

    void findPeers(int num);
    void findInit();

    void new_message(Task task, int send_to, bool compute_to_compute=false); // produce -- new message
    void new_message(refactoring_data task, int send_to, bool compute_to_compute=false, bool rooting_table_=false); // produce -- new message
    
    void put_internal_task(Task task);
    void put_internal_task(refactoring_data task);

    //Task check_new_task(); //consumer - new task
    Task check_new_task(); //consumer - new task
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