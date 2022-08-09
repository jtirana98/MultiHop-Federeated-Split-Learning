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
//#include "Message.h"

class network_layer {
 public:
    int myid;
    // pending messages 
    std::map<int, std::pair<std::string, int>> rooting_table; // (node_id, (ip, port))
    std::mutex m_mutex_new_message;
    std::condition_variable m_cv_new_message;

    // pending tasks for the APP
    std::queue<Task> pending_tasks;
    std::mutex m_mutex_new_task;
    std::condition_variable m_cv_new_task;

    // pending refactor messages for the APP
    std::queue<refactoring_data> pending_refactor_tasks;
    std::mutex m_mutex_new_refactor_task;
    std::condition_variable m_cv_new_refactor_task;

    network_layer(int myid) : myid(myid) {
        rooting_table.insert({0, std::pair<std::string, int>("localhost", 8081)});
        rooting_table.insert({1, std::pair<std::string, int>("localhost", 8082)});
        rooting_table.insert({2, std::pair<std::string, int>("localhost", 8083)});
        rooting_table.insert({3, std::pair<std::string, int>("localhost", 8084)});
    }

    void new_message(Task task, int send_to, bool compute_to_compute=false); // produce -- new message
    void new_message(refactoring_data task, int send_to, bool compute_to_compute=false); // produce -- new message
    
    void put_internal_task(Task task);
    void put_internal_task(refactoring_data task);

    Task check_new_task(); //consumer - new task
    refactoring_data check_new_refactor_task(); //consumer - new task

    //threads
    void receiver(); // producer -- new task
    void sender(); // consumer -- new message

};

#endif