#ifndef _NETWORK_LAYER_H_
#define _NETWORK_LAYER_H_

#include <iostream> 
#include <iterator>
#include <condition_variable>
#include <map>
#include <string>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>
#include <queue>

#include "Task.h"

class network_layer {
 public:
    // pending messages 
    std::map<int, std::pair<std::string, int>> rooting_table; // (node_id, (ip, port))
    std::queue<Task> pending_messages;
    std::mutex m_mutex_new_message;
    std::condition_variable m_cv_new_message;

    // pending tasks for the APP
    int f=1234;
    std::queue<Task> pending_tasks;
    std::mutex m_mutex_new_task;
    std::condition_variable m_cv_new_task;

    network_layer() {
        //std::cout << "hello" << std::endl;
    }
    void new_message(Task task, int sender); // produce -- new message
    void put_internal_task(Task task);
    Task check_new_task(); //consumer - new task

    //threads
    void receiver(); // producer -- new task
    void sender(); // consumer -- new message

};

#endif