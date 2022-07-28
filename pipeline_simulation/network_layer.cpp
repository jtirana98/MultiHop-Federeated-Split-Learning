#include "network_layer.h"

void network_layer::new_message(Task task, int sender) { // produce -- new message
    Message msg;

    {
    std::unique_lock<std::mutex> lock(m_mutex_new_message);
    pending_messages.push(msg);
    }

     m_cv_new_message.notify_one();

}

void network_layer::new_message(refactoring_data task, int sender){ // produce -- new message
    Message msg;

    {
    std::unique_lock<std::mutex> lock(m_mutex_new_message);
    pending_messages.push(msg);
    }

     m_cv_new_message.notify_one();
}

void network_layer::put_internal_task(Task task) {
    
    {
    std::unique_lock<std::mutex> lock(m_mutex_new_task);
    pending_tasks.push(task);
    }

    m_cv_new_task.notify_one();
}

void network_layer::put_internal_task(refactoring_data task) {
    
    {
    std::unique_lock<std::mutex> lock(m_mutex_new_refactor_task);
    pending_refactor_tasks.push(task);
    }

    m_cv_new_refactor_task.notify_one();
}

Task network_layer::check_new_task() { //consumer
    Task new_task;

    std::unique_lock<std::mutex> lock(m_mutex_new_task);
    while (pending_tasks.empty()) {
        m_cv_new_task.wait(lock, [&](){ return !pending_tasks.empty(); }); // predicate an while loop - protection from spurious wakeups
    }
    
    new_task = pending_tasks.front();
    pending_tasks.pop();

    //std::cout << "Consumer Thread, queue element: " << new_task << " size: " << pending_tasks.size() << std::endl;
    return new_task;
}

refactoring_data network_layer::check_new_refactor_task() {
    refactoring_data new_task;

    std::unique_lock<std::mutex> lock(m_mutex_new_refactor_task);
    while (pending_refactor_tasks.empty()) {
        m_cv_new_refactor_task.wait(lock, [&](){ return !pending_refactor_tasks.empty(); }); // predicate an while loop - protection from spurious wakeups
    }
    
    new_task = pending_refactor_tasks.front();
    pending_refactor_tasks.pop();

    //std::cout << "Consumer Thread, queue element: " << new_task << " size: " << pending_tasks.size() << std::endl;
    return new_task;
}


void network_layer::receiver() {
    Task task;

    while(true) {
        // communtication protocol ...

        // new task
        put_internal_task(task);

        // TODO or put sto allo

        //std::this_thread::sleep_for(std::chrono::seconds(1));
    }
        
}


void network_layer::sender() { // consumer -- new message
    Message new_msg;

    while(true) {
        std::unique_lock<std::mutex> lock(m_mutex_new_message);
        while (pending_messages.empty()) {
            m_cv_new_message.wait(lock, [&](){ return !pending_messages.empty(); });
        }
    
        new_msg = pending_messages.front();
        pending_messages.pop();

        // communication protocol ...
    }

}