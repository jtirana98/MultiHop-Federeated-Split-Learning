#include "network_layer.h"

void network_layer::new_message(Task task, int sender) { // produce -- new message
    
    {
    std::unique_lock<std::mutex> lock(m_mutex_new_message);
    pending_messages.push(task);
    }

     m_cv_new_message.notify_one();

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


void network_layer::receiver() {
    Task task;

    while(true) {
        // communtication protocol ...

        // new task
        {
        std::unique_lock<std::mutex> lock(m_mutex_new_task);
        pending_tasks.push(task);
        }

        m_cv_new_task.notify_one();
        //std::this_thread::sleep_for(std::chrono::seconds(1));
    }
        
}


void network_layer::sender() { // consumer -- new message
    Task new_msg;

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