#include "network_layer.h"
//#include "my_json.h"
#include "Message.h"

std::queue<Message> pending_messages;
void network_layer::new_message(Task task, int send_to, bool compute_to_compute) { // produce -- new message
    Message msg;

    // make task a message
    msg.type = OPERATION;
    msg.prev_node = task.prev_node;
    msg.client_id = task.client_id;
    msg.size_ = task.size_;
    msg.type_op = task.type;
    //TODO: msg.values
    msg.save_connection = (compute_to_compute) ? 1 : 0;

    msg.dest = send_to;

    {
    std::unique_lock<std::mutex> lock(m_mutex_new_message);
    pending_messages.push(msg);
    }

     m_cv_new_message.notify_one();

}

void network_layer::new_message(refactoring_data task, int send_to, bool compute_to_compute){ // produce -- new message
    Message msg;

    // make task a message
    msg.type = (task.to_data_onwer) ? REFACTOR_DATA_OWNER : REFACTOR_COMPUTE_NODE;
    msg.start = task.start;
    msg.end = task.end;
    msg.prev = task.prev;
    msg.next = task.next;
    // TODO list data owners
    msg.dataset = task.dataset;
    msg.num_classes = task.num_class;
    msg.model_name = task.model_name_;
    msg.model_type = task.model_type_;
    msg.save_connection = (compute_to_compute) ? 1 : 0;

    msg.dest = send_to;

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
    refactoring_data refactor_obj;
    Message new_msg;
    std::map<int, int> open_connections; //client_id --> socket_fd
    struct sockaddr_in serv_addr, cli_addr;
    socklen_t clientlen;
    char buffer[256];
    int my_socket, my_port, maxfd, len=0, num, n;
    fd_set readset;

    std::pair<std::string, int> my_addr = rooting_table.find(myid)->second;
    my_port = my_addr.second;
    std::cout << "@ " << my_port << std::endl;
    my_socket =  socket(AF_INET, SOCK_STREAM, 0);
    if (my_socket < 0) 
        perror("ERROR opening socket");

    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;  
    serv_addr.sin_addr.s_addr = INADDR_ANY;  //my_addr.first
    //std::cout << "server ip: " << inet_ntoa(serv_addr.sin_addr) << std::endl;
    serv_addr.sin_port = htons(my_port);

    if (bind(my_socket, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) 
        perror("ERROR on binding");

    listen(my_socket,5);
    clientlen = sizeof(cli_addr);

    while(true) {
        FD_ZERO(&readset);
        FD_SET(my_socket, &readset);

        maxfd = my_socket;
        for (auto it = open_connections.begin(); it != open_connections.end(); it++) {
            if (maxfd < it->second)
                maxfd = it->second;
            
            FD_SET(it->second, &readset);
        }

        if (((num = select(maxfd+1, &readset, NULL, NULL, NULL)) == -1) && (errno == EINTR))
            continue;
        std::cout << "!" << std::endl;

        std::vector<int> to_remove;
        for (auto it = open_connections.begin(); it != open_connections.end(); it++) {
            if (FD_ISSET(it->second, &readset)) {
                bzero(buffer,256);
                n = read(it->second,buffer,255);
                
                // CHANGEEE
                if (n==0) { // remove socket
                    to_remove.push_back(it->first);
                    close(it->second);
                }
                
                if (n < 0) perror("ERROR reading from socket");
                
                printf("Here is the message: %s\n",buffer);
                 // CHANGEEE

                if (new_msg.type == OPERATION) { // create new Task object
                    // from message to task object
                    put_internal_task(task);
                }
                else {
                    // from message to refactor object
                    put_internal_task(refactor_obj);
                }


            }

        }

        for (int i=0; i<to_remove.size(); i++) {
            open_connections.erase(to_remove[i]);
        }
        

        if (FD_ISSET(my_socket, &readset)) {
            int newsockfd = accept(my_socket, (struct sockaddr *) &cli_addr, &clientlen);
            if (newsockfd < 0) {
                perror("ERROR on accept");
                continue;
            }

            printf("server: got connection from %s port %d\n",
                inet_ntoa(cli_addr.sin_addr), ntohs(cli_addr.sin_port));

            // receive 

            if (new_msg.type == OPERATION) { // create new Task object
                // from message to task object
                put_internal_task(task);
            }
            else {
                // from message to refactor object
                put_internal_task(refactor_obj);
            }

            //check an prepei na mpei lista i prepei na antikatastisei allo.
            //open_connections.insert({});  
            len += 1;
        }
        //std::this_thread::sleep_for(std::chrono::seconds(1));
    }
        
}


void network_layer::sender() { // consumer -- new message
    Message new_msg;
    std::map<int, int> open_connections;
    std::map<int,int>::iterator it;
    
    int sockfd, portno, n;
    struct sockaddr_in serv_addr;
    struct hostent *receiver;
    //char buffer[256];
    bool store;
    int len=0;

    while(true) {
        store = false;
        std::unique_lock<std::mutex> lock(m_mutex_new_message);
        while (pending_messages.empty()) {
            m_cv_new_message.wait(lock, [&](){ return !pending_messages.empty(); });
        }
        
        std::cout << "new message" << std::endl;

        new_msg = pending_messages.front();
        pending_messages.pop();

        // message to json to string
        Json::Value jsonMsg = toJson(new_msg);
        // json to string
        std::string data = fromJson_toStr<Message>(jsonMsg);
        len = data.size() + 1;
        char buffer[len];
        strcpy(buffer, data.c_str());
        std::cout << "!! " << new_msg.dest << std::endl;
        it = open_connections.find(new_msg.dest);
        if (it != open_connections.end()) {
            int client_sock = it->second;
            send(client_sock, buffer, len, 0);
            // TODO check mipws to connection exei kleisei
        }
        else{
            auto client_addr = rooting_table.find(new_msg.dest)->second;

            portno = client_addr.second;
            sockfd = socket(AF_INET, SOCK_STREAM, 0);
            if (sockfd < 0) 
                std::cerr << "ERROR opening socket";

            int len_ = client_addr.first.size() + 1;
            char addr[len_];
            strcpy(addr, client_addr.first.c_str());

            receiver = gethostbyname(addr);
            if (receiver == NULL) {
                fprintf(stderr,"ERROR, no such host\n");
                //exit(0);
            }

            bzero((char *) &serv_addr, sizeof(serv_addr));
            serv_addr.sin_family = AF_INET;
            bcopy((char *)receiver->h_addr, 
                (char *)&serv_addr.sin_addr.s_addr,
                receiver->h_length);
            serv_addr.sin_port = htons(portno);

            if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) 
                std::cerr << "ERROR connecting";
            
            //bzero(buffer,256);
            n = send(sockfd, buffer, strlen(buffer), 0);
            if (n < 0) 
                std::cerr << "ERROR writing to socket";

            if (store) {
                open_connections.insert({new_msg.dest, sockfd});
            }
            else {
                close(sockfd);
            }
        }

    }

}