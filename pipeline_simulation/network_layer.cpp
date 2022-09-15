#include "network_layer.h"
#include "Message.h"

std::queue<Message> pending_messages;

int my_send(int socket_fd, std::string& data) {
    const char* data_ptr  = data.data();
    int data_size = data.size();

    std::string len = std::to_string(data_size);
    send(socket_fd, &data_size, sizeof(int), 0);
    int bytes_sent;
    
    while (data_size > 0) {
        bytes_sent = send(socket_fd, data_ptr, data_size, 0);

        data_ptr += bytes_sent;
        data_size -= bytes_sent;
    }
    
    return 1;
}

std::string my_receive(int socket_fd) {
    int max = 4096;
    int expected_input=0, bytes_recv = 0, n, len=max;
    std::string json_format;
    std::string leader_board_package= "";

    read(socket_fd,&expected_input,sizeof(int));
    //std::cout << "I expect: " << expected_input << std::endl;
    if (expected_input == 0)
        return leader_board_package;

    while(bytes_recv < expected_input) {

        if(expected_input - bytes_recv <= max) {
            len = expected_input - bytes_recv;
        }
        std::vector<char> leader_board_buffer(len);
        n = read(socket_fd, leader_board_buffer.data(), len);
        bytes_recv = bytes_recv + n;
        if (bytes_recv == -1) {
            std::cout << "Communication error...";
            return leader_board_package;
        }
        else {
            for (int i =0; i<leader_board_buffer.size(); i++) {
                leader_board_package = leader_board_package + leader_board_buffer[i];
            }
        }
    }

    return leader_board_package; 
}

void network_layer::new_message(Task task, int send_to, bool compute_to_compute) { // produce -- new message
    Message msg;

    // make task a message
    msg.type = OPERATION;
    msg.prev_node = task.prev_node;
    msg.client_id = task.client_id;
    msg.size_ = task.size_;
    msg.type_op = task.type;
    auto data = torch::pickle_save(task.values);
    std::string s(data.begin(), data.end());
    msg.values = s;
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

    msg.dataset = task.dataset;
    msg.num_classes = task.num_class;
    msg.model_name = task.model_name_;
    msg.model_type = task.model_type_;
    msg.data_owners = task.data_owners;
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
        m_cv_new_task.wait(lock, [&](){ return !pending_tasks.empty(); });
    }
    
    new_task = pending_tasks.front();
    pending_tasks.pop();

    return new_task;
}

refactoring_data network_layer::check_new_refactor_task() {
    refactoring_data new_task;

    std::unique_lock<std::mutex> lock(m_mutex_new_refactor_task);
    while (pending_refactor_tasks.empty()) {
        m_cv_new_refactor_task.wait(lock, [&](){ return !pending_refactor_tasks.empty(); });
    }
    
    new_task = pending_refactor_tasks.front();
    pending_refactor_tasks.pop();

    return new_task;
}


void network_layer::receiver() {
    Message new_msg;
    std::map<int, int> open_connections; //client_id --> socket_fd
    struct sockaddr_in serv_addr, cli_addr;
    socklen_t clientlen;
    char buffer[1024];
    int my_socket, my_port, maxfd, num, n;
    fd_set readset;

    std::pair<std::string, int> my_addr = rooting_table.find(myid)->second;
    my_port = my_addr.second;
    my_socket =  socket(AF_INET, SOCK_STREAM, 0);
    if (my_socket < 0) 
        perror("ERROR opening socket");

    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;  
    serv_addr.sin_addr.s_addr = INADDR_ANY;  //my_addr.first
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

        std::vector<int> to_remove;
        for (auto it = open_connections.begin(); it != open_connections.end(); it++) {
            if (FD_ISSET(it->second, &readset)) {
                auto json_format_str = my_receive(it->second);
                if (json_format_str.size()==0 || json_format_str == "") { // remove socket
                    to_remove.push_back(it->first);
                    close(it->second);
                    continue;
                }

                auto json_format = fromStr_toJson<Message>(json_format_str);
                new_msg = fromJson<Message>(json_format);

                if (new_msg.type == OPERATION) { // create new Task object
                    // from message to task object
                    Task task(new_msg.client_id, (operation)new_msg.type_op, new_msg.prev_node);
                    task.size_ = new_msg.size_;
                    std::vector<char> v(new_msg.values.begin(), new_msg.values.end());
                    task.values = torch::pickle_load(v).toTensor();
                    
                    // POINT Network layer: received message
                    newPoint(NT_RECEIVED_MSG, task.client_id);
                    put_internal_task(task);
                }
                else {
                    // from message to refactor object
                    refactoring_data refactor_obj;
                    refactor_obj.to_data_onwer = (new_msg.type == REFACTOR_DATA_OWNER) ? true : false;
                    refactor_obj.start = new_msg.start;
                    refactor_obj.end = new_msg.end;
                    refactor_obj.prev = new_msg.prev;
                    refactor_obj.next = new_msg.next;
                    refactor_obj.data_owners = new_msg.data_owners;
                    refactor_obj.dataset = new_msg.dataset;
                    refactor_obj.num_class = new_msg.num_classes;
                    refactor_obj.model_name_ = new_msg.model_name;
                    refactor_obj.model_type_ = new_msg.model_type;
                    
                    // POINT Network layer: received message
                    newPoint(NT_RECEIVED_MSG, refactor_obj.client_id);

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

           /* printf("server: got connection from %s port %d\n",
                inet_ntoa(cli_addr.sin_addr), ntohs(cli_addr.sin_port));
            */
            
            auto json_format_str = my_receive(newsockfd);
            if(json_format_str == "")
                    continue;
            
            auto json_format = fromStr_toJson<Message>(json_format_str);
            new_msg = fromJson<Message>(json_format);
            if (new_msg.type == OPERATION) { // create new Task object
                // from message to task object
                Task task(new_msg.client_id, (operation)new_msg.type_op, new_msg.prev_node);
                task.size_ = new_msg.size_;
                std::vector<char> v(new_msg.values.begin(), new_msg.values.end());
                task.values = torch::pickle_load(v).toTensor();
                
                // POINT Network layer: received message
                newPoint(NT_RECEIVED_MSG, task.client_id);
                
                put_internal_task(task);
                
            }
            else {
                // from message to refactor object
                refactoring_data refactor_obj;
                refactor_obj.to_data_onwer = (new_msg.type == REFACTOR_DATA_OWNER) ? true : false;
                refactor_obj.start = new_msg.start;
                refactor_obj.end = new_msg.end;
                refactor_obj.prev = new_msg.prev;
                refactor_obj.next = new_msg.next;
                refactor_obj.data_owners = new_msg.data_owners;
                refactor_obj.dataset = new_msg.dataset;
                refactor_obj.num_class = new_msg.num_classes;
                refactor_obj.model_name_ = new_msg.model_name;
                refactor_obj.model_type_ = new_msg.model_type;
                
                // POINT Network layer: received message
                newPoint(NT_RECEIVED_MSG, refactor_obj.client_id);

                put_internal_task(refactor_obj);
            }

            if (new_msg.save_connection == 1) {
                open_connections.insert(std::pair<int,int>{new_msg.prev_node, newsockfd});
            }
            else {
                close(newsockfd);
            } 
        }
    }
}


void network_layer::sender() { // consumer -- new message
    Message new_msg;
    std::map<int, int> open_connections;
    std::map<int,int>::iterator it;
    
    int sockfd, portno, n;
    struct sockaddr_in serv_addr;
    struct hostent *receiver;
    char buffer[1024];
    int len=0, len_;

    while(true) {
        std::unique_lock<std::mutex> lock(m_mutex_new_message);
        while (pending_messages.empty()) {
            m_cv_new_message.wait(lock, [&](){ return !pending_messages.empty(); });
        }

        new_msg = pending_messages.front();
        pending_messages.pop();

        // POINT 2 Network layer: preparing to send
        newPoint(NT_PREPARE_MSG, new_msg.dest);

        // message to json
        Json::Value jsonMsg = toJson(new_msg);
        // json to string
        std::string data = fromJson_toStr<Message>(jsonMsg);

        it = open_connections.find(new_msg.dest);
        if (it != open_connections.end()) {
            int client_sock = it->second;
            // TODO: check mipws to connection exei kleisei
            
            // POINT 3 Network layer: starts message transmission
            newPoint(NT_START_SENDING, new_msg.dest);

            n = my_send(client_sock, data);
            
            // POINT 4 Network layer: completes message transmission
            newPoint(NT_STOP_SENDING, new_msg.dest);
            
        }
        else{
            auto client_addr = rooting_table.find(new_msg.dest)->second;
            portno = client_addr.second;
            
            while(true) {
                sockfd = socket(AF_INET, SOCK_STREAM, 0);

                if (sockfd < 0) {
                    std::cerr << "ERROR opening socket " << new_msg.dest << " e:" << sockfd;
                    printf("The last error message is: %s\n", strerror(errno));
                    break;
                }
                else
                    break;
            }

            int len_ = client_addr.first.size() + 1;
            char addr[len_];
            strcpy(addr, client_addr.first.c_str());

            receiver = gethostbyname(addr);
            if (receiver == NULL) {
                fprintf(stderr, "ERROR, no such host\n");
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
            
            // POINT 3  Network layer: starts message transmission
            newPoint(NT_START_SENDING, new_msg.dest);

            n = my_send(sockfd, data);
            
            // POINT 4 Network layer: completes message transmission
            newPoint(NT_STOP_SENDING, new_msg.dest);
            
            if (new_msg.save_connection) {
                open_connections.insert({new_msg.dest, sockfd});
            }
            else {
                close(sockfd);
            }
        }

    }

}