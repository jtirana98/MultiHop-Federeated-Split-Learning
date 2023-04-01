#include "network_layer.h"
#include "Message.h"

std::queue<Message> pending_messages;

int my_send(int socket_fd, std::string& data, int dest) {
    const char* data_ptr  = data.data();
    int data_size = data.size();
   
    if (data_size < 1000)
        std::cout << "-->" << data << std::endl;
    

    auto timestamp1 = std::chrono::steady_clock::now();
    std::string len = std::to_string(data_size);
    send(socket_fd, &data_size, sizeof(int), 0);
    int bytes_sent;
    
    while (data_size > 0) {
        bytes_sent = send(socket_fd, data_ptr, data_size, 0);

        data_ptr += bytes_sent;
        data_size -= bytes_sent;
    }
    
    auto timestamp2 = std::chrono::steady_clock::now();
    auto optim_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (timestamp2 - timestamp1).count();

    return 1;
}

std::vector<std::string> my_receive(int socket_fd) {
    int max = 4096;
    int expected_input=0, bytes_recv = 0, n, len=max;
    std::string json_format;
    std::string leader_board_package= "";
    std::vector<std::string> to_return;

    read(socket_fd,&expected_input,sizeof(int));
    to_return.push_back(std::to_string(expected_input));
    auto timestamp1 = std::chrono::steady_clock::now();
    if (expected_input == 0){
        to_return.push_back(leader_board_package);
        return to_return; 
    }
    
    char* buffer = new char[expected_input];
    if (buffer == NULL) {
        std::cout << "bad..." << std::endl;
    }
    auto buffer_ptr = buffer;
    while(bytes_recv < expected_input) {

        len = expected_input - bytes_recv;
        n = read(socket_fd, buffer_ptr, len);
        bytes_recv = bytes_recv + n;
        buffer_ptr = buffer_ptr + n;
        if (bytes_recv == -1) {
            std::cout << "Communication error...";
            to_return.push_back("");
            return to_return;
        }
    }
    leader_board_package.assign(buffer, expected_input);
    auto timestamp2 = std::chrono::steady_clock::now();
    auto optim_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (timestamp2 - timestamp1).count();

    delete[] buffer;
    
    to_return.push_back(leader_board_package);
    return to_return; 
}

void network_layer::findPeers(int num, bool aggr) {
    int completed = num;
    // mulitcast address
    std::string s = "130.0.0.0";
    char group_[s.length() + 1];
    strcpy(group_, s.c_str());
    char *group = group_;
    int port = 4321;
    std::set<int> registered;
    std::cout << "searching" << std::endl;
    if(aggr) {
        port = 4322;
    }
    std::cout << port << std::endl;
    // open a multicast socket
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        perror("socket");
        return ;
    }

    u_int yes = 1;
    if (
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (char*) &yes, sizeof(yes)) < 0
    ) {
       perror("Reusing ADDR failed");
       return ;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    // bind to receive address
    if (bind(fd, (struct sockaddr*) &addr, sizeof(addr)) < 0) {
        perror("bind");
        return ;
    }

    // use setsockopt() to request that the kernel join a multicast group
    struct ip_mreq mreq;
    mreq.imr_multiaddr.s_addr = inet_addr(group);
    mreq.imr_interface.s_addr = htonl(INADDR_ANY);
    if (
        setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*) &mreq, sizeof(mreq)) < 0
    ){
        perror("setsockopt");
        return ;
    }

    while (num > 0) {
        socklen_t addrlen = sizeof(addr);
        fflush(stdout);
        int id=0;
        int nbytes = recvfrom(fd,&id,sizeof(int), MSG_WAITALL,(struct sockaddr *) &addr, &addrlen);
        if (nbytes < 0) {
            perror("recvfrom");
            return ;
        }
        if (registered.find(id) != registered.end())
            continue;
        registered.insert(id);
        std::cout << "NODE: " << id << " just registered" << std::endl;

        
        // get node's ip address
        char str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(addr.sin_addr), str, INET_ADDRSTRLEN);

        std::map<int, std::pair<std::string, int>>::iterator itr;
        itr = rooting_table.find(id);
        int port_n = itr->second.second;
        //if(itr != m.end()) 
        itr->second = std::pair<std::string, int>(str, port_n);
        //else
        //    rooting_table.insert({if, std::pair<std::string, int>(str, 8081)}); // wrong

        
        // send ACK and let them know the init's node ip
        sleep(2);
        // open socket:
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
                std::cerr << "ERROR opening socket e:" << sockfd;
                printf("The last error message is: %s\n", strerror(errno));
        }

          struct hostent *receiver;
        receiver = gethostbyname(str);

        if (receiver == NULL) {
            fprintf(stderr, "ERROR, no such host\n");
            //exit(0);
        }
        struct sockaddr_in serv_addr;
        bzero((char *) &serv_addr, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        bcopy((char *)receiver->h_addr, 
            (char *)&serv_addr.sin_addr.s_addr,
            receiver->h_length);
        serv_addr.sin_port = htons(port_n);

        if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) 
            std::cerr << "ERROR connecting";

        std::string data = "ACK";
        const char* data_ptr  = data.data();
        int data_size = data.size();

        std::string len = std::to_string(data_size);
        nbytes = send(sockfd, data_ptr, data_size, 0); 
        close(sockfd);
        num--;
    }
    if(aggr)
        put_internal_task(Task());

}

void network_layer::findInit(bool aggr) {
    // mulitcast address
    std::string s = "130.0.0.0";
    char group_[s.length() + 1];
    strcpy(group_, s.c_str());
    char *group = group_;
    int port = 4321;

    if(aggr) {
        port = 4322;
    }

    int id=myid;
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        perror("socket");
        return ;
    }

    // set up destination address
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(group);
    addr.sin_port = htons(port);

    int nbytes = sendto(fd,&id,sizeof(int),0,(struct sockaddr*) &addr,sizeof(addr));
    if (nbytes < 0) {
        perror("sendto");
        return ;
    }

    int my_port = rooting_table.find(id)->second.second;
    int my_socket =  socket(AF_INET, SOCK_STREAM, 0);
    
    struct sockaddr_in serv_addr, cli_addr;
        socklen_t clientlen;
        bzero((char *) &serv_addr, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;  
        serv_addr.sin_addr.s_addr = INADDR_ANY;  //my_addr.first
        serv_addr.sin_port = htons(my_port);

    if (bind(my_socket, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) 
        perror("ERROR on binding!");

    u_int yes = 1;
    if (
        setsockopt(my_socket, SOL_SOCKET, SO_REUSEADDR, (char*) &yes, sizeof(yes)) < 0
    ) {
       perror("Reusing ADDR failed");
       return ;
    }

    listen(my_socket,5);
    clientlen = sizeof(cli_addr);

    int newsockfd = accept(my_socket, 
            (struct sockaddr *) &cli_addr, &clientlen);
    if (newsockfd < 0) 
        perror("ERROR on accept");

    printf("server: got connection from %s port %d\n",
        inet_ntoa(cli_addr.sin_addr), ntohs(cli_addr.sin_port));
    
    char str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(cli_addr.sin_addr), str, INET_ADDRSTRLEN);

    // update init's node ip address
    std::map<int, std::pair<std::string, int>>::iterator itr;
    
    int id_ = 0;
    if(aggr)
        id_ = -1;
    
    itr = rooting_table.find(id_);
    int port_n = itr->second.second;
    itr->second = std::pair<std::string, int>(str, port_n);
    std::cout << "addr " << str << std::endl;
    char buffer[256];
    send(newsockfd, buffer, 10, 0);

    bzero(buffer,256);
    socklen_t addrlen = sizeof(addr);
    //int n = read(newsockfd,buffer,255);
    int n = recvfrom(newsockfd,buffer,255, MSG_WAITALL,(struct sockaddr *) &addr, &addrlen);
    if (n < 0) perror("ERROR reading from socket");
    printf("Here is the message: %s\n",buffer);

    close(newsockfd);
    close(my_socket);
    if(!is_data_owner || (is_data_owner && aggr)){
        std::cout << "free" << std::endl;
        put_internal_task(Task());
    }
}

void network_layer::new_message(Task task, int send_to, bool compute_to_compute) { // produce -- new message
    Message msg;

    // make task a message
    msg.type = OPERATION;
    msg.prev_node = task.prev_node;
    msg.client_id = task.client_id;
    msg.size_ = task.size_;
    msg.type_op = task.type;
    msg.t_start = task.t_start;
    msg.batch0 = task.batch0;
    
    if(msg.type_op == operation::aggregation_) {
        msg.model_part = task.model_part;
        std::stringstream s;
        torch::save(task.model_part_, s);
        /*if(task.check_) {
            msg.values = task.model_parts;
        }
        else*/
        msg.values = s.str();
        msg.save_connection = (compute_to_compute) ? 1 : 0;
        msg.dest = send_to;
    }
    else{
        std::stringstream s;
        torch::save(task.values, s);
        msg.values = s.str();
        
        msg.save_connection = (compute_to_compute) ? 1 : 0;
        msg.dest = send_to;
    }

    {
    std::unique_lock<std::mutex> lock(m_mutex_new_message);
    pending_messages.push(msg);
    }

     m_cv_new_message.notify_one();

}

void network_layer::new_message(refactoring_data task, int send_to, bool compute_to_compute, bool rooting_table_){ // produce -- new message
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
    
    if (rooting_table_) {
        std::vector<std::pair<int, std::string>> temp_root;
        for(auto itr = rooting_table.begin(); itr != rooting_table.end(); itr++) {
            temp_root.push_back(std::pair<int, std::string>(itr->first, itr->second.first));
        }

        msg.rooting_table =  temp_root;
        msg.read_table = 1;
    }
    
    msg.dest = send_to;

    {
    std::unique_lock<std::mutex> lock(m_mutex_new_message);
    pending_messages.push(msg);
    }

     m_cv_new_message.notify_one();
}

void network_layer::put_internal_task(Task task, long timestamp, bool back) {
  
    {
    std::unique_lock<std::mutex> lock(m_mutex_new_task);
    pending_tasks.push_back(std::pair<long, Task>(timestamp, task));
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

Task network_layer::check_new_task(bool back) { //consumer
    Task new_task;
    
    {
    std::unique_lock<std::mutex> lock(m_mutex_new_task);
    while (pending_tasks.empty()) {
        m_cv_new_task.wait(lock, [&](){ return !pending_tasks.empty(); });
    }
    }
    m_mutex_new_task.lock();
    if (is_data_owner) {
        auto new_task_pair = pending_tasks[0];
        new_task = new_task_pair.second;
        pending_tasks.erase(pending_tasks.begin());

        m_mutex_new_task.unlock();
        return new_task;
    }
    else {
        bool not_ready = false;
        
        while(!not_ready) {
            int it_best;
            long best_time=-1, min=100;
            auto p1 = std::chrono::system_clock::now();
            auto my_time = std::chrono::duration_cast<std::chrono::milliseconds>(p1.time_since_epoch());

            // case 1 - tasks that have exceed
            int i = 0;
            for (std::vector<std::pair<long, Task>>::iterator it = pending_tasks.begin(); it != pending_tasks.end(); it++) {
                
                if(it->first == -1) { // early exit
                    new_task = it->second;
                    pending_tasks.erase(pending_tasks.begin()+i);

                    m_mutex_new_task.unlock();
                    
                    return new_task;
                }

                if(my_time.count() >= it->first) {  
                    auto time_tmp = my_time.count() - it->first;
                    if (best_time < time_tmp) {
                        
                        it_best = i;
                        best_time = time_tmp;
                    }
                }
                i++;
            }

            if (best_time != -1) {
                new_task = pending_tasks[it_best].second;
                
                pending_tasks.erase(pending_tasks.begin()+it_best);
                m_mutex_new_task.unlock();
                return new_task;
            }
            
            // case 2 - task that are near threadhold
            best_time = 3000000;
            i = 0;
            for (std::vector<std::pair<long, Task>>::iterator it = pending_tasks.begin(); it != pending_tasks.end(); it++) {
                if((it->first > my_time.count()) && ((it->first - my_time.count()) <= min)) {  
                    auto time_tmp = it->first - my_time.count();
                    if (best_time > time_tmp) {
                        it_best = i;
                        best_time = time_tmp;
                    }
                }
                i++;
            }

            if (best_time != 3000000) {
                new_task = pending_tasks[it_best].second;
                pending_tasks.erase(pending_tasks.begin()+it_best);
                m_mutex_new_task.unlock();
                return new_task;
            }

            // case 3 - not ready yet sleep
            m_mutex_new_task.unlock();
            usleep(min);
            m_mutex_new_task.lock();
        }

    }
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
    auto p_prev = std::chrono::system_clock::now();
    // lock
    //auto dump = check_new_task();
    std::cout << "let's go " << myid << " " << rooting_table.size() << std::endl;
    sleep(1);

    if(myid > 3 && myid < 18) {
        std::pair<std::string, int> my_addr = rooting_table.find(0)->second;
        my_port = my_addr.second;
        my_port = my_port + (myid +3);
    }
    else if (myid >= 18) {
        std::pair<std::string, int> my_addr = rooting_table.find(18)->second;
        my_port = my_addr.second;
        my_port = my_port + (myid - 18);
    }
    else{
        std::pair<std::string, int> my_addr = rooting_table.find(myid)->second;
        my_port = my_addr.second;
    }
    
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
                auto json_format_str = my_receive(it->second)[1];
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
                    task.t_start = new_msg.t_start;
                    task.batch0 = new_msg.batch0;

                    std::stringstream ss(std::string(new_msg.values.begin(), new_msg.values.end()));
                    torch::load(task.values, ss);
                    
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
                    
                    if(new_msg.read_table == 1) {
                        refactor_obj.rooting_table = new_msg.rooting_table;
                     }

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

            auto received = my_receive(newsockfd);
            auto json_format_str = received[1];

            std::stringstream load(received[0]);
            int load_received = 0;
            load >>  load_received;

            if(json_format_str == "")
                    continue;

            auto json_format = fromStr_toJson<Message>(json_format_str);
            new_msg = fromJson<Message>(json_format);
            if (new_msg.type == OPERATION) { // create new Task object
                // from message to task object
                Task task(new_msg.client_id, (operation)new_msg.type_op, new_msg.prev_node);
                task.size_ = new_msg.size_;
                task.t_start = new_msg.t_start;
                task.batch0 = new_msg.batch0;
                
                if((operation)new_msg.type_op == operation::aggregation_) {
                    std::stringstream ss(std::string(new_msg.values.begin(), new_msg.values.end()));
                    //torch::load(task.model_part_/*task.model_part_*/, ss);
                    task.model_part = new_msg.model_part;
                    task.model_parts = new_msg.values;

                    auto p1 = std::chrono::system_clock::now();
                    auto my_time = std::chrono::duration_cast<std::chrono::milliseconds>(p1.time_since_epoch());
                    long real_duration = ((load_received* 0.000008)/my_rpi.rpi_to_vm)*1000;

                    if (my_time.count()-task.t_start > real_duration) {
                        std::cout << "Network: Cannot Simulate " << (my_time.count()-task.t_start - real_duration) << std::endl;
                    } 
                    else{
                        //std::cout << "go to sleep " << real_duration-(my_time.count()-task.t_start) << " " << load_received << std::endl;
                        
                        usleep(real_duration-(my_time.count()-task.t_start));
                    }

                    put_internal_task(task);
                }
                else{
                    std::stringstream ss(std::string(new_msg.values.begin(), new_msg.values.end()));
                    torch::load(task.values, ss);
                    if(!is_data_owner) {
                        if((sim_forw && (operation)task.type == operation::forward_ ) || (sim_back && (operation)task.type == operation::backward_)){ 
                            auto p1 = std::chrono::system_clock::now();
                            auto my_time = std::chrono::duration_cast<std::chrono::milliseconds>(p1.time_since_epoch());
                            long real_duration = ((load_received* 0.000008)/my_rpi.rpi_to_vm)*1000;
                            
                            if (my_time.count()-task.t_start > real_duration) {
                                std::cout << "Network: Cannot Simulate for " << (my_time.count()-task.t_start - real_duration) << std::endl;
                                put_internal_task(task);
                            }
                            else {
                              
                                auto prev = std::chrono::duration_cast<std::chrono::milliseconds>(p_prev.time_since_epoch());
                                //std::cout << "received task for " << task.t_start+real_duration  << " " << my_time.count()-prev.count() << " " << real_duration - (my_time.count() - task.t_start) << std::endl;
                                put_internal_task(task, task.t_start+real_duration);
                            }
                            p_prev = p1;
                        }
                        else { // the compute nodes that communicate with each other, should not simulate the network delay
                            put_internal_task(task);
                        }
                    }
                    else{ // data owner -- simulate transfer
                        auto p1 = std::chrono::system_clock::now();
                        auto my_time = std::chrono::duration_cast<std::chrono::milliseconds>(p1.time_since_epoch());
                        long real_duration = ((load_received* 0.000008)/my_rpi.rpi_to_vm)*1000;

                        if (my_time.count()-task.t_start > real_duration) {
                            std::cout << "Network: Cannot Simulate " << (my_time.count()-task.t_start - real_duration) << std::endl;
                        } 
                        else{
                            usleep(real_duration-(my_time.count()-task.t_start));
                        }

                        put_internal_task(task);
                    }
                }
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
                
                if(new_msg.read_table == 1) {
                        refactor_obj.rooting_table = new_msg.rooting_table;
                     }
                
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
        new_msg.t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        // message to json
        Json::Value jsonMsg = toJson(new_msg);
        // json to string
        std::string data = fromJson_toStr<Message>(jsonMsg);

        it = open_connections.find(new_msg.dest);
        if (it != open_connections.end()) {
            int client_sock = it->second;
            // TODO: check mipws to connection exei kleisei
            
            n = my_send(client_sock, data, new_msg.dest);
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
            int res=0;
            int k = 0;
            do {
                bzero((char *) &serv_addr, sizeof(serv_addr));
                serv_addr.sin_family = AF_INET;
                bcopy((char *)receiver->h_addr, 
                    (char *)&serv_addr.sin_addr.s_addr,
                    receiver->h_length);
                serv_addr.sin_port = htons(portno);

                res = connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr));
                //std::cerr << "ERROR connecting";
                if (res < 0 && (k<100)) {
                    std::cout << "server not found (" << client_addr << "," << new_msg.dest << ")" << std::endl;
                    sleep(4);
                }
                k++;
            } while (res<0);

            n = my_send(sockfd, data, new_msg.dest);
            
            if (new_msg.save_connection) {
                open_connections.insert({new_msg.dest, sockfd});
            }
            else {
                close(sockfd);
            }
        }

    }

}