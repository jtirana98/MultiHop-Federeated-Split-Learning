#ifndef PIPELINE_LOGGING_H_
#define PIPELINE_LOGGING_H_

#include <iostream>
#include <map>
#include <string>
#include <sstream> 
#include <chrono>
#include <ctime>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <deque> 
#include <filesystem>
#include <fstream>

#define NT_RECEIVED_MSG 1 
#define NT_PREPARE_MSG 2
#define NT_START_SENDING 3
#define NT_STOP_SENDING 4

#define INIT_START_MSG_PREP 5
#define INIT_END_PREP_START_DO_BCAST 6
#define INIT_END_BCAST_START_REFACTOR 7
#define INIT_END_REFACTOR_START_CN_BCAST 8
#define INIT_END_BCAST_CN 9
// 
#define INIT_WAIT_FOR_REFACTOR 10
#define INIT_END_W_REFACTOR 11
#define INIT_END_INIT 12

// execution compute node
#define CN_START_WAIT 13
#define CN_START_EXEC 14
#define CN_END_EXEC 15

// one batch data owner
#define DO_START_BATCH 16
#define DO_FRWD_FIRST_PART 17
#define DO_END_WAIT 18
#define DO_FRWD_BCKWD_SECOND_PART 19
#define DO_END_WAIT2 20
#define DO_END_BATCH 21

// init phase steps
// init node:               5 - 6 - for each dO: {2 - 3 - 4} - 7 - 8 - for each CN: {2 - 3 - 4} - 9
// other data owners:       10 - . - . - . - . . - . - . - 1 - 11 - 12
// other compute node:      10 - . - . - . - . - . - . - . -. - . - . - . - . - . - . - . - . - 1 - 11

inline std::string const Path = "/Users/joanatirana/Documents/git_repos/testing_environment/pipeline_simulation/logs/";
inline std::map<int, std::string> const point_description = {
    {NT_RECEIVED_MSG, "Network layer: received message"},
    {NT_PREPARE_MSG, "Network layer: preparing to send"},
    {NT_START_SENDING, "Network layer: starts message transmission"},
    {NT_STOP_SENDING, "Network layer: completes message transmission"},
    {INIT_START_MSG_PREP, "Initialization phase: init node starts preperation"},
    {INIT_END_PREP_START_DO_BCAST, "Initialization phase: init node completes preperation - start bcast to dataowners"},
    {INIT_END_BCAST_START_REFACTOR, "Initialization phase: bcast to do completed - start refactoring"},
    {INIT_END_REFACTOR_START_CN_BCAST, "Initialization phase: completes refactoring - start bcast to cn"},
    {INIT_END_BCAST_CN, "Initialization phase: bcast to cn completed"},
    {INIT_WAIT_FOR_REFACTOR, "Initialization phase: do/cn waiting for refactor message"},
    {INIT_END_W_REFACTOR, "Initialization phase: do/cn end waiting for refactor message"},
    {INIT_END_INIT, "Initialization phase: completed"},
    {CN_START_WAIT, "Execution phase: CN waits for new task"},
    {CN_START_EXEC, "Execution phase: CN starts executing task"},
    {CN_END_EXEC, "Execution phase: CN completed a task"},
    {DO_START_BATCH, "Execution phase: DO starts new batch"},
    {DO_FRWD_FIRST_PART, "Execution phase: DO produced activations from first part"},
    {DO_END_WAIT, "Execution phase: DO received activations from CN"},
    {DO_FRWD_BCKWD_SECOND_PART, "Execution phase: DO completed training of last part"},
    {DO_END_WAIT2, "Execution phase: DO received gradients"},
    {DO_END_BATCH, "Execution phase: DO completed training for first part"},
};

class Point {
 public:
    std::chrono::steady_clock::time_point timestamp;
    int msg_id;
    
    int client_id; // owner of msg or task
    int point;
    //std::stringstream timeline;
    std::string timeline;
    
    Point() {};
    Point(/*int msg_id, */int client_id, int point) :
        //msg_id(msg_id),
        client_id(client_id),
        point(point) {
            //std::cout << std::chrono::steady_clock::now();
            timestamp = std::chrono::steady_clock::now();
    };

    void update_timeline(std::chrono::steady_clock::time_point begin) {
        auto interval = std::chrono::duration_cast<std::chrono::milliseconds>
                            (timestamp - begin).count();

        std::stringstream timeline_stream;
        timeline_stream << interval;
        timeline = timeline_stream.str();

    }
};

enum interval_type {
    fwd_only,           // 0
    bwd_only,          // 1
    opz_only,         // 2
    fwd_bwd_opz,     // 1
    bwd_opz         // 2
};


class logger {
 public:
    int node_id;
    int fd_logs;
    std::string const parent_dir = "logs";
    std::string dir_name;
    std::string f_name, interval1, interval2, interval3;

    std::mutex m_mutex, interval_mutex;
    std::condition_variable m_cv, interval_cv;
    std::deque<Point> points;
    std::vector<std::vector<std::string>> intervals{
                                                    std::vector<std::string>(), 
                                                    std::vector<std::string>(), 
                                                    std::vector<std::string>()
                                                };

    std::chrono::steady_clock::time_point big_bang;

    logger(int node_id, std::string log_name): node_id(node_id) {
        big_bang = std::chrono::steady_clock::now();
        dir_name = parent_dir + "/" + log_name + "/";
        // create log directory
        namespace fs = std::filesystem;
        if (!fs::is_directory(dir_name) || !fs::exists(dir_name)) { // Check if src folder exists
            fs::create_directory(dir_name);
        }

        f_name = dir_name + "_node" + std::to_string(node_id) + ".log";
        interval1 = dir_name + "_node" + std::to_string(node_id) + "_i1" + ".log";
        interval2 = dir_name + "_node" + std::to_string(node_id) + "_i2" + ".log";
        interval3 = dir_name + "_node" + std::to_string(node_id) + "_i3" + ".log";

    };

    void add_point(Point newPoint); // producer a
    void add_interval(Point& start, Point& end, interval_type type); // producer b
    void logger_(); // consumer
};

#endif