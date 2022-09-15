#include "mylogging.h"

Event::Event(measure_type type, std::string message, int layer) : type(type), message(message), layer(layer){
    timestamp = std::chrono::steady_clock::now();
}

std::chrono::steady_clock::time_point Event::getTimestamp() {
    return timestamp;
}

measure_type Event::getType() {
    return type;
}

void Total::addNew(Event part1, Event part2, Event part3) {
    auto forw_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (part2.getTimestamp() - part1.getTimestamp()).count();

    forward_.push_back(forw_time);
    auto back_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (part3.getTimestamp() - part2.getTimestamp()).count();

    backprop_.push_back(back_time);
    
}

void Total::addNew(Event part1, Event part2, Event part3, Event part4) {
    auto forw_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (part2.getTimestamp() - part1.getTimestamp()).count();

    forward_.push_back(forw_time);
   
    auto back_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (part3.getTimestamp() - part2.getTimestamp()).count();

    backprop_.push_back(back_time);

    auto optim_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (part4.getTimestamp() - part3.getTimestamp()).count();
    
    optimizer_.push_back(optim_time);
}


Total::Total() {
    
}

void Total::printRes() {
    int count = 0, max, min, sum;
    std::vector<std::vector<int>> atr{forward_, backprop_};
    std::vector<std::string> str_print{"FORWARD:", "BACKPROP: "};

    if (optimizer_.size() != 0) {
        atr.push_back(optimizer_);
        str_print.push_back("OPTIMIZER: ");
    }

    for (int i = 0; i < atr.size(); i++) {
        count = 0;
        sum = 0;
        std::cout << str_print[i] << "\t";
        
        for (int interval : atr[i]) {
            if (count == 0) {
                max = interval;
                min = interval;
            }
            else{
                if (interval > max) {
                    sum += max;
                    max = interval;
                }
                else if (interval < min) {
                    sum += min;
                    min = interval;
                }
                else {
                    sum += interval;
                }
            }
            count += 1;
        }

        double avg = (double)sum/(count-2);
        std::cout << avg << std::endl;
    }
}

void Total::addEvent(Event event) {
    if (event.getType() == forward) {
        
        forward_timestamps.push_back(event);
    }
    else if (event.getType() == backprop){
        backprop_timestamps.push_back(event);
    }
    else {
        optimize_timestamps.push_back(event);
    }
}

void Total::computeIntervals() {
    std::vector<int> new_batch_forward, new_batch_backprop, new_batch_optimizer;

    // forward
    for (int i=0; i< forward_timestamps.size(); i++) {
        auto part1 = forward_timestamps[i];

        if (i == forward_timestamps.size()-1) {
            auto forw_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (backprop_timestamps[0].getTimestamp() - part1.getTimestamp()).count();
            new_batch_forward.push_back(forw_time);
        }
        else {
            auto forw_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (forward_timestamps[i+1].getTimestamp() - part1.getTimestamp()).count();
            new_batch_forward.push_back(forw_time);
        }

    }
    // backprop and optimizer
    for (int i = backprop_timestamps.size() - 1; i > 0; i--) {
        auto part1 = backprop_timestamps[i-1];
        auto part2 = optimize_timestamps[i-1];
        auto part3 = backprop_timestamps[i];

        auto back_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (part2.getTimestamp() - part1.getTimestamp()).count();
        new_batch_backprop.push_back(back_time);

        auto opti_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (part3.getTimestamp() - part2.getTimestamp()).count();
        new_batch_optimizer.push_back(opti_time);
    }
    

    forward_timestamps.clear();
    backprop_timestamps.clear();
    optimize_timestamps.clear();

    forward_split.push_back(new_batch_forward);
    backprop_split.push_back(new_batch_backprop);
    optimize_split.push_back(new_batch_optimizer);
}

void Total::printRes_intervals() {
    int count = 0, max, min, sum;
    std::vector<std::vector<std::vector<int>>> atr{forward_split, backprop_split, optimize_split};
    std::vector<std::string> str_print{"FORWARD: ", "BACKPROP: ", "OPTIMIZER: "};

    std::vector<double> forw, back, opt;
    std::vector<std::vector<double>> log_{forw, back, opt};

    for (int i = 0; i < 3; i++) {
        std::cout << str_print[i] << std::endl;
        for (int j=0; j<forward_split[0].size(); j++) { // for each layer
            count = 0;
            sum = 0;
            for (int k = 0; k<forward_split.size(); k++) {  // for each batch
                auto interval = atr[i][k][j];
                if (count == 0) {
                    max = interval;
                    min = interval;
                }
                else{
                    if (interval > max) {
                        sum += max;
                        max = interval;
                    }
                    else if (interval < min) {
                        sum += min;
                        min = interval;
                    }
                    else {
                        sum += interval;
                    }
                }
                count += 1;
            }

            double avg = (double)sum/(count-2);
            log_[i].push_back(avg);
        }
        
    }

    
    for (int i = 0; i < 3; i++) {
        for (int j=0; j<log_[i].size(); j++) {
            std::cout << log_[i][j] << "\t";
        }

        std::cout << std::endl;
        std::cout << std::endl;
    }
    
}

