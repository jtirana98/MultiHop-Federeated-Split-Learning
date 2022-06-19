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
   // std::cout << forw_time << std::endl;
    auto back_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (part3.getTimestamp() - part2.getTimestamp()).count();

    backprop_.push_back(back_time);
    //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(part3.getTimestamp() - part2.getTimestamp()).count() << std::endl;
    //std::cout << back_time << std::endl;
}

Total::Total() {
    
}

void Total::printRes() {
    int count = 0, max, min, sum;
    std::vector<std::vector<int>> atr{forward_, backprop_};
    std::vector<std::string> str_print{"FORWARD:", "BACKPROP: "};

    for (int i = 0; i < 2; i++) {
        count = 0;
        sum = 0;
        std::cout << str_print[i] << std::endl;
        
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
            std::cout << interval << std::endl;
        }

        double avg = (double)sum/count;
        
        std::cout << "average time: " << avg << std::endl;
    }
}

void Total::addEvent(Event event) {
    if (event.getType() == forward) {
        //std::cout << "ok" << std::endl;
        forward_timestamps.push_back(event);
    }
    else {
        backprop_timestamps.push_back(event);
    }
}

void Total::computeIntervals() {
    std::vector<int> new_batch_forward, new_batch_backprop;

    // forward
    for (int i=0; i< forward_timestamps.size(); i++) {
        auto part1 = forward_timestamps[i];

        if (i == forward_timestamps.size()-1) {
            auto forw_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (backprop_timestamps[0].getTimestamp() - part1.getTimestamp()).count();
            //std::cout << "f " << forw_time << std::endl;
            new_batch_forward.push_back(forw_time);
        }
        else {
            auto forw_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (forward_timestamps[i+1].getTimestamp() - part1.getTimestamp()).count();
            //std::cout << "f " << forw_time << std::endl;
            new_batch_forward.push_back(forw_time);
        }
        
        //std::cout << "f " << forw_time << std::endl;
        
    }

    // backprop
    for (int i= backprop_timestamps.size() - 1; i > 0; i--) {
        auto part1 = backprop_timestamps[i-1];
        auto part2 = backprop_timestamps[i];

        auto forw_time = std::chrono::duration_cast<std::chrono::milliseconds>
                        (part2.getTimestamp() - part1.getTimestamp()).count();
        //std::cout << "b " << forw_time << std::endl;
        new_batch_backprop.push_back(forw_time);
    }
    

    forward_timestamps.clear();
    backprop_timestamps.clear();

    forward_split.push_back(new_batch_forward);
    backprop_split.push_back(new_batch_backprop);
}

void Total::printRes_intervals() {
    int count = 0, max, min, sum;
    std::vector<std::vector<std::vector<int>>> atr{forward_split, backprop_split};
    std::vector<std::string> str_print{"FORWARD: ", "BACKPROP: "};

    std::vector<double> forw, back;
    std::vector<std::vector<double>> log_{forw, back};

    //std::cout << "-: " << forward_split.size() << std::endl;
    //std::cout << "-: " << backprop_split.size() << std::endl;
    //std::cout << "-: " << forward_split[0].size() << std::endl;
    //std::cout << "-: " << backprop_split[0].size() << std::endl;

    for (int i = 0; i < 2; i++) {
        std::cout << str_print[i] << std::endl;
        for (int j=0; j<forward_split[0].size(); j++) { // for each layer
            count = 0;
            sum = 0;
            std::cout << "layer: " << j << std::endl;
            for (int k = 0; k<forward_split.size(); k++) {  // for each batch
                auto interval = atr[i][k][j];
                std::cout << "int: " << interval << std::endl;
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
                //std::cout << interval << std::endl;
            }

            double avg = (double)sum/(count-2);
            std::cout << "average time: " << avg << std::endl;
            log_[i].push_back(avg);
        }
        
    }

    
    for (int i = 0; i < 2; i++) {
        //std::cout << log_[i].size() << std::endl;
        for (int j=0; j<log_[i].size(); j++) {
            std::cout << log_[i][j] << "\t";
        }

        std::cout << std::endl;
    }
    
}

