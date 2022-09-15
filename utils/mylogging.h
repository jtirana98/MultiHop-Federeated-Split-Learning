#pragma once

#include <chrono>
#include <ctime>
#include <ratio>
#include <string>
#include <vector>
#include <iostream>

enum measure_type {
    forward,
    backprop,
    optimize,
    activations_load,
    gradients_load
};


struct dataload{
   int layer;
   measure_type type; // activations_load or gradients_load
   long long data_load;
};

class Event {
   public:
    Event(measure_type type, std::string message, int layer);
    Event() {}
    std::chrono::steady_clock::time_point getTimestamp();
    measure_type getType();
   private:
    measure_type type; // forward or backpop
    std::chrono::steady_clock::time_point timestamp;
    std::string message;
    int layer;
};

struct gatherd_data{
    std::vector<dataload> activations;
    std::vector<dataload> gradients;
};

class Total {
  public:
    Total();
    void addNew(Event part1, Event part2, Event part3);
    void addNew(Event part1, Event part2, Event part3, Event part4);
    void addEvent(Event event);
    void computeIntervals();
    void printRes();
    void printRes_intervals();
  private:
    std::vector<int> forward_;
    std::vector<int> backprop_;
    std::vector<int> optimizer_;
    std::vector<Event> forward_timestamps;
    std::vector<Event> backprop_timestamps;
    std::vector<Event> optimize_timestamps;
    std::vector<std::vector<int>> forward_split;
    std::vector<std::vector<int>> backprop_split;
    std::vector<std::vector<int>> optimize_split;
    std::vector<dataload> activations_;
    std::vector<dataload> gradients_;
};

