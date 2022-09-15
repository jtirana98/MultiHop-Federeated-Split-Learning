#include "pipeline_logging.h"

std::ostream& operator<<(std::ostream& os, const Point& pt) {
    os << std::to_string(pt.client_id); 
    os << ",";
    os << pt.timeline;
    os << ",";
    os << point_description.find(pt.point)->second << "," << std::to_string(pt.point);
    os << "," << pt.msg_id;
    return os;
}


void logger::add_point(Point newPoint) { // producer
    {
    std::unique_lock<std::mutex> lock(m_mutex);
    newPoint.update_timeline(big_bang);
    points.push_back(newPoint);
    
    /*
    std::deque<Point>::iterator it;
    std::cout << "START\n";
    for (it = points.begin(); it != points.end(); ++it) {
        std::cout << *it;
        std::cout << "\n";
    }
    std::cout << "DONE\n";
    */
    }
    m_cv.notify_one();

}

void logger::add_interval(Point& start, Point& end, interval_type type) {// producer b
    std::string intraval_;
    auto interval = std::chrono::duration_cast<std::chrono::milliseconds>
                            (end.timestamp - start.timestamp).count();
    
    std::stringstream timeline_stream;
    timeline_stream << interval;
    intraval_ = timeline_stream.str();
    {
    std::unique_lock<std::mutex> lock(interval_mutex);
    switch (type) {
    case fwd_only:  // 0
        intervals[0].push_back(intraval_);
        break;           
    case bwd_only:  // 1
    case fwd_bwd_opz:
        intervals[1].push_back(intraval_);
        break;
    case opz_only:  // 2
    case bwd_opz:
        intervals[2].push_back(intraval_);
        break;
    default:
        break;
    }

    }
    interval_cv.notify_one();
}

void logger::logger_() { // consumer
    std::vector<Point> to_write_points;
    std::vector<std::string> to_write_int1, to_write_int2, to_write_int3 ;
    int full_list = 10;
    std::ofstream log_file, log_int1, log_int2, log_int3;

    log_file.open(f_name, std::ios::out);
    log_int1.open(interval1, std::ios::out);
    log_int2.open(interval2, std::ios::out);
    log_int3.open(interval3, std::ios::out);

    bool interval = false;
    while(true) {
        if (interval) {
            auto timeout = std::chrono::system_clock::now() + std::chrono::seconds(2);
            std::unique_lock<std::mutex> lock_(interval_mutex);
            while (intervals[0].empty() && intervals[1].empty() && intervals[2].empty()) {
                if (interval_cv.wait_until(lock_, timeout) == std::cv_status::timeout) { }
                break;
            }
            
            if (intervals[0].size() > 0) {
                to_write_int1.push_back(intervals[0].front());
                intervals[0].erase(intervals[0].begin());
                
                if (to_write_int1.size() >= full_list) {
                    for (int i = 0; i < full_list; i++) {
                        log_int1 << to_write_int1[0] << std::endl;
                        to_write_int1.erase(to_write_int1.begin());
                    }
                }
            }

            if (intervals[1].size() > 0) {
                to_write_int2.push_back(intervals[1].front());
                intervals[1].erase(intervals[1].begin());
                
                if (to_write_int2.size() >= full_list) {
                    for (int i = 0; i < full_list; i++) {
                        log_int2 << to_write_int2[0] << std::endl;
                        to_write_int2.erase(to_write_int2.begin());
                    }
                }
            }

            if (intervals[2].size() > 0) {
                to_write_int3.push_back(intervals[2].front());
                intervals[2].erase(intervals[2].begin());
                
                if (to_write_int3.size() >= full_list) {
                    for (int i = 0; i < full_list; i++) {
                        log_int3 << to_write_int3[0] << std::endl;
                        to_write_int3.erase(to_write_int3.begin());
                    }
                }
            }

            interval = false;
        }
        else {
            std::unique_lock<std::mutex> lock(m_mutex);
            while (points.empty()) {
                m_cv.wait(lock, [&](){ return !points.empty(); });
            }

            to_write_points.push_back(points.front());
            points.pop_front();
            
            if (to_write_points.size() >= full_list) {
                for (int i = 0; i < full_list; i++) {
                    log_file << to_write_points[0] << std::endl;
                    to_write_points.erase(to_write_points.begin());
                }
            }
            interval = true;
        }
    }
}
