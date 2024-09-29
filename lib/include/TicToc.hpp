#pragma once

#include <iostream>
#include <chrono>

class TicToc {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double>;

    TicToc() : start_(Clock::now()) {}

    void tic() {
        start_ = Clock::now();
    }

    double toc(const std::string& message = "") {
        TimePoint end = Clock::now();
        Duration duration = end - start_;
        double elapsed_seconds = duration.count();
        if (!message.empty()) {
            std::cout << message << ": " << elapsed_seconds << " seconds" << std::endl;
        }
        return elapsed_seconds;
    }

private:
    TimePoint start_;  // 记录开始时间
};