#pragma once
#include <iostream>
#include <chrono>

class TimestampLogger {
public:
    // Static method to get the singleton instance
    static TimestampLogger& getInstance() {
        static TimestampLogger instance; // Guaranteed to be destroyed.
                                        // Instantiated on first use.
        return instance;
    }

    // Method to get the program start time
    std::chrono::steady_clock::time_point getProgramStartTime() {
        return programStartTime;
    }

    

    // Method to calculate and return the elapsed time since program start
    u_int64_t getElapsedTime() {
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - programStartTime);
        return elapsedTime.count();
    }

private:
    // Private constructor to prevent instantiation
    TimestampLogger() {
        // Initialize the program start time
        programStartTime = std::chrono::steady_clock::now();
    }

    // Private destructor to prevent deletion of the instance
    ~TimestampLogger() {}

    // Private copy constructor and assignment operator to prevent cloning
    TimestampLogger(const TimestampLogger&);
    TimestampLogger& operator=(const TimestampLogger&);

    // Member variable to store the program start time
    std::chrono::steady_clock::time_point programStartTime;
};

