//
// Created by luca on 01/06/18.
//

#ifndef ACA2018_LOGGING_H
#define ACA2018_LOGGING_H

#include "mkldnn.hpp"
#include "../mem_management/mem_base.h"
#include <fstream>


void log(std::string str);
void log(mkldnn::memory::dims d, int size);
void log(long long n);
void log(std::string label, long long n);
void log(membase* mb);
void log(std::vector<int> v);

class Logger{
public:
    explicit Logger(std::string&& path);
    template <class T>
    void logValue(T value) {
        if (!lineBegin)
            logfile << ",";
        logfile << value;
        lineBegin = false;
    }
    void endLine();
    ~Logger();
private:
    std::ofstream& logfile;
    bool lineBegin;
};
std::string error_message(int status);

#endif //ACA2018_LOGGING_H
