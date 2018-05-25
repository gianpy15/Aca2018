//
// Created by luca on 25/05/18.
//

#include <cstdlib>
#include <cstdio>
#include <cstring>

size_t parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    size_t i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = std::strtoul(p, nullptr, 10);
    return i;
}

size_t getCurrentMemUsage(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    size_t result = 0;
    char line[128];

    while (fgets(line, 128, file) != nullptr){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}