#include <iostream>
#include "scoped_timer.h"

ScopedTimer::ScopedTimer(const char* func)
    : function_{func}, start_{ClockType::now()}
{
}

ScopedTimer::~ScopedTimer()
{
    using namespace std::chrono;
    auto stop = ClockType::now();
    auto duration = (stop - start_);
    auto ms = duration_cast<milliseconds>(duration).count();
    std::cout << ms << " ms " << function_ <<  '\n';
}

