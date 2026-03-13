// Heavily inspired by https://github.com/PacktPublishing/Cpp-High-Performance/blob/master/Chapter03/scoped_timer.cpp


#include <chrono>

#define USE_TIMER 1

#if USE_TIMER
#define MEASURE_FUNCTION() ScopedTimer timer{__func__}
#else
#define MEASURE_FUNCTION()
#endif

class ScopedTimer {

public:
    using ClockType = std::chrono::steady_clock;

  ScopedTimer(const char* func);

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer(ScopedTimer&&) = delete;
  auto operator=(const ScopedTimer&) -> ScopedTimer& = delete;
  auto operator=(ScopedTimer&&) -> ScopedTimer& = delete;

  ~ScopedTimer();

private:
  const char* function_ = {};
  const ClockType::time_point start_ = {};
};

