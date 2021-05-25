#ifndef CHRONOBENCH_H
#define CHRONOBENCH_H

#include <array>
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include <cmath>

template<size_t SubTimes>
class ChronoBench
{
public:
    typedef std::array<double,SubTimes+1> Times;
    struct FilterType {double mean; double stddev;};

    ChronoBench() { reset(); }

    void reset() {
        allTimes.clear();
        for (auto &t: curTimes) t=Time({0,0});
        for (auto &r: result) r=0;
        for (auto &r: fltResult) r=0;
    }

    void start() { _start(0);}
    void stop() { _stop(0);}
    void start(size_t n) { _start(n);}
    void stop(size_t n) { _stop(n);}
    void stopStart(size_t n0, size_t n1) { _stopStart(n0,n1);}

    void next() {
        Times times = currents();

        allTimes.push_back(times);
        size_t n=allTimes.size();
        for (size_t i=0; i<result.size(); i++)
            result[i] = (result[i] * (n-1) + times[i]) / n;

        for (auto &t: curTimes) t=Time({0,0});
    }

    FilterType filter() {
        FilterType filter={0,0};

        if (allTimes.size()<10)
            return filter;

        std::sort(allTimes.begin(), allTimes.end(), [](const Times& a, const Times &b) {
            return a[0] < b[0];
        });

        for (auto &r: fltResult) r=0;
        size_t nb = allTimes.size()/2;
        for (size_t i=0; i<nb; i++) {
            filter.mean += allTimes[2+i][0];
            filter.stddev += allTimes[2+i][0] * allTimes[2+i][0];
            for (size_t j=0; j<result.size(); j++)
                fltResult[j] += allTimes[2+i][j];
        }
        filter.stddev = std::sqrt((filter.stddev - filter.mean*filter.mean/nb) / (nb-1));
        filter.mean /= nb;
        for (auto &r : fltResult)
            r /= nb;
        return filter;
    }

    double current(size_t n) const {
        return (curTimes[n].sec + curTimes[n].nsec * 1e-9) * 1e3;
    }

    Times currents() const {
        Times times;
        for (size_t i=0; i<times.size(); i++)
            times[i] = current(i);
        return times;
    }

    double mean(size_t n) const { return result[n]; }
    Times means() const { return result; }
    double fltMean(size_t n) const { return fltResult[n]; }
    Times fltMeans() const { return fltResult; }

private:
    struct Time {
        int64_t sec;
        int64_t nsec;
    };
    std::array<Time,SubTimes+1> curTimes;
    Times result;
    Times fltResult;
    std::vector<Times> allTimes;

    inline void _start(size_t n)
    {
        struct timespec tp;
        clock_gettime(CLOCK_MONOTONIC_RAW,&tp);
        curTimes[n].sec -= tp.tv_sec;
        curTimes[n].nsec -= tp.tv_nsec;
    }
    inline void _stop(size_t n)
    {
        struct timespec tp;
        clock_gettime(CLOCK_MONOTONIC_RAW,&tp);
        curTimes[n].sec += tp.tv_sec;
        curTimes[n].nsec += tp.tv_nsec;
    }
    inline void _stopStart(size_t n0,size_t n1)
    {
        struct timespec tp;
        clock_gettime(CLOCK_MONOTONIC_RAW,&tp);
        curTimes[n0].sec += tp.tv_sec;
        curTimes[n0].nsec += tp.tv_nsec;
        curTimes[n1].sec -= tp.tv_sec;
        curTimes[n1].nsec -= tp.tv_nsec;
    }

};

#endif // CHRONOBENCH_H
