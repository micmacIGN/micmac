#ifndef __GPGPU_MULTITHREADING_CPU_H__
#define __GPGPU_MULTITHREADING_CPU_H__

#include <stdio.h>

#include <boost/thread/thread.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/atomic.hpp>

#include "GpGpu/GpGpuTools.h"
#include "GpGpu/GpGpuOptimisation.h"


class GpGpuMultiThreadingCpu
{
public:
    GpGpuMultiThreadingCpu();

    ~GpGpuMultiThreadingCpu();


    void producer(void);
    void ConsumerProducer(void);
    void Consumer(void);

    int producer_count;
    const int iterations;
    boost::atomic_int consumer_count;
    boost::atomic_int consumerProducer_count;


    boost::lockfree::spsc_queue<int, boost::lockfree::capacity<2> > spsc_queue_1;
    boost::lockfree::spsc_queue<int, boost::lockfree::capacity<2> > spsc_queue_2;
    boost::atomic<bool> done;
    boost::atomic<bool> done_2;

    boost::thread *producer_thread;
    boost::thread *consumerProducer_thread;
    boost::thread *consumer_thread;

    void createThread();

private:




    void precomputeCpu();
    void copyResult();
    void threadComputeGpGpu();
    void launchKernel();


};



#endif //__GPGPU_MULTITHREADING_CPU_H__

