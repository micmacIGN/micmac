#ifndef __GPGPU_MULTITHREADING_CPU_H__
#define __GPGPU_MULTITHREADING_CPU_H__

#include <stdio.h>

#include <boost/thread/thread.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/atomic.hpp>

#include "GpGpu/GpGpuTools.h"
#include "GpGpu/GpGpuOptimisation.h"


#define ITERACUDA 8

#define SIZERING 2


template< class H, class D >
class GpGpuMultiThreadingCpu
{
public:

    GpGpuMultiThreadingCpu();
    ~GpGpuMultiThreadingCpu();

    void createThread();

private:

    void Precompute(uint idBuffer);

    void producer(void);
    void ConsProd(void);
    void Consumer(void);

    int producer_count;
    const int iterations;
    boost::atomic_int consumer_count;
    boost::atomic_int consumerProducer_count;

    boost::lockfree::spsc_queue<CuHostData3D<uint> *, boost::lockfree::capacity<SIZERING> > spsc_queue_1;
    boost::lockfree::spsc_queue<CuHostData3D<uint> *, boost::lockfree::capacity<SIZERING> > spsc_queue_2;
    boost::atomic<bool> done;
    boost::atomic<bool> done_2;

    boost::thread *producer_thread;
    boost::thread *consProd_thread;
    boost::thread *consumer_thread;

};

template< class H, class D >
GpGpuMultiThreadingCpu<H,D>::GpGpuMultiThreadingCpu():
    producer_count(0),
    iterations(ITERACUDA),
    consumer_count(0),
    consumerProducer_count(0),
    done(false),
    done_2(false)
{}

template< class H, class D >
GpGpuMultiThreadingCpu<H,D>::~GpGpuMultiThreadingCpu(){}

template< class H, class D >
void GpGpuMultiThreadingCpu<H,D>::producer()
{

    srand (time(NULL));

    H rdVal[SIZERING+1];

    for(int i = 0 ; i < SIZERING + 1; i++)
    {
        rdVal[i].Realloc(NWARP * WARPSIZE);
        rdVal[i].FillRandom((uint)0,(uint)128);
    }

    int Idbuf = 0;

    for (int i = 0; i != iterations; ++i)
    {

        Precompute(Idbuf);

        while (!spsc_queue_1.push(rdVal + Idbuf))
            ;

        Idbuf = (Idbuf + 1)%(SIZERING+1);
    }

}

template< class H, class D >
void GpGpuMultiThreadingCpu<H,D>::ConsProd()
{
    H *value;
    H *result = new CuHostData3D<uint>((uint)NWARP * WARPSIZE);
    D devValue(NWARP * WARPSIZE,"devValue");

    while (!done) {
        while (spsc_queue_1.pop(value))
        {

            devValue.CopyHostToDevice(value->pData());

            Launch(devValue.pData());

            devValue.CopyDevicetoHost(result->pData());

            while (!spsc_queue_2.push(result))
                ;

            ++consumerProducer_count;
        }
    }

    while (spsc_queue_1.pop(value))
    {

        while (!spsc_queue_2.push(value))
            ;
        //printf("ConsProd 2 : %d\n",(*value)[0]);
        ++consumerProducer_count;
    }

    devValue.Dealloc();

}

template< class H, class D >
void GpGpuMultiThreadingCpu<H,D>::Consumer()
{
    H  *value;

    while (!done_2) {

        while (spsc_queue_2.pop(value))
        {
            ++consumer_count;
        }
    }

    //value->OutputValues();

}

template< class H, class D >
void GpGpuMultiThreadingCpu<H,D>::Precompute(uint idBuffer)
{
}


template< class H, class D >
void GpGpuMultiThreadingCpu<H,D>::createThread()
{

    producer_thread = new boost::thread(&GpGpuMultiThreadingCpu::producer,this);
    consProd_thread = new boost::thread(&GpGpuMultiThreadingCpu::ConsProd,this);
    consumer_thread = new boost::thread(&GpGpuMultiThreadingCpu::Consumer,this);

    producer_thread->join();
    done = true;
    consProd_thread->join();
    done_2 = true;
    consumer_thread->join();

}
#endif //__GPGPU_MULTITHREADING_CPU_H__

