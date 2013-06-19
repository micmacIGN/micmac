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

    void launchJob();
    D&  GetDataDevice(){return _devicePrecompute;}

private:

    virtual void InitPrecompute()           = 0;
    virtual void Precompute(uint idBuffer)  = 0;
    virtual void GpuCompute(H* host_In,H* host_Out) = 0;

    void producer(void);
    void ConsProd(void);
    void Consumer(void);

    int                 producer_count;
    const int           iterations;
    boost::atomic_int   consumer_count;
    boost::atomic_int   consumerProducer_count;

    boost::lockfree::spsc_queue<CuHostData3D<uint> *, boost::lockfree::capacity<SIZERING> > spsc_queue_1;
    boost::lockfree::spsc_queue<CuHostData3D<uint> *, boost::lockfree::capacity<SIZERING> > spsc_queue_2;
    boost::atomic<bool> done_PreComp;
    boost::atomic<bool> done_GPU;

    boost::thread *producer_thread;
    boost::thread *consProd_thread;
    boost::thread *consumer_thread;

    H _ringBuffer[SIZERING+1];

    H *_hostPrecompute;
    H *_hostResult;
    D _devicePrecompute;

};

template< class H, class D >
GpGpuMultiThreadingCpu<H,D>::GpGpuMultiThreadingCpu():
    //producer_count(0),
    iterations(ITERACUDA),
    consumer_count(0),
    consumerProducer_count(0),
    done_PreComp(false),
    done_GPU(false)
{

    srand (time(NULL));
    for(int i = 0 ; i < SIZERING + 1; i++)
    {
        _ringBuffer[i].Realloc(NWARP * WARPSIZE);
        _ringBuffer[i].FillRandom((uint)0,(uint)128);
    }

    _hostResult = new H((uint)NWARP * WARPSIZE);
    _devicePrecompute.Realloc(NWARP * WARPSIZE);
}

template< class H, class D >
GpGpuMultiThreadingCpu<H,D>::~GpGpuMultiThreadingCpu()
{

}

template< class H, class D >
void GpGpuMultiThreadingCpu<H,D>::producer()
{
    int Idbuf = 0;

    for (int i = 0; i != iterations; ++i)
    {
        Precompute(Idbuf);

        while (!spsc_queue_1.push(_ringBuffer + Idbuf));

        //(_ringBuffer + Idbuf)->OutputValues();

        Idbuf = (Idbuf + 1)%(SIZERING+1);

    }

}

template< class H, class D >
void GpGpuMultiThreadingCpu<H,D>::ConsProd()
{


    while (!done_PreComp) {
        while (spsc_queue_1.pop(_hostPrecompute))
        {

//            _devicePrecompute.CopyHostToDevice(_hostPrecompute->pData());

//            Launch(_devicePrecompute.pData());

//            _devicePrecompute.CopyDevicetoHost(_hostResult->pData());

            GpuCompute(_hostPrecompute,_hostResult);

            while (!spsc_queue_2.push(_hostResult))
                ;

            ++consumerProducer_count;
        }
    }

    while (spsc_queue_1.pop(_hostPrecompute))
    {

        while (!spsc_queue_2.push(_hostPrecompute))
            ;
        //printf("ConsProd 2 : %d\n",(*value)[0]);
        ++consumerProducer_count;
    }

    _devicePrecompute.Dealloc();

}

template< class H, class D >
void GpGpuMultiThreadingCpu<H,D>::Consumer()
{
    H  *result;

    while (!done_GPU)
        while (spsc_queue_2.pop(result))
            ++consumer_count;

    //result->OutputValues();
}


template< class H, class D >
void GpGpuMultiThreadingCpu<H,D>::Precompute(uint idBuffer)
{
}


template< class H, class D >
void GpGpuMultiThreadingCpu<H,D>::launchJob()
{

    producer_thread = new boost::thread(&GpGpuMultiThreadingCpu::producer,this);
    consProd_thread = new boost::thread(&GpGpuMultiThreadingCpu::ConsProd,this);
    consumer_thread = new boost::thread(&GpGpuMultiThreadingCpu::Consumer,this);

    producer_thread->join();
    done_PreComp = true;
    consProd_thread->join();
    done_GPU = true;
    consumer_thread->join();

}

#define UINT3D CuHostData3D<uint>

class JobCpuGpuTest : public GpGpuMultiThreadingCpu< UINT3D, CuDeviceData3D<uint> >
{
public:

    JobCpuGpuTest(){}
    ~JobCpuGpuTest(){}

 private:
    virtual void InitPrecompute();
    virtual void Precompute(uint idBuffer);
    virtual void GpuCompute(UINT3D* host_In,UINT3D* host_Out)
    {

        GetDataDevice().CopyHostToDevice(host_In->pData());

        //Launch((uint*)GetDataDevice().pData());

        GetDataDevice().CopyDevicetoHost(host_Out->pData());

     }

};



#endif //__GPGPU_MULTITHREADING_CPU_H__

