#ifndef __GPGPU_MULTITHREADING_CPU_H__
#define __GPGPU_MULTITHREADING_CPU_H__

#include <stdio.h>

#include <boost/thread/thread.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/atomic.hpp>

#include "GpGpu/GpGpuTools.h"
#include "GpGpu/GpGpuOptimisation.h"

#define ITERACUDA   2
#define SIZERING    2

template< class H, class D >
class GpGpuMultiThreadingCpu
{
public:

    GpGpuMultiThreadingCpu();
    ~GpGpuMultiThreadingCpu();

    void launchJob();
    D&  GetDeviIN() {return _devi_IN;}
    H*  GetHostIn() {return _host_IN;}
    H*  GetHostOut(){return _host_OUT;}

private:

    virtual void InitPrecompute()           = 0;
    virtual void Precompute(H* hostIn)      = 0;
    virtual void GpuCompute()               = 0;

    void producer(void);
    void ConsProd(void);
    void Consumer(void);

    const int           iterations;

    boost::lockfree::spsc_queue<H*, boost::lockfree::capacity<SIZERING> > spsc_queue_1;
    boost::lockfree::spsc_queue<H*, boost::lockfree::capacity<SIZERING> > spsc_queue_2;
    boost::atomic<bool> done_PreComp;
    boost::atomic<bool> done_GPU;

    boost::thread*  _producer_thread;
    boost::thread*  _consProd_thread;
    boost::thread*  _consumer_thread;

    H               _ringBuffer[SIZERING+1];

    H*              _host_IN;
    H*              _host_OUT;
    D               _devi_IN;
    D               _devi_OUT;

};

template< class H, class D >
GpGpuMultiThreadingCpu<H,D>::GpGpuMultiThreadingCpu():    
    iterations(ITERACUDA),
    done_PreComp(false),
    done_GPU(false)
{

    srand (time(NULL));
    for(int i = 0 ; i < SIZERING + 1; i++)
    {
        _ringBuffer[i].Realloc(NWARP * WARPSIZE);
       // _ringBuffer[i].FillRandom((uint)0,(uint)128);
    }

    _host_OUT = new H((uint)NWARP * WARPSIZE);
    _devi_IN.Realloc(NWARP * WARPSIZE);
}

template< class H, class D >
GpGpuMultiThreadingCpu<H,D>::~GpGpuMultiThreadingCpu(){}

template< class H, class D >
void GpGpuMultiThreadingCpu<H,D>::producer()
{
    int Idbuf = 0;

    for (int i = 0; i != iterations; ++i)
    {
        Precompute(_ringBuffer + Idbuf);

        (_ringBuffer + Idbuf)->OutputValues();

        while (!spsc_queue_1.push(_ringBuffer + Idbuf));

        Idbuf = (Idbuf + 1)%(SIZERING+1);
    }   
}

template< class H, class D >
void GpGpuMultiThreadingCpu<H,D>::ConsProd()
{
    while (!done_PreComp) {
        while (spsc_queue_1.pop(_host_IN))
        {
            GetDeviIN().CopyHostToDevice(_host_IN->pData());

            GpuCompute();

            GetDeviIN().CopyDevicetoHost(_host_OUT->pData());

            while (!spsc_queue_2.push(_host_OUT))
                ;         
        }
    }

    while (spsc_queue_1.pop(_host_IN))
        while (!spsc_queue_2.push(_host_IN));

    _devi_IN.Dealloc();

}

template< class H, class D >
void GpGpuMultiThreadingCpu<H,D>::Consumer()
{
    H  *result;

    while (!done_GPU)
        while (spsc_queue_2.pop(result));

    result->OutputValues();
}

template< class H, class D >
void GpGpuMultiThreadingCpu<H,D>::launchJob()
{

    _producer_thread = new boost::thread(&GpGpuMultiThreadingCpu::producer,this);
    _consProd_thread = new boost::thread(&GpGpuMultiThreadingCpu::ConsProd,this);
    _consumer_thread = new boost::thread(&GpGpuMultiThreadingCpu::Consumer,this);

    _producer_thread->join();
    done_PreComp = true;
    _consProd_thread->join();
    done_GPU = true;
    _consumer_thread->join();

}

#define UINT3D CuHostData3D<uint>

class JobCpuGpuTest : public GpGpuMultiThreadingCpu< UINT3D, CuDeviceData3D<uint> >
{
public:

    JobCpuGpuTest(){}
    ~JobCpuGpuTest(){}

 private:
    virtual void InitPrecompute();
    virtual void Precompute(UINT3D* hostIn);
    virtual void GpuCompute(){ Launch((uint*)GetDeviIN().pData()); }

};

void JobCpuGpuTest::InitPrecompute(){}

void JobCpuGpuTest::Precompute(UINT3D* hostIn)
{

    hostIn->FillRandom((uint)0,(uint)128);

}


#endif //__GPGPU_MULTITHREADING_CPU_H__

