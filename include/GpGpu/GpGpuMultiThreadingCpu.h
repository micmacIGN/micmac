#ifndef __GPGPU_MULTITHREADING_CPU_H__
#define __GPGPU_MULTITHREADING_CPU_H__

#include <stdio.h>

#include <boost/thread/thread.hpp>
#include <boost/signal.hpp>
#include <boost/asio/io_service.hpp>

class GpGpuMultiThreadingCpu
{
public:
    GpGpuMultiThreadingCpu();

    ~GpGpuMultiThreadingCpu();
    void doSomething()
        {
            _service.post(boost::bind(&GpGpuMultiThreadingCpu::doSomethingOp, this));
        }
    void loop()
    {
        printf("Run service\n");
        _service.run();
    }


    void doSomethingOp()
    {
        printf("Ehhhh!\n");
    }


private:






    void precomputeCpu();
    void copyResult();
    void threadComputeGpGpu();
    void launchKernel();

    boost::mutex                _mutexDataIN;
    boost::asio::io_service     _service;

};



#endif //__GPGPU_MULTITHREADING_CPU_H__

