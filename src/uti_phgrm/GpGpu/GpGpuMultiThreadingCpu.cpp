#include "GpGpu/GpGpuMultiThreadingCpu.h"



GpGpuMultiThreadingCpu::GpGpuMultiThreadingCpu():
    producer_count(0),
    iterations(10),
    consumer_count(0),
    consumerProducer_count(0),
    done(false),
    done_2(false)
{

}

GpGpuMultiThreadingCpu::~GpGpuMultiThreadingCpu()
{
}

void GpGpuMultiThreadingCpu::producer()
{
    srand (time(NULL));
    for (int i = 0; i != iterations; ++i) {
        //int value = ++producer_count;
           int rdVal  = rand()%((int)1024);

        while (!spsc_queue_1.push(rdVal))
            ;
        printf("producer : %d\n",rdVal);
    }
}

void GpGpuMultiThreadingCpu::ConsumerProducer()
{
    int value;
    while (!done) {
        while (spsc_queue_1.pop(value))
        {

            while (!spsc_queue_2.push(value))
                ;
            printf("ConsumerProducer : %d\n",value);
            ++consumerProducer_count;
        }
    }

    while (spsc_queue_1.pop(value))
    {

        while (!spsc_queue_2.push(value))
            ;
        printf("ConsumerProducer : %d\n",value);
        ++consumerProducer_count;
    }
}

void GpGpuMultiThreadingCpu::Consumer()
{
    int value;
    while (!done_2) {
        while (spsc_queue_2.pop(value))
        {

            printf("Consumer : %d\n",value);
            ++consumer_count;
        }
    }

    while (spsc_queue_2.pop(value))
    {
        printf("Consumer : %d\n",value);
        ++consumer_count;
    }

}


void GpGpuMultiThreadingCpu::createThread()
{

    producer_thread = new boost::thread(&GpGpuMultiThreadingCpu::producer,this);
    consumerProducer_thread = new boost::thread(&GpGpuMultiThreadingCpu::ConsumerProducer,this);
    consumer_thread = new boost::thread(&GpGpuMultiThreadingCpu::Consumer,this);

    producer_thread->join();
    done = true;
    consumerProducer_thread->join();
    done_2 = true;
    consumer_thread->join();

}

void GpGpuMultiThreadingCpu::precomputeCpu()
{
}

void GpGpuMultiThreadingCpu::copyResult()
{
}

void GpGpuMultiThreadingCpu::threadComputeGpGpu()
{
}

void GpGpuMultiThreadingCpu::launchKernel()
{
}

