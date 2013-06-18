#include "GpGpu/GpGpuMultiThreadingCpu.h"



GpGpuMultiThreadingCpu::GpGpuMultiThreadingCpu():
    producer_count(0),
    iterations(ITERACUDA),
    consumer_count(0),
    consumerProducer_count(0),
    done(false),
    done_2(false),
    producResult((uint)ITERACUDA),
    consumResult((uint)ITERACUDA)
{

}

GpGpuMultiThreadingCpu::~GpGpuMultiThreadingCpu()
{
}

void GpGpuMultiThreadingCpu::producer()
{


    srand (time(NULL));

    for (int i = 0; i != iterations; ++i) {
        ++producer_count;
        //int rdVal  = rand()%((int)1024);

        CuHostData3D<uint>  rdVal((uint)SIZECU);

        rdVal.FillRandom((uint)0,(uint)128);

        producResult[i] = rdVal[0];

        while (!spsc_queue_1.push(rdVal))
            ;

        //boost::this_thread::sleep(boost::posix_time::microsec(1));
        //printf("producer 1  : ");
        //rdVal.OutputValues();
        //printf("producer : %d\n",rdVal[0]);

    }

        producResult.OutputValues();
}

void GpGpuMultiThreadingCpu::ConsumerProducer()
{
    CuHostData3D<uint> value;
    while (!done) {
        while (spsc_queue_1.pop(value))
        {

            CuDeviceData3D<uint> devValue(SIZECU,"devValue");

            devValue.CopyHostToDevice(value.pData());

            Launch(devValue.pData());

            devValue.CopyDevicetoHost(value.pData());

            cudaDeviceSynchronize();

            while (!spsc_queue_2.push(value))
                ;

//            printf("ConsProd 1 : ");
//            value.OutputValues();

            ++consumerProducer_count;

            //delete pvalue;
        }
    }

    while (spsc_queue_1.pop(value))
    {

        while (!spsc_queue_2.push(value))
            ;
        printf("ConsProd 2 : %d\n",value[0]);
        ++consumerProducer_count;
    }
}

void GpGpuMultiThreadingCpu::Consumer()
{
    CuHostData3D<uint>  value;
    while (!done_2) {

        while (spsc_queue_2.pop(value))
        {
            consumResult[consumer_count] = value[0];
            //printf("Consumer 1 : %d\n",value[0]);
//            printf("Consumer 1  : ");
//            boost::this_thread::sleep(boost::posix_time::microsec(300));
//            value.OutputValues();
            ++consumer_count;
        }
    }

    consumResult.OutputValues();

//    while (spsc_queue_2.pop(value))
//    {
//        printf("Consumer 2 : %d\n",value[0]);
//        ++consumer_count;
//    }

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

