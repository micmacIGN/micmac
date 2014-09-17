#ifndef __GPGPU_MULTITHREADING_CPU_H__
#define __GPGPU_MULTITHREADING_CPU_H__

#include <stdio.h>

#include <boost/thread/thread.hpp>
#include <boost/progress.hpp>
#include <boost/timer.hpp>

#include "GpGpu/GpGpu_Data.h"

template< class T >
///
/// \brief The CSimpleJobCpuGpu class
///
class CSimpleJobCpuGpu
{
public:

    ///
    /// \brief CSimpleJobCpuGpu
    /// \param useMultiThreading
    ///
    CSimpleJobCpuGpu(bool useMultiThreading = true);
    ~CSimpleJobCpuGpu();

    ///
    /// \brief SetCompute indique au thread Gpu s'il doit traiter les données
    /// \param toBeComputed
    ///
    void            SetCompute(T toBeComputed);
    ///
    /// \brief GetCompute : savoir si le Gpu doit traiter des données
    /// \return
    ///
    T               GetCompute();

    ///
    /// \brief SetDataToCopy
    /// \param toBeCopy
    ///
    void            SetDataToCopy(T toBeCopy);

    ///
    /// \brief GetDataToCopy
    /// \return
    ///
    T               GetDataToCopy();

    ///
    /// \brief SetPreComp
    /// \param canBePreCompute
    ///
    void            SetPreComp(bool canBePreCompute);

    ///
    /// \brief GetPreComp
    /// \return
    ///
    bool            GetPreComp();

    bool            UseMultiThreading();

    bool            GetIdBuf();
    void            SwitchIdBuffer();
    void            ResetIdBuffer();
    virtual void    freezeCompute() = 0;

    void            CreateJob();
    void            KillJob();

    void            SetProgress(unsigned long expected_count);

    void            IncProgress(uint inc = 1);

protected:

    void            SetThread(boost::thread* Thread);


private:

    boost::thread*  _gpGpuThread;

    virtual void    threadCompute() = 0;


    void            LaunchJob();

    bool            _useMultiThreading;

    boost::mutex    _mutexCompu;
    boost::mutex    _mutexCopy;
    boost::mutex    _mutexPreCompute;

    T               _compute;
    T               _copy;
    bool            _precompute;

    bool            _idBufferHostIn;

    boost::progress_display *_show_progress;

    bool            _show_progress_console;

};

template< class T >
CSimpleJobCpuGpu<T>::CSimpleJobCpuGpu(bool useMultiThreading):
    _gpGpuThread(NULL),
    _useMultiThreading(useMultiThreading),
    _idBufferHostIn(false),
    _show_progress(NULL),
    _show_progress_console(false)
{}

template< class T >
CSimpleJobCpuGpu<T>::~CSimpleJobCpuGpu()
{
    KillJob();
    if(_show_progress)
        delete _show_progress;
}

template< class T >
void CSimpleJobCpuGpu<T>::SetCompute(T toBeComputed)
{
    boost::lock_guard<boost::mutex> guard(_mutexCompu);
    _compute = toBeComputed;
}

template< class T >
T CSimpleJobCpuGpu<T>::GetCompute()
{
    boost::lock_guard<boost::mutex> guard(_mutexCompu);
    return _compute;
}

template< class T >
void CSimpleJobCpuGpu<T>::SetDataToCopy(T toBeCopy)
{
    boost::lock_guard<boost::mutex> guard(_mutexCopy);
    _copy = toBeCopy;

}

template< class T >
T CSimpleJobCpuGpu<T>::GetDataToCopy()
{
    boost::lock_guard<boost::mutex> guard(_mutexCopy);
    return _copy;
}

template< class T >
void CSimpleJobCpuGpu<T>::SetPreComp(bool canBePreCompute)
{
    boost::lock_guard<boost::mutex> guard(_mutexPreCompute);
    _precompute = canBePreCompute;
}

template< class T >
bool CSimpleJobCpuGpu<T>::GetPreComp()
{
    boost::lock_guard<boost::mutex> guard(_mutexPreCompute);
    return _precompute;
}

template< class T >
bool CSimpleJobCpuGpu<T>::UseMultiThreading()
{
    return _useMultiThreading;
}

template< class T >
bool CSimpleJobCpuGpu<T>::GetIdBuf()
{
    return _idBufferHostIn;
}

template< class T >
void CSimpleJobCpuGpu<T>::SwitchIdBuffer()
{
    _idBufferHostIn = !_idBufferHostIn;
}

template< class T >
void CSimpleJobCpuGpu<T>::ResetIdBuffer()
{
    _idBufferHostIn = false;
}

template< class T >
void CSimpleJobCpuGpu<T>::SetThread(boost::thread *Thread)
{
    _gpGpuThread = Thread;
}

template< class T >
void CSimpleJobCpuGpu<T>::SetProgress(unsigned long expected_count)
{
    if(_show_progress_console)
    {
        if(_show_progress == NULL)
            _show_progress = new boost::progress_display(expected_count);
        else
            _show_progress->restart(expected_count);
    }
}

template< class T >
void CSimpleJobCpuGpu<T>::IncProgress(uint inc)
{
    if(_show_progress_console)
        (*_show_progress) += inc;
}

template< class T >
void CSimpleJobCpuGpu<T>::CreateJob()
{
    if(UseMultiThreading())
    {
        SetThread(new boost::thread(&CSimpleJobCpuGpu::LaunchJob,this));
        freezeCompute();
    }
}

template< class T >
void CSimpleJobCpuGpu<T>::KillJob()
{
    if(UseMultiThreading())
    {
        if(_gpGpuThread)
        {
            _gpGpuThread->interrupt();
            delete _gpGpuThread;
            _gpGpuThread = NULL;
        }
    }
//    _mutexCompu.unlock();
 //   _mutexCopy.unlock();
 //   _mutexPreCompute.unlock();
}

template< class T >
void CSimpleJobCpuGpu<T>::LaunchJob()
{
    threadCompute();
}

/*

class DataBuffer
{
    //  Gerer les donnees d entres au niveau du host
//        Les donnes a traiter
//       les parametres
//            - les constants
//           - les non constants
     
     
    

    virtual void AllocHostIn()   = 0;
    virtual void AllocDeviceIn() = 0;

};

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
        _ringBuffer[i].Realloc(NWARP * WARPSIZE);

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

    //result->OutputValues();
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

#define HOST_UINT3D CuHostData3D<uint>
#define DEVI_UINT3D CuDeviceData3D<uint>

class JobCpuGpuTest : public GpGpuMultiThreadingCpu< HOST_UINT3D, DEVI_UINT3D >
{
public:

    JobCpuGpuTest(){}
    ~JobCpuGpuTest(){}

 private:
    virtual void InitPrecompute(){}
    virtual void Precompute(HOST_UINT3D* hostIn){hostIn->FillRandom((uint)0,(uint)128);}
    virtual void GpuCompute(){Launch((uint*)GetDeviIN().pData());}
};
*/

#endif //__GPGPU_MULTITHREADING_CPU_H__

