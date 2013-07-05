#include "GpGpu/GpGpuMultiThreadingCpu.h"

template< class T >
CSimpleJobCpuGpu<T>::CSimpleJobCpuGpu(bool useMultiThreading):
    _useMultiThreading(useMultiThreading)
{
    if(UseMultiThreading())
        _gpGpuThread = new boost::thread(&CSimpleJobCpuGpu::threadCompute,this);
}

template< class T >
CSimpleJobCpuGpu<T>::~CSimpleJobCpuGpu()
{
    if(UseMultiThreading())
    {
        _gpGpuThread->interrupt();
        //_gpGpuThread->join();
        delete _gpGpuThread;
    }
    _mutexCompu.unlock();
    _mutexCopy.unlock();
    _mutexPreCompute.unlock();
}

template< class T >
void CSimpleJobCpuGpu<T>::SetCompute(T toBeComputed)
{
    boost::lock_guard<boost::mutex> guard(_mutexCompu);
    _compute = toBeComputed;
}

template< class T >
bool CSimpleJobCpuGpu<T>::GetCompute()
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
bool CSimpleJobCpuGpu<T>::GetDataToCopy()
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

