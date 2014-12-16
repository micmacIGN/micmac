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

    void            SetProgress(unsigned long expected_count);

    void            IncProgress(uint inc = 1);

    void            simpleJob();

private:


    void            simpleCompute();

    virtual void    simpleWork()    = 0;

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
    _useMultiThreading(useMultiThreading),
    _idBufferHostIn(false),
    _show_progress(NULL),
    _show_progress_console(false)
{}

template< class T >
CSimpleJobCpuGpu<T>::~CSimpleJobCpuGpu()
{
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
void CSimpleJobCpuGpu<T>::simpleCompute()
{
    while(!GetCompute())
        boost::this_thread::sleep(boost::posix_time::microsec(1));

    SetCompute(false);

    simpleWork();

    while(GetDataToCopy());
//        boost::this_thread::sleep(boost::posix_time::microsec(5));

    SwitchIdBuffer();
    SetDataToCopy(true);
    SetCompute(true);

}

template< class T >
void CSimpleJobCpuGpu<T>::simpleJob()
{
    boost::thread tOpti(&CSimpleJobCpuGpu<T>::simpleCompute,this);
    tOpti.detach();
}


#endif //__GPGPU_MULTITHREADING_CPU_H__

