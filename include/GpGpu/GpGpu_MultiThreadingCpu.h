#ifndef __GPGPU_MULTITHREADING_CPU_H__
#define __GPGPU_MULTITHREADING_CPU_H__

#include <stdio.h>

#include "GpGpu/GpGpu_Data.h"

#ifdef CPP11THREAD_NOBOOSTTHREAD
#define CPP11_THREAD
#endif

#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
        #include <chrono>
        #include <thread>
        #include <mutex>
    #endif
#else
#include <boost/thread/thread.hpp>
#include <boost/progress.hpp>
#include <boost/timer.hpp>
#endif


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

	///
	/// \brief UseMultiThreading
	/// \return La valeur de l'option sur l'utilisation du parallélisme CPU
	///
    bool            UseMultiThreading();

	///
	/// \brief GetIdBuf
	/// \return L'identifiant du buffer courrant
	///
    bool            GetIdBuf();
	///
	/// \brief SwitchIdBuffer
	/// Changer de buffer
    void            SwitchIdBuffer();
	///
	/// \brief ResetIdBuffer
	/// Réinitialise l'identifiant du buffer
    void            ResetIdBuffer();

	///
	/// \brief freezeCompute
	/// Stoppe le calcul
    virtual void    freezeCompute() = 0;


	///
	/// \brief SetProgress
	/// \param expected_count
	/// Définir la progression
    void            SetProgress(unsigned long expected_count);

	///
	/// \brief IncProgress Incrémenter la progression
	/// \param inc
	///
    void            IncProgress(uint inc = 1);

	///
	/// \brief simpleJob
	/// Lance le processus de gpu
    void            simpleJob();

private:


    void            simpleCompute();

    virtual void    simpleWork()    = 0;

    bool            _useMultiThreading;

#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
    std::mutex    _mutexCompu;
    std::mutex    _mutexCopy;
	std::mutex    _mutexPreCompute;
    #endif
#else
    boost::mutex    _mutexCompu;
    boost::mutex    _mutexCopy;
    boost::mutex    _mutexPreCompute;
#endif

    T               _compute;
    T               _copy;
    bool            _precompute;

    bool            _idBufferHostIn;
#ifndef CPP11_THREAD
    boost::progress_display *_show_progress;
#endif
    bool            _show_progress_console;

};

template< class T >
CSimpleJobCpuGpu<T>::CSimpleJobCpuGpu(bool useMultiThreading):
    _useMultiThreading(useMultiThreading),
    _idBufferHostIn(false),
    #ifndef CPP11_THREAD
    _show_progress(NULL),
    #endif
    _show_progress_console(false)
{}

template< class T >
CSimpleJobCpuGpu<T>::~CSimpleJobCpuGpu()
{
#ifndef CPP11_THREAD
    if(_show_progress)
        delete _show_progress;
#endif
}

template< class T >
void CSimpleJobCpuGpu<T>::SetCompute(T toBeComputed)
{
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
    std::lock_guard<std::mutex> guard(_mutexCompu);
    #endif
#else
    boost::lock_guard<boost::mutex> guard(_mutexCompu);
#endif

    _compute = toBeComputed;
}

template< class T >
T CSimpleJobCpuGpu<T>::GetCompute()
{
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
    std::lock_guard<std::mutex> guard(_mutexCompu);
    #endif
#else
    boost::lock_guard<boost::mutex> guard(_mutexCompu);
#endif
    return _compute;
}

template< class T >
void CSimpleJobCpuGpu<T>::SetDataToCopy(T toBeCopy)
{
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
    std::lock_guard<std::mutex> guard(_mutexCopy);
    #endif
#else
    boost::lock_guard<boost::mutex> guard(_mutexCopy);
#endif

    _copy = toBeCopy;

}

template< class T >
T CSimpleJobCpuGpu<T>::GetDataToCopy()
{
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
    std::lock_guard<std::mutex> guard(_mutexCopy);
    #endif
#else
    boost::lock_guard<boost::mutex> guard(_mutexCopy);
#endif
    return _copy;
}

template< class T >
void CSimpleJobCpuGpu<T>::SetPreComp(bool canBePreCompute)
{
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
    std::lock_guard<std::mutex> guard(_mutexPreCompute);
    #endif
#else
    boost::lock_guard<boost::mutex> guard(_mutexPreCompute);
#endif

    _precompute = canBePreCompute;
}

template< class T >
bool CSimpleJobCpuGpu<T>::GetPreComp()
{
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
    std::lock_guard<std::mutex> guard(_mutexPreCompute);
    #endif
#else
    boost::lock_guard<boost::mutex> guard(_mutexPreCompute);
#endif
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
#ifndef CPP11_THREAD
    if(_show_progress_console)
    {
        if(_show_progress == NULL)
            _show_progress = new boost::progress_display(expected_count);
        else
            _show_progress->restart(expected_count);
    }
#endif
}

template< class T >
void CSimpleJobCpuGpu<T>::IncProgress(uint inc)
{
#ifndef CPP11_THREAD
    if(_show_progress_console)
        (*_show_progress) += inc;
#endif
}

template< class T >
void CSimpleJobCpuGpu<T>::simpleCompute()
{
    while(!GetCompute())
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
        std::this_thread::sleep_for(std::chrono::microseconds(1));
#endif
     #else
        boost::this_thread::sleep(boost::posix_time::microsec(1));
#endif
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
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
        std::thread tOpti(&CSimpleJobCpuGpu<T>::simpleCompute,this);
        tOpti.detach();
    #endif
#else
        boost::thread tOpti(&CSimpleJobCpuGpu<T>::simpleCompute,this);
        tOpti.detach();
#endif


}


#endif //__GPGPU_MULTITHREADING_CPU_H__

