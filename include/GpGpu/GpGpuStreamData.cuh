#ifndef _GPGPUSTREAMDATA_H_
#define _GPGPUSTREAMDATA_H_

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include "GpGpu/GpGpuTools.h"
#include "GpGpu/helper_math_extented.cuh"

using namespace std;

/// \brief Tableau des penalites pre-calculees
#define PENALITE 7
#define WARPSIZE 32
#define NAPPEMAX 256

#define eAVANT      true
#define eARRIERE    false

static __constant__ float   penalite[PENALITE];
static __constant__ ushort  dMapIndex[WARPSIZE];


template< class T >
class CDeviceStream
{
public:

    __device__ CDeviceStream(T* buf,T* stream):
        _bufferData(buf),
        _streamData(stream),
        _curStreamId(0),
        _curBufferId(WARPSIZE)
    {}

    __device__ virtual short getLengthToRead(short2 &index,bool sens)
    {
        index = make_short2(0,0);
        return 1;
    }

    __device__ short2 read(T* destData, ushort tid, bool sens, T def, bool waitSync = true);

private:

    __device__ short vec(bool sens){ return 1 - 2 * !sens; }

    __device__ short GetNbToCopy(ushort nTotal,ushort nCopied, bool sens)
    {
        return min(nTotal - nCopied , MaxReadBuffer(sens));
    }

    __device__ ushort MaxReadBuffer(bool sens)
    {
        return sens ? ((ushort)WARPSIZE - _curBufferId) : _curBufferId;
    }

    T*                          _bufferData;
    T*                          _streamData;
    uint                        _curStreamId;
    ushort                      _curBufferId;
};

template< class T > __device__
short2 CDeviceStream<T>::read(T *destData, ushort tid, bool sens, T def, bool waitSync)
{
    short2  index;
    ushort  NbCopied    = 0 , NbTotalToCopy = getLengthToRead(index, sens);
    short   PitSens     = !sens * WARPSIZE;

    //bool AA = (blockIdx.x==0 && threadIdx.x == 1  && NbTotalToCopy  != 1 && !sens);

    while(NbCopied < NbTotalToCopy)
    {
        ushort NbToCopy = GetNbToCopy(NbTotalToCopy,NbCopied, sens);

        if(!NbToCopy)
        {
            _bufferData[threadIdx.x] = _streamData[_curStreamId + threadIdx.x - 2 * PitSens];
            _curBufferId   = PitSens;
            _curStreamId   = _curStreamId  + vec(sens) * WARPSIZE;

            NbToCopy = GetNbToCopy(NbTotalToCopy,NbCopied, sens);
            __syncthreads();
        }

        ushort idDest = tid + (sens ? NbCopied : NbTotalToCopy - NbCopied - NbToCopy);

        if(idDest < NAPPEMAX)
        {
            if (tid < NbToCopy && idDest < NbTotalToCopy)
                destData[idDest] = _bufferData[_curBufferId + tid - !sens * NbToCopy];
            else if (idDest >= NbTotalToCopy)
                destData[idDest] = def;
        }
        if(waitSync) __syncthreads();

        _curBufferId  = _curBufferId + vec(sens) * NbToCopy;
        NbCopied     += NbToCopy;

   }
   return index;
}

template< class T >
class CDeviceDataStream : public CDeviceStream<T>
{
public:

    __device__ CDeviceDataStream(T* buf,T* stream,short2* bufId,short2* streamId):
        CDeviceStream<T>(buf,stream),
        _streamIndex(bufId,streamId)
    {}

    __device__ short getLengthToRead(short2 &index, bool sens)
    {
        _streamIndex.read(&index,0,sens,make_short2(0,0),false);
        const short leng = diffYX(index) + 1;
        return leng;
    }

private:

    CDeviceStream<short2>       _streamIndex;

};


#endif
