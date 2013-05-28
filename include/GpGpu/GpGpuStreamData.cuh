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

#define NAPPEMAX 256

#define eAVANT      true
#define eARRIERE    false

static __constant__ float   penalite[PENALITE];
static __constant__ ushort  dMapIndex[WARPSIZE];

/// \class CDeviceStream
/// \brief Classe gerant un fluc de données en memoire video
template< class T >
class CDeviceStream
{
public:

    __device__ CDeviceStream(T* buf,T* stream, uint sizeStream):
        _bufferData(buf),
        _streamData(stream),
        _curStreamId(0),
        _curBufferId(WARPSIZE),
        _sizeStream(sizeStream)
    {}

    __device__ virtual short getLengthToRead(short2 &index, bool sens)
    {
        index = make_short2(0,0);
        return 1;
    }

    template<bool sens> __device__ short2 read(T* destData, ushort tid, T def);

private:

    template<bool sens> __device__ short vec(){ return 1 - 2 * !sens; }

    template<bool sens> __device__ short GetNbToCopy(ushort nTotal,ushort nCopied)
    {
        return min(nTotal - nCopied , MaxReadBuffer<sens>());
    }

    template<bool sens> __device__ ushort MaxReadBuffer()
    {
        return sens ? ((ushort)WARPSIZE - _curBufferId) : _curBufferId;
    }

    T*                          _bufferData;
    T*                          _streamData;
    uint                        _curStreamId;
    ushort                      _curBufferId;
    uint                        _sizeStream;
};

template< class T > template<bool sens>  __device__
short2 CDeviceStream<T>::read(T *destData, ushort tid, T def)
{
    short2  index;
    ushort  NbCopied    = 0 , NbTotalToCopy = getLengthToRead(index,sens);
    short   PitSens     = !sens * WARPSIZE;

    while(NbCopied < NbTotalToCopy)
    {
        ushort NbToCopy = GetNbToCopy<sens>(NbTotalToCopy,NbCopied);

        if(!NbToCopy)
        {
            uint idStream =_curStreamId + threadIdx.x - 2 * PitSens;
            if(idStream < _sizeStream)
                _bufferData[threadIdx.x] = _streamData[idStream];
            _curBufferId   = PitSens;            
            _curStreamId   = _curStreamId  + (vec<sens>()<<5); // * 32

            NbToCopy = GetNbToCopy<sens>(NbTotalToCopy,NbCopied);
        }

        ushort idDest = tid + (sens ? NbCopied : NbTotalToCopy - NbCopied - NbToCopy);

        if(!(idDest>>8)) // < 256
        {
            if (tid < NbToCopy && idDest < NbTotalToCopy)
                destData[idDest] = _bufferData[_curBufferId + tid - !sens * NbToCopy];
            else if (idDest >= NbTotalToCopy)
                destData[idDest] = def;
        }
        _curBufferId  = _curBufferId + vec<sens>() * NbToCopy;
        NbCopied     += NbToCopy;

   }
   return index;
}

template< class T >
class CDeviceDataStream : public CDeviceStream<T>
{
public:

    __device__ CDeviceDataStream(T* buf,T* stream,short2* bufId,short2* streamId, uint sizeStream, uint sizeStreamId):
        CDeviceStream<T>(buf,stream,sizeStream),
        _streamIndex(bufId,streamId,sizeStreamId)
    {}

    __device__ short getLengthToRead(short2 &index, bool sens)
    {
        if(sens == eAVANT)
            _streamIndex.read<eAVANT>(&index,0,make_short2(0,0));
        else
            _streamIndex.read<eARRIERE>(&index,0,make_short2(0,0));

        return diffYX(index) + 1;
    }

private:

    CDeviceStream<short2>       _streamIndex;

};


#endif
