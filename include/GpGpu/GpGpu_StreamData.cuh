#ifndef _GPGPUSTREAMDATA_H_
#define _GPGPUSTREAMDATA_H_

#include "GpGpu/GpGpu_Data.h"

using namespace std;

static __constant__ ushort  dMapIndex[WARPSIZE];

/// @cond DEV
/// \class CDeviceStream
/// \brief Classe gerant un flux de données en memoire video
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


	template<bool sens>
	__device__ short getLen2Read(short2 &index)
	{
		index = make_short2(0,0);
		return 1;
	}

//    __device__ virtual short getLen2ReadAR(short2 &index)
//    {
//        index = make_short2(0,0);
//        return 1;
//    }

	template<bool sens, class D> __device__ short2 read(D* destData, ushort tid, T def);

private:

	template<bool sens> __device__ inline short vec();//{ return 1 - 2 * !sens; }

	template<bool sens> __device__ short GetNbToCopy(ushort nTotal,ushort nCopied)
	{
		return min(nTotal - nCopied , MaxReadBuffer<sens>());
	}

	template<bool sens> __device__ inline ushort MaxReadBuffer();




	T*                          _bufferData;
	T*                          _streamData;
	uint                        _curStreamId;
	ushort                      _curBufferId;
	uint                        _sizeStream;
};
/// @endcond

template<bool sens>
__device__ inline ushort __mxReadBuffer(ushort &curBufferId)
{
	return 0;
}

template<>
__device__ inline ushort __mxReadBuffer<true>(ushort &curBufferId)
{
	return ((ushort)WARPSIZE - curBufferId);
}

template<>
__device__ inline ushort __mxReadBuffer<false>(ushort &curBufferId)
{
	return curBufferId;
}

template< class T > template<bool sens>
__device__ inline ushort CDeviceStream<T>::MaxReadBuffer()
{
	return __mxReadBuffer<sens>(_curBufferId);
}

template<bool sens>
__device__ inline short __vec()
{
	return 0;
}

template<>
__device__ inline short __vec<true>()
{
	return 1;
}


template<>
__device__ inline short __vec<false>()
{
	return -1;
}

template< class T >
template <bool sens> __device__ inline short CDeviceStream<T>::vec()
{
	return __vec<sens>();
}



template< class T > template<bool sens, class D>  __device__
short2 CDeviceStream<T>::read(D *destData, ushort tid, T def)
{
	short2  index;
	ushort  NbCopied    = 0 , NbTotalToCopy = getLen2Read<sens>(index);// : getLen2ReadAR(index);
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


/// @cond DEV
template< class T >
///
/// \brief The CDeviceDataStream class
/// Cette classe est un lecteur de données GpGpu  de la mémoire.
/// Elle lit comme un flux de données.
/// Les données sont lues par paquet de 32. Elle permet d'optimiser les requètes dans la mémoire globale
class CDeviceDataStream : public CDeviceStream<T>
{
public:

	///
	/// \brief CDeviceDataStream Constructeur
	/// \param buf Buffer pour mettre en cache les données lues. Cette espace mémoire doit etre en mémoire partagée
	/// \param stream pointeur sur le début du flux de donnée en mémoire globale
	/// \param bufId
	/// \param streamId
	/// \param sizeStream
	/// \param sizeStreamId
	/// \return
	///
	__device__ CDeviceDataStream(T* buf,T* stream,short2* bufId,short2* streamId, uint sizeStream, uint sizeStreamId):
		CDeviceStream<T>(buf,stream,sizeStream),
		_streamIndex(bufId,streamId,sizeStreamId)
	{}

	template<bool sens>
	///
	/// \brief getLen2Read
	/// \param index
	/// \return
	///
	__device__ inline  short getLen2Read(short2 &index);

private:

	CDeviceStream<short2>       _streamIndex;

};

/// @endcond

#define sgn s<sens>
#define max_cost 1e9

template<bool sens>

__device__ inline short __getLen2Read(short2 &index,CDeviceStream<short2>   &streamIndex)
{
	streamIndex.read<sens>(&index,0,make_short2(0,0));

	return diffYX(index);
}

template<class T>
template<bool sens>
__device__ inline  short CDeviceDataStream<T>::getLen2Read(short2 &index)
{
	return __getLen2Read<sens>(index,_streamIndex);
}

template<bool sens>
__device__ inline int s(int v)
{
	return 0;
}

template<>
__device__ inline int s<true>(int v)
{
	return v;
}

template<>
__device__ inline int s<false>(int v)
{
	return -v ;
}

template<class T>
///
/// \brief The SimpleStream class
/// Strucure de donnée 1D avec lecture par flux
/// Permet une lecture optimisée des données en mémoire globale
/// Les Données sont mis en cache en mémoire partagée
class SimpleStream
{
public:

	///
	/// \brief SimpleStream
	/// \param globalStream Pointeur sur les données en mémoire globale
	/// \param sizeBuffer Dimension du buffer en mémoire partagée
	/// \return
	///
    __device__ SimpleStream(   T*      globalStream,ushort  sizeBuffer);


	template<bool sens> __device__
	///
	/// \brief read Lire les données et les mettre dans le buffer de la mémoire partagée
	/// \param sharedBuffer
	///
	void read(T* sharedBuffer);

	template<bool sens,class S> __device__
	///
	/// \brief readFrom Lire les données et les mettre dans le buffer de la mémoire partagée avec un decalage delta
	/// \param sharedBuffer
	/// \param delta Décalage
	///
	void readFrom(S* sharedBuffer,uint delta = 0);

	template<bool sens> __device__
	///
	/// \brief incre Incrémente le pointeur de la mémoire globale de la taille du buffer
	///
	void incre();

	template<bool sens> __device__
	///
	/// \brief ReverseIncre Décrémente le pointeur de la mémoire globale de la taille du buffer
	///
	void ReverseIncre();
	T                   __device__
	///
	/// \brief GetValue Obtenir la valeur à l'index id
	/// \param id
	/// \return Retourne la valeur
	GetValue(int id);

	void                __device__
	///
	/// \brief SetValue
	/// \param id
	/// \param value
	///
	SetValue(int id, T value);

    void                __device__  SubValue(int id, T value);

    void                __device__  AddValue(int id, T value);

    template<bool sens> __device__ void SetOrAddValue(int id, T value);

    template<bool sens> __device__ void SetOrAddValue(int id, T setValue,T addValue);

    //long int                __device__  GetGiD(){return _idG;}

    template<bool sens> __device__  void reverse();

    void                __device__  output();

private:

    template<class S>
    void   __device__  copyThread(T* p1, S* p2);

    T*      _globalStream;
    //long int    _idG;
    ushort  _sizeBuffer;
};

template<class T> __device__
SimpleStream<T>::SimpleStream( T *globalStream, ushort sizeBuffer):
    _globalStream(globalStream + threadIdx.x),
    //_idG(0),
    _sizeBuffer(sizeBuffer)
{}

template<class T> __device__
T SimpleStream<T>::GetValue(int id)
{
    return _globalStream[id];
}

template<class T> __device__
void SimpleStream<T>::SetValue(int id, T value)
{
    _globalStream[id] = value;
}

template<class T> __device__
void SimpleStream<T>::SubValue(int id, T value)
{
    _globalStream[id] -= value;
}

template<class T> __device__
void SimpleStream<T>::AddValue(int id, T value)
{
    _globalStream[id] += value;
}

template<class T> template<bool sens> __device__
void SimpleStream<T>::SetOrAddValue(int id, T value)
{
    if(sens)
        SetValue(id,value);
    else
        AddValue(id,value);
}



template<class T> template<bool sens> __device__
void SimpleStream<T>::SetOrAddValue(int id, T setValue, T addValue)
{
    if(sens)
        SetValue(id,setValue);
    else
        AddValue(id,addValue);
}


template<class T> __device__
void SimpleStream<T>::output()
{
    if(!threadIdx.x)
    {
        printf("----------------------------------\n");
        printf("_sizeBuffer = %d\n",_sizeBuffer);
        //printf("_idG        = %d\n",_idG);
        printf("----------------------------------\n");
    }
}

template<class T> template<bool sens> __device__
void SimpleStream<T>::reverse()
{
    _globalStream += sgn(WARPSIZE);
}

template<class T> template<bool sens> __device__
void SimpleStream<T>::read(T *sharedBuffer)
{        
    readFrom<sens>(sharedBuffer);
    incre<sens>();
}

template<class T> template<bool sens,class S> __device__
void SimpleStream<T>::readFrom(S *sharedBuffer,uint delta)
{
    T* gLocal = _globalStream + sgn(delta);

    for(ushort i = 0; i < _sizeBuffer; i += WARPSIZE)
        *(sharedBuffer + sgn(i)) = *(gLocal+sgn(i));
}


template<class T>
template<class S> __device__
void SimpleStream<T>::copyThread(T* p1,S* p2)
{
    *p2 = *p1;
}

template<>
template<> __device__ inline
void SimpleStream<ushort2>::copyThread(ushort2* p1,uint* p2)
{
    *p2 = (*p1).x;
}

template<>
template<bool sens,class S> __device__
void SimpleStream<ushort2>::readFrom(S *sharedBuffer,uint delta)
{
    T* gLocal = _globalStream + sgn(delta);

    for(ushort i = 0; i < _sizeBuffer; i += WARPSIZE)

           copyThread(gLocal+sgn(i),sharedBuffer + sgn(i));


}

template<class T> template<bool sens> __device__
void SimpleStream<T>::incre()
{
    _globalStream += sgn(_sizeBuffer);
}

template<class T> template<bool sens> __device__
void SimpleStream<T>::ReverseIncre()
{
    reverse<sens>();
    incre<sens>();
}

#endif
