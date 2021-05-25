#ifndef _GPGPUSTREAMDATA_H_
#define _GPGPUSTREAMDATA_H_

#include "GpGpu/GpGpu_Data.h"

#define sgn s<sens>
#define max_cost 1e9

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
	/// \brief SetValue Affecte la valeur à l'index id
	/// \param id
	/// \param value La valeur
	///
	SetValue(int id, T value);

	void                __device__
	///
	/// \brief SubValue Soustrait la valeur à la donnée situé à l'index id
	/// \param id
	/// \param value
	///
	SubValue(int id, T value);

	void                __device__
	///
	/// \brief AddValue Additione la valeur à la donnée situé à l'index id
	/// \param id
	/// \param value
	///
	AddValue(int id, T value);

	template<bool sens> __device__ void
	///
	/// \brief SetOrAddValue Affecte si sens == true et additionne si sens == false
	/// \param id
	/// \param value
	///
	SetOrAddValue(int id, T value);

	template<bool sens> __device__
	///
	/// \brief SetOrAddValue Affecte si sens == true et additionne si sens == false
	/// \param id
	/// \param setValue
	/// \param addValue
	///
	void SetOrAddValue(int id, T setValue,T addValue);

    //long int                __device__  GetGiD(){return _idG;}

	template<bool sens> __device__
	///
	/// \brief reverse Prépare pour le sens de lecture
	///
	void reverse();

	void                __device__
	///
	/// \brief output Affiche dans la console les caractéristiques du stream
	///
	output();

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
/// \cond

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
/// \endcond
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
