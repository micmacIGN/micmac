#ifndef GPGPU_DATA_H
#define GPGPU_DATA_H

#include "GpGpu/GpGpu_Object.h"
#include "GpGpu/GpGpu_Context.h"
#include "GpGpu/GpGpu_Tools.h"
#include <xmmintrin.h>

#include <stdio.h>
#include <stdarg.h>

/** @addtogroup GpGpuDoc */
/*@{*/


/// \cond
#define TPL_T template<class T>
/// \endcond

/// \class CData
/// \brief Classe Abstraite de donnees
template<class T>
class CData : public CGObject
{

    friend class    DecoratorImageCuda;
    template<class M,class context> friend class    DecoratorDeviceData;

public:

    CData();
    ~CData()        { Dealloc();}

    /// \brief      Allocation memoire
    bool            Malloc();

    /// \brief      Initialise toutes les elements avec la valeur val
    /// \param      val : valeur d initialisation
    virtual bool	Memset(int val) = 0;

    /// \brief      Desalloue la memoire alloue
    bool            Dealloc();

    /// \brief      Sortie console de la classe
    virtual void	OutputInfo()	= 0;

    /// \brief      Renvoie le pointeur des donnees
    T*              pData(){ return _data; }

    /// \brief      Sortie console de l allocation memoire globale Gpu

    void            MallocInfo();

    /// \brief      Obtenir une valeur aleatoire comprise entre min et max
    static T        GetRandomValue(T min, T max);

#if OPENCL_ENABLED
    cl_mem          clMem() const{return _clMem;}
#endif
protected:

    /// \brief      Renvoie la taille de la memoire alloue
    uint            GetSizeofMalloc(){ return _sizeofMalloc; }

    /// \brief      Initialise la taille de la memoire alloue
    /// \brief      Renvoie le pointeur du pointeur des donnees
    T**             ppData(){ return &_data; }

    /// \brief      Init le pointeur des donnees
    void            SetPData(T *p){ _data = p;}

	///
	/// \brief abDealloc Désallocation de la mémoire
	/// \return true si la désallocation est reussie
	///
    virtual bool    abDealloc(){ return false;} // TODO pour le rendre completement virtuelle il faut reimplementer les destructeurs...

	///
	/// \brief abMalloc Allocation de la mémoire
	/// \return
	///
    virtual bool    abMalloc(){ return false;}

	///
	/// \brief Sizeof retourn le taille de la structure en mémoire
	/// \return
	///
    virtual uint    Sizeof(){return 0;}

    /// \brief      Sortie console des erreurs Cuda
    /// \param      err :  erreur cuda rencontree
    /// \param      fonctionName : nom de la fonction ou se trouve l erreur
    virtual bool	ErrorOutput(cudaError_t err,const char* fonctionName);


#if OPENCL_ENABLED
    void            setClMem(const cl_mem &clMem){_clMem = clMem;}
#endif

private:

    uint            _memoryOc;

    T*              _data;
#if OPENCL_ENABLED
    cl_mem          _clMem;
#endif
    uint            _sizeofMalloc;

    /// \brief      Suppression de memoire alloue
    void            SubMemoryOc(uint m) { _memoryOc -= m; }

    /// \brief      Ajout de memoire alloue
    void            AddMemoryOc(uint m) { _memoryOc += m; }

    /// \brief      Initialise a NULL le pointeur des donnees
    void            dataNULL(){ _data = NULL;}

    /// \brief      Renvoie True si le pointeur des donnees est NULL
    bool            isNULL(){return (_data == NULL);}

    /// \param      sizeofmalloc : Taille de l allocation
	uint            SetSizeofMalloc(uint sizeofmalloc){ return _sizeofMalloc = sizeofmalloc; }

};

TPL_T void CData<T>::MallocInfo()
{
    std::cout << "Malloc Info " << CGObject::Name() << " Size of Malloc | Memory Oc en Mo     : "<<  _sizeofMalloc / pow(2.0,20) << " | " << _memoryOc / pow(2.0,20) << "\n";
    std::cout << "Malloc Info " << CGObject::Name() << " Size of Malloc | Memory Oc en octets : "<<  _sizeofMalloc << " | " << _memoryOc  << "\n";
}

TPL_T bool CData<T>::ErrorOutput( cudaError_t err,const char* fonctionName )
{
    if (!CGpGpuContext<cudaContext>::errorDump(err,fonctionName))
    {
        std::cout << "Object " + CGObject::Id() << "\n";
        OutputInfo();
        std::cout << "Pointeur de donnees : " << CData<T>::pData()  << "\n";
        std::cout << "Memoire allouee     : " << _memoryOc / pow(2.0,20) << " Mo | " << _memoryOc / pow(2.0,10) << " ko | " << _memoryOc  << " octets \n";
        std::cout << "Taille des donnees  : " << CData<T>::GetSizeofMalloc()  / pow(2.0,20) << " Mo | " << CData<T>::GetSizeofMalloc()  / pow(2.0,10) << " ko | " << CData<T>::GetSizeofMalloc() << " octets \n";
        CGpGpuContext<cudaContext>::OutputInfoGpuMemory();
        std::cout << "--------------------------------------------------------------------------------------\n";
        exit(1);
        return false;
    }
    return true;
}



TPL_T CData<T>::CData():
    _memoryOc(0),
    _data(NULL),
#ifdef OPENCL_ENABLED
    _clMem(NULL),
#endif
    _sizeofMalloc(0)
{
#ifdef      NOCUDA_X11	
	CGObject::ClassTemplate(AutoStringClass(_data));
#else
	CGObject::ClassTemplate(CGObject::StringClass<T>(pData()));
#endif
}

TPL_T T CData<T>::GetRandomValue(T min, T max)
{
    T mod = abs(max - min);
    int rdVal  = rand()%((int)mod);
    double dRdVal = (float)rand() / std::numeric_limits<int>::max();
    return min + rdVal + (T)dRdVal;
}

TPL_T bool CData<T>::Malloc()
{
    AddMemoryOc(SetSizeofMalloc(Sizeof()));
    return abMalloc();
}

TPL_T bool CData<T>::Dealloc()
{
    bool op = false;
    SubMemoryOc(GetSizeofMalloc());
    SetSizeofMalloc(0);
#ifdef OPENCL_ENABLED
    if (!isNULL() || _clMem!=NULL) op = abDealloc();
    _clMem = NULL;
#else
    if (!isNULL()) op = abDealloc();

#endif
    dataNULL();
    return op;
}


/// \cond
template<bool var>
uint  ttGetDimension(uint* dimension,ushort idDim)
{
    return dimension[idDim];
}

template<> inline
uint  ttGetDimension<false>(uint* dimension,ushort idDim)
{
    return 1;
}

template<uint dim,ushort idDim>
uint  tgetDimension(uint* dimension)
{
    return ttGetDimension<(idDim<dim)> (dimension,idDim);
}

template<bool var>
void  ttSetDimension(uint* dimension,ushort idDim, uint val)
{
    dimension[idDim] = val;
}

template<> inline
void  ttSetDimension<false>(uint* dimension,ushort idDim, uint val)
{

    DUMP("WARNING Set dimension --> ")
            DUMP(idDim)

}

template<uint dim,ushort idDim>
void  tSetDimension(uint* dimension, uint val)
{
    ttSetDimension<(idDim<dim)> (dimension,idDim, val);
}

int inline _foo(size_t n, int xs[])
{
    int i;
    for(i=0 ; i < (int)n ; i++ ) {
        int x = xs[i];
        printf("%d\n", x);
    }
    return (int)n;
}

#define foo(arg1, ...) ({              \
   int _x[] = { arg1, __VA_ARGS__ };   \
   _foo(sizeof(_x)/sizeof(_x[0]), _x); \
})

template<ushort dim = 3>
class CStructure
{
public:

    CStructure()
    {
        DUMP("Constructeur\n")
		_dimension = new uint[dim];

        for (int i = 0; i < dim; ++i)
            setDim(i,1);
    }

#ifdef NOCUDA_X11
    template<typename ... Types>
    void setDimension ( Types ... rest)
    {
        _setDimension(rest...);
    }
#else

    template<class T>
    void setDimension(T x = 1 ,T y = 1)
    {
        _setDimension((uint)x,(uint)y);
    }


    template<class T>
    void setDimension(T x,T y,T z)
    {
        _setDimension((uint)x,(uint)y,(uint)z);
    }

#endif

    void setDimension(uint2 d)
    {
        setDimension(d.x,d.y);
    }

    void setDimension(uint3 d)
    {
        setDimension(d.x,d.y,d.z);
    }

    uint2 getDimension()
    {        
        return make_uint2(getDimX(),getDimY());
    }

    uint getNbLayer()
    {
        return getDimZ();
    }

    uint getSize()
    {
        int size = _dimension[0];

        for (int id = 1; id < dim; ++id)
        {
            size *= _dimension[id];
        }

        return size;
    }

private:

#ifdef NOCUDA_X11

    template<typename ... Types>
    void _setDimension ( Types ... rest)
    {
        return ___setDimension((uint)0,rest...);
    }

    void ___setDimension(ushort id){}

    template<typename T,typename ... Types>
    void ___setDimension(ushort id, T &first, Types& ... rest)
    {
        if(id<dim)
            setDim(id,first);
        else
            return;

        return ___setDimension(++id,rest...);
    }

#else
    void _setDimension(uint dx,uint dy = 1,uint dz = 1)
    {
        setDimX(dx);
        setDimY(dy);
        setDimZ(dz);
    }
#endif

    uint  getDimX(){ return getDim<0>();}
    uint  getDimY(){ return getDim<1>();}
    uint  getDimZ(){ return getDim<2>();}

    void  setDimX(uint val){ return setDim<0>(val);}
    void  setDimY(uint val){ return setDim<1>(val);}
    void  setDimZ(uint val){ return setDim<2>(val);}

    template<ushort id>
    uint  getDim(){ return tgetDimension<dim,id> (_dimension);}

    template<ushort id>
    void  setDim(uint val){ return tSetDimension<dim,id> (_dimension,val);}


    void  setDim(ushort id, uint val){ _dimension[id]=val;}



    uint *_dimension;
};

template<> inline
uint CStructure<0>::getSize()
{
    return 0;
}

template <class T, int dim = 3>
///
/// \brief The CStructuredData class
/// meta programation element
class CStructuredData : public CData<T>, public CStructure<dim>
{

};

template <class T, int dim = 3, class structuringClass = struct2DLayered>
class deviceStructuredData : public CStructuredData<T,dim>
{

public:

    bool	Memset(int val)
    {
        DUMP("Memset device")
                return false;
    }

    void	OutputInfo(){}

protected:

    bool    abDealloc()
    {
        DUMP("abDealloc device")
                return false;
    }

    bool    abMalloc()
    {
        DUMP("abMalloc device")

                return false;
    }

    uint    Sizeof(){return 0;}
};

/// \endcond


/// \class CData2D
/// \brief Classe abstraite d un tableau d elements structuree en deux dimensions
template <class T>
class CData2D : public struct2D, virtual public CData<T>
{

public:

    CData2D(){}
    /// \brief      constructeur avec initialisation de la dimension de la structure
    /// \param      dim : Dimension a initialiser
    CData2D(uint2 dim) { Malloc(dim); }

    /// \brief      Initialise les elements des images a val
    /// \param      val : Valeur d initialisation
    virtual bool	Memset(int val) = 0;

    void			OutputInfo();

    /// \brief       Allocation memoire pour les tous les elements de la structures avec initialisation de la dimension de la structure
    /// \param      dim : Dimension 2D a initialiser
    bool			Malloc(uint2 dim);
    /// \brief      Desallocation puis re-allocation memoire pour les tous les elements
    ///             de la structures avec initialisation de la dimension
    /// \param      dim : Dimension 2D a initialiser
    bool			Realloc(uint2 dim);

	///
	/// \brief ReallocIfDim Reallocation mémoire si le dimension est supèrieure à la dimension actuelle
	/// \param dim Taille de l'allocation en 2 dimensions
	/// \return
	///
    bool            ReallocIfDim(uint2 dim);

protected:

    virtual bool    abDealloc() = 0;

    virtual bool    abMalloc()  = 0;

    uint            Sizeof(){return sizeof(T) * struct2D::GetSize();}

};

TPL_T void CData2D<T>::OutputInfo()
{

    struct2D::Output();
}


/// Specialisation pour cudaArray la taille memoire
template <> inline
uint CData2D<cudaArray>::Sizeof()
{
    return struct2D::GetSize();
}

TPL_T bool CData2D<T>::Realloc( uint2 dim )
{
    CData<T>::Dealloc();
    Malloc(dim);
    return true;
}

TPL_T bool CData2D<T>::ReallocIfDim(uint2 dim)
{
    if(oI(struct2D::GetMaxDimension(),dim))
        return CData2D<T>::Realloc(dim);
    else
        CData2D<T>::SetDimension(dim);

    return true;
}

TPL_T bool CData2D<T>::Malloc( uint2 dim )
{
    SetDimension(dim);
    return CData<T>::Malloc();
}

/// \class CData3D
/// \brief Classe abstraite d un tableau d elements structuree en trois dimensions
template <class T>
class CData3D : public struct2DLayered, public CData<T>
{
public:

    CData3D(){}

    /// \brief constructeur avec initialisation de la dimension de la structure
    /// \param dim : Dimension 2D a initialiser
    /// \param l : Taille de la 3eme dimension
    CData3D(uint2 dim, uint l){ Malloc(dim,l); }

    /// \brief      Initialise toutes les elements avec la valeur val
    /// \param      val : valeur d initialisation
    virtual bool	Memset(int val) = 0;

    void			OutputInfo();

    /// \brief      Allocation memoire pour les tous les elements de la structures avec initialisation de la dimension de la structure
    /// \param      dim : Dimension 2D a initialiser
    /// \param      l : Taille de la 3eme dimension
    bool			Malloc(uint2 dim, uint l);

    /// \brief      Desallocation puis re-allocation memoire pour les tous les elements de la structures avec initialisation de la dimension de la structure
    /// \param      dim : Dimension 2D a initialiser
    /// \param      l : Taille de la 3eme dimension
    bool			Realloc(uint2 dim, uint l);

	///
	/// \brief Realloc reallocation de taille size
	/// \param size Taille de l'allocation
	/// \return
	///
    bool			Realloc(uint size){return Realloc(make_uint2(size,1),1);}

	///
	/// \brief Realloc Reallocation de la mémoire de taille dim
	/// \param dim Taille de l'allocation de dimension 3
	/// \return
	///
    bool			Realloc(uint3 dim){return Realloc(make_uint2(dim.x,dim.y),dim.z);}

	///
	/// \brief ReallocIf Reallocation si la nouvelle taille est supèrieure à l'actuelle
	/// \param dim1D Taille de l'allocation
	/// \return
	///
    bool			ReallocIf(uint dim1D);

	///
	/// \brief ReallocIf Reallocation si la nouvelle taille est supèrieure à l'actuelle
	/// \param dim taille de dimension 2
	/// \param l taille sur la dimension Z
	/// \return
	///
    bool			ReallocIf(uint2 dim, uint l = 1);

	///
	/// \brief ReallocIf Reallocation si la nouvelle taille est supèrieure à l'actuelle
	/// \param dimX Taille sur X
	/// \param dimY Taille sur Y
	/// \param l Taille sur Z
	/// \return
	///
    bool			ReallocIf(uint dimX, uint dimY, uint l = 1);

	///
	/// \brief ReallocIfDim Reallocation si la nouvelle taille est supèrieure à l'actuelle
	/// \param dim taille de dimension 2
	/// \param l taille sur la dimension Z
	/// \return
	///
    bool            ReallocIfDim(uint2 dim,uint l);

	///
	/// \brief operator []
	/// \param pt coordonnées de la données requeter
	/// \return retourne la valeur
	///
    T&              operator[](uint2 pt);

	///
	/// \brief operator []
	/// \param pt
	/// \return
	///
    T&              operator[](uint3 pt);

	///
	/// \brief operator []
	/// \param pt1D
	/// \return  La valeur en position pt1D
	///
    T&              operator[](uint pt1D)   {   return (CData<T>::pData())[pt1D];       }

	///
	/// \brief operator []
	/// \param pt1D
	/// \return La valeur en position pt1D
	///
    T&              operator[](int pt1D)    {   return (CData<T>::pData())[(uint)pt1D]; }

protected:

	/// \cond
    virtual bool    abMalloc()  = 0;

    virtual bool    abDealloc() = 0;

    /// \brief      Nombre d elements de la structure
    uint			Sizeof(){return GetSize() * sizeof(T);}

    void            bInit(uint2 dim = make_uint2(0), uint l = 0);
	/// \endcond
};

TPL_T void CData3D<T>::OutputInfo()
{
    std::cout << "Structure 3D        : " << CGObject::Id() << "\n";
    struct2DLayered::Output();
}

TPL_T bool CData3D<T>::Malloc( uint2 dim, uint l )
{
    SetDimension(dim,l);
    return CData<T>::Malloc();
}

TPL_T bool CData3D<T>::Realloc( uint2 dim, uint l )
{
    bool dB = CData<T>::Dealloc();
    bool dM = Malloc(dim,l);
    return (dB && dM);
}

TPL_T bool CData3D<T>::ReallocIf(uint dim1D)
{
    uint2 dim2D = make_uint2(dim1D,1);
    return ReallocIf(dim2D,1);
}

TPL_T inline bool CData3D<T>::ReallocIfDim(uint2 dim,uint l)
{

//    DUMP_UINT2(dim)
//    DUMP_UINT2(struct2DLayered::GetMaxDimension())
//            DUMP_UINT(l)
//            DUMP_UINT(struct2DLayered::GetNbLayer())
    if( oI(struct2DLayered::GetMaxDimension(),dim) || l > struct2DLayered::GetNbLayer())
    {
//        printf("REALLOC :\n");

        return CData3D<T>::Realloc(dim,l);
    }
    else
    {
//        printf("SET DIMENSION :\n");
        CData3D<T>::SetDimension(dim,l);
    }
    return true;
}

TPL_T inline bool CData3D<T>::ReallocIf(uint2 dim, uint l)
{
    //if( size(dim) * l * sizeof(T) > CData3D<T>::GetSizeofMalloc())

//    printf("REALLOC IF ========================================= BEGIN\n");
//    DUMP_UINT2(dim)
//    DUMP_UINT(l)
//    DUMP_UINT(struct2DLayered::GetMaxSize())
//    OutputInfo();
    if( size(dim) * l > struct2DLayered::GetMaxSize())//|| l > struct2DLayered::GetNbLayer())
    {
        //printf("realloc\n");
        return CData3D<T>::Realloc(dim,l);
    }
    else
    {
        //printf("SetDimension\n");
        CData3D<T>::SetDimension(dim,l);
    }

    return true;
}

TPL_T inline bool CData3D<T>::ReallocIf(uint dimX, uint dimY, uint l)
{
    uint2 dim2D = make_uint2(dimX,dimY);
    return ReallocIf(dim2D,l);
}

template <> inline
uint CData3D<cudaArray>::Sizeof(){   return GetSize();  }

TPL_T T &CData3D<T>::operator [](uint2 pt)
{
    return (CData<T>::pData())[to1D(pt,GetDimension())];
}

TPL_T T &CData3D<T>::operator [](uint3 pt)
{
    return (CData<T>::pData())[pt.z * struct2D::GetSize() + to1D(make_uint2(pt.x,pt.y),GetDimension())];
}

/// \cond
TPL_T void CData3D<T>::bInit(uint2 dim, uint l)
{
    if(size(dim) && l) Malloc(dim,l);
}
/// \endcond

/// \class CuHostData3D
/// \brief Tableau 3D d elements contenue la memoire du Host.
/// La gestion memoire est realise par l API Cuda.
template <class T>
class CuHostData3D : public CData3D<T>
{
public:

	///
	/// \brief CuHostData3D Constructeur
	/// \param pgLockMem Option de mémoire paginée
	/// \param alignMemory Option de mémoire alignée
	///
	CuHostData3D(bool pgLockMem = NOPAGLOCKMEM,bool alignMemory = NOALIGNM128){init(pgLockMem,alignMemory);}

	/// \brief CuHostData3D
	/// \param dimX Dimension 1D a initialiser
	/// \param dimY Dimension 1D a initialiser
	/// \param l Taille de la 3eme dimension
	/// \param pgLockMem Option de mémoire paginée
	/// \param alignMemory Option de mémoire alignée
	///
	CuHostData3D(uint dimX, uint dimY = 1, uint l = 1, bool pgLockMem = NOPAGLOCKMEM,bool alignMemory = NOALIGNM128){init(pgLockMem,alignMemory,make_uint2(dimX,dimY),l);}

	///
	/// \brief CuHostData3D
	/// \param dim Dimension 2D a initialiser
	/// \param l
	/// \param pgLockMem
	/// \param alignMemory
	///
	CuHostData3D(uint2 dim, uint l = 1, bool pgLockMem = NOPAGLOCKMEM,bool alignMemory = NOALIGNM128){ init(pgLockMem,alignMemory,dim,l);}


	/// \brief CuHostData3D
	/// \param dim
	/// \param pgLockMem
	/// \param alignMemory
	///
	CuHostData3D(uint3 dim,bool pgLockMem = NOPAGLOCKMEM ,bool alignMemory = NOALIGNM128 ){init(pgLockMem,alignMemory,make_uint2(dim.x,dim.y),dim.z);}

    bool Memset(int val);

    /// \brief Remplie le tableau avec la valeur Value
    /// \param Value : valeur a remplir
    void Fill(T Value);

    /// \brief Remplie le tableau avec la valeur aleatoire pour chaque element
    /// \param min : valeur a remplir minimum
    /// \param max : valeur a remplir maximum
    void FillRandom(T min, T max);

    /// \brief Affiche un Z du tableau dans la console
    void OutputValues(uint level = 0, uint plan = XY, Rect rect = NEGARECT, uint offset = 3, T defaut = GpGpuTools::SetValue<T>(), float sample = 1.0f, float factor = 1.0f);

	///
	/// \brief pLData
	/// \param layer
	/// \return  le pointeur du calque pointée
	///
    T*   pLData(uint layer){ return CData<T>::pData() + layer*size(CData3D<T>::GetDimension());}

	///
	/// \brief saveImage Sauvegaerder l'image l sur le disque
	/// \param nameImage Nom de la sauvergarde
	/// \param layer Identifiant de calque
	/// \return
	///
    bool saveImage(string nameImage,ushort layer = 0);

	///
	/// \brief pgLockMem
	/// \return Option de mémoire paginée
	///
	bool pgLockMem() const;
	///
	/// \brief setPgLockMem
	/// \param pgLockMem
	///
	void setPgLockMem(bool pgLockMem);

	///
	/// \brief alignMemory
	/// \return Option de mémoire alignée
	///
	bool alignMemory() const;

	///
	/// \brief setAlignMemory
	/// \param alignMemory
	///
	void setAlignMemory(bool alignMemory);

protected:

	virtual bool    abDealloc() ;

	virtual bool    abMalloc();

private:

    bool    _pgLockMem;
	bool	_alignMemory;

	void    init(bool pgLockMem = NOPAGLOCKMEM,bool alignMemory = NOALIGNM128, uint2 dim = make_uint2(0), uint l = 0);

};

TPL_T  bool CuHostData3D<T>::pgLockMem() const
{
	return _pgLockMem;
}

TPL_T void CuHostData3D<T>::setPgLockMem(bool pgLockMem)
{
	_pgLockMem = pgLockMem;
}
TPL_T bool CuHostData3D<T>::alignMemory() const
{
	return _alignMemory;
}

TPL_T void CuHostData3D<T>::setAlignMemory(bool alignMemory)
{
	_alignMemory = alignMemory;
}

TPL_T void CuHostData3D<T>::init(bool pgLockMem,bool alignMemory, uint2 dim, uint l)
{

#ifdef NOCUDA_X11
	CGObject::SetType(CGObject::AutoStringClass(this));
#else
	CGObject::SetType("CuHostData3D");
#endif
	_pgLockMem		= pgLockMem;
	_alignMemory	= alignMemory;
    CData3D<T>::bInit(dim,l); // ATTENTION PROBLEME : Pure virtual method called
}

TPL_T bool CuHostData3D<T>::Memset( int val )
{
    if (CData<T>::GetSizeofMalloc() < CData3D<T>::Sizeof())
        return false;

    memset(CData3D<T>::pData(),val,CData3D<T>::Sizeof());
    return true;
}

TPL_T void CuHostData3D<T>::Fill(T Value)
{
    T* data = CData3D<T>::pData();

    data[0] = Value;

    uint sizeFilled = 1;

    while( sizeFilled < CData3D<T>::GetSize() - sizeFilled)
    {
        memcpy(data + sizeFilled, data, sizeof(T) * sizeFilled);
        sizeFilled <<= 1;
    }

    memcpy(data + sizeFilled, data, sizeof(T) * (CData3D<T>::GetSize() - sizeFilled));
}

TPL_T void CuHostData3D<T>::FillRandom(T min, T max)
{
    T mod = (T)abs((float)max - (float)min);

    for(uint i=0;i<CData3D<T>::GetSize();i++)
    {
        int rdVal  = rand()%((int)mod);
        double dRdVal = (float)rand() / std::numeric_limits<int>::max();
        CData3D<T>::pData()[i] = min + rdVal + (T)dRdVal;
    }
}

TPL_T void CuHostData3D<T>::OutputValues(uint level, uint plan,  Rect rect, uint offset, T defaut, float sample, float factor)
{
    GpGpuTools::OutputArray(CData3D<T>::pData(), CData3D<T>::GetDimension3D(), plan, level, rect, offset, defaut, sample, factor);
}

TPL_T bool CuHostData3D<T>::abDealloc()
{
    bool  error = true;
    if(_pgLockMem)
        error = CData<T>::ErrorOutput(cudaFreeHost(CData3D<T>::pData()),__FUNCTION__);
    else
        free(CData3D<T>::pData());

    struct2D::SetMaxSize(0);
    struct2D::SetMaxDimension();

    return error;
}

TPL_T bool CuHostData3D<T>::abMalloc()
{
    if(_pgLockMem)
        return CData<T>::ErrorOutput(cudaMallocHost(CData3D<T>::ppData(),CData3D<T>::Sizeof()),__FUNCTION__);
	else if(_alignMemory)
	{
		T* data;
#ifdef _MSC_VER
	if ((data = (T*)_aligned_malloc(CData3D<T>::Sizeof(), sizeof(__m128i))) != NULL)
#else
		if(!posix_memalign((void **) &data, sizeof(__m128i), CData3D<T>::Sizeof()))
#endif
			CData3D<T>::SetPData(data);
	}
    else
        CData3D<T>::SetPData((T*)malloc(CData3D<T>::Sizeof()));

    return true;
}

TPL_T bool CuHostData3D<T>::saveImage(string nameImage,ushort layer)
{
    std::string numec(GpGpuTools::conca(nameImage.c_str(),layer));
    std::string nameFile = numec + std::string(".pgm");
    return GpGpuTools::Array1DtoImageFile(pLData(layer) ,nameFile.c_str(),CData3D<T>::GetDimension());
}


///
///
///

template<class T, class gpsdk = cudaContext> class DecoratorDeviceData{};

/// \cond
template<class T>
class DecoratorDeviceData<T,cudaContext>
{
public:

    bool    CopyDevicetoHost(T* hostData)
    {
        return _dD->ErrorOutput(cudaMemcpy( hostData, _dD->pData(), _dD->Sizeof(), cudaMemcpyDeviceToHost),__FUNCTION__);
    }

    bool    Memset( int val )
    {

        return _dD->ErrorOutput(cudaMemset( _dD->pData(), val, _dD->Sizeof()),__FUNCTION__);
    }

    ///     \brief  Copie asynchrone de toutes les valeurs du tableau dans un tableau du host
    ///     \param  hostData : tableau destination
    ///     \param stream : flux cuda de gestion des appels asynchrone
    bool    CopyDevicetoHostASync( T* hostData, cudaStream_t stream )
    {

        return _dD->ErrorOutput(cudaMemcpyAsync ( hostData, _dD->pData(), _dD->Sizeof(), cudaMemcpyDeviceToHost, stream),__FUNCTION__);

    }

    /// \brief  Copie toutes les valeurs d un tableau dans la structure de donnee de la classe (dans la memoire globale GPU)
    /// \param  hostData : tableau cible
    bool    CopyHostToDevice(T *hostData)
    {

        return _dD->ErrorOutput(cudaMemcpy( _dD->pData(),hostData, _dD->Sizeof(), cudaMemcpyHostToDevice),__FUNCTION__);
    }

    bool    MemsetAsync(int val, cudaStream_t stream)
    {
        return  _dD->ErrorOutput(cudaMemsetAsync(_dD->pData(), val, _dD->Sizeof(), stream ),__FUNCTION__);
    }

protected:

    DecoratorDeviceData(CData<T> *dataDevice):_dD(dataDevice){}

    bool    dabDealloc(){ return _dD->ErrorOutput(cudaFree(_dD->pData()),__FUNCTION__);}

    bool    dabMalloc(){ return _dD->ErrorOutput(cudaMalloc((void **)_dD->ppData(), _dD->Sizeof()),__FUNCTION__);}

private:

    CData<T>* _dD;
};


#if OPENCL_ENABLED

template<class T>
class DecoratorDeviceData<T,openClContext>
{
public:

    bool    CopyDevicetoHost(T* hostData)
    {

        return clEnqueueReadBuffer(CGpGpuContext<openClContext>::commandQueue(),_dD->clMem(),CL_FALSE,0,_dD->Sizeof(),hostData,0,NULL,NULL) == CL_SUCCESS;
    }

    bool    Memset( int val ){

#if     CL_VERSION_1_2 == 1
        const cl_int pat = val;
        return clEnqueueFillBuffer(CGpGpuContext<openClContext>::commandQueue(),_dD->clMem(),&pat, sizeof(cl_uint), 0, _dD->Sizeof(), 0, NULL, NULL) == CL_SUCCESS;
#elif   CL_VERSION_1_1 == 1
        // A implemeter
        // 2 alternatives
        //  * write buffer  ....
        //  * kernel        ....
        return false;
#endif

    }

    /// \brief  Copie toutes les valeurs d un tableau dans la structure de donnee de la classe (dans la memoire globale GPU)
    /// \param  hostData : tableau cible
    bool    CopyHostToDevice(T *hostData){     return clEnqueueWriteBuffer(CGpGpuContext<openClContext>::commandQueue(),_dD->clMem(),CL_FALSE,0,_dD->Sizeof(),hostData,0,NULL,NULL) == CL_SUCCESS;}

protected:

    DecoratorDeviceData(CData<T> *dataDevice):_dD(dataDevice){}

    bool    dabDealloc(){
        cl_int errorCode = clReleaseMemObject(_dD->clMem());

        CGpGpuContext<openClContext>::errorDump(errorCode,"Dealloc buffer");

        return  errorCode == CL_SUCCESS;

    }

    bool    dabMalloc()
    {

        cl_int errorCode = -1;
        _dD->setClMem(clCreateBuffer(CGpGpuContext<openClContext>::contextOpenCL(),CL_MEM_READ_WRITE,_dD->Sizeof(),NULL,&errorCode));

        CGpGpuContext<openClContext>::errorDump(errorCode,"malloc buffer");

        return errorCode == CL_SUCCESS;
    }

private:

    CData<T>* _dD;
};

#endif
/// \endcond


/// \class CuDeviceData2D
/// \brief Cette classe est un tableau de donnee 2D situee dans memoire globale de la carte video
///
template <class T, class gpsdk = cudaContext >
class CuDeviceData2D : public CData2D<T>, public DecoratorDeviceData<T,gpsdk>
{
public:

    CuDeviceData2D():DecoratorDeviceData<T,gpsdk>((CData2D<T>*)this){}

    bool        Memset(int val){return DecoratorDeviceData<T,gpsdk>::Memset(val);}

protected:

    bool        abDealloc(){

        struct2D::SetMaxSize(0);
        struct2D::SetMaxDimension();

        return DecoratorDeviceData<T,gpsdk>::dabDealloc();

    }

    bool        abMalloc(){return DecoratorDeviceData<T,gpsdk>::dabMalloc();}

};


/// \class CuDeviceData3D
/// \brief Structure 3d de données instanciées dans la mémoire globale vidéo
template <class T>
class CuDeviceData3D : public CData3D<T>, public DecoratorDeviceData<T,cudaContext>
{
public:

    CuDeviceData3D():DecoratorDeviceData<T,cudaContext>(this){init("No Name");}

	///
	/// \brief CuDeviceData3D Constructeur de la classe
	/// \param dim	Dimension 2D de la structure de données
	/// \param l	Taille de la structure dans la dimansion 3 (Z)
	/// \param name Nommer la sctructure pour le débogage
	///
    CuDeviceData3D(uint2 dim,uint l, string name = "NoName"):DecoratorDeviceData<T,cudaContext>(this) { init(name,dim,l);}

	///
	/// \brief CuDeviceData3D
	/// \param dim	Dimension totale de la structure de données
	/// \param name Nommer la sctructure pour le débogage
    CuDeviceData3D(uint dim, string name = "NoName"):DecoratorDeviceData<T,cudaContext>(this){init(name,make_uint2(dim,1),1);}

    /// \brief Initialise toutes les valeurs du tableau a val
    /// \param val : valeur d initialisation
    bool        Memset(int val){return DecoratorDeviceData<T,cudaContext>::Memset(val);}

	///
	/// \brief CopyDevicetoHost Copier le contenue de la mémoire globale de la structure vers l'hote
	/// \param hostData pointeur hote de destination
	/// \return vrais si l'opération a réussi
	///
    bool        CopyDevicetoHost(CuHostData3D<T> &hostData){return  DecoratorDeviceData<T,cudaContext>::CopyDevicetoHost(hostData.pData());}

protected:

    bool        abDealloc(){

        struct2D::SetMaxSize(0);
        struct2D::SetMaxDimension();
        return DecoratorDeviceData<T,cudaContext>::dabDealloc();

    }

    bool        abMalloc(){return DecoratorDeviceData<T,cudaContext>::dabMalloc();}

private:

    void        init(string name,uint2 dim = make_uint2(0), uint l = 0);

};

TPL_T void CuDeviceData3D<T>::init(string name, uint2 dim, uint l)
{	
#ifdef NOCUDA_X11
	CGObject::SetType(CGObject::AutoStringClass(this));
#else
	CGObject::SetType("CuDeviceData3D");
#endif

    CGObject::SetName(name);
    if(size(dim) && l)
        CData3D<T>::Malloc(dim,l);
}

template<class context = cudaContext>
class DecoratorImage{};

/// \cond
/// \class DecoratorImage
/// \brief Decorateur pour imageCuda
template<>
class DecoratorImage<cudaContext> : public CData3D<cudaArray>
{
public:

    /// \brief  Lie l image a une texture Gpu
    /// \param  texRef : reference de la texture a lier
    bool		bindTexture(textureReference& texRef)
    {
        _textureReference = &texRef;

        cudaChannelFormatDesc desc;

        bool bCha	= !cudaGetChannelDesc(&desc, GetCudaArray());
        bool bBind	= !cudaBindTextureToArray(&texRef,GetCudaArray(),&desc);

        return bCha && bBind;
    }

    bool        UnbindDealloc(){

        if(_textureReference) cudaUnbindTexture(_textureReference);

        _textureReference = NULL;

        return CData3D<cudaArray>::Dealloc();
    }

protected:

    DecoratorImage(){}

    /// \brief  Initialisation de toutes les valeurs du tableau a val
    /// \param  val : valeur d initialisation
    bool		Memset(int val)
    {
        std::cout << "PAS DE MEMSET POUR CUDA ARRAY" << "\n";
        return true;
    }

    /// \brief  renvoie le tableau cuda contenant les valeurs de l'image
    cudaArray*	GetCudaArray()
    {
        return pData();
    }


    bool    abDealloc()
    {

        struct2D::SetMaxSize(0);
        struct2D::SetMaxDimension();
        return (cudaFreeArray( GetCudaArray()) == cudaSuccess) ? true : false;
    }

private:

    textureReference*   _textureReference;

};

#if OPENCL_ENABLED
template<>
class DecoratorImage<openClContext>
{
public:

    bool        Dealloc()
    {
        return abDealloc();
    }

protected:

    DecoratorImage(CData<cl_mem> *buffer):
        _buffer(buffer){}

    /// \brief  Initialisation de toutes les valeurs du tableau a val
    /// \param  val : valeur d initialisation
    bool		Memset(int val)
    {
        std::cout << "PAS DE MEMSET POUR OPENCL" << "\n";
        return true;
    }

    bool        abDealloc()
    {
        cl_mem* buf = _buffer->pData();

        return clReleaseMemObject(*buf) == CL_SUCCESS;
    }

private:

    CData<cl_mem>*      _buffer;

};
#endif

template <class T,class context> class ImageGpGpu
{};
/// \endcond

template <class T>
///
/// \brief The ImageGpGpu<T, cudaContext> class
/// Texture device pour le GpGpu dans le contexte CUDA
class ImageGpGpu<T,cudaContext > : public DecoratorImage<cudaContext>
{
public:

    ImageGpGpu<T,cudaContext> ()
    {
		#ifdef NOCUDA_X11
			DecoratorImage<cudaContext>::SetType(CGObject::AutoStringClass(this));
		#else
			DecoratorImage<cudaContext>::SetType("ImageCuda");
		#endif

        DecoratorImage<cudaContext>::ClassTemplate(DecoratorImage<cudaContext>::ClassTemplate() + " " + DecoratorImage<cudaContext>::StringClass<T>(_ClassData));
    }

    /// \brief Initialise les valeurs de l image avec un tableau de valeur du Host
    /// \param data : Donnees cible a copier
    bool	copyHostToDevice(T* data){
        return DecoratorImage<cudaContext>::ErrorOutput(cudaMemcpyToArray(DecoratorImage<cudaContext>::pData(), 0, 0, data, sizeof(T)*size(GetDimension()), cudaMemcpyHostToDevice),__FUNCTION__);
    }

	///
	/// \brief SetNameImage Nommer l'image pour le débogage
	/// \param name Le nom a donner
	///
    void SetNameImage(string name)
    {
        DecoratorImage<cudaContext>::SetName(name);
    }

	///
	/// \brief syncDevice Copier les données de l'hote vers le device
	/// \param hostData Pointeur source
	/// \param texture Texture destinataire
	/// \return
	///
    bool    syncDevice(CuHostData3D<T> &hostData,textureReference&  texture)
    {
        CData3D::ReallocIfDim(hostData.GetDimension(),1);
        bool resultSync = copyHostToDevice(hostData.pData());
        bindTexture(texture);

        return resultSync;
    }

protected:

	/// \cond
    bool    abMalloc()
    {

        cudaChannelFormatDesc channelDesc =  cudaCreateChannelDesc<T>();
        return DecoratorImage<cudaContext>::ErrorOutput(cudaMallocArray(DecoratorImage<cudaContext>::ppData(),&channelDesc,DecoratorImage<cudaContext>::GetDimension().x,DecoratorImage<cudaContext>::GetDimension().y),__FUNCTION__);
    }

    uint	Sizeof(){ return DecoratorImage<cudaContext>::GetSize() * sizeof(T);}

private:

    T*		_ClassData;
	/// \endcond
};


#if OPENCL_ENABLED
template <class T>
class ImageGpGpu<T,openClContext> : public CData2D<cl_mem>, public DecoratorImage<openClContext>
{
    ImageGpGpu():
        DecoratorImage<openClContext>(this)
    {

#ifdef NOCUDA_X11
	CData2D::SetType(CGObject::AutoStringClass(this));
#else
	CData2D::SetType("Image OpenCL");
#endif

        CData2D::ClassTemplate(CData2D::ClassTemplate() + " " + CData2D::StringClass<T>(_ClassData));
    }

    /// \brief Initialise les valeurs de l image avec un tableau de valeur du Host
    /// \param data : Donnees cible a copier
    bool	copyHostToDevice(T* data)
    {
        cl_mem* image = CData2D<cl_mem>::pData();

        size_t origin[] = {0,0,0}; // Defines the offset in pixels in the image from where to write.
        size_t region[] = {struct2D::GetDimension().x, struct2D::GetDimension().y, 1}; // Size of object to be transferred

        cl_int err = clEnqueueWriteImage(CGpGpuContext<openClContext>::contextOpenCL(), *image, CL_TRUE, origin, region,0,0, data, 0, NULL,NULL);

        return err == CL_SUCCESS;
    }

protected:

    bool    abDealloc(){

        struct2D::SetMaxSize(0);
        struct2D::SetMaxDimension();

        return DecoratorImage<openClContext>::abDealloc();}

    bool    abMalloc(){

        cl_image_format img_fmt;
        img_fmt.image_channel_order = CL_RGBA; // --> en fonction de je ne sait quoi!!!
        img_fmt.image_channel_data_type = CL_FLOAT; // --> en fonction de T
        cl_int err;
        cl_mem *image = new cl_mem;
        CData2D<cl_mem>::SetPData(image);

        *image = clCreateImage2D(CGpGpuContext<openClContext>::contextOpenCL(), CL_MEM_READ_ONLY, &img_fmt, struct2D::GetDimension().x, struct2D::GetDimension().y, 0, 0, &err);
        return err == CL_SUCCESS;
    }

    uint	Sizeof(){ return CData2D::GetSize() * sizeof(T);}

private:

    T*		_ClassData;

};
#endif

template <class T, class context> class ImageLayeredGpGpu {};

template <class T>
///
/// \brief The ImageLayeredGpGpu<T, cudaContext> class
///  Cette classe est une pile d'image 2D directement liable a une texture dans un context CUDA
class ImageLayeredGpGpu <T, cudaContext> :  public DecoratorImage<cudaContext>
{

public:

    ImageLayeredGpGpu()
    {

    #ifndef _WIN32
        CData3D::SetType(__CLASS_NAME__);
    #endif

        CData3D::ClassTemplate(CData3D::ClassTemplate() + " " + CData3D::StringClass<T>(_ClassData));
    }

    /// \brief Copie des valeurs des images avec un tableau 3D de valeur du Host
    /// \param data : Donnees cible a copier
    bool	copyHostToDevice(T* data)
    {

        cudaMemcpy3DParms	p = CudaMemcpy3DParms(data,cudaMemcpyHostToDevice);

        return CData3D::ErrorOutput(cudaMemcpy3D(&p),__FUNCTION__) ;

    }
    /// \brief Copie des valeurs des images vers un tableau 3D du Host
    /// \param data : tableau de destination
    bool	copyDeviceToDevice(T* data)
    {
        cudaMemcpy3DParms	p = CudaMemcpy3DParms(data,cudaMemcpyDeviceToDevice);

        return CData3D::ErrorOutput(cudaMemcpy3D(&p),__FUNCTION__) ;
    }
    /// \brief Copie asynchrone des valeurs des images avec un tableau 3D de valeur du Host
    /// \param data : Donnees cible a copierAImageCuda
    /// \param stream : flux cuda
    bool	copyHostToDeviceASync(T* data, cudaStream_t stream = 0){
        cudaMemcpy3DParms	p = CudaMemcpy3DParms(data,cudaMemcpyHostToDevice);

        return CData3D::ErrorOutput( cudaMemcpy3DAsync (&p, stream),__FUNCTION__);
    }


	///
	/// \brief syncDevice Copier les données de l'hote vers le device
	/// \param hostData Pointeur source
	/// \param texture Texture destinataire
	/// \return
	///
    bool    syncDevice(CuHostData3D<T> &hostData,textureReference&  texture)
    {
        CData3D::ReallocIfDim(hostData.GetDimension(),hostData.GetNbLayer());
        bool resultSync = copyHostToDevice(hostData.pData());
        bindTexture(texture);

        return resultSync;
    }

protected:

	///
	/// \cond
    bool    abMalloc()
    {

        cudaChannelFormatDesc channelDesc =	cudaCreateChannelDesc<T>();

        return CData3D::ErrorOutput(cudaMalloc3DArray(ppData(),&channelDesc,CudaExtent(),cudaArrayLayered),__FUNCTION__);
    }

    uint	Sizeof(){ return CData3D::GetSize() * sizeof(T);}

private:

    T*	_ClassData;

    cudaMemcpy3DParms CudaMemcpy3DParms(T *data, cudaMemcpyKind kind)
    {
        cudaExtent sizeImgsLay      = CudaExtent();

        // Déclaration des parametres de copie 3D
        cudaMemcpy3DParms	p		= { 0 };
        cudaPitchedPtr		pitch	= make_cudaPitchedPtr(data, sizeImgsLay.width * sizeof(T), sizeImgsLay.width, sizeImgsLay.height);

        p.dstArray	= pData();   // Pointeur du tableau de destination
        p.srcPtr	= pitch;                        // Pitch
        p.extent	= sizeImgsLay;                  // Taille du cube
        p.kind      = kind;                         // Type de copie

        return p;
    }

    cudaExtent  CudaExtent()
    {
        return make_cudaExtent( CData3D::GetDimension().x, CData3D::GetDimension().y, CData3D::GetNbLayer());
    }
	/// \endcond

};

template<class T>
///
/// \brief The CuUnifiedData3D struct
/// Structure de données unifiées
/// Cette structure de données est constituées d'un espace mémoire sur l'hote et un espace identique sur le device
struct CuUnifiedData3D
{
	///
	/// \brief deviceData les données sur le device
	///
    CuDeviceData3D<T>   deviceData;
	///
	/// \brief hostData les données sur l'hote
	///
    CuHostData3D<T>     hostData;

	///
	/// \brief Malloc Allocation de  la mémoire pour l'hote et le device
	/// \param dim Taille 2d des structures de données
	/// \param l Taille en Z des structures de données
	///
    void Malloc( uint2 dim, uint l )
    {
        deviceData.Malloc(dim,l);
        hostData.Malloc(dim,l);
    }

	///
	/// \brief syncDevice
	/// Copier les données de l'hote vers le device
    void syncDevice()
    {
        deviceData.CopyHostToDevice(hostData.pData());
    }


	///
	/// \brief syncHost
	/// Copier les données du device vers l'hote
    void syncHost()
    {
        deviceData.CopyDevicetoHost(hostData);
    }


	///
	/// \brief ReallocIfDim Reallocation mémoire si la dimension totale est supérieur à la précédente
	/// \param dim Taille 2d des structures de données
	/// \param l Taille en Z des structures de données
	///
    void ReallocIfDim(uint2 dim, uint l)
    {
        deviceData. ReallocIfDim(dim,l);
        hostData.ReallocIfDim(dim,l);
    }

	///
	/// \brief Reallocation mémoire si la dimension totale est supérieur à la précédente
	/// \param size Taille totale de la nouvelle allocation
	///
	void ReallocIfDim(uint size)
	{
		uint2 sizeDim = make_uint2(size,1);

		deviceData. ReallocIfDim(sizeDim,1);
		hostData.ReallocIfDim(sizeDim,1);
	}

	///
	/// \brief Dealloc Désalocation de la mémoire des deux structures
	///
    void Dealloc()
    {
        deviceData. Dealloc();
        hostData.Dealloc();
    }

	///
	/// \brief pData
	/// \return le pointeur sur les données du device
	///
    T* pData()
    {
       return deviceData.pData();
    }

	///
	/// \brief SetName Nommer la structure pour le débogage
	/// \param name Nom
	/// \param id Agreger un identifiant à la suite du nom
	///
	void SetName(string name,ushort id = 0)
	{
		deviceData.SetName("uDevice_" + name + "_",id);
		hostData.SetName("uHost_" + name + "_",id);
	}
};


/*@}*/

#endif //GPGPU_DATA_H
