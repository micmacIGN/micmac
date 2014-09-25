#ifndef GPGPU_DATA_H
#define GPGPU_DATA_H

#include "GpGpu/GpGpu_Object.h"
#include "GpGpu/GpGpu_Context.h"
#include "GpGpu/GpGpu_Tools.h"

#define TPL_T template<class T>

/// \class CData
/// \brief Classe Abstraite de donnees
template<class T>
class CData : public CGObject
{

    friend class    DecoratorImageCuda;
    template<class M,int sdkGPU> friend class    DecoratorDeviceData;

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

    /// \brief      Sortie consolle de l allocation memoire globale Gpu

    void            MallocInfo();

    /// \brief      Obtenir une valeur aleatoire comprise entre min et max
    static T        GetRandomValue(T min, T max);

protected:

    /// \brief      Renvoie la taille de la memoire alloue
    uint            GetSizeofMalloc(){ return _sizeofMalloc; }

    /// \brief      Initialise la taille de la memoire alloue
    /// \brief      Renvoie le pointeur du pointeur des donnees
    T**             ppData(){ return &_data; }

    /// \brief      Init le pointeur des donnees
    void            SetPData(T *p){ _data = p;}

    virtual bool    abDealloc(){ return false;}

    virtual bool    abMalloc(){ return false;}

    virtual uint    Sizeof(){return 0;}

    /// \brief      Sortie console des erreurs Cuda
    /// \param      err :  erreur cuda rencontree
    /// \param      fonctionName : nom de la fonction ou se trouve l erreur
    virtual bool	ErrorOutput(cudaError_t err,const char* fonctionName);

    cl_mem          clMem() const{return _clMem;}

    void            setClMem(const cl_mem &clMem){_clMem = clMem;}

private:

    uint            _memoryOc;

    T*              _data;

    cl_mem          _clMem;

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
    if (err != cudaSuccess)
    {
        std::cout << "--------------------------------------------------------------------------------------\n";
        std::cout << "Erreur Cuda         : " <<  fonctionName  << "() | Object " + CGObject::Id() << "\n";        
        OutputInfo();
        std::cout << "Pointeur de donnees : " << CData<T>::pData()  << "\n";
        std::cout << "Memoire allouee     : " << _memoryOc / pow(2.0,20) << " Mo | " << _memoryOc / pow(2.0,10) << " ko | " << _memoryOc  << " octets \n";
        std::cout << "Taille des donnees  : " << CData<T>::GetSizeofMalloc()  / pow(2.0,20) << " Mo | " << CData<T>::GetSizeofMalloc()  / pow(2.0,10) << " ko | " << CData<T>::GetSizeofMalloc() << " octets \n";
        checkCudaErrors( err );
        GpGpuTools::OutputInfoGpuMemory();
        std::cout << "--------------------------------------------------------------------------------------\n";
        exit(1);
        return false;
    }
    return true;
}

TPL_T CData<T>::CData():
    _memoryOc(0),
    _data(NULL),
    _sizeofMalloc(0)
{
    CGObject::ClassTemplate(CGObject::StringClass<T>(pData()));
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
    if (!isNULL()) op = abDealloc();
    dataNULL();
    return op;
}

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

    bool            ReallocIfDim(uint2 dim);

protected:

    virtual bool    abDealloc() = 0;

    virtual bool    abMalloc()  = 0;

    uint            Sizeof(){return sizeof(T) * struct2D::GetSize();}

};

TPL_T void CData2D<T>::OutputInfo()
{
    std::cout << "Structure 2D : \n";
    struct2D::Output();
}

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

    bool			Realloc(uint size){return Realloc(make_uint2(size,1),1);}

    bool			Realloc(uint3 dim){return Realloc(make_uint2(dim.x,dim.y),dim.z);}

    bool			ReallocIf(uint dim1D);

    bool			ReallocIf(uint2 dim, uint l = 1);

    bool			ReallocIf(uint dimX, uint dimY, uint l = 1);

    bool            ReallocIfDim(uint2 dim,uint l);



    T&              operator[](uint2 pt);

    T&              operator[](uint3 pt);

    T&              operator[](uint pt1D)   {   return (CData<T>::pData())[pt1D];       }

    T&              operator[](int pt1D)    {   return (CData<T>::pData())[(uint)pt1D]; }

protected:

    virtual bool    abMalloc()  = 0;

    virtual bool    abDealloc() = 0;

    /// \brief      Nombre d elements de la structure
    uint			Sizeof(){return GetSize() * sizeof(T);}

    void            bInit(uint2 dim = make_uint2(0), uint l = 0);
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

TPL_T void CData3D<T>::bInit(uint2 dim, uint l)
{
    if(size(dim) && l) Malloc(dim,l);
}

/// \class CuHostData3D
/// \brief Tableau 3D d elements contenue la memoire du Host.
/// La gestion memoire est realise par l API Cuda.
template <class T>
class CuHostData3D : public CData3D<T>
{
public:

    CuHostData3D(bool pgLockMem = NOPAGLOCKMEM){init(pgLockMem);}

    /// \brief constructeur avec initialisation de la dimension de la structure
    /// \param dimX : Dimension 1D a initialiser
    /// \param dimY : Dimension 1D a initialiser
    /// \param l : Taille de la 3eme dimension
    CuHostData3D(uint dimX, uint dimY = 1, uint l = 1, bool pgLockMem = NOPAGLOCKMEM){init(pgLockMem,make_uint2(dimX,dimY),l);}

    /// \brief constructeur avec initialisation de la dimension de la structure
    /// \param dim : Dimension 2D a initialiser
    /// \param l : Taille de la 3eme dimension
    CuHostData3D(uint2 dim, uint l = 1, bool pgLockMem = NOPAGLOCKMEM){ init(pgLockMem,dim,l);}

    /// \brief constructeur avec initialisation de la dimension de la structure
    /// \param dim : Dimension 3D a initialiser
    CuHostData3D(uint3 dim,bool pgLockMem = NOPAGLOCKMEM ){init(pgLockMem,make_uint2(dim.x,dim.y),dim.z);}

    bool Memset(int val);

    /// \brief Remplie le tableau avec la valeur Value
    /// \param Value : valeur a remplir
    void Fill(T Value);

    /// \brief Remplie le tableau avec la valeur aleatoire pour chaque element
    /// \param min : valeur a remplir minimum
    /// \param max : valeur a remplir maximum
    void FillRandom(T min, T max);

    /// \brief Affiche un Z du tableau dans la console
    void OutputValues(uint level = 0, uint plan = XY, Rect rect = NEGARECT, uint offset = 3, T defaut = (T)0.0f, float sample = 1.0f, float factor = 1.0f);

    void SetPageLockedMemory(bool page){ _pgLockMem = page; }

protected:

    virtual bool    abDealloc() ;

    virtual bool    abMalloc();

private:

    bool    _pgLockMem;

    void    init(bool pgLockMem = NOPAGLOCKMEM, uint2 dim = make_uint2(0), uint l = 0);

};

TPL_T void CuHostData3D<T>::init(bool pgLockMem, uint2 dim, uint l)
{
    CGObject::SetType("CuHostData3D");
    _pgLockMem = pgLockMem;
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
    else
        CData3D<T>::SetPData((T*)malloc(CData3D<T>::Sizeof()));

    return true;
}

template<class T, int gpsdk = CUDASDK> class DecoratorDeviceData{};

template<class T>
class DecoratorDeviceData<T,CUDASDK>
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
class DecoratorDeviceData<T,OPENCLSDK>
{
public:

    bool    CopyDevicetoHost(T* hostData)
    {

        return clEnqueueReadBuffer(CGpGpuContext<OPENCLSDK>::commandQueue(),_dD->clMem(),CL_FALSE,0,_dD->Sizeof(),hostData,0,NULL,NULL) == CL_SUCCESS;
    }

    bool    Memset( int val ){

#if     CL_VERSION_1_2 == 1
        const cl_int pat = val;
        return clEnqueueFillBuffer(CGpGpuContext<OPENCLSDK>::commandQueue(),_dD->clMem(),&pat, sizeof(cl_uint), 0, _dD->Sizeof(), 0, NULL, NULL);) == CL_SUCCESS;
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
    bool    CopyHostToDevice(T *hostData){     return clEnqueueWriteBuffer(CGpGpuContext<OPENCLSDK>::commandQueue(),_dD->clMem(),CL_FALSE,0,_dD->Sizeof(),hostData,0,NULL,NULL) == CL_SUCCESS;}

protected:

    DecoratorDeviceData(CData<T> *dataDevice):_dD(dataDevice){}

    bool    dabDealloc(){ return clReleaseMemObject(_dD->clMem()) == CL_SUCCESS;}

    bool    dabMalloc()
    {
        cl_int errorCode = -1;
        _dD->setClMem(clCreateBuffer(CGpGpuContext<OPENCLSDK>::contextOpenCL(),CL_MEM_READ_WRITE,_dD->Sizeof(),NULL,&errorCode));
        return errorCode == CL_SUCCESS;
    }

private:

    CData<T>* _dD;
};
#endif

/// \class CuDeviceData2D
/// \brief Cette classe est un tableau de donnee 2D situee dans memoire globale de la carte video
template <class T>
class CuDeviceData2D : public CData2D<T>, public DecoratorDeviceData<T,CUDASDK>
{
public:

    CuDeviceData2D():DecoratorDeviceData<T,CUDASDK>((CData2D<T>*)this){}

    bool        Memset(int val){return DecoratorDeviceData<T,CUDASDK>::Memset(val);}

protected:

    bool        abDealloc(){

        struct2D::SetMaxSize(0);
        struct2D::SetMaxDimension();

        return DecoratorDeviceData<T,CUDASDK>::dabDealloc();

    }

    bool        abMalloc(){return DecoratorDeviceData<T,CUDASDK>::dabMalloc();}

};

/// \class CuDeviceData3D
/// \brief Structure 3d de données instanciées dans la mémoire globale vidéo
template <class T>
class CuDeviceData3D : public CData3D<T>, public DecoratorDeviceData<T,CUDASDK>
{
public:

    CuDeviceData3D():DecoratorDeviceData<T,CUDASDK>(this){init("No Name");}

    CuDeviceData3D(uint2 dim,uint l, string name = "NoName"):DecoratorDeviceData<T,CUDASDK>(this) { init(name,dim,l);}

    CuDeviceData3D(uint dim, string name = "NoName"):DecoratorDeviceData<T,CUDASDK>(this){init(name,make_uint2(dim,1),1);}

    /// \brief Initialise toutes les valeurs du tableau a val
    /// \param val : valeur d initialisation
    bool        Memset(int val){return DecoratorDeviceData<T,CUDASDK>::Memset(val);}

    bool        CopyDevicetoHost(CuHostData3D<T> &hostData){return  DecoratorDeviceData<T,CUDASDK>::CopyDevicetoHost(hostData.pData());}

protected:

    bool        abDealloc(){

        struct2D::SetMaxSize(0);
        struct2D::SetMaxDimension();
        return DecoratorDeviceData<T,CUDASDK>::dabDealloc();

    }

    bool        abMalloc(){return DecoratorDeviceData<T,CUDASDK>::dabMalloc();}

private:

    void        init(string name,uint2 dim = make_uint2(0), uint l = 0);

};

TPL_T void CuDeviceData3D<T>::init(string name, uint2 dim, uint l)
{
    CGObject::SetType("CuDeviceData3D");
    CGObject::SetName(name);
    if(size(dim) && l)
        CData3D<T>::Malloc(dim,l);
}

template<int gpusdk = CUDASDK>
class DecoratorImage{};


/// \class DecoratorImage
/// \brief Decorateur pour imageCuda
template<>
class DecoratorImage<CUDASDK> : public CData3D<cudaArray>
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
class DecoratorImage<OPENCLSDK>
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

template <class T,int SDKGPU> class ImageGpGpu
{};

template <class T>
class ImageGpGpu<T,CUDASDK> : public DecoratorImage<CUDASDK>
{
public:

    ImageGpGpu<T,CUDASDK> ()
    {
        DecoratorImage<CUDASDK>::SetType("ImageCuda");
        DecoratorImage<CUDASDK>::ClassTemplate(DecoratorImage<CUDASDK>::ClassTemplate() + " " + DecoratorImage<CUDASDK>::StringClass<T>(_ClassData));
    }

    /// \brief Initialise les valeurs de l image avec un tableau de valeur du Host
    /// \param data : Donnees cible a copier
    bool	copyHostToDevice(T* data){
        return DecoratorImage<CUDASDK>::ErrorOutput(cudaMemcpyToArray(DecoratorImage<CUDASDK>::pData(), 0, 0, data, sizeof(T)*size(GetDimension()), cudaMemcpyHostToDevice),__FUNCTION__);
    }

    void SetNameImage(string name)
    {
        DecoratorImage<CUDASDK>::SetName(name);
    }

protected:

    bool    abMalloc()
    {

        cudaChannelFormatDesc channelDesc =  cudaCreateChannelDesc<T>();
        return DecoratorImage<CUDASDK>::ErrorOutput(cudaMallocArray(DecoratorImage<CUDASDK>::ppData(),&channelDesc,DecoratorImage<CUDASDK>::GetDimension().x,DecoratorImage<CUDASDK>::GetDimension().y),__FUNCTION__);
    }

    uint	Sizeof(){ return DecoratorImage<CUDASDK>::GetSize() * sizeof(T);}

private:

    T*		_ClassData;
};

#if OPENCL_ENABLED
template <class T>
class ImageGpGpu<T,OPENCLSDK> : public CData2D<cl_mem>, public DecoratorImage<OPENCLSDK>
{
    ImageGpGpu():
        DecoratorImage<OPENCLSDK>(this)
    {
        CData2D::SetType("Image OpenCL");
        CData2D::ClassTemplate(CData2D::ClassTemplate() + " " + CData2D::StringClass<T>(_ClassData));
    }

    /// \brief Initialise les valeurs de l image avec un tableau de valeur du Host
    /// \param data : Donnees cible a copier
    bool	copyHostToDevice(T* data)
    {
        cl_mem* image = CData2D<cl_mem>::pData();

        size_t origin[] = {0,0,0}; // Defines the offset in pixels in the image from where to write.
        size_t region[] = {struct2D::GetDimension().x, struct2D::GetDimension().y, 1}; // Size of object to be transferred

        cl_int err = clEnqueueWriteImage(CGpGpuContext<OPENCLSDK>::contextOpenCL(), *image, CL_TRUE, origin, region,0,0, data, 0, NULL,NULL);

        return err == CL_SUCCESS;
    }

protected:

    bool    abDealloc(){

        struct2D::SetMaxSize(0);
        struct2D::SetMaxDimension();

        return DecoratorImage<OPENCLSDK>::abDealloc();}

    bool    abMalloc(){

        cl_image_format img_fmt;
        img_fmt.image_channel_order = CL_RGBA; // --> en fonction de je ne sait quoi!!!
        img_fmt.image_channel_data_type = CL_FLOAT; // --> en fonction de T
        cl_int err;
        cl_mem *image = new cl_mem;
        CData2D<cl_mem>::SetPData(image);

        *image = clCreateImage2D(CGpGpuContext<OPENCLSDK>::contextOpenCL(), CL_MEM_READ_ONLY, &img_fmt, struct2D::GetDimension().x, struct2D::GetDimension().y, 0, 0, &err);
        return err == CL_SUCCESS;
    }

    uint	Sizeof(){ return CData2D::GetSize() * sizeof(T);}

private:

    T*		_ClassData;

};
#endif

template <class T, int sdkgpu> class ImageLayeredGpGpu {};

/// \class ImageLayeredGpGpu
/// \brief Cette classe est une pile d'image 2D directement liable a une texture GpGpu
template <class T>
class ImageLayeredGpGpu <T, CUDASDK> :  public DecoratorImage<CUDASDK>
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


protected:

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
};

#endif //GPGPU_DATA_H
