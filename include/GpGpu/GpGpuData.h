#ifndef GPGPU_DATA_H
#define GPGPU_DATA_H

#include "GpGpu/GpGpuObject.h"
#include "GpGpu/GpGpuTools.h"

/// \class CData
/// \brief Classe Abstraite de donnees
template <class T>
class CData : public CGObject
{

public:

    CData();
    ~CData();
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
    T*              pData();
    /// \brief      Sortie console des erreurs Cuda
    /// \param      err :  erreur cuda rencontree
    /// \param      fonctionName : nom de la fonction ou se trouve l erreur
    virtual bool	ErrorOutput(cudaError_t err,const char* fonctionName);
    /// \brief      Sortie consolle de l allocation memoire globale Gpu
    void            MallocInfo();
    /// \brief      Obtenir une valeur aleatoire comprise entre min et max
    static T        GetRandomValue(T min, T max);

protected:

    /// \brief      Renvoie la taille de la memoire alloue
    uint            GetSizeofMalloc();

    /// \brief      Initialise la taille de la memoire alloue
    /// \brief      Renvoie le pointeur du pointeur des donnees
    T**             ppData();

    /// \brief      Init le pointeur des donnees
    void            SetPData(T *p);

    virtual bool    abDealloc() = 0;

    virtual bool    abMalloc()  = 0;

    virtual uint    Sizeof()    = 0;

private:

    uint            _memoryOc;
    T*              _data;
    uint            _sizeofMalloc;

    /// \brief      Suppression de memoire alloue
    void            SubMemoryOc(uint m);

    /// \brief      Ajout de memoire alloue
    void            AddMemoryOc(uint m);

    /// \brief      Initialise a NULL le pointeur des donnees
    void            dataNULL();
    /// \brief      Renvoie True si le pointeur des donnees est NULL
    bool            isNULL();

    /// \param      sizeofmalloc : Taille de l allocation
    void            SetSizeofMalloc(uint sizeofmalloc);
};

template <class T>
void CData<T>::MallocInfo()
{
    std::cout << "Malloc Info " << CGObject::Name() << " Size of Malloc | Memory Oc en Mo     : "<<  _sizeofMalloc / pow(2.0,20) << " | " << _memoryOc / pow(2.0,20) << "\n";
    std::cout << "Malloc Info " << CGObject::Name() << " Size of Malloc | Memory Oc en octets : "<<  _sizeofMalloc << " | " << _memoryOc  << "\n";
}

template <class T>
void CData<T>::SubMemoryOc( uint m )
{
    _memoryOc -=m;
}

template <class T>
void CData<T>::AddMemoryOc( uint m )
{
    _memoryOc +=m;
}

template <class T>
bool CData<T>::ErrorOutput( cudaError_t err,const char* fonctionName )
{
    if (err != cudaSuccess)
    {

        std::cout << "--------------------------------------------------------------------------------------\n";
        std::cout << "\n";
        std::cout << "Erreur Cuda         : " <<  fonctionName  << "() | Object " + CGObject::Id() << "\n";
        GpGpuTools::OutputInfoGpuMemory();
        OutputInfo();
        std::cout << "Pointeur de donnees : " << CData<T>::pData()  << "\n";
        std::cout << "Memoire allouee     : " << _memoryOc / pow(2.0,20) << " Mo | " << _memoryOc / pow(2.0,10) << " ko | " << _memoryOc  << " octets \n";
        std::cout << "Taille des donnees  : " << CData<T>::GetSizeofMalloc()  / pow(2.0,20) << " Mo | " << CData<T>::GetSizeofMalloc()  / pow(2.0,10) << " ko | " << CData<T>::GetSizeofMalloc() << " octets \n";
        checkCudaErrors( err );
        std::cout << "\n";
        std::cout << "--------------------------------------------------------------------------------------\n";
        exit(1);
        return false;
    }

    return true;
}

template <class T>
void CData<T>::SetSizeofMalloc( uint sizeofmalloc )
{
    _sizeofMalloc = sizeofmalloc;
}

template <class T>
T CData<T>::GetRandomValue(T min, T max)
{
    T mod = abs(max - min);
    int rdVal  = rand()%((int)mod);
    double dRdVal = (float)rand() / std::numeric_limits<int>::max();
    return min + rdVal + (T)dRdVal;
}

template <class T>
uint CData<T>::GetSizeofMalloc()
{
    return _sizeofMalloc;
}

template <class T>
CData<T>::CData():
    _memoryOc(0),
    _data(NULL),
    _sizeofMalloc(0)
{
    CGObject::ClassTemplate(CGObject::StringClass<T>(pData()));
}

template <class T>
CData<T>::~CData()
{
    Dealloc();
}

template <class T>
bool CData<T>::Malloc()
{
    SetSizeofMalloc(Sizeof());
    AddMemoryOc(GetSizeofMalloc());
    return abMalloc();
}

template <class T>
bool CData<T>::Dealloc()
{
    bool op = false;
    SubMemoryOc(GetSizeofMalloc());
    SetSizeofMalloc(0);
    if (!isNULL()) op = abDealloc();
    CData<T>::dataNULL();
    return op;
}

template <class T>
T** CData<T>::ppData()
{
    return &_data;
}
template <class T>
bool CData<T>::isNULL()
{
    return (_data == NULL);
}

template <class T>
void CData<T>::dataNULL()
{
    _data = NULL;
}

template <class T>
T* CData<T>::pData()
{
    return _data;
}
template <class T>
void CData<T>::SetPData(T* p)
{
    _data = p;
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
    CData2D(uint2 dim);

    /// \brief Alloue la memoire neccessaire
    //virtual bool	Malloc()        = 0;
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

protected:

    virtual bool    abDealloc() = 0;

    virtual bool    abMalloc()  = 0;

    /// \brief      Nombre d elements de la structure

    uint            Sizeof(){return sizeof(T) * struct2D::GetSize();}

};

template <class T>
void CData2D<T>::OutputInfo()
{
    std::cout << "Structure 2D : \n";
    struct2D::Output();
}

template <class T>
CData2D<T>::CData2D(uint2 dim)
{
    Malloc(dim);
}

template <> inline
uint CData2D<cudaArray>::Sizeof()
{
    return struct2D::GetSize();
}

template <class T>
bool CData2D<T>::Realloc( uint2 dim )
{
    CData<T>::Dealloc();
    Malloc(dim);
    return true;
}

template <class T>
bool CData2D<T>::Malloc( uint2 dim )
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

    CData3D(uint2 dim, uint l);

    ~CData3D(){}

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

    bool			Realloc(uint size);

    bool			ReallocIf(uint dim1D);

    bool			ReallocIf(uint2 dim, uint l);


    T&              operator[](uint2 pt);
    T&              operator[](uint3 pt);
    T&              operator[](uint pt1D);
    T&              operator[](int pt1D);

protected:

    virtual bool    abMalloc()  = 0;

    virtual  bool   abDealloc() = 0;

    /// \brief      Nombre d elements de la structure
    uint			Sizeof();
};

template <class T>
void CData3D<T>::OutputInfo()
{

    std::cout << "Structure 3D        : " << CGObject::Id() << "\n";
    struct2DLayered::Output();
    std::cout << "\n";
}

template <class T>
CData3D<T>::CData3D( uint2 dim, uint l )
{
    Malloc(dim,l);
}

template <class T>
bool CData3D<T>::Malloc( uint2 dim, uint l )
{
    SetDimension(dim,l);
    return CData<T>::Malloc();
}

template <class T>
bool CData3D<T>::Realloc( uint2 dim, uint l )
{
    bool dB = CData<T>::Dealloc();
    bool dM = Malloc(dim,l);
    return (dB && dM);
}

template <class T>
bool CData3D<T>::Realloc(uint size)
{
    return Realloc(make_uint2(size,1),1);
}

template <class T>
bool CData3D<T>::ReallocIf(uint dim1D)
{
    uint2 dim2D = make_uint2(dim1D,1);
    return ReallocIf(dim2D,1);
}

template <class T> inline
bool CData3D<T>::ReallocIf(uint2 dim, uint l)
{
    if( size(dim) * l * sizeof(T) > CData3D<T>::GetSizeofMalloc())
        return CData3D<T>::Realloc(dim,l);
    else
        CData3D<T>::SetDimension(dim,l);

    return true;
}

template <class T>
uint CData3D<T>::Sizeof()
{
    return GetSize() * sizeof(T);
}

template <> inline
uint CData3D<cudaArray>::Sizeof()
{
    return GetSize();
}

template <class T>
T &CData3D<T>::operator [](uint2 pt)
{
    return (CData<T>::pData())[to1D(pt,GetDimension())];
}
template <class T>
T &CData3D<T>::operator [](uint3 pt)
{
    return (CData<T>::pData())[pt.z * struct2D::GetSize() + to1D(make_uint2(pt.x,pt.y),GetDimension())];
}
template <class T>
T &CData3D<T>::operator [](uint pt1D)
{
    return (CData<T>::pData())[pt1D];
}
template <class T>
T &CData3D<T>::operator [](int pt1D)
{
    return (CData<T>::pData())[(uint)pt1D];
}

/// \class CuHostData3D
/// \brief Tableau 3D d elements contenue la memoire du Host.
/// La gestion memoire est realise par l API Cuda.
/// La memoire allouee n'est pas pagine.

template <class T>

class CuHostData3D : public CData3D<T>
{

public:

    CuHostData3D(bool pageLockedmemory = NOPAGELOCKEDMEMORY);

    /// \brief constructeur avec initialisation de la dimension de la structure
    /// \param dimX : Dimension 1D a initialiser
    /// \param dimY : Dimension 1D a initialiser
    /// \param l : Taille de la 3eme dimension
    CuHostData3D(uint dimX, uint dimY = 1, uint l = 1, bool pageLockedmemory = NOPAGELOCKEDMEMORY);

    /// \brief constructeur avec initialisation de la dimension de la structure
    /// \param dim : Dimension 2D a initialiser
    /// \param l : Taille de la 3eme dimension
    CuHostData3D(uint2 dim, uint l = 1, bool pageLockedmemory = NOPAGELOCKEDMEMORY);

    /// \brief constructeur avec initialisation de la dimension de la structure
    /// \param dim : Dimension 3D a initialiser
    CuHostData3D(uint3 dim,bool pageLockedmemory = NOPAGELOCKEDMEMORY );

    ~CuHostData3D(){}

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

    void SetPageLockedMemory(bool page);

protected:

    bool    abDealloc() ;

    bool    abMalloc();

private:

    bool    _pageLockedMemory;

    void    init(bool pageLockedmemory);

};

template <class T>
void CuHostData3D<T>::init(bool pageLockedmemory)
{
    CGObject::SetType("CuHostData3D");
    _pageLockedMemory = pageLockedmemory;
}

template <class T>
CuHostData3D<T>::CuHostData3D(bool pageLockedmemory)
{
    init(pageLockedmemory);
}

template <class T>
CuHostData3D<T>::CuHostData3D(uint dimX, uint dimY, uint l, bool pageLockedmemory)
{
    init(pageLockedmemory);
    CData3D<T>::Malloc(make_uint2(dimX,dimY),l);
}

template <class T>
CuHostData3D<T>::CuHostData3D(uint2 dim, uint l, bool pageLockedmemory )
{
    init(pageLockedmemory);
    CData3D<T>::Malloc(dim,l);
}

template <class T>
CuHostData3D<T>::CuHostData3D(uint3 dim, bool pageLockedmemory)
{
    CData3D<T>::Malloc(make_uint2(dim.x,dim.y),dim.z);
}

template <class T>
bool CuHostData3D<T>::Memset( int val )
{
    if (CData<T>::GetSizeofMalloc() < CData3D<T>::Sizeof())
    {
        std::cout << "Memset, Allocation trop petite !!!" << "\n";
        return false;
    }
    memset(CData3D<T>::pData(),val,CData3D<T>::Sizeof());
    return true;
}

template <class T>
void CuHostData3D<T>::Fill(T Value)
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

template <class T>
void CuHostData3D<T>::FillRandom(T min, T max)
{
    T mod = abs((float)max - (float)min);
    //srand (time(NULL));
    for(uint i=0;i<CData3D<T>::GetSize();i++)
    {
        int rdVal  = rand()%((int)mod);
        double dRdVal = (float)rand() / std::numeric_limits<int>::max();
        CData3D<T>::pData()[i] = min + rdVal + (T)dRdVal;
    }
}

template <class T>
void CuHostData3D<T>::OutputValues(uint level, uint plan,  Rect rect, uint offset, T defaut, float sample, float factor)
{
    GpGpuTools::OutputArray(CData3D<T>::pData(), CData3D<T>::GetDimension3D(), plan, level, rect, offset, defaut, sample, factor);
}

template <class T>
void CuHostData3D<T>::SetPageLockedMemory(bool page)
{
    _pageLockedMemory = page;
}

template <class T>
bool CuHostData3D<T>::abDealloc()
{
    bool  error = true;
    if(_pageLockedMemory)
        error = CData<T>::ErrorOutput(cudaFreeHost(CData3D<T>::pData()),"Dealloc");
    else
        free(CData3D<T>::pData());

    return error;
}

template <class T>
bool CuHostData3D<T>::abMalloc()
{
    if(_pageLockedMemory)
        return CData<T>::ErrorOutput(cudaMallocHost(CData3D<T>::ppData(),CData3D<T>::Sizeof()),"Malloc");
    else
        CData3D<T>::SetPData((T*)malloc(CData3D<T>::Sizeof()));

    return true;
}


/// \class CuDeviceData2D
/// \brief Cette classe est un tableau de donnee 2D situee dans memoire globale de la carte video
template <class T>
class CuDeviceData2D : public CData2D<T>
{

public:

    CuDeviceData2D(){}
    ~CuDeviceData2D(){}

    /// \brief  Initialise toutes les valeurs du tableau avec la valeur val
    /// \param  val : valeur d initialisation
    bool        Memset(int val);
    /// \brief  Copie toutes les valeurs du tableau dans un tableau du host
    /// \param  hostData : tableau destination
    bool        CopyDevicetoHost(T* hostData);

protected:

    bool        abDealloc() ;

    bool        abMalloc();

};

template <class T>
bool CuDeviceData2D<T>::CopyDevicetoHost( T* hostData )
{
    cudaError_t err = cudaMemcpy( hostData, CData2D<T>::pData(), CData2D<T>::Sizeof(), cudaMemcpyDeviceToHost);

    return CData<T>::ErrorOutput(err,"CopyDevicetoHost");
}

template <class T>
bool CuDeviceData2D<T>::Memset( int val )
{
    checkCudaErrors(cudaMemset( CData2D<T>::pData(), val, CData2D<T>::Sizeof()));
    return true;
}

template <class T>
bool CuDeviceData2D<T>::abDealloc()
{
    return (cudaFree(CData2D<T>::pData()) == cudaSuccess) ? true : false;
}

template <class T>
bool CuDeviceData2D<T>::abMalloc()
{
    return ErrorOutput(cudaMalloc((void **)CData2D<T>::ppData(), CData2D<T>::Sizeof()),"Malloc");
}

/*! \class CuDeviceData3D
 *  \brief Donnees structurees en 3 dimensions
 *
 *   La classe gere la memoire
 */


/// \class CuDeviceData3D
/// \brief Cette classe est un tableau de donnee 3D situee dans memoire globale de la carte video
template <class T>
class CuDeviceData3D : public CData3D<T>
{
public:

    CuDeviceData3D();
    CuDeviceData3D(uint2 dim,uint l, string name = "NoName");
    CuDeviceData3D(uint dim, string name = "NoName");
    ~CuDeviceData3D(){}


    /// \brief Initialise toutes les valeurs du tableau a val
    /// \param val : valeur d initialisation
    bool        Memset(int val);
    /// \brief Initialisation asynchrone de toutes les valeurs du tableau a val
    /// \param val : valeur d initialisation
    /// \param stream : flux cuda de gestion des appels asynchrone
    bool        MemsetAsync(int val, cudaStream_t stream );
    /// \brief  Copie toutes les valeurs du tableau dans un tableau du host
    /// \param  hostData : tableau destination
    bool        CopyDevicetoHost(T* hostData);

    bool        CopyDevicetoHost(CuHostData3D<T> &hostData);

    /// \brief  Copie toutes les valeurs d un tableau dans la structure de donnee de la classe (dans la memoire globale GPU)
    /// \param  hostData : tableau cible
    bool        CopyHostToDevice(T* hostData);
    /// \brief  Copie asynchrone de toutes les valeurs du tableau dans un tableau du host
    /// \param  hostData : tableau destination
    /// \param stream : flux cuda de gestion des appels asynchrone
    bool        CopyDevicetoHostASync(T* hostData, cudaStream_t stream = 0);

protected:

    bool        abDealloc() ;

    bool        abMalloc();

private:

    void        init(string name);

};

template <class T>
bool CuDeviceData3D<T>::CopyDevicetoHostASync( T* hostData, cudaStream_t stream )
{
    return CData<T>::ErrorOutput(cudaMemcpyAsync ( hostData, CData3D<T>::pData(), CData3D<T>::Sizeof(), cudaMemcpyDeviceToHost, stream),"CopyDevicetoHostASync");
}

template <class T>
bool CuDeviceData3D<T>::CopyDevicetoHost( T* hostData )
{
    return CData<T>::ErrorOutput(cudaMemcpy( hostData, CData3D<T>::pData(), CData3D<T>::Sizeof(), cudaMemcpyDeviceToHost),"CopyDevicetoHost");
}

template <class T>
bool CuDeviceData3D<T>::CopyDevicetoHost(CuHostData3D<T> &hostData)
{
    return CopyDevicetoHost(hostData.pData());
}

template <class T>
bool CuDeviceData3D<T>::CopyHostToDevice(T *hostData)
{
    return CData<T>::ErrorOutput(cudaMemcpy( CData3D<T>::pData(),hostData, CData3D<T>::Sizeof(), cudaMemcpyHostToDevice),"CopyHostToDevice");
}

template <class T>
bool CuDeviceData3D<T>::Memset( int val )
{
    if (CData<T>::GetSizeofMalloc() < CData3D<T>::Sizeof())
        std::cout << "Memset : Allocation trop petite !!!" << "\n";

    return CData<T>::ErrorOutput(cudaMemset( CData3D<T>::pData(), val, CData3D<T>::Sizeof()),"Memset");
}

template <class T>
bool CuDeviceData3D<T>::MemsetAsync(int val, cudaStream_t stream)
{
    if (CData<T>::GetSizeofMalloc() < CData3D<T>::Sizeof())
        std::cout << "MemsetAsync : Allocation trop petite !!!" << "\n";

    return CData<T>::ErrorOutput(cudaMemsetAsync(CData3D<T>::pData(), val, CData3D<T>::Sizeof(), stream ),"MemsetAsync");
}

template <class T>
CuDeviceData3D<T>::CuDeviceData3D()
{
    init("No Name");
}

template <class T>
CuDeviceData3D<T>::CuDeviceData3D(uint2 dim, uint l, string name)
{
    init(name);
    CData3D<T>::Malloc(dim,l);
}

template <class T>
CuDeviceData3D<T>::CuDeviceData3D(uint dim, string name)
{
    init(name);
    CData3D<T>::Realloc(make_uint2(dim,1),1);
}

template <class T>
bool CuDeviceData3D<T>::abDealloc()
{
    return (cudaFree(CData<T>::pData()) == cudaSuccess) ? true : false;
}

template <class T>
bool CuDeviceData3D<T>::abMalloc()
{
    return CData<T>::ErrorOutput(cudaMalloc((void **)CData3D<T>::ppData(), CData3D<T>::Sizeof()),"Malloc");
}

template <class T>
void CuDeviceData3D<T>::init(string name)
{
    CGObject::SetType("CuDeviceData3D");
    CGObject::SetName(name);
}

/// \class  AImageCuda
/// \brief Cette classe abstraite est une image directement liable a une texture GpGpu
class AImageCuda : virtual public CData<cudaArray>
{
public:

    AImageCuda(){}
    ~AImageCuda(){}

    /// \brief  Lie l image a une texture Gpu
    /// \param  texRef : reference de la texture a lier
    bool		bindTexture(textureReference& texRef);
    /// \brief  renvoie le tableau cuda contenant les valeurs de l'image
    cudaArray*	GetCudaArray();

    /// \brief  Initialisation de toutes les valeurs du tableau a val
    /// \param  val : valeur d initialisation
    bool		Memset(int val);

protected:

    bool        abDealloc() ;

    virtual bool  abMalloc() = 0;

};


/// \class  ImageCuda
/// \brief Cette classe est une image 2D directement liable a une texture GpGpu
template <class T>
class ImageCuda : public CData2D<cudaArray>, virtual public AImageCuda
{

public:

    ImageCuda();
    ~ImageCuda(){}
    /// \brief Initialise la dimension et les valeurs de l image
    /// \param dimension : la dimension d initialisation
    /// \param data : Donnees a copier dans l image
    bool	InitImage(uint2 dimension, T* data);

    /// \brief Initialise les valeurs de l image avec un tableau de valeur du Host
    /// \param data : Donnees cible a copier
    bool	copyHostToDevice(T* data);
    /// \brief Initialise les valeurs de l image a val
    /// \param val : Valeur d initialisation
    bool	Memset(int val){return AImageCuda::Memset(val);}
    //bool	Dealloc(){return AImageCuda::Dealloc();}
    /// \brief Sortie console de la classe
    void	OutputInfo(){CData2D::OutputInfo();}

protected:

    bool    abDealloc(){return AImageCuda::abDealloc();}

    bool    abMalloc();

    uint	Sizeof(){ return CData2D::Sizeof() * sizeof(T);}

private:

    T*		_ClassData;

};

template <class T>
ImageCuda<T>::ImageCuda()
{
    CData2D::SetType("ImageLayeredCuda");
    CData2D::ClassTemplate(CData2D::ClassTemplate() + " " + CData2D::StringClass<T>(_ClassData));
}

template <class T>
bool ImageCuda<T>::copyHostToDevice( T* data )
{
    // Copie des données du Host dans le tableau Cuda
    return CData2D::ErrorOutput(cudaMemcpyToArray(AImageCuda::pData(), 0, 0, data, sizeof(T)*size(GetDimension()), cudaMemcpyHostToDevice),"copyHostToDevice");
}

template <class T>
bool ImageCuda<T>::abMalloc()
{
    cudaChannelFormatDesc channelDesc =  cudaCreateChannelDesc<T>();
    return CData2D::ErrorOutput(cudaMallocArray(AImageCuda::ppData(),&channelDesc,struct2D::GetDimension().x,struct2D::GetDimension().y),"Malloc");
}

template <class T>
bool    ImageCuda<T>::InitImage(uint2 dimension, T* data)
{
    SetDimension(dimension);
    CData<T>::Malloc();
    return copyHostToDevice(data);
}

//-----------------------------------------------------------------------------------------------
//									CLASS IMAGE LAYARED CUDA
//-----------------------------------------------------------------------------------------------

/// \class ImageLayeredCuda
/// \brief Cette classe est une pile d'image 2D directement liable a une texture GpGpu
template <class T>
class ImageLayeredCuda : public CData3D<cudaArray>, virtual public AImageCuda
{

public:

    ImageLayeredCuda();
    ~ImageLayeredCuda(){}

    /// \brief Initialise les valeurs des images a val
    /// \param val : Valeur d initialisation
    bool	Memset(int val){return AImageCuda::Memset(val);}

    /// \brief Copie des valeurs des images avec un tableau 3D de valeur du Host
    /// \param data : Donnees cible a copier
    bool	copyHostToDevice(T* data);
    /// \brief Copie des valeurs des images vers un tableau 3D du Host
    /// \param data : tableau de destination
    bool	copyDeviceToDevice(T* data);
    /// \brief Copie asynchrone des valeurs des images avec un tableau 3D de valeur du Host
    /// \param data : Donnees cible a copier
    /// \param stream : flux cuda
    bool	copyHostToDeviceASync(T* data, cudaStream_t stream = 0);
    /// \brief Sortie console de la classe
    void	OutputInfo(){CData3D::OutputInfo();}

protected:

    bool    abDealloc(){return AImageCuda::abDealloc();}

    bool    abMalloc();

    uint	Sizeof(){ return CData3D::Sizeof() * sizeof(T);}

private:

    T*	_ClassData;

    cudaMemcpy3DParms CudaMemcpy3DParms(T *data, cudaMemcpyKind kind);

    cudaExtent        CudaExtent();
};

template <class T>
ImageLayeredCuda<T>::ImageLayeredCuda()
{
    CData3D::SetType("ImageLayeredCuda");
    CData3D::ClassTemplate(CData3D::ClassTemplate() + " " + CData3D::StringClass<T>(_ClassData));
}

template <class T>
cudaMemcpy3DParms ImageLayeredCuda<T>::CudaMemcpy3DParms(T *data, cudaMemcpyKind kind)
{
    cudaExtent sizeImgsLay      = CudaExtent();

    // Déclaration des parametres de copie 3D
    cudaMemcpy3DParms	p		= { 0 };
    cudaPitchedPtr		pitch	= make_cudaPitchedPtr(data, sizeImgsLay.width * sizeof(T), sizeImgsLay.width, sizeImgsLay.height);

    p.dstArray	= AImageCuda::GetCudaArray();   // Pointeur du tableau de destination
    p.srcPtr	= pitch;                        // Pitch
    p.extent	= sizeImgsLay;                  // Taille du cube
    p.kind      = kind;                         // Type de copie

    return p;
}

template <class T>
cudaExtent ImageLayeredCuda<T>::CudaExtent()
{
    return make_cudaExtent( CData3D::GetDimension().x, CData3D::GetDimension().y, CData3D::GetNbLayer());
}

template <class T>
bool ImageLayeredCuda<T>::copyHostToDevice( T* data )
{
    cudaMemcpy3DParms	p = CudaMemcpy3DParms(data,cudaMemcpyHostToDevice);
    // Copie des images du Host vers le Device
    return CData3D::ErrorOutput(cudaMemcpy3D(&p),"copyHostToDevice") ;
}

template <class T>
bool ImageLayeredCuda<T>::copyDeviceToDevice(T *data)
{
    cudaMemcpy3DParms	p = CudaMemcpy3DParms(data,cudaMemcpyDeviceToDevice);

    // Copie des images du Host vers le Device
    return CData3D::ErrorOutput(cudaMemcpy3D(&p),"copyDeviceToDevice") ;
}

template <class T>
bool ImageLayeredCuda<T>::copyHostToDeviceASync( T* data, cudaStream_t stream /*= 0*/ )
{
    cudaMemcpy3DParms	p = CudaMemcpy3DParms(data,cudaMemcpyHostToDevice);

    // Copie des images du Host vers le Device
    return CData3D::ErrorOutput( cudaMemcpy3DAsync (&p, stream),"copyHostToDeviceASync");
}

template <class T>
bool ImageLayeredCuda<T>::abMalloc()
{
    cudaChannelFormatDesc channelDesc =	cudaCreateChannelDesc<T>();
    // Allocation memoire GPU du tableau des calques d'images
    return CData3D::ErrorOutput(cudaMalloc3DArray(AImageCuda::ppData(),&channelDesc,CudaExtent(),cudaArrayLayered),"Malloc");
}


#endif //GPGPU_DATA_H
