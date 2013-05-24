#ifndef GPGPUTOOLS_H
#define GPGPUTOOLS_H

#include "helper_math_extented.cuh"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <sstream>     // for ostringstream
#include <string>
#include <iostream>
#include <limits>
#ifdef _WIN32
#include <Lmcons.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <cmath>
#endif

using namespace std;
typedef unsigned char pixel;

#define NOPAGELOCKEDMEMORY false


#define DISPLAYOUTPUT
#define TexFloat2Layered texture<float2,cudaTextureType2DLayered>

enum Plans {XY,XZ,YZ,YX,ZX,ZY};

template<class T> class CuHostData3D;

/// \class GpGpuTools
/// \brief classe d outils divers
/// La classe gere la restructuration de donnees, des outils d'affichages console
class GpGpuTools
{

public:

    GpGpuTools(){}

    ~GpGpuTools(){}


    ///  \brief         Convertir array 2D en tableau lineaire
    template <class T>
    static void			Memcpy2Dto1D(T** dataImage2D, T* dataImage1D, uint2 dimDest, uint2 dimSource);

    ///  \brief         Sauvegarder tableau de valeur dans un fichier PGN
    ///  \param         dataImage : Donnees images a ecrire
    ///  \param         fileName : nom du fichier a ecrire
    ///  \param         dimImage : dimension de l image
    ///  \return        true si l ecriture reussie
    template <class T>
    static bool			Array1DtoImageFile(T* dataImage,const char* fileName, uint2 dimImage);

    ///  \brief			Sauvegarder tableau de valeur (multiplier par un facteur) dans un fichier PGN
    ///  \param         dataImage : Donnees images a ecrire
    ///  \param         fileName : nom du fichier a ecrire
    ///  \param         dimImage : dimension de l image
    ///  \param         factor : facteur multiplicatif
    ///  \return        true si l ecriture reussie
    template <class T>
    static bool			Array1DtoImageFile(T* dataImage,const char* fileName, uint2 dimImage, float factor );

    ///  \brief			Retourne la dossier image de l'utilisateur
    ///  \return        renvoie un string
    static std::string	GetImagesFolder();

    ///  \brief			Divise toutes les valeurs du tableau par un facteur
    ///  \param         data : Donnees images a ecrire
    ///  \param         dimImage : dimension du tableau
    ///  \param         factor : facteur multiplicatif
    ///  \return        renvoie un pointeur sur le tableau resultant
    template <class T>
    static T*			MultArray(T* data, uint2 dimImage, float factor);

    ///	\brief			Sortie console d'une donnees
    ///  \param         data : Donnees du tableau a afficher
    ///  \param         dim : dimension du tableau
    ///  \param         offset : nombre de chiffre apres la virgule
    ///  \param         defaut : valeur affichee par un caractere speciale
    ///  \param         sample : saut dans l'affichage
    ///  \param         factor : facteur multiplicatif
    ///  \return        renvoie un pointeur sur le tableau resultant


    ///	\brief			Obtenir la valeur dans un tableau en fonction de ses coordonnees
    template <class T>
    static T			GetArrayValue(T* data, uint3 pt, uint3 dim);


    template <class T>
    static void			OutputArray(T* data, uint3 dim, uint plan = XY, uint level = 0, Rect rect = NEGARECT, uint offset = 3, T defaut = (T)0.0f, float sample = 1.0f, float factor = 1.0f);

    ///	\brief			Sortie console d'un tableau de donnees host cuda
    ///  \param         data : tableau host cuda
    ///  \param         Z : profondeur du tableau a afficher
    ///  \param         offset : nombre de chiffre apres la virgule
    ///  \param         defaut : valeur affichee par un caractere speciale
    ///  \param         sample : saut dans l'affichage
    ///  \param         factor : facteur multiplicatif
    ///  \return        renvoie un pointeur sur le tableau resultant
    template <class T>
    static void			OutputArray(CuHostData3D<T> &data, uint Z = 0, uint offset = 3, T defaut = (T)0.0f, float sample = 1.0f, float factor = 1.0f);

    ///	\brief			Sortie console formater d'une valeur
    /// \param          value : valeur a afficher
    ///  \param         offset : nombre de chiffre apres la virgule
    ///  \param         defaut : valeur affichee par un caractere speciale
    ///  \param         factor : facteur multiplicatif
    template <class T>
    static void			OutputValue(T value, uint offset = 3, T defaut = (T)0.0f, float factor = 1.0f);

    ///	\brief			Retour chariot
    static void			OutputReturn(char * out /*= ""*/);

    ///	\brief			multiplie par un facteur
    static float		fValue( float value,float factor );

    ///	\brief			multiplie par un facteur
    static float		fValue( float2 value,float factor );

    ///	\brief			Convertie un uint2 en string
    static std::string	toStr(uint2 tt);

    ///	\brief			Affiche les parametres GpGpu de correlation multi-images
    static void			OutputInfoGpuMemory();

    ///	\brief			(X)
    static void			OutputGpu();

};

template <class T>
void GpGpuTools::Memcpy2Dto1D( T** dataImage2D, T* dataImage1D, uint2 dimDest, uint2 dimSource )
{
    for (uint j = 0; j < dimSource.y ; j++)
        memcpy(  dataImage1D + dimDest.x * j , dataImage2D[j],  dimSource.x * sizeof(T));
}

template <class T>
void GpGpuTools::OutputValue( T value, uint offset, T defaut, float factor)
{
#ifndef DISPLAYOUTPUT
    return;
#endif



    std::string S2	= "    ";
    std::string ES	= "";
    std::string S1	= " ";

    float outO	= fValue((float)value,factor);
    float p		= pow(10.0f,(float)(offset-1));
    if(p < 1.0f ) p = 1.0f;
    float out	= floor(outO*p)/p;

    std::string valS;
    stringstream sValS (stringstream::in | stringstream::out);

    sValS << abs(out);
    long sizeV = (long)sValS.str().length();

    if (sizeV == 5) ES = ES + "";
    else if (sizeV == 4) ES = ES + " ";
    else if (sizeV == 3) ES = ES + "  ";
    else if (sizeV == 2) ES = ES + "   ";
    else if (sizeV == 1) ES = ES + "    ";

    if (outO == 0.0f)
        std::cout << S1 << "0" << S2;
    else if (outO == defaut)
        std::cout << S1 << "!" + S2;
    else if (outO == -1000.0f)
        std::cout << S1 << "." << S2;
    else if (outO == 2*defaut)
        std::cout << S1 << "s" << S2;
    else if (outO == 3*defaut)
        std::cout << S1 << "z" << S2;
    else if (outO == 4*defaut)
        std::cout << S1 << "s" << S2;
    else if (outO == 5*defaut)
        std::cout << S1 << "v" << S2;
    else if (outO == 6*defaut)
        std::cout << S1 << "e" << S2;
    else if (outO == 7*defaut)
        std::cout << S1 << "c" << S2;
    else if (outO == 8*defaut)
        std::cout << S1 << "?" << S2;
    else if (outO == 9*defaut)
        std::cout << S1 << "¤" << S2;
    else if ( outO < 0.0f)
        std::cout << out << ES;
    else
        std::cout << S1 << out << ES;

}

template<> inline void GpGpuTools::OutputValue( short2 value, uint offset, short2 defaut, float factor)
{
    std::cout << "[" << value.x << "," << value.y << "]";
}

template <class T>
T GpGpuTools::GetArrayValue(T* data, uint3 pt, uint3 dim)
{
    return data[to1D(pt,dim)];
}


template <class T>
void GpGpuTools::OutputArray(T* data, uint3 dim, uint plan, uint level, Rect rect, uint offset, T defaut, float sample, float factor )
{
#ifndef DISPLAYOUTPUT
    return;
#endif
    if(rect == NEGARECT)
    {
        rect.pt0 = make_int2(0,0);
        switch (plan) {
        case XY:
            rect.pt1.x = dim.x;
            rect.pt1.y = dim.y;
            break;
        case XZ:
            rect.pt1.x = dim.x;
            rect.pt1.y = dim.z;
            break;
        case YZ:
            rect.pt1.x = dim.y;
            rect.pt1.y = dim.z;
            break;
        case YX:
            rect.pt1.x = dim.y;
            rect.pt1.y = dim.x;
            break;
        case ZX:
            rect.pt1.x = dim.z;
            rect.pt1.y = dim.x;
            break;
        case ZY:
            rect.pt1.x = dim.z;
            rect.pt1.y = dim.y;
            break;
        default:
            break;
        }
    }

    uint2 p;

    for (p.y = (uint)rect.pt0.y ; p.y < (uint)rect.pt1.y; p.y+= (int)sample)
    {
        for (p.x = (uint)rect.pt0.x; p.x < (uint)rect.pt1.x ; p.x+= (int)sample)
        {
            T value;
            switch (plan) {
            case XY:
                value = GetArrayValue(data,make_uint3(p.x,p.y,level),dim);
                break;
            case XZ:
                value = GetArrayValue(data,make_uint3(p.x,level,p.y),dim);
                break;
            case YZ:
                value = GetArrayValue(data,make_uint3(level,p.x,p.y),dim);
                break;
            case YX:
                value = GetArrayValue(data,make_uint3(p.y,p.x,level),dim);
                break;
            case ZX:
                value = GetArrayValue(data,make_uint3(p.y,level,p.x),dim);
                break;
            case ZY:
                value = GetArrayValue(data,make_uint3(level,p.y,p.x),dim);
                break;
            default:
                value = defaut;
                break;
            }

            OutputValue(value,offset,defaut,factor);
        }
        std::cout << "\n";
    }
    std::cout << "==================================================================================\n";
}	


template <class T>
static void OutputArray(CuHostData3D<T> &data, uint Z, uint offset, float defaut, float sample, float factor)
{

    OutputArray(data.pData() + Z * Sizeof(data.Dimension()),data.Dimension(),offset, defaut, sample, factor );

}

template <class T>
T* GpGpuTools::MultArray( T* data, uint2 dim, float factor )
{
    if (factor == 0) return NULL;

    int sizeData = size(dim);

    T* image = new T[sizeData];

    for (int i = 0; i < sizeData ; i++)
        image[i] = data[i] * (T)factor;

    return image;

}

template <class T>
bool GpGpuTools::Array1DtoImageFile( T* dataImage,const char* fileName, uint2 dimImage )
{
    std::string pathfileImage = std::string(GetImagesFolder()) + std::string(fileName);

    std::cout << pathfileImage << "\n";
    return sdkSavePGM<T>(pathfileImage.c_str(), dataImage, dimImage.x,dimImage.y);
}

template <class T>
bool GpGpuTools::Array1DtoImageFile(T* dataImage,const char* fileName, uint2 dimImage, float factor)
{
    T* image = MultArray(dataImage, dimImage, factor);

    bool r = Array1DtoImageFile( image, fileName, dimImage );

    delete[] image;

    return r;
}

//-----------------------------------------------------------------------------------------------
//									CLASS CUDA
//-----------------------------------------------------------------------------------------------

/// \class CGObject
/// \brief Classe de gestion des types
class CGObject
{
public:

    CGObject();
    ~CGObject();

    /// \brief  renvoie les caracteristiques de l objets en string
    std::string Id();
    /// \brief  renvoie le nom de l objet en string
    std::string	Name();
    /// \brief  affecte le nom
    /// \param  name : le nom a affecte
    void		SetName(std::string name);
    /// \brief  renvoie le type de l objet en string
    std::string	Type();
    /// \brief  affecte le type de l objet
    void		SetType(std::string type);
    /// \brief  renvoie la classe du template de l objet en string
    std::string	ClassTemplate();
    /// \brief  Affecte la classe du template de l objet
    void		ClassTemplate(std::string classTemplate);

    /// \brief  renvoie la classe T en string
    template<class T>
    const char* StringClass(T* tt){ return "T";}

private:

    std::string _name;
    std::string _type;
    std::string _classTemplate;

};
/// \brief  renvoie la classe float en char*
template<> inline const char* CGObject::StringClass( float* t ){return "float*";}
/// \brief  renvoie la classe pixel en char*
template<> inline const char* CGObject::StringClass( pixel* t ){return "pixel*";}
/// \brief  renvoie la classe uint en char*
template<> inline const char* CGObject::StringClass( uint* t ){	return "uint*";}
/// \brief  renvoie la classe float en char*
template<> inline const char* CGObject::StringClass(struct float2* t ){	return "float2*";}
/// \brief  renvoie la classe cudaArray en char*
template<> inline const char* CGObject::StringClass(cudaArray* t ){	return "cudaArray*";}


/// \class struct2D
/// \brief classe structure de donnees de dimension 2
class struct2D
{
public:

    struct2D(){}
    ~struct2D(){}
    /// \brief  Renvoie la dimension de la structure 2D
    uint2		GetDimension();
    /// \brief  Initialise la dimension de la structure 2D
    /// \param  dimension : Dimension d initialisation
    uint2		SetDimension(uint2 dimension);
    /// \brief  Initialise la dimension de la structure 2D
    /// \param  dimX : Dimension X d initialisation
    /// \param  dimY : Dimension Y d initialisation
    uint2		SetDimension(int dimX,int dimY);
    /// \brief  Initialise la dimension de la structure 2D
    /// \param  dimX : Dimension X d initialisation
    /// \param  dimY : Dimension Y d initialisation
    uint2		SetDimension(uint dimX,uint dimY);
    /// \brief  Renvoie le nombre d elements de la structure
    uint		GetSize();
    /// \brief  Sortie console de la structure
    void		Output();

private:

    uint2		_dimension;

};


/// \class struct2DLayered
/// \brief classe pile de tableau 2D d elements
class struct2DLayered : public struct2D
{

public:

    struct2DLayered();
    ~struct2DLayered(){}
    /// \brief Renvoie le nombre de tableau 2D
    uint        GetNbLayer();
    /// \brief Initialise le nombre de tableau 2D
    void        SetNbLayer(uint nbLayer);
    /// \brief  Initialise la dimension de la structure 2D et le nombre de tableau
    /// \param  dimension : Dimension d initialisation de la structure 2D
    /// \param  nbLayer : nombre de tableau
    void        SetDimension(uint2 dimension, uint nbLayer);
    /// \brief  Initialise la dimension de la structure 2D et le nombre de tableau
    /// \param  dimension : Dimension d initialisation de la structure 3D
    void        SetDimension(uint3 dimension);
    /// \brief  Renvoie la dimension de la structure 3D
    uint3       GetDimension3D();

    uint        GetSize();

    void        Output();

private:

    uint _nbLayers;
};


/// \class CData
/// \brief Classe Abstraite de donnees
template <class T> 
class CData : public CGObject
{

public:

    CData();
    ~CData(){}
    /// \brief      Allocation memoire
    virtual bool	Malloc()		= 0;
    /// \brief      Initialise toutes les elements avec la valeur val
    /// \param      val : valeur d initialisation
    virtual bool	Memset(int val) = 0;
    /// \brief      Desalloue la memoire alloue
    virtual bool	Dealloc()		= 0;
    /// \brief      Sortie console de la classe
    virtual void	OutputInfo()	= 0;
    /// \brief      Renvoie le pointeur des donnees
    T*              pData();
    /// \brief      Init le pointeur des donnees
    void            SetPData(T *p);
    /// \brief      Renvoie le pointeur du pointeur des donnees
    T**             ppData();
    /// \brief      Sortie console des erreurs Cuda
    /// \param      err :  erreur cuda rencontree
    /// \param      fonctionName : nom de la fonction ou se trouve l erreur
    virtual bool	ErrorOutput(cudaError_t err,const char* fonctionName);
    /// \brief      Sortie consolle de l allocation memoire globale Gpu
    void            MallocInfo();
    /// \brief      Obtenir une valeur aleatoire comprise entre min et max
    static T        GetRandomValue(T min, T max);
    /// \brief      Renvoie la taille de la memoire alloue
    uint            GetSizeofMalloc();

protected:

    /// \brief      Initialise a NULL le pointeur des donnees
    void            dataNULL();
    /// \brief      Renvoie True si le pointeur des donnees est NULL
    bool            isNULL();
    /// \brief      Ajout de memoire alloue
    void            AddMemoryOc(uint m);
    /// \brief      Suppression de memoire alloue
    void            SubMemoryOc(uint m);
    /// \brief      Initialise la taille de la memoire alloue
    /// \param      sizeofmalloc : Taille de l allocation
    void            SetSizeofMalloc(uint sizeofmalloc);

private:

    uint            _memoryOc;
    T*              _data;
    uint            _sizeofMalloc;

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
    _memoryOc(0)
{
    dataNULL();
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

    CData2D();
    /// \brief      constructeur avec initialisation de la dimension de la structure
    /// \param      dim : Dimension a initialiser
    CData2D(uint2 dim);
    ~CData2D(){}
    /// \brief Alloue la memoire neccessaire
    virtual bool	Malloc()        = 0;
    /// \brief      Initialise les elements des images a val
    /// \param      val : Valeur d initialisation
    virtual bool	Memset(int val) = 0;
    virtual bool	Dealloc()       = 0;
    void			OutputInfo();
    /// \brief       Allocation memoire pour les tous les elements de la structures avec initialisation de la dimension de la structure
    /// \param      dim : Dimension 2D a initialiser
    bool			Malloc(uint2 dim);
    /// \brief      Desallocation puis re-allocation memoire pour les tous les elements
    ///             de la structures avec initialisation de la dimension
    /// \param      dim : Dimension 2D a initialiser
    bool			Realloc(uint2 dim);
    /// \brief      Nombre d elements de la structure
    uint			Sizeof();

};

template <class T>
void CData2D<T>::OutputInfo()
{
    std::cout << "Structure 2D : \n";
    struct2D::Output();
}

template <class T>
CData2D<T>::CData2D()
{
    CGObject::ClassTemplate(CGObject::StringClass<T>(CData2D::pData()));
}
template <class T>
CData2D<T>::CData2D(uint2 dim)
{
    CGObject::ClassTemplate(CGObject::StringClass<T>(CData2D::pData()));
    Realloc(dim);
}

template <class T>
uint CData2D<T>::Sizeof()
{
    return sizeof(T) * struct2D::GetSize();
}

template <class T>
bool CData2D<T>::Realloc( uint2 dim )
{
    Dealloc();
    Malloc(dim);
    return true;
}

template <class T>
bool CData2D<T>::Malloc( uint2 dim )
{
    SetDimension(dim);
    Malloc();
    return true;
}

/// \class CData3D
/// \brief Classe abstraite d un tableau d elements structuree en trois dimensions
template <class T> 
class CData3D : public struct2DLayered, public CData<T>
{
public:

    CData3D();
    /// \brief constructeur avec initialisation de la dimension de la structure
    /// \param dim : Dimension 2D a initialiser
    /// \param l : Taille de la 3eme dimension
    CData3D(uint2 dim, uint l);
    ~CData3D(){}
    /// \brief      Allocation memoire pour les tous les elements de la structures
    virtual bool	Malloc() = 0;
    /// \brief      Initialise toutes les elements avec la valeur val
    /// \param      val : valeur d initialisation
    virtual bool	Memset(int val) = 0;
    /// \brief      Desalloue la memoire alloue
    virtual bool	Dealloc() = 0;

    void			OutputInfo();
    /// \brief      Allocation memoire pour les tous les elements de la structures avec initialisation de la dimension de la structure
    /// \param      dim : Dimension 2D a initialiser
    /// \param      l : Taille de la 3eme dimension
    bool			Malloc(uint2 dim, uint l);
    /// \brief      Desallocation puis re-allocation memoire pour les tous les elements de la structures avec initialisation de la dimension de la structure
    /// \param      dim : Dimension 2D a initialiser
    /// \param      l : Taille de la 3eme dimension
    bool			Realloc(uint2 dim, uint l);
    /// \brief      Nombre d elements de la structure
    uint			Sizeof();

    T&              operator[](uint2 pt);
    T&              operator[](uint3 pt);
    T&              operator[](uint pt1D);
    T&              operator[](int pt1D);

};

template <class T>
void CData3D<T>::OutputInfo()
{

    std::cout << "Structure 3D        : " << CGObject::Id() << "\n";
    struct2DLayered::Output();
    std::cout << "\n";
}

template <class T>
CData3D<T>::CData3D()
{
    CGObject::ClassTemplate(CGObject::StringClass<T>(CData3D::pData()));
}

template <class T>
CData3D<T>::CData3D( uint2 dim, uint l )
{
    if(!Malloc(dim,l))
        std::cout << "ERROR -> CData3D( uint2 dim, uint l )\n";
}

template <class T>
bool CData3D<T>::Malloc( uint2 dim, uint l )
{
    SetDimension(dim,l);
    Malloc();
    return true;
}

template <class T>
bool CData3D<T>::Realloc( uint2 dim, uint l )
{
    bool dB = Dealloc();
    bool dM = Malloc(dim,l);
    return (dB && dM);
}

template <class T>
uint CData3D<T>::Sizeof()
{
    return GetSize() * sizeof(T);
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

    CuHostData3D(bool pageLockedmemory);

    /// \brief constructeur avec initialisation de la dimension de la structure
    /// \param dim : Dimension 2D a initialiser
    /// \param l : Taille de la 3eme dimension
    CuHostData3D(bool pageLockedmemory,uint2 dim, uint l = 1);

    /// \brief constructeur avec initialisation de la dimension de la structure
    /// \param dim : Dimension 3D a initialiser
    CuHostData3D(bool pageLockedmemory,uint3 dim);

    ~CuHostData3D(){}

    bool Dealloc();

    bool Malloc();

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

private:

    bool _pageLockedmemory;

};

template <class T>
CuHostData3D<T>::CuHostData3D(bool pageLockedmemory):
    _pageLockedmemory(pageLockedmemory)
{
    CData<T>::SetSizeofMalloc(0);
    CGObject::SetType("CuHostData3D");
}

template <class T>
CuHostData3D<T>::CuHostData3D(bool pageLockedmemory, uint2 dim, uint l ):
    _pageLockedmemory(pageLockedmemory)
{
    CData<T>::SetSizeofMalloc(0);
    CGObject::SetType("CuHostData3D");
    CData3D<T>::Realloc(dim,l);
}

template <class T>
CuHostData3D<T>::CuHostData3D(bool pageLockedmemory, uint3 dim):
    _pageLockedmemory(pageLockedmemory)
{
    CData<T>::SetSizeofMalloc(0);
    CGObject::SetType("CuHostData3D");
    CData3D<T>::Realloc(make_uint2(dim.x,dim.y),dim.z);
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
    T mod = abs(max - min);
    srand (time(NULL));
    for(int i=0;i<CData3D<T>::GetSize();i++)
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
bool CuHostData3D<T>::Malloc()
{
    CData3D<T>::SetSizeofMalloc(CData3D<T>::Sizeof());
    CData3D<T>::AddMemoryOc(CData3D<T>::GetSizeofMalloc());
    if(_pageLockedmemory)
        return CData<T>::ErrorOutput(cudaMallocHost(CData3D<T>::ppData(),CData3D<T>::Sizeof()),"Malloc");
    else
        CData3D<T>::SetPData((T*)malloc(CData3D<T>::Sizeof()));

    return true;
}

template <class T>
bool CuHostData3D<T>::Dealloc()
{
    CData3D<T>::SubMemoryOc(CData3D<T>::GetSizeofMalloc());
    CData3D<T>::SetSizeofMalloc(0);
    if(_pageLockedmemory)
        return CData<T>::ErrorOutput(cudaFreeHost(CData3D<T>::pData()),"Dealloc");
    else
        free(CData3D<T>::pData());
    return true;
}

/// \class CuDeviceData2D
/// \brief Cette classe est un tableau de donnee 2D situee dans memoire globale de la carte video
template <class T> 
class CuDeviceData2D : public CData2D<T> 
{

public:

    CuDeviceData2D();
    ~CuDeviceData2D(){}
    /// \brief  Desalloue la memoire globale alloue a ce tableau
    bool        Dealloc();
    /// \brief  Alloue la memoire globale pour ce tableau en fonction de sa dimension
    bool        Malloc();
    /// \brief  Initialise toutes les valeurs du tableau avec la valeur val
    /// \param  val : valeur d initialisation
    bool        Memset(int val);
    /// \brief  Copie toutes les valeurs du tableau dans un tableau du host
    /// \param  hostData : tableau destination
    bool        CopyDevicetoHost(T* hostData);
    //void CopyDevicetoHostASync(T* hostData, cudaStream_t stream = 0);

};

template <class T>
CuDeviceData2D<T>::CuDeviceData2D()
{
    CData2D<T>::dataNULL();
}

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
bool CuDeviceData2D<T>::Malloc()
{
    SetSizeofMalloc(CData2D<T>::Sizeof());
    CData2D<T>::AddMemoryOc(CData2D<T>::GetSizeofMalloc());
    return ErrorOutput(cudaMalloc((void **)CData2D<T>::ppData(), CData2D<T>::Sizeof()),"Malloc");
}

template <class T>
bool CuDeviceData2D<T>::Dealloc()
{
    cudaError_t erC = cudaSuccess;
    CData2D<T>::SubMemoryOc(CData2D<T>::GetSizeofMalloc());
    CData2D<T>::SetSizeofMalloc(0);
    if (!CData2D<T>::isNULL()) erC = cudaFree(CData2D<T>::pData());
    CData2D<T>::dataNULL();
    return erC == cudaSuccess ? true : false;
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
    ~CuDeviceData3D(){}
    /// \brief Desallocation memoire globale
    /// \return true si la desallocation a reussie false sinon
    bool        Dealloc();
    /// \brief Allocation de memoire globale
    bool        Malloc();
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
    /// \brief  Copie toutes les valeurs d un tableau dans la structure de donnee de la classe (dans la memoire globale GPU)
    /// \param  hostData : tableau cible
    bool        CopyHostToDevice(T* hostData);
    /// \brief  Copie asynchrone de toutes les valeurs du tableau dans un tableau du host
    /// \param  hostData : tableau destination
    /// \param stream : flux cuda de gestion des appels asynchrone
    bool        CopyDevicetoHostASync(T* hostData, cudaStream_t stream = 0);

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
bool CuDeviceData3D<T>::CopyHostToDevice(T *hostData)
{
    return CData<T>::ErrorOutput(cudaMemcpy( CData3D<T>::pData(),hostData, CData3D<T>::Sizeof(), cudaMemcpyHostToDevice),"CopyHostToDevice");
}

template <class T>
bool CuDeviceData3D<T>::Memset( int val )
{
    if (CData<T>::GetSizeofMalloc() < CData3D<T>::Sizeof())
        std::cout << "Allocation trop petite !!!" << "\n";

    return CData<T>::ErrorOutput(cudaMemset( CData3D<T>::pData(), val, CData3D<T>::Sizeof()),"Memset");
}

template <class T>
bool CuDeviceData3D<T>::MemsetAsync(int val, cudaStream_t stream)
{
    if (CData<T>::GetSizeofMalloc() < CData3D<T>::Sizeof())
        std::cout << "Allocation trop petite !!!" << "\n";

    return CData<T>::ErrorOutput(cudaMemsetAsync(CData3D<T>::pData(), val, CData3D<T>::Sizeof(), stream ),"MemsetAsync");
}

template <class T>
CuDeviceData3D<T>::CuDeviceData3D()
{
    CData3D<T>::dataNULL();
    CGObject::SetType("CuDeviceData3D");
}

template <class T>
CuDeviceData3D<T>::CuDeviceData3D(uint2 dim, uint l, string name)
{
    CData3D<T>::dataNULL();
    CGObject::SetType(name);
    CData3D<T>::Realloc(dim,l);
}

template <class T>
bool CuDeviceData3D<T>::Malloc()
{
    CData<T>::SetSizeofMalloc(CData3D<T>::Sizeof());
    CData<T>::AddMemoryOc(CData3D<T>::GetSizeofMalloc());
    return CData<T>::ErrorOutput(cudaMalloc((void **)CData3D<T>::ppData(), CData3D<T>::Sizeof()),"Malloc");
}

template <class T>
bool CuDeviceData3D<T>::Dealloc()
{
    cudaError_t erC = cudaSuccess;
    CData<T>::SubMemoryOc(CData3D<T>::GetSizeofMalloc());
    CData3D<T>::SetSizeofMalloc(0);
    if (!CData3D<T>::isNULL()) erC = cudaFree(CData3D<T>::pData());
    CData3D<T>::dataNULL();
    return erC == cudaSuccess ? true : false;
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
    /// \brief  Desalloue la memoire globale
    bool		Dealloc();
    /// \brief  Initialisation de toutes les valeurs du tableau a val
    /// \param  val : valeur d initialisation
    bool		Memset(int val);

};


/// \class  ImageCuda
/// \brief Cette classe est une image 2D directement liable a une texture GpGpu
template <class T> 
class ImageCuda : public CData2D<cudaArray>, public AImageCuda
{

public:

    ImageCuda();
    ~ImageCuda(){}
    /// \brief Initialise la dimension et les valeurs de l image
    /// \param dimension : la dimension d initialisation
    /// \param data : Donnees a copier dans l image
    bool	InitImage(uint2 dimension, T* data);
    /// \brief Alloue la memoire globale neccessaire
    bool	Malloc();
    /// \brief Initialise les valeurs de l image avec un tableau de valeur du Host
    /// \param data : Donnees cible a copier
    bool	copyHostToDevice(T* data);
    /// \brief Initialise les valeurs de l image a val
    /// \param val : Valeur d initialisation
    bool	Memset(int val){return AImageCuda::Memset(val);}
    bool	Dealloc(){return AImageCuda::Dealloc();}
    /// \brief Sortie console de la classe
    void	OutputInfo(){CData2D::OutputInfo();}

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
bool	 ImageCuda<T>::InitImage(uint2 dimension, T* data)
{
    SetDimension(dimension);
    Malloc();
    return copyHostToDevice(data);
}

template <class T>
bool ImageCuda<T>::Malloc()
{
    cudaChannelFormatDesc channelDesc =  cudaCreateChannelDesc<T>();
    CData2D::SetSizeofMalloc(CData2D::GetSize()*sizeof(T));
    CData2D::AddMemoryOc(CData2D::GetSizeofMalloc());
    // Allocation mémoire du tableau cuda
    return CData2D::ErrorOutput(cudaMallocArray(AImageCuda::ppData(),&channelDesc,struct2D::GetDimension().x,struct2D::GetDimension().y),"Malloc");
}

//-----------------------------------------------------------------------------------------------
//									CLASS IMAGE LAYARED CUDA
//-----------------------------------------------------------------------------------------------

/// \class ImageLayeredCuda
/// \brief Cette classe est une pile d'image 2D directement liable a une texture GpGpu
template <class T> 
class ImageLayeredCuda : public CData3D<cudaArray>, public AImageCuda
{

public:

    ImageLayeredCuda();
    ~ImageLayeredCuda(){}
    /// \brief Alloue la memoire globale neccessaire
    bool	Malloc();
    /// \brief Initialise les valeurs des images a val
    /// \param val : Valeur d initialisation
    bool	Memset(int val){return AImageCuda::Memset(val);}
    bool	Dealloc(){return AImageCuda::Dealloc();}
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

private:

    T*	_ClassData;
};

template <class T>
ImageLayeredCuda<T>::ImageLayeredCuda()
{
    CData3D::SetType("ImageLayeredCuda");
    CData3D::ClassTemplate(CData3D::ClassTemplate() + " " + CData3D::StringClass<T>(_ClassData));
}

template <class T>
bool ImageLayeredCuda<T>::copyHostToDeviceASync( T* data, cudaStream_t stream /*= 0*/ )
{
    cudaExtent sizeImagesLayared = make_cudaExtent( CData3D::GetDimension().x, CData3D::GetDimension().y, CData3D::GetNbLayer());

    // Déclaration des parametres de copie 3D
    cudaMemcpy3DParms	p		= { 0 };
    cudaPitchedPtr		pitch	= make_cudaPitchedPtr(data, sizeImagesLayared.width * sizeof(T), sizeImagesLayared.width, sizeImagesLayared.height);

    p.dstArray	= AImageCuda::GetCudaArray();	// Pointeur du tableau de destination
    p.srcPtr	= pitch;						// Pitch
    p.extent	= sizeImagesLayared;			// Taille du cube
    p.kind		= cudaMemcpyHostToDevice;		// Type de copie

    // Copie des images du Host vers le Device
    return CData3D::ErrorOutput( cudaMemcpy3DAsync (&p, stream),"copyHostToDeviceASync");
}

template <class T>
bool ImageLayeredCuda<T>::copyHostToDevice( T* data )
{
    cudaExtent sizeImagesLayared = make_cudaExtent( CData3D::GetDimension().x, CData3D::GetDimension().y, CData3D::GetNbLayer());

    // Déclaration des parametres de copie 3D
    cudaMemcpy3DParms	p		= { 0 };
    cudaPitchedPtr		pitch	= make_cudaPitchedPtr(data, sizeImagesLayared.width * sizeof(T), sizeImagesLayared.width, sizeImagesLayared.height);

    p.dstArray	= AImageCuda::GetCudaArray();   // Pointeur du tableau de destination
    p.srcPtr	= pitch;                        // Pitch
    p.extent	= sizeImagesLayared;            // Taille du cube
    p.kind		= cudaMemcpyHostToDevice;       // Type de copie

    // Copie des images du Host vers le Device
    return CData3D::ErrorOutput(cudaMemcpy3D(&p),"copyHostToDevice") ;
}

template <class T>
bool ImageLayeredCuda<T>::copyDeviceToDevice(T *data)
{
    cudaExtent sizeImagesLayared = make_cudaExtent( CData3D::GetDimension().x, CData3D::GetDimension().y, CData3D::GetNbLayer());

    // Déclaration des parametres de copie 3D
    cudaMemcpy3DParms	p		= { 0 };
    cudaPitchedPtr		pitch	= make_cudaPitchedPtr(data, sizeImagesLayared.width * sizeof(T), sizeImagesLayared.width, sizeImagesLayared.height);

    p.dstArray	= AImageCuda::GetCudaArray();   // Pointeur du tableau de destination
    p.srcPtr	= pitch;                        // Pitch
    p.extent	= sizeImagesLayared;            // Taille du cube
    p.kind	= cudaMemcpyDeviceToDevice;         // Type de copie

    // Copie des images du Host vers le Device
    return CData3D::ErrorOutput(cudaMemcpy3D(&p),"copyDeviceToDevice") ;
}

template <class T>
bool ImageLayeredCuda<T>::Malloc()
{

    CData3D::SetSizeofMalloc(CData3D::GetSize()*sizeof(T));
    CData3D::AddMemoryOc(CData3D::GetSizeofMalloc());

    cudaExtent sizeImagesLayared = make_cudaExtent( CData3D::GetDimension().x, CData3D::GetDimension().y,CData3D::GetNbLayer());

    // Définition du format des canaux d'images
    cudaChannelFormatDesc channelDesc =	cudaCreateChannelDesc<T>();

    // Allocation memoire GPU du tableau des calques d'images
    return CData3D::ErrorOutput(cudaMalloc3DArray(AImageCuda::ppData(),&channelDesc,sizeImagesLayared,cudaArrayLayered),"Malloc");
}

#endif /*GPGPUTOOLS_H*/
