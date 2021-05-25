#ifndef GPGPUTOOLS_H
#define GPGPUTOOLS_H

/** @addtogroup GpGpuDoc */
/*@{*/

#include "GpGpu/GpGpu_CommonHeader.h"

using namespace std;

template<class T> class CuHostData3D;

/// \class GpGpuTools
/// \brief classe d outils divers
/// La classe gere la restructuration de donnees, des outils d'affichages console
class GpGpuTools
{

public:

    GpGpuTools(){}

    ~GpGpuTools(){}

    /// \brief          parametre texture
    static void			SetParamterTexture(textureReference &textRef);

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


    template <class T>
	///
	/// \brief AddArray Additionner une valeur a toutes composantes d'un tableau
	/// \param data
	/// \param dimImage
	/// \param factor
	/// \return
	///
    static T*			AddArray(T* data, uint2 dimImage, float factor);

    ///	\brief			Sortie console d'une donnees
    ///  \param         data : Donnees du tableau a afficher
    ///  \param         dim : dimension du tableau
    ///  \param         offset : nombre de chiffre apres la virgule
    ///  \param         defaut : valeur affichee par un caractere speciale
    ///  \param         sample : saut dans l'affichage
    ///  \param         factor : facteur multiplicatif
    ///  \return        renvoie un pointeur sur le tableau resultant

	/// \cond

	template <class T>
	///
	/// \brief GetArrayValue Obtenir la valeur dans un tableau en fonction de ses coordonnees
	/// \param data
	/// \param pt
	/// \param dim
	/// \return
	///
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


    template <class T>
    static T			SetValue(float defaut = 0.0f){return (T)defaut;}

    template <class T>
    static void			OutputValue(T value, uint offset = 3, T defaut = SetValue<T>(0.0f), float factor = 1.0f);
/// \endcond

	///
	/// \brief OutputReturn Retour chariot
	/// \param out
	///
	static void			OutputReturn(char * out /*= ""*/);

    ///	\brief			multiplie par un facteur
    static float		fValue( float value,float factor );

    ///	\brief			multiplie par un facteur
    static float		fValue( float2 value,float factor );

    ///	\brief			Convertie un uint2 en string
    static std::string	toStr(uint2 tt);

    ///	\brief			Convertie un uint2 en string
    static const char* 	conca(const char* texte, int t = 0);

    ///	\brief			Affiche les parametres GpGpu de correlation multi-images
    //static void			OutputInfoGpuMemory();

    ///	\brief			(X)
    static void			OutputGpu();

    //static void			check_Cuda();

#ifdef NVTOOLS
	///
	/// \brief NvtxR_Push Pousser une fonction pour le profiling
	/// \param message
	/// \param color
	///
	static void  NvtxR_Push(const char* message, int32_t color);
#else
	///
	/// \brief NvtxR_Push
	/// \param message
	/// \param color
	///
	static void  NvtxR_Push(const char* message, int color){}
#endif

	///
	/// \brief Nvtx_RangePop Retirer une fonction du profiling
	///
	static void	Nvtx_RangePop();

	/// \cond
    template <class T>
    static T            getMaxArray(T *data, uint2 dim);

    template <class T>
	static T            getMinArray(T *data, uint2 dim);
	/// \endcond
};

template <class T>
void GpGpuTools::Memcpy2Dto1D( T** dataImage2D, T* dataImage1D, uint2 dimDest, uint2 dimSource )
{
    OMP_NT1
    for (uint j = 0; j < dimSource.y ; j++)
        memcpy(  dataImage1D + dimDest.x * j , dataImage2D[j],  dimSource.x * sizeof(T));
}

/// \cond
// TODO ???
template <> inline
uint2    GpGpuTools::SetValue(float defaut){return make_uint2((uint)defaut);}

template <> inline
int2    GpGpuTools::SetValue(float defaut){return make_int2((int)defaut);}


template <> inline
float2    GpGpuTools::SetValue(float defaut){return make_float2(defaut);}

template <> inline
short2    GpGpuTools::SetValue(float defaut){return make_short2((short)defaut);}


template <> inline
ushort2    GpGpuTools::SetValue(float defaut){return make_ushort2(defaut);}


template <class T>
void GpGpuTools::OutputValue( T value, uint offset, T defaut, float factor)
{
   DUMPI(value)
   std::cout << "\t";
}

template <> inline
///
/// \brief GpGpuTools::OutputValue Affiche une valeur
/// \param value
/// \param offset
/// \param defaut
/// \param factor
///
void GpGpuTools::OutputValue( float value, uint offset, float defaut, float factor)
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

	//std::string valS;
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

template<> inline
///
/// \brief GpGpuTools::OutputValue Affiche une valeur
/// \param value
/// \param offset
/// \param defaut
/// \param factor
///
void GpGpuTools::OutputValue( short2 value, uint offset, short2 defaut, float factor)
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
/// \endcond

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
T* GpGpuTools::AddArray( T* data, uint2 dim, float factor )
{
    if (factor == 0) return NULL;

    int sizeData = size(dim);

    T* image = new T[sizeData];

    for (int i = 0; i < sizeData ; i++)
        image[i] = data[i] + (T)factor;

    return image;

}

/// \cond
template <class T>
T GpGpuTools::getMinArray( T* data, uint2 dim )
{

    int sizeData = size(dim);

    T min = 0;

    for (int i = 0; i < sizeData ; i++)
        if(data[i] < min)
            min = data[i];

    return min;
}

template <class T>
T GpGpuTools::getMaxArray( T* data, uint2 dim )
{

    int sizeData = size(dim);

    T max = 0;

    for (int i = 0; i < sizeData ; i++)
        if(data[i] > max)
            max = data[i];

    return max;
}
/// \endcond

template <class T>
bool GpGpuTools::Array1DtoImageFile( T* dataImage,const char* fileName, uint2 dimImage )
{
    std::string pathfileImage = std::string(GetImagesFolder()) + std::string(fileName);
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

/*@}*/

#endif /*GPGPUTOOLS_H*/
