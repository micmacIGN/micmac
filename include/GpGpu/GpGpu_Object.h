#ifndef GPGPU_OBJECT_H
#define GPGPU_OBJECT_H

#include "GpGpu/GpGpu_eLiSe.h"
#include "GpGpu/GpGpu_CommonHeader.h"
#include "GpGpu/GpGpu_Tools.h"

using namespace std;

#ifdef NOCUDA_X11
#ifndef _WIN32 
#include <cxxabi.h>
#endif
#endif

/** @addtogroup GpGpuDoc */
/*@{*/

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

	/// \brief SetName  affecte le nom
	/// \param name : le nom a affecte
	/// \param id ajout identifiant
	///
	void		SetName(std::string name, int id);
    /// \brief  renvoie le type de l objet en string
    std::string	Type();
    /// \brief  affecte le type de l objet
	void		SetType(string type);
    /// \brief  renvoie la classe du template de l objet en string
    std::string	ClassTemplate();
    /// \brief  Affecte la classe du template de l objet
	void		ClassTemplate(string classTemplate);


    /// \brief  renvoie la classe T en string
    template<class T>
	const char* StringClass(T* tt){ return "T";}

#ifdef NOCUDA_X11

	template<class T>
	static string AutoStringClass(T* _data)
	{
		string sCT(CGObject::demangle(typeid(_data).name()));
		return sCT.substr(0, sCT.size()-1);
	}

	static inline const char* demangle(const char* name)
	{
#ifndef _WIN32 
		char buf[1024];
		size_t size=1024;
		int status;
		char* res = abi::__cxa_demangle (name,buf,&size,
										 &status);
		return res;
#else
		return name;
#endif

	}
#endif


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

/// \cond
template<class CDimension>
class CStructuring
{

public:

    CDimension dimension() const;
    void setDimension(const CDimension &dimension);
    virtual uint GetSize() = 0;
private:

    CDimension _dimension;

};

/// \enndcond
template<class CDimension>
CDimension CStructuring<CDimension>::dimension() const
{
    return _dimension;
}
template<class CDimension>
void CStructuring<CDimension>::setDimension(const CDimension &dimension)
{
    _dimension = dimension;
}

/// \class struct2D
/// \brief classe structure de donnees de dimension 2
class struct2D
{
public:

    struct2D();
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

protected:

/// \cond
    uint2		SetDimensionOnly(uint2 dimension);

    uint        GetMaxSize();

    virtual     void RefreshMaxSize();

    void        RefreshMaxDim();

    void        SetMaxSize(uint size);

    uint2       GetMaxDimension();

    void        SetMaxDimension(uint2 dim = make_uint2(0,0));


private:

    uint2		_dimension;

    uint        _m_maxsize;

    uint2		_m_maxdimension;

/// \endcond
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

	///
	/// \brief GetSize
	/// \return La taille en 1 dimension de la structure
	///
    uint        GetSize();

	///
	/// \brief Output
	/// Retour console des informations de la structure
    void        Output();

protected:

	///
	/// \brief RefreshMaxSize rafraichir la valeur de la taille maximal
	///
    virtual     void RefreshMaxSize();

private:

    uint _nbLayers;
};

/*@}*/

#endif  //GPGPU_OBJECT_H
