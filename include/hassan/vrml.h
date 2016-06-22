/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#ifndef __VRML_H__
#define __VRML_H__
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

template <class T>
class TPoint3D_vrml
{
	public:
	T x;
	T y;
	T z;

	TPoint3D_vrml < T > & operator += (const TPoint3D_vrml<T> &b) {
		x += b.x ;
		y += b.y ;
		z += b.z ;
    return *this ;
	}

	TPoint3D_vrml(T const vx,T const vy,T const vz)
	{
		x=vx;
		y=vy;
		z=vz;
	}

	TPoint3D_vrml(){;}
};

template <class T>
class TPoint2D_vrml
{
	public:
	T x;
	T y;
};

class TVRML
{
public:

	enum BlocType { SHAPE, VIEWPOINT, BLOC };

	enum GeometryType { INDEXEDLINESET, INDEXEDFACESET, POINTSET, ELEVATIONGRID, GEOMETRY };

class TBloc
{
public:
	TBloc(){}
	virtual void Sauver(std::ofstream &fic)const =0;
	virtual void SauverTRAPU(std::ofstream &fic,int const index)const =0;
	virtual ~TBloc(){};
	virtual BlocType GetType()const {return BLOC;}
};

class TGeometry
{
public:
	TGeometry(){}
	virtual void Sauver(std::ofstream &fic)const =0;
	virtual void SauverTRAPU(std::ofstream &fic,int const index)const =0;
	virtual ~TGeometry(){};
	virtual GeometryType GetType() const {return GEOMETRY;};
};

class TImageTexture
{
private:
	char url[256];//GIF JPEG ou PNG
	
public:
	void GetUrl(char * nom)const;

	void SetUrl(char * const nom);

	TImageTexture();

	TImageTexture(char * const nom);

	TImageTexture(std::ifstream &fic);

	void Sauver(std::ofstream &fic)const;
};

class TcoordIndex : public std::vector< std::vector< int> >
{
public:
	TcoordIndex():std::vector<std::vector<int> >(){}

	TcoordIndex (std::ifstream & fic);

	void Sauver (std::ofstream &fic)const;
};

class TtexCoordIndex : public std::vector< std::vector< int> >
{
public:
	TtexCoordIndex():std::vector<std::vector<int> >(){}

	TtexCoordIndex (std::ifstream & fic);

	void Sauver (std::ofstream &fic)const;
};

class TcolorIndex : public std::vector< int>
{
public:
	TcolorIndex():std::vector<int> (){}

	TcolorIndex (std::ifstream & fic);

	void Sauver (std::ofstream &fic)const;
};

template <class T>
class TPileCoordinate
{
private:
	std::vector< T > point;
	virtual T LireT(std::ifstream &fic)const =0;
	virtual void EcrireT(std::ofstream &fic,T const &t)const =0;
	virtual void EcrireType(std::ofstream &fic)const =0;
	virtual void EcriressType(std::ofstream &fic)const =0;
protected:
	void Lire(std::ifstream &fic)
	{
		char ch[256];
		char car;
		fic >> car;//{
		fic >> ch;//type de point
		fic >> car;//[
		while(car!=']')
		{
			point.push_back(LireT(fic));
			fic >> car;
		}
		fic >> car;//}
	}
	
public:
	virtual ~TPileCoordinate(){};
	std::vector<T> GetPoint()const{return point;}
	void SetPoint(std::vector<T>const & p){point=p;}	
	T GetPoint(int const i)const{return point[i];}
	void push_back(T const &p){point.push_back(p);}
	int size()const{return point.size();}
	typename std::vector<T>::const_iterator end()const{return point.end();}
	typename std::vector<T>::const_iterator begin()const{return point.begin();}

	TPileCoordinate(){point=std::vector<T>();}	

	void Sauver(std::ofstream &fic) const
	{
		EcrireType(fic);
		fic << "{" << std::endl;
		EcriressType(fic);
		typename std::vector<T>::const_iterator it,fin=point.end();
		--fin;
		for(it=point.begin();it<fin;++it)
		{
			EcrireT(fic,(*it));
			fic << " ," << std::endl;
		}
		EcrireT(fic,(*it));
		fic << std::endl;
		fic << "	]" << std::endl;
		fic << "}" << std::endl;
	}
};

class TCoordinate:public TPileCoordinate<TPoint3D_vrml<double> >
{
private:
	virtual TPoint3D_vrml<double> LireT(std::ifstream &fic)const
	{
		TPoint3D_vrml<double> P;
		 fic >> P.x >> P.y >> P.z;
		 return P;
	}
	virtual void EcrireT(std::ofstream &fic,TPoint3D_vrml<double>const & p)const
	{
		fic << "	" << p.x << " " << p.y << " " << p.z;
	}
	virtual void EcrireType(std::ofstream &fic)const
	{
		fic << "Coordinate" << std::endl;
	}
	virtual void EcriressType(std::ofstream &fic)const
	{
		fic << "	point [" << std::endl;
	}
 	
public:
	TCoordinate():TPileCoordinate<TPoint3D_vrml<double> > (){}; 
	TCoordinate(std::ifstream &fic):TPileCoordinate<TPoint3D_vrml<double> >(){Lire(fic);};
};

class TTextureCoordinate:public TPileCoordinate<TPoint2D_vrml<double> >
{
private:
	virtual TPoint2D_vrml<double> LireT(std::ifstream &fic)const
	{
		TPoint2D_vrml<double> P;
		 fic >> P.x >> P.y;
		 return P;
	}
	virtual void EcrireT(std::ofstream &fic,TPoint2D_vrml<double>const & p)const
	{
		fic << "	" << p.x << " " << p.y;
	}
	virtual void EcrireType(std::ofstream &fic)const
	{
		fic << "TextureCoordinate" << std::endl;
	}
	virtual void EcriressType(std::ofstream &fic)const
	{
		fic << "	point [" << std::endl;
	}
	
public:
	TTextureCoordinate():TPileCoordinate<TPoint2D_vrml<double> > (){}; 
	TTextureCoordinate(std::ifstream &fic):TPileCoordinate<TPoint2D_vrml<double> >(){Lire(fic);};
};

class TColor:public TPileCoordinate<TPoint3D_vrml<double> >
{
private:
	virtual TPoint3D_vrml<double> LireT(std::ifstream &fic)const
	{
		TPoint3D_vrml<double> P;
		 fic >> P.x >> P.y >> P.z;
		 return P;
	}
	virtual void EcrireT(std::ofstream &fic,TPoint3D_vrml<double>const & p)const
	{
		fic << "	" << p.x << " " << p.y << " " << p.z;
	}
	virtual void EcrireType(std::ofstream &fic)const
	{
		fic << "Color" << std::endl;
	}
	virtual void EcriressType(std::ofstream &fic)const
	{
		fic << "	color [" << std::endl;
	}
	
public:
	TColor():TPileCoordinate<TPoint3D_vrml<double> > (){}; 
	TColor(std::ifstream &fic):TPileCoordinate<TPoint3D_vrml<double> >(){Lire(fic);};
};

class TIndexedFaceSet:public TGeometry
{
private:
	unsigned char solid;
	TcoordIndex coordIndex;
	TCoordinate Coordinate;
	double creaseAngle;
	unsigned char colorPerVertex;
	TColor Color;
	TcolorIndex colorIndex;
	TTextureCoordinate TextureCoordinate;
	TtexCoordIndex texCoordIndex;


public:
	unsigned char GetSolid()const{return solid;}

	void SetSolid(unsigned char const s){solid=s;}

	unsigned char GetColorPerVertex()const{return colorPerVertex;}

	void SetColorPerVertex(unsigned char const c){colorPerVertex=c;}

	double GetCreaseAngle()const{return creaseAngle;}

	void SetCreaseAngle(double const &ca){creaseAngle=ca;}

	TPoint3D_vrml<double> GetPoint(int const i)const{return Coordinate.GetPoint(i);}

	int GetNbColor()const{return Color.size();}

	int iGetNbPoint()const{return Coordinate.size();}

	void GetFace(int const i,std::vector<int> &p)const{p=coordIndex[i];}

	int iGetNbFace()const{return coordIndex.size();}

	void push_backPoint(TPoint3D_vrml<double> const &P){Coordinate.push_back(P);}

	void push_backPoint(TPoint3D_vrml<double> const &P,int const c)
	{
		Coordinate.push_back(P);
		colorIndex.push_back(c);
	}

	TPoint2D_vrml<double> GetTexturePoint(int const i)const{return TextureCoordinate.GetPoint(i);}

	void push_backTexturePoint(TPoint2D_vrml<double> const &P){TextureCoordinate.push_back(P);}

	TPoint3D_vrml<double> GetColor(int const i)const{return Color.GetPoint(i);}

	void push_backColor(TPoint3D_vrml<double> const &C){Color.push_back(C);}

	void push_backTriangle(TPoint3D_vrml<int> const &tri)
	{
		std::vector<int> P;
		P.push_back(tri.x);
		P.push_back(tri.y);
		P.push_back(tri.z);
		coordIndex.push_back(P);
	}

	void push_backTriangle(TPoint3D_vrml<int>const & tri,int const c)
	{
		push_backTriangle(tri);
		colorIndex.push_back(c);
	}

	void push_backFace(std::vector<int>const & face)
	{
		coordIndex.push_back(face);
	}

	void push_backFace(std::vector<int>const & face,int const c)
	{
		coordIndex.push_back(face);
		colorIndex.push_back(c);
	}

	void push_backFace(std::vector<int>const & face,std::vector<int>const& tex)
	{
		coordIndex.push_back(face);
		texCoordIndex.push_back(tex);
	}

	void push_backTriangle(int const a,int const b,int const c)
	{
		std::vector<int> P;
		P.push_back(a);
		P.push_back(b);
		P.push_back(c);
		coordIndex.push_back(P);
	}

	TIndexedFaceSet();

	TIndexedFaceSet(std::ifstream &fic);

	void Sauver(std::ofstream &fic)const;

	void SauverTRAPU(std::ofstream &fic,int const index)const;

	virtual GeometryType GetType()const {return INDEXEDFACESET;}
};

class THeight
{
	
public:

  int NC,NL;
  double * tab;


	THeight()
	{
		NC=0;
		NL=0;
		tab=NULL;
	}
	~THeight()
	{
		delete(tab);
	}

	THeight(char * const nom,int const NC,int const NL);

	THeight(std::ifstream &fic,int const NC,int const NL);

	THeight(THeight const &h)
	{
		NC=h.NC;
		NL=h.NL;
		tab=new double [NC*NL];
		int i;
		for(i=0;i<NC*NL;++i)
		{
			tab[i]=h.tab[i];
		}
	}

	THeight & operator = (THeight const & h) 
	{ 
		NC=h.NC;
		NL=h.NL;
		tab=new double [NC*NL];
		int i;
		for(i=0;i<NC*NL;++i)
		{
			tab[i]=h.tab[i];
		}
		return *this ;
	}

	void Sauver(std::ofstream &fic)const;
};

class TElevationGrid:public TGeometry
{
private:
	unsigned char solid;
	double creaseAngle;
	unsigned char colorPerVertex;
	TColor Color;
	TTextureCoordinate TextureCoordinate;
	int xDimension;
	double xSpacing;
	int zDimension;
	double zSpacing;
	THeight height;


public:
	void SetGrid(char * const nom,int const NC,int const NL,int const pas)
	{
		height=THeight(nom,NC,NL);
		xDimension=height.NC;
		zDimension=height.NL;
		xSpacing=pas;
		zSpacing=xSpacing;
	}

	TElevationGrid(char * const nom,int const NC,int const NL,int const pas):height(nom, NC, NL)
	{
		xDimension=height.NC;
		zDimension=height.NL;
		xSpacing=pas;
		zSpacing=xSpacing;
		creaseAngle=0.;
		
		solid=0;
		colorPerVertex=1;
		Color=TVRML::TColor();
		TextureCoordinate=TVRML::TTextureCoordinate();

	}

	TElevationGrid(TElevationGrid const &eg):height(eg.height)
	{
		solid=eg.solid;
		creaseAngle=eg.creaseAngle;
		colorPerVertex=eg.colorPerVertex;
		Color=eg.Color;
		TextureCoordinate=eg.TextureCoordinate;
		xDimension=eg.xDimension;
		xSpacing=eg.xSpacing;
		zDimension=eg.zDimension;
		zSpacing=eg.zSpacing;
	}

	TElevationGrid & operator = (TElevationGrid const & eg) 
	{ 
		solid=eg.solid;
		creaseAngle=eg.creaseAngle;
		colorPerVertex=eg.colorPerVertex;
		Color=eg.Color;
		TextureCoordinate=eg.TextureCoordinate;
		xDimension=eg.xDimension;
		xSpacing=eg.xSpacing;
		zDimension=eg.zDimension;
		zSpacing=eg.zSpacing;
		height=eg.height;
		return *this ;
	}

	unsigned char GetSolid()const{return solid;}

	void SetSolid(unsigned char const s){solid=s;}

	unsigned char GetColorPerVertex()const{return colorPerVertex;}

	void SetColorPerVertex(unsigned char const c){colorPerVertex=c;}

	double GetCreaseAngle()const{return creaseAngle;}

	void SetCreaseAngle(double const ca){creaseAngle=ca;}

	TPoint2D_vrml<double> GetTexturePoint(int const i)const{return TextureCoordinate.GetPoint(i);}

	void push_backTexturePoint(TPoint2D_vrml<double> const &P){TextureCoordinate.push_back(P);}

	TPoint3D_vrml<double> GetColor(int const i)const{return Color.GetPoint(i);}

	void push_backColor(TPoint3D_vrml<double> const &C){Color.push_back(C);}

	TElevationGrid();

	TElevationGrid(std::ifstream &fic);

	void Sauver(std::ofstream &fic)const;

	void SauverTRAPU(std::ofstream &fic,int const index)const;

	virtual GeometryType GetType()const {return ELEVATIONGRID;}
};


class TIndexedLineSet:public TGeometry
{
private:
	TcoordIndex coordIndex;
	TCoordinate Coordinate;
	TColor Color;
	TcolorIndex colorIndex;
	unsigned char colorPerVertex;

public:
	TIndexedLineSet(TIndexedLineSet const &lset):	coordIndex(lset.coordIndex),
													Coordinate(lset.Coordinate),
													Color(lset.Color),
													colorIndex(lset.colorIndex){;}

	unsigned char GetColorPerVertex()const{return colorPerVertex;}

	void SetColorPerVertex(unsigned char const c){colorPerVertex=c;}

	TPoint3D_vrml<double> GetPoint(int const i)const{return Coordinate.GetPoint(i);}

	void push_backPoint(TPoint3D_vrml<double> const &P){Coordinate.push_back(P);}

	void push_backPoint(TPoint3D_vrml<double> const &P,int const c)
	{
		Coordinate.push_back(P);
		colorIndex.push_back(c);
	}

	TPoint3D_vrml<double> GetColor(int const i)const{return Color.GetPoint(i);}

	int GetNbColor()const{return Color.size();}

	void push_backColor(TPoint3D_vrml<double> const &C){Color.push_back(C);}

	void push_backLine(std::vector<int>const & line)
	{
		coordIndex.push_back(line);
	}

	void push_backLine(std::vector<int>const & line,int const c)
	{
		coordIndex.push_back(line);
		colorIndex.push_back(c);
	}

	TIndexedLineSet();

	TIndexedLineSet(std::ifstream &fic);

	void Sauver(std::ofstream &fic)const;

	void SauverTRAPU(std::ofstream &fic,int const index)const;

	virtual GeometryType GetType()const {return INDEXEDLINESET;}
};

class TPointSet:public TGeometry
{
private:
	TCoordinate Coordinate;
	TColor Color;

public:

	TPoint3D_vrml<double> GetPoint(int const i)const{return Coordinate.GetPoint(i);}

	void push_backPoint(TPoint3D_vrml<double> const &P){Coordinate.push_back(P);}

	void push_backPoint(TPoint3D_vrml<double> const &P,TPoint3D_vrml<double> const &c)
	{
		Coordinate.push_back(P);
		Color.push_back(c);
	}

	void push_backColor(TPoint3D_vrml<double> const &c)
	{
		Color.push_back(c);
	}

	int GetNbColor()const{return Color.size();}

	TPoint3D_vrml<double> GetColor(int const i)const{return Color.GetPoint(i);}

	TPointSet();

	TPointSet(std::ifstream &fic);

	void Sauver(std::ofstream &fic)const;

	void SauverTRAPU(std::ofstream &fic,int const index)const;

	virtual GeometryType GetType() const {return POINTSET;}
};

class TMaterial
{
private:
	double ambientIntensity;
	TPoint3D_vrml<double> diffuseColor;
	TPoint3D_vrml<double> emissiveColor;
	double shininess;
	TPoint3D_vrml<double> specularColor;
	double transparency;

public:
	double GetAmbientIntensity()const{return ambientIntensity;}

	void SetAmbientIntensity(double const a){ambientIntensity=a;}

	TPoint3D_vrml<double> GetDiffuseColor()const{return diffuseColor;}

	void SetDiffuseColor(TPoint3D_vrml<double> const &C){diffuseColor=C;}

	TPoint3D_vrml<double> GetEmissiveColor()const{return emissiveColor;}

	void SetEmissiveColor(TPoint3D_vrml<double> const &C){emissiveColor=C;}

	double GetShininess()const{return shininess;}

	void SetShininess(double const s){shininess=s;}

	TPoint3D_vrml<double> GetSpecularColor()const{return specularColor;}
	
	void SetSpecularColor(TPoint3D_vrml<double> const &C){specularColor=C;}

	double GetTransparency()const{return transparency;}

	void SetTransparency(double const t){transparency=t;}

	TMaterial();
	
	TMaterial(std::ifstream &fic);

	TMaterial(TPoint3D_vrml<double> const &dc,TPoint3D_vrml<double> const &ec,TPoint3D_vrml<double> const &sc,double const sh,double const ai,double const t)
	{
		diffuseColor=dc;
		emissiveColor=ec;
		specularColor=sc;
		shininess=sh;
		ambientIntensity=ai;
		transparency=t;
	}

	void Sauver(std::ofstream &fic)const;
};

class TAppearance
{
private:
	TMaterial material;
	TImageTexture texture;
//	TtextureTransform textureTransform;

public:
	void GetUrl(char *nom)const{texture.GetUrl(nom);}

	void SetUrl(char * const &url){texture.SetUrl(url);}

	double GetAmbientIntensity()const{return material.GetAmbientIntensity();}

	void SetAmbientIntensity(double const ai){material.SetAmbientIntensity(ai);}

	double GetShininess()const{return material.GetShininess();}

	void SetShininess(double const sh){material.SetShininess(sh);}

	double GetTransparency()const{return material.GetTransparency();}

	void SetTransparency(double const t){material.SetTransparency(t);}

	TPoint3D_vrml<double> GetDiffuseColor()const {return material.GetDiffuseColor();}

	void SetDiffuseColor(TPoint3D_vrml<double> const &C){material.SetDiffuseColor(C);}

	TPoint3D_vrml<double> GetEmissiveColor()const{return material.GetEmissiveColor();}

	void SetEmissiveColor(TPoint3D_vrml<double> const &C){material.SetEmissiveColor(C);}

	TPoint3D_vrml<double> GetSpecularColor()const{return material.GetSpecularColor();}

	void SetSpecularColor(TPoint3D_vrml<double> const &C){material.SetSpecularColor(C);}

	TMaterial GetMaterial()const{return material;}

	void SetMaterial(TMaterial const &mat){material=mat;}

	TImageTexture GetImageTexture()const{return texture;}

	void SetImageTexture(TImageTexture const &t){texture=t;}

	TAppearance()
	{
		material=TMaterial();
		texture=TImageTexture();
	}

	TAppearance(TMaterial const &mat,TImageTexture const &tex=TImageTexture())
	{
		material=mat;
		texture=tex;
	}

	TAppearance(std::ifstream &fic);

	void Sauver(std::ofstream &fic)const;
};

class TViewpoint: public TBloc
{
public:
	double fieldOfView;
	double orientation[4];
	TPoint3D_vrml<double> position;
	char * description;

	TViewpoint();

	TViewpoint(TPoint3D_vrml<double> const&P,char * const & d);

	TViewpoint(std::ifstream &fic);

	void Sauver(std::ofstream &fic)const;

	void SauverTRAPU(std::ofstream &fic,int const index)const
	{
		printf("Attention les Viewpoints ne sont pas sauv�s en TRAPU;\n");
	}

	BlocType GetType()const{return VIEWPOINT;}
};

class TShape: public TBloc
{
private:
	TAppearance appearance;
	TGeometry * geometry;

public:
	TShape(TShape const &shp):appearance(shp.appearance)
	{
		//{ INDEXEDLINESET, INDEXEDFACESET, POINTSET, ELEVATIONGRID, GEOMETRY };

		if (shp.geometry->GetType()==INDEXEDLINESET)
			geometry= new TIndexedLineSet(*((TIndexedLineSet *)shp.geometry));
		else if (shp.geometry->GetType()==INDEXEDFACESET)
			geometry= new TIndexedFaceSet(*((TIndexedFaceSet *)shp.geometry));
		else if (shp.geometry->GetType()==POINTSET)
			geometry= new TPointSet(*((TPointSet *)shp.geometry));
		else if (shp.geometry->GetType()==ELEVATIONGRID)
			geometry= new TElevationGrid(*((TElevationGrid *)shp.geometry));
		else
			printf("Erreur dans la construction par recopie de TShape\n");
	}

	double GetAmbientIntensity()const{return appearance.GetAmbientIntensity();}
	
	void SetAmbientIntensity(double const ai){appearance.SetAmbientIntensity(ai);}

	double GetShininess()const{return appearance.GetShininess();}

	void SetShininess(double const s){appearance.SetShininess(s);}

	double GetTransparency()const{return appearance.GetTransparency();}

	void SetTransparency(double const t){appearance.SetTransparency(t);}

	TPoint3D_vrml<double> GetDiffuseColor()const{return appearance.GetDiffuseColor();}

	void SetDiffuseColor(TPoint3D_vrml<double> const &DC){appearance.SetDiffuseColor(DC);}

	TPoint3D_vrml<double> GetEmissiveColor()const{return appearance.GetEmissiveColor();}

	void SetEmissiveColor(TPoint3D_vrml<double> const &DC){appearance.SetEmissiveColor(DC);}

	TPoint3D_vrml<double> GetSpecularColor()const{return appearance.GetSpecularColor();}

	void SetSpecularColor(TPoint3D_vrml<double> const &DC){appearance.SetSpecularColor(DC);}

	TImageTexture GetImageTexture()const{return appearance.GetImageTexture();}

	void SetImageTexture(TImageTexture const &tex){appearance.SetImageTexture(tex);}

	TMaterial GetMaterial()const{return appearance.GetMaterial();}

	void SetMaterial(TMaterial const &mat){appearance.SetMaterial(mat);}

	TAppearance GetAppearance()const{return appearance;}

	void SetAppearance(TAppearance const &A){appearance=A;}

	TGeometry * GetGeometry()const
	{
		if (!(geometry)) throw "Geometry non initialisee"; 
		return geometry;
	}

	void SetGeometry(TIndexedFaceSet const &G){geometry=new TIndexedFaceSet(G);}

	void SetGeometry(TIndexedLineSet const &G){geometry=new TIndexedLineSet(G);}

	void SetGeometry(TPointSet const &G){geometry=new TPointSet(G);}

	void SetGeometry(TElevationGrid const &G){geometry=new TElevationGrid(G);}


	TShape()
	{
		appearance=TAppearance();
		geometry=NULL;
	}

	TShape(std::ifstream &fic);

	void Sauver(std::ofstream &fic)const;

	void SauverTRAPU(std::ofstream &fic,int const i)const
	{	
		if (!(geometry)) throw "Geometry non initialisee"; 
		geometry->SauverTRAPU(fic,i);
	}

	BlocType GetType()const {return SHAPE;}
};


class TTransform 
{
private:	
	TPoint3D_vrml<double> center;
	double rotation[4];
	double scaleOrientation[4];
	TPoint3D_vrml<double> scale;
	TPoint3D_vrml<double> translation;
	TPoint3D_vrml<double> bboxCenter;
	TPoint3D_vrml<double> bboxSize;
	std::vector<TBloc *> children;

public:
	TTransform();
	
	TTransform(std::ifstream &fic);

	void Sauver(std::ofstream &fic)const;

	void push_backShape(TShape const &shp)
	{
		children.push_back(new TShape(shp));
	}

	void push_backViewpoint(TViewpoint const &vp)
	{
		children.push_back(new TViewpoint(vp));
	}

	void push_backChildren(TShape const &shp)
	{
		children.push_back(new TShape(shp));
	}

	void push_backChildren(TViewpoint const &vp)
	{
		children.push_back(new TViewpoint(vp));
	}

	int iGetNbChildren()const{return children.size();}

	void GetChildren(int const n,TBloc * & bloc)const{bloc=children[n];}

	void Translate(TPoint3D_vrml<double> const & t){translation+=t;}

	TPoint3D_vrml<double> GetTranslation()const{return translation;}
	
	void SetTranslation(TPoint3D_vrml<double>const & t){translation=t;}

	void SetCenter(TPoint3D_vrml<double> const & c){center=c;}

	TPoint3D_vrml<double> GetCenter()const{return center;}

	void GetRotation(double &x,double &y,double &z,double &angle)const
	{
		x=rotation[0];
		y=rotation[1];
		z=rotation[2];
		angle=rotation[3];
	}

	void SetRotation(double const &x,double const &y,double const &z,double const &angle)
	{
		rotation[0]=x;
		rotation[1]=y;
		rotation[2]=z;
		rotation[3]=angle;
	}
	
	TPoint3D_vrml<double> GetScale()const{return scale;}

	void SetScale(TPoint3D_vrml<double> const & s){scale=s;}

	void GetScaleOrientation(double &x,double &y,double &z,double &angle)const
	{
		x=scaleOrientation[0];
		y=scaleOrientation[1];
		z=scaleOrientation[2];
		angle=scaleOrientation[3];
	}

	void SetScaleOrientation(double const &x,double const &y,double const &z,double const &angle)
	{
		scaleOrientation[0]=x;
		scaleOrientation[1]=y;
		scaleOrientation[2]=z;
		scaleOrientation[3]=angle;
	}

	void Zoom(double const z)
	{
		scale.x=z*scale.x;
		scale.y=z*scale.y;
		scale.z=z*scale.z;
	}

	void Zoom(TPoint3D_vrml<double> const z)
	{
		scale.x=z.x*scale.x;
		scale.y=z.y*scale.y;
		scale.z=z.z*scale.z;
	}
};


	std::vector<TTransform> Ptransform;
	std::vector<TShape> Pshape;
	std::vector<TViewpoint> Pviewpoint;

	TVRML()
	{}

	TVRML(char * const &nom);

	void Sauver(char * const &nom)const;

	void SauverTRAPU(char * const &nom)const;

	void Translate(TPoint3D_vrml<double> const &t);

	void Zoom(double const z);

	void Zoom(TPoint3D_vrml<double> const c);

};

void exemple_vrml();

char * FHNomType(TVRML::TViewpoint * );

char * FHNomType(TVRML::TShape * );

char * FHNomType(TVRML::TTransform * );

#endif

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
