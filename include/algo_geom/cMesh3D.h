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

#ifndef CMESH
#define CMESH

#include "general/ptxd.h"
#include "../private/cElNuage3DMaille.h"

/*struct TriangleAttribute
{
	double angle;			//angle between image idx and triangle normale
	double correlation;		//correlation in image idx
};*/

class cMesh;
class cVertex;
class cTriangle;
class cZBuf;

class cMesh
{
	friend class cTriangle;

	public:
						cMesh(const string & Filename);
		
						~cMesh();

		int			getVertexNumber() const	{return (int) mVertexes.size();}
		int			getFacesNumber() const	{return (int) mTriangles.size();}
	
		void		getVertexes(vector <Pt3dr> &vPts) const {vPts = mVertexes;}
		void		getTriangles(vector <cTriangle> &vTriangles) const {vTriangles = mTriangles;}
		void		getEdges(vector <int> &vEdges);
	
		Pt3dr		getVertex(int idx) const;
		cTriangle*	getTriangle(int idx);

		void		writePly(const string &Filename, int AttributeAsRGB);

		void		addPt(const Pt3dr &aPt);
		void		addTriangle(const cTriangle &aTri);

		void		setTrianglesAttribute(int img_idx, Pt3dr Dir, vector <unsigned int> const &TriIdx);
	
	private:

		vector <Pt3dr>		mVertexes;
		vector <cTriangle>	mTriangles;
};

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

class cVertex
{
	public:
					cVertex(const Pt3dr & pt);
			
					~cVertex();

		void		getPos(Pt3dr &pos){pos = mPos;}
		int			getIndex(){return mIndex;}
		//VertexAttribute getAttribute();

	private:

		int			mIndex;
		Pt3dr		mPos;
};

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

class cTriangle
{
	public:
				cTriangle(vector <int> const &idx);
				cTriangle(int idx1, int idx2, int idx3);

				~cTriangle();

		Pt3dr	getNormale(cMesh const &elMesh, bool normalize = false) const;
		void	getVertexes(cMesh const &elMesh, vector <Pt3dr> &vList) const;
		
		void	getVertexesIndexes(vector <int> &vList) const {vList = mIndexes;}
		void	getVoisins(vector <int> &vList) const;
		bool	getAttributes(int image_idx, vector <float> &ta) const;

		void	setAttributes(int image_idx, const vector <float> &ta);

	private:

		vector <int>				mIndexes;		// index of vertexes
		map <int, vector<float> >	mAttributes;	// map between image index and triangle attributes
};

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

class cZBuf
{
	public:
				cZBuf(Pt2di aSz);

				~cZBuf();

		Im2D_REAL4	BasculerUnMaillage(cMesh &aMesh, cElNuage3DMaille const &aNuage, float aDef);			//Projection du maillage dans la geometrie de aNuage, aDef: valeur par defaut de l'image resultante

		void		BasculerUnTriangle(cTriangle &aTri, cMesh &aMesh, cElNuage3DMaille const &aNuage, bool doMask);

		set <unsigned int> getTrianglesIndexes();

		void		ComputeMask(int img_idx, cMesh &aMesh, cElNuage3DMaille const &aNuage, vector <unsigned int> const &TriIdx);

		Im2D_Bits<1> getMaskImg() { return mImMask; }

	private:

		Pt2di			mSzRes;			//size result

		Im2D_U_INT1     mImTriIdx;		//triangle index image (label image)
		Im2D_Bits<1>    mImMask;		//mask image

		Im2D_REAL4		mRes;			//Zbuffer
        float **		mDataRes;

		float			mDef;			//default value for Depth img
};

#endif
