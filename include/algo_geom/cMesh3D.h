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

#ifndef _ELISE_CMESH
#define _ELISE_CMESH

#include "general/ptxd.h"
#include "../private/cElNuage3DMaille.h"

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
	
		Pt3dr		getVertex(unsigned int idx) const;
		cTriangle*	getTriangle(unsigned int idx);

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

	private:

		int			mIndex;
		Pt3dr		mPos;
};

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

class cTriangle
{
	public:
				cTriangle(vector <int> const &idx, int TriIdx);
				cTriangle(int idx1, int idx2, int idx3, int TriIdx);

				~cTriangle();

		Pt3dr	getNormale(cMesh const &elMesh, bool normalize = false) const;
		void	getVertexes(cMesh const &elMesh, vector <Pt3dr> &vList) const;
		
		void	getVertexesIndexes(vector <int> &vList) const {vList = mIndexes;}
		void	getVoisins(vector <int> &vList) const;
		bool	getAttributes(int image_idx, vector <double> &ta) const;
		map <int, vector <REAL> >	getAttributesMap() const {return mAttributes;}
		int		getIdx() const {return mTriIdx;}

		void	setAttributes(int image_idx, const vector <double> &ta);

		bool	hasAttributes() { return (mAttributes.size() != 0); }
		
	private:

		int							mTriIdx;		// triangle index
		vector <int>				mIndexes;		// index of vertexes
		map <int, vector <REAL> >	mAttributes;	// map between image index and triangle attributes
};

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

class cZBuf
{
	public:
				cZBuf();

				~cZBuf();

		Im2D_REAL4	BasculerUnMaillage(cMesh &aMesh);			//Projection du maillage dans la geometrie de aNuage, aDef: valeur par defaut de l'image resultante

		void		BasculerUnTriangle(cTriangle &aTri, cMesh &aMesh, bool doMask = false); //soit on calcule le ZBuffer, soit le Masque (true)
		
		void		ComputeVisibleTrianglesIndexes();
		Im2D_BIN	ComputeMask(int img_idx, cMesh &aMesh);

		Im2D_U_INT2				getIndexImage() const {return mImTriIdx;}
		vector <unsigned int>	getVisibleTrianglesIndexes() const {return vTri;}
		
		cElNuage3DMaille * &	Nuage() {return mNuage;}

		void					setSelfSz(){mSzRes = mNuage->Sz();} //temp
		void					setMaxAngle(double aAngle){mMaxAngle = aAngle;}

		Pt2di					Sz(){return mSzRes;}

	private:

		double					mMaxAngle;		//threshold on angle between surface and viewing direction

		Pt2di					mSzRes;			//size result

		Im2D_U_INT2				mImTriIdx;		//triangle index image (label image)
		Im2D_BIN				mImMask;		//mask image

		Im2D_REAL4				mRes;			//Zbuffer
        float **				mDataRes;

		float					mDpDef;			//default value for depth img (mRes)
		unsigned int			mIdDef;			//default value for index img (mImTriIdx)

		vector <unsigned int>	vTri;			//list of visible triangles (contained in the index image)

		cElNuage3DMaille *		mNuage;			
};

#endif // _ELISE_CMESH

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
