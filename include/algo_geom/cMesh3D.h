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

struct TriangleAttribute
{
	double angle;			//angle between image idx and triangle normale
	double correlation;		//correlation in image idx
};

// MODID MPD : pb compile LINUX
class cTriangle;
class cMesh;
class cVertex;

class cMesh
{
	friend class cTriangle;

	public:
							cMesh();
							cMesh(const string & Filename);
				
							~cMesh();

				int			getVertexNumber()	{return mVertexNumber;}
				int			getFacesNumber()	{return mFacesNumber;}
				int			getEdgesNumber()	{return mEdgesNumber;}

				void		getVertexes(vector <Pt3dr> &vPts) {vPts = mPts;}
				void		getFaces(vector <int> &vFaces);
				void		getEdges(vector <int> &vEdges);
			
				Pt3dr		getPt (int idx);
				cTriangle	getTriangle(int idx);

				void		writePly(const string & Filename, int AttributeAsRGB);
	
	private:
				int		mVertexNumber;		//size of mPts
				int		mFacesNumber;		//size of mTriangles
				int		mEdgesNumber;

				vector <Pt3dr>		mPts;
				vector <cTriangle>	mTriangles;
};

class cVertex
{
	public:
				void				getPos(Pt3dr &pos){pos = mPos;}
				int					getIndex(){return mIndex;}
				//VertexAttribute getAttribute();

	private:
				int					mIndex;
				Pt3dr				mPos;
};

class cTriangle
{
	public:

				Pt3dr				getNormale(cMesh &elMesh, bool normalize = false);
				void				getPoints(cMesh &elMesh, vector <Pt3dr> &vList);
				
				void				getPointsIndexes(vector <int> &vList){vList = mIndexes;}
				void				getVoisins(vector <int> &vList);
				bool				getAttribute(int image_idx, TriangleAttribute &ta);

	private:
				vector <int>					mIndexes;		// index of vertexes
				map <int, TriangleAttribute>	mAttributes;	// map between image index and triangle attribute
};

#endif
