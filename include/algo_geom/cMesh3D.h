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

/*struct TriangleAttribute
{
	double angle;			//angle between image idx and triangle normale
	double correlation;		//correlation in image idx
};*/

class cMesh
{
	friend class cTriangle;

	public:
					cMesh(const string & Filename);
		
					~cMesh();

		int			getVertexNumber()	{return mVertexes.size();}
		int			getFacesNumber()	{return mTriangles.size();}
		//int			getEdgesNumber()	{return mEdgesNumber;}

		void		getVertexes(vector <Pt3dr> &vPts) {vPts = mVertexes;}
		void		getTriangles(vector <cTriangle> &vTriangles){vTriangles = mTriangles;}
		void		getEdges(vector <int> &vEdges);
	
		Pt3dr		getVertex  (int idx);
		cTriangle	getTriangle(int idx);

		void		writePly(const string &Filename, int AttributeAsRGB);

		void		addPt(const Pt3dr &aPt);
		void		addTriangle(const cTriangle &aTri);
	
	private:

		vector <Pt3dr>		mVertexes;
		vector <cTriangle>	mTriangles;
};

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

class cTriangle
{
	public:
				cTriangle(vector <int> const &idx);
				cTriangle(int idx1, int idx2, int idx3);

				~cTriangle();

		Pt3dr	getNormale(cMesh &elMesh, bool normalize = false);
		void	getVertexes(cMesh &elMesh, vector <Pt3dr> &vList);
		
		void	getVertexesIndexes(vector <int> &vList){vList = mIndexes;}
		void	getVoisins(vector <int> &vList);
		bool	getAttributes(int image_idx, vector <float> &ta);

		void	setAttributes(int image_idx, const vector <float> &ta);

	private:

		vector <int>				mIndexes;		// index of vertexes
		map <int, vector<float> >	mAttributes;	// map between image index and triangle attributes
};

#endif
