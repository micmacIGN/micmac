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

#include "StdAfx.h"
#include "../uti_phgrm/MergePly/ply.h"

static const REAL Eps = 1e-7;

//static const REAL MESH_MAX_ANGLE = 0.80901699437494742410229341718282; // PI/5
static const REAL MESH_MAX_ANGLE = 0.70710678118654752440084436210485; // PI/4

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// cTriangle
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTriangle::~cTriangle(){}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTriangle::cTriangle(vector <int> const &idx)
{
	mIndexes = idx;
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTriangle::cTriangle(int idx1, int idx2, int idx3)
{
	mIndexes.push_back(idx1);
	mIndexes.push_back(idx2);
	mIndexes.push_back(idx3);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

Pt3dr cTriangle::getNormale(cMesh const &elMesh, bool normalize) const
{
	vector <Pt3dr> vPts;
	getVertexes(elMesh, vPts);

	if (normalize)
	{
		Pt3dr v1n = PointNorm1(vPts[1]-vPts[0]);
		Pt3dr v2n = PointNorm1(vPts[2]-vPts[0]);
		return v1n^v2n;
	}
	else
	{
		return (vPts[1]-vPts[0])^(vPts[2]-vPts[0]);
	}
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cTriangle::getVertexes(cMesh const &elMesh, vector <Pt3dr> &vList) const
{
	for (unsigned int aK =0; aK < mIndexes.size(); ++aK)
	{
		vList.push_back(elMesh.getVertex(mIndexes[aK]));
	}
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

bool cTriangle::getAttributes(int image_idx, vector <float> &ta) const
{
	map <int, vector <float>>::const_iterator it;

	it = mAttributes.find(image_idx);
	if (it != mAttributes.end())
	{
		ta = it->second;
		return true;
	}
	else
		return false;
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// cMesh
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cMesh::~cMesh(){}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

Pt3dr cMesh::getVertex(int idx) const
{
	ELISE_ASSERT(idx < mVertexes.size(), "cMesh3D.cpp cMesh::getPt, out of vertex array");
	
	return mVertexes[idx];
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTriangle* cMesh::getTriangle(int idx) 
{
	ELISE_ASSERT(idx < mTriangles.size(), "cMesh3D.cpp cMesh::getTriangle, out of faces array");
	
	return &(mTriangles[idx]);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

// list of property information for a vertex
PlyProperty props[] = 
{ 
	{"x", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,x), 0, 0, 0, 0},
	{"y", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,y), 0, 0, 0, 0},
	{"z", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,z), 0, 0, 0, 0},
};

// list of property information for a face
PlyProperty face_props[] = 
{ 
  {"vertex_indices", PLY_INT, PLY_INT, offsetof(Face,verts),
   1, PLY_UINT, PLY_UINT, offsetof(Face,nverts)},
};

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cMesh::cMesh(const std::string & Filename)
{
	PlyFile * thePlyFile;
	int nelems;
	char **elist;
	int file_type;
	float version;
	int nprops;
	int num_elems;
	char *elem_name;
	PlyProperty **plist;

	thePlyFile = ply_open_for_reading( const_cast<char *>(Filename.c_str()), &nelems, &elist, &file_type, &version);
	
	ELISE_ASSERT(thePlyFile != NULL, "cMesh3D.cpp: cMesh::cMesh, cannot open ply file for reading");

	for (int i = 0; i < nelems; i++) 
	{
		// get the description of the first element
		elem_name = elist[i];
		plist = ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);
				
		//printf ("element %s %d\n", elem_name, num_elems);
					
		if (equal_strings ("vertex", elem_name)) 
		{		
			// set up for getting vertex elements
			ply_get_property (thePlyFile, elem_name, &props[0]);
			ply_get_property (thePlyFile, elem_name, &props[1]);
			ply_get_property (thePlyFile, elem_name, &props[2]);
			
			// grab all the vertex elements
			for (int j = 0; j < num_elems; j++) 
			{
				Vertex *vert = (Vertex *) malloc (sizeof Vertex);
				
				ply_get_element (thePlyFile, vert);
									
				addPt(Pt3dr(vert->x, vert->y, vert->z));

				//printf ("vertex: %g %g %g\n", vert->x, vert->y, vert->z);
			}
		}
		else if (equal_strings ("face", elem_name)) 
		{
			ply_get_property ( thePlyFile, elem_name, &face_props[0]);
	
			for (int j = 0; j < num_elems; j++) 
			{
				Face *theFace = (Face *) malloc (sizeof (Face));
				ply_get_element (thePlyFile, theFace);

				vector <int> vIndx;
				for (int aK =0; aK < theFace->nverts; ++aK)
				{
					vIndx.push_back(theFace->verts[aK]);
				}

				addTriangle(cTriangle(vIndx));
			}
		}
	}
	
	ply_close (thePlyFile);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cMesh::addPt(const Pt3dr &aPt)
{
	mVertexes.push_back(aPt);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cMesh::addTriangle(const cTriangle &aTri)
{
	mTriangles.push_back(aTri);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

//Sets angle between Dir and Triangle if in TriIdx
void cMesh::setTrianglesAttribute(int img_idx, Pt3dr Dir, vector <unsigned int> const &TriIdx)
{
	for (int aK=0 ; aK < TriIdx.size() ; aK++)
	{
		cTriangle *aTri = getTriangle(TriIdx[aK]);
	
		Pt3dr A, B, C;

		Pt3dr aNormale = aTri->getNormale(*this, true);

		float cosAngle = scal(Dir, aNormale) / euclid(Dir);

		//aTri->setAttribute(pair <int, float> (img_idx, cosAngle));
	}
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// cZBuf
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cZBuf::cZBuf
(
Pt2di aSz
) :
mImMask     (1,1),
mImTriIdx   (1,1),
mRes        (1,1),
mDataRes    (0),
mSzRes		(aSz)
{
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cZBuf::~cZBuf(){}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

Im2D_REAL4 cZBuf::BasculerUnMaillage(cMesh &aMesh, cElNuage3DMaille const &aNuage, float aDef)
{
	mRes = Im2D_REAL4(mSzRes.x,mSzRes.y,aDef);
    mDataRes = mRes.data();

	vector <cTriangle> vTriangles;
	aMesh.getTriangles(vTriangles);
	
	for (int aK =0; aK<vTriangles.size();++aK)
	{
		BasculerUnTriangle(vTriangles[aK], aMesh, aNuage, true);
	}

	return mRes;
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cZBuf::BasculerUnTriangle(cTriangle &aTri, cMesh &aMesh, cElNuage3DMaille const &aNuage, bool doMask)
{
	vector <Pt3dr> Sommets;
	aTri.getVertexes(aMesh, Sommets);
	
	if (Sommets.size() == 3 )
	{
		//Projection du terrain dans l'image
		Pt2dr A2  =  aNuage.Terrain2Index(Sommets[0]);
		Pt2dr B2  =  aNuage.Terrain2Index(Sommets[1]);
		Pt2dr C2  =  aNuage.Terrain2Index(Sommets[2]);
		 
		Pt2dr AB = B2-A2;
		Pt2dr AC = C2-A2;
		REAL aDet = AB^AC;

		if (aDet==0) return;

		Pt2di A2i = round_down(A2);
		Pt2di B2i = round_down(B2);
		Pt2di C2i = round_down(C2);

		 //On verifie que le triangle se projete entierement dans l'image
		 //TODO: gerer les triangles de bord
		if (A2i.x < 0 || B2i.x < 0 || C2i.x < 0 || A2i.y < 0 || B2i.y < 0 || C2i.y < 0 || A2i.x >= mSzRes.x || B2i.x >= mSzRes.x || C2i.x >= mSzRes.x || A2i.y >= mSzRes.y  || B2i.y >= mSzRes.y  || C2i.y >= mSzRes.y)
			 return;

		REAL zA = aNuage.ProfOfIndex(A2i); //A verifier
		REAL zB = aNuage.ProfOfIndex(B2i);
		REAL zC = aNuage.ProfOfIndex(C2i);

		Pt2di aP0 = round_down(Inf(A2,Inf(B2,C2)));
		aP0 = Sup(aP0,Pt2di(0,0));
		Pt2di aP1 = round_up(Sup(A2,Sup(B2,C2)));
		aP1 = Inf(aP1,mSzRes-Pt2di(1,1));

		for (INT x=aP0.x ; x<= aP1.x ; x++)
			for (INT y=aP0.y ; y<= aP1.y ; y++)
			{
				Pt2dr AP = Pt2dr(x,y)-A2;

				// Coordonnees barycentriques DE P(x,y)
				REAL aPdsB = (AP^AC) / aDet;
				REAL aPdsC = (AB^AP) / aDet;
				REAL aPdsA = 1 - aPdsB - aPdsC;
				if ((aPdsA>-Eps) && (aPdsB>-Eps) && (aPdsC>-Eps))
				{
					REAL4 aZ = (float) (zA *aPdsA  + zB* aPdsB + zC *aPdsC);
					if (aZ>mDataRes[y][x])
					{
						mDataRes[y][x] = aZ;
						//mImTriInv.set(x,y,aDet<0);
					   
						/* for (int aK=0;aK<attributes.size(); ++aK)
						{
							mImAttrOut[aK]->SetR(Pt2di(x,y),attributes[aK]);
						}*/
					}
				}
			}
	}
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

set <unsigned int> cZBuf::getTrianglesIndexes()
{
	set <unsigned int> Res;

	for (int aK=0 ; aK < mSzRes.x ; aK++)
	{
		for (int bK=0 ; bK < mSzRes.y ; bK++)
		{
			unsigned int Idx; //= mImTriIdx.get(aK,bK);
			if ((Idx != mDef) && (Res.find(Idx) == Res.end())) Res.insert(Idx);
		}
	}

	return Res;
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cZBuf::ComputeMask(int img_idx, cMesh &aMesh, cElNuage3DMaille const &aNuage, vector <unsigned int> const &TriIdx)
{
	for (int aK=0 ; aK < TriIdx.size() ; aK++)
	{
		cTriangle *aTri = aMesh.getTriangle(TriIdx[aK]);

		//if (aTri->Attributes.size())
		{
			float bestAngle = MESH_MAX_ANGLE;  //on force l'angle à être au moins inférieur à PI/4
			int   bestImage  = -1;

			//for (int bK=0; bK < aTri->Attributes.size(); ++bK)
			{
				pair <int, float> aP;// = aTri->Attributes[bK];

				if (aP.second > bestAngle)
				{
					bestAngle = aP.second;
					bestImage = aP.first;
				}
			}

			if ((bestAngle > MESH_MAX_ANGLE) && (bestImage == img_idx))
			{
				BasculerUnTriangle(*aTri, aMesh, aNuage, false); //soit on bascule en mode ZBuffer (true), soit en mode Masque (false)
			}
		}
	}
}
