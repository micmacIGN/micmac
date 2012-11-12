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

Pt3dr cTriangle::getNormale(cMesh &elMesh, bool normalize)
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

void cTriangle::getVertexes(cMesh &elMesh, vector <Pt3dr> &vList)
{
	for (unsigned int aK =0; aK < mIndexes.size(); ++aK)
	{
		vList.push_back(elMesh.getVertex(mIndexes[aK]));
	}
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

bool cTriangle::getAttributes(int image_idx, vector <float> &ta)
{
	map <int, vector <float>>::iterator it;

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

Pt3dr cMesh::getVertex(int idx)
{
	ELISE_ASSERT(idx < mVertexes.size(), "cMesh3D.cpp cMesh::getPt, out of vertex array");
	
	return mVertexes[idx];
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTriangle cMesh::getTriangle(int idx)
{
	ELISE_ASSERT(idx < mTriangles.size(), "cMesh3D.cpp cMesh::getTriangle, out of faces array");
	
	return mTriangles[idx];
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
