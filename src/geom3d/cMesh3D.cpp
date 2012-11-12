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

Pt3dr cTriangle::getNormale(cMesh &elMesh, bool normalize)
{
	vector <Pt3dr> vPts;
	getPoints(elMesh, vPts);

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

void cTriangle::getPoints(cMesh &elMesh, vector <Pt3dr> &vList)
{
	for (unsigned int aK =0; aK < mIndexes.size(); ++aK)
	{
		vList.push_back(elMesh.getPt(mIndexes[aK]));
	}
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

bool cTriangle::getAttribute(int image_idx, TriangleAttribute &ta)
{
	map <int, TriangleAttribute>::iterator it;

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

cMesh::cMesh(){}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

Pt3dr cMesh::getPt (int idx)
{
	ELISE_ASSERT(idx < mVertexNumber, "cMesh3D.cpp cMesh::getPt, out of vertex array");
	
	return mPts[idx];
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTriangle cMesh::getTriangle(int idx)
{
	ELISE_ASSERT(idx < mFacesNumber, "cMesh3D.cpp cMesh::getTriangle, out of faces array");
	
	return mTriangles[idx];
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

PlyProperty props[] = { /* list of property information for a vertex */
	{"x", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,x), 0, 0, 0, 0},
	{"y", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,y), 0, 0, 0, 0},
	{"z", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,z), 0, 0, 0, 0},
};

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
	
	for (int i = 0; i < nelems; i++) 
	{
		/* get the description of the first element */
		elem_name = elist[i];
		plist = ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);
				
		/* print the name of the element, for debugging */
		//printf ("element %s %d\n", elem_name, num_elems);
					
		/* if we're on vertex elements, read them in */
		if (equal_strings ("vertex", elem_name)) 
		{		
			/* set up for getting vertex elements */
			ply_get_property (thePlyFile, elem_name, &props[0]);
			ply_get_property (thePlyFile, elem_name, &props[1]);
			ply_get_property (thePlyFile, elem_name, &props[2]);
			
			/* grab all the vertex elements */
			for (int j = 0; j < num_elems; j++) 
			{
				/* grab and element from the file */
				//vlist[j] = (Vertex *) malloc (sizeof (Vertex));
				
				//ply_get_element (thePlyFile, (void *) vlist[j]);
									
				
				
				/* print out vertex x,y,z for debugging */
				//printf ("vertex: %g %g %g %g %g %g\n", vlist[j]->x, vlist[j]->y, vlist[j]->z, vlist[j]->nx, vlist[j]->ny, vlist[j]->nz);
			}
		}
	}
	
	ply_close (thePlyFile);
}