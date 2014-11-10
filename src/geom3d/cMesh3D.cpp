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

static const REAL Eps = 1e-7;

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// cTriangle
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTriangle::~cTriangle(){}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTriangle::cTriangle(vector <int> const &idx, int TriIdx)
{
    mInside = false;
    mIndexes = idx;
    mTriIdx  = TriIdx;
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTriangle::cTriangle(int idx1, int idx2, int idx3, int TriIdx)
{
    mInside = false;

    mIndexes.push_back(idx1);
    mIndexes.push_back(idx2);
    mIndexes.push_back(idx3);

    mTriIdx  = TriIdx;
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

bool cTriangle::getAttributes(int image_idx, vector <REAL> &ta) const
{
    map <int, vector <REAL> >::const_iterator it;

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

void cTriangle::setAttributes(int image_idx, const vector <REAL> &ta)
{
    mAttributes.insert(pair <int, vector <REAL> > (image_idx, ta) );
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cTriangle::getVertexesIndexes(int &v1, int &v2, int &v3)
{
    //ELISE_ASSERT
    v1 = mIndexes[0];
    v2 = mIndexes[1];
    v3 = mIndexes[2];
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

REAL cTriangle::computeEnergy(int img_idx)
{
    REAL diff;

    //angle entre la direction de visee de l'image idx et la normale au triangle
    REAL angle0 = (mAttributes.find(img_idx))->second[0];

    REAL min = PI;
    map<int,vector <REAL> >::const_iterator it;
    for (it = mAttributes.begin(); it != mAttributes.end(); it++)
    {
        if (it->first != img_idx)
        {
            diff = abs(angle0 - ((*it).second)[0]);

            if (diff < min) min = diff;
        }
    }

    return PI - min;
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// cMesh
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cMesh::~cMesh(){}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

Pt3dr cMesh::getVertex(unsigned int idx) const
{
    ELISE_ASSERT(idx < mVertexes.size(), "cMesh3D.cpp cMesh::getPt, out of vertex array");

    return mVertexes[idx];
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTriangle* cMesh::getTriangle(unsigned int idx)
{
    ELISE_ASSERT(idx < mTriangles.size(), "cMesh3D.cpp cMesh::getTriangle, out of faces array");

    return &(mTriangles[idx]);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cEdge cMesh::getEdge(unsigned int idx) const
{
    ELISE_ASSERT(idx < mEdges.size(), "cMesh3D.cpp cMesh::getEdge, out of edges array");

    return mEdges[idx];
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

static PlyProperty face_props[] =
{
    { "vertex_indices" , PLY_INT , PLY_INT , offsetof(PlyFace,vertices) , 1 , PLY_UCHAR, PLY_UCHAR , offsetof(PlyFace,nr_vertices) },
};

PlyProperty props[] =
{
    {"x", PLY_FLOAT, PLY_FLOAT, offsetof(sVertex,x), 0, 0, 0, 0},
    {"y", PLY_FLOAT, PLY_FLOAT, offsetof(sVertex,y), 0, 0, 0, 0},
    {"z", PLY_FLOAT, PLY_FLOAT, offsetof(sVertex,z), 0, 0, 0, 0},
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
    int cpt;
    int id0, id1, id2;
    int idc0, idc1; //index des sommets communs
    id0 = id1 = id2 = idc0 = idc1 = -1;

    thePlyFile = ply_open_for_reading( const_cast<char *>(Filename.c_str()), &nelems, &elist, &file_type, &version);

    ELISE_ASSERT(thePlyFile != NULL, "cMesh3D.cpp: cMesh::cMesh, cannot open ply file for reading");

    for (int i = 0; i < nelems; i++)
    {
        elem_name = elist[i];
        ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);

        //printf ("element %s %d\n", elem_name, num_elems);

        if (equal_strings ("vertex", elem_name))
        {
            ply_get_property (thePlyFile, elem_name, &props[0]);
            ply_get_property (thePlyFile, elem_name, &props[1]);
            ply_get_property (thePlyFile, elem_name, &props[2]);

            for (int j = 0; j < num_elems; j++)
            {
                sVertex *vert = (sVertex *) malloc (sizeof(sVertex));

                ply_get_element (thePlyFile, vert);

                //ajout du point
                addPt(Pt3dr(vert->x, vert->y, vert->z));

                //printf ("vertex: %g %g %g\n", vert->x, vert->y, vert->z);
            }
        }
        else if (equal_strings ("face", elem_name))
        {
            ply_get_property ( thePlyFile, elem_name, &face_props[0]);
            vector <int> vIndx;

            for (int j = 0; j < num_elems; j++)
            {
                sFace *theFace = (sFace *) malloc (sizeof (sFace));
                ply_get_element (thePlyFile, theFace);

                vIndx.clear();
                for (int aK =0; aK < theFace->nverts; ++aK)
                {
                    vIndx.push_back(theFace->verts[aK]);
                }

                //remplissage du graphe d'adjacence
                for (int k = 0; k < getFacesNumber(); ++k)
                {
                    mTriangles[k].getVertexesIndexes(id0, id1, id2);

                    cpt = 0;
                    if((vIndx[0] == id0)||(vIndx[1] == id0)||(vIndx[2] == id0)) {cpt++; idc0 = id0;}
                    if((vIndx[0] == id1)||(vIndx[1] == id1)||(vIndx[2] == id1))
                    {
                        if (cpt) idc1 = id1;
                        else	 idc0 = id1;

                        cpt++;
                    }
                    if((vIndx[0] == id2)||(vIndx[1] == id2)||(vIndx[2] == id2))
                    {
                        if (cpt) idc1 = id2;
                        else	 idc0 = id2;

                        cpt++;
                    }

                    if (cpt == 2)
                    {
                        #ifdef _DEBUG
                            printf ("found adjacent triangles : %d %d - vertex : %d %d\n", j, k, idc0, idc1);
                        #endif

                        addEdge(cEdge(k, j, idc0, idc1));
                    }
                }

                //ajout du triangle
                addTriangle(cTriangle(vIndx, j));
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

void cMesh::addEdge(const cEdge &aEdge)
{
    mEdges.push_back(aEdge);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

//Calcule et stocke l'angle entre Dir et Triangle (appartenant a TriIdx)
void cMesh::setTrianglesAttribute(int img_idx, Pt3dr Dir, vector <unsigned int> const &aTriIdx)
{
    for (unsigned int aK=0; aK < aTriIdx.size(); aK++)
    {
        cTriangle *aTri = getTriangle(aTriIdx[aK]);

        Pt3dr aNormale = aTri->getNormale(*this, true);

        double cosAngle = scal(Dir, aNormale) / euclid(Dir);

        vector <double> vAttr;
        vAttr.push_back(cosAngle);

        aTri->setAttributes(img_idx, vAttr);
    }
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cMesh::setGraph(int img_idx, RGraph &aGraph, vector <int> &aTriInGraph, vector <unsigned int> const &aVisTriIdx)
{
    int id1, id2, pos1, pos2;
    float E0, E1, E2;

    //parcours des aretes du graphe d'adjacence
    for (unsigned int aK=0; aK < mEdges.size(); aK++)
    {
        id1 = mEdges[aK].n1();
        id2 = mEdges[aK].n2();

        //on recherche id1 et id2 parmi les triangles visibles
        if ((find(aVisTriIdx.begin(), aVisTriIdx.end(), id1)!=aVisTriIdx.end()) &&
            (find(aVisTriIdx.begin(), aVisTriIdx.end(), id2)!=aVisTriIdx.end()))
        {
            //on ajoute seulement les triangles qui ne sont pas encore presents dans le graphe
            if (find(aTriInGraph.begin(), aTriInGraph.end(), id1) == aTriInGraph.end())
            {
                aTriInGraph.push_back(id1);
                aGraph.add_node();
            }

            if (find(aTriInGraph.begin(), aTriInGraph.end(), id2) == aTriInGraph.end())
            {
                aTriInGraph.push_back(id2);
                aGraph.add_node();
            }
        }
    }

    cEdge elEdge;

    //creation des aretes et calcul de leur energie
    for (unsigned int aK=0; aK < mEdges.size(); aK++)
    {
        elEdge = mEdges[aK];

        id1 = elEdge.n1();
        id2 = elEdge.n2();

        vector<int>::iterator it1 = find(aTriInGraph.begin(), aTriInGraph.end(), id1);
        vector<int>::iterator it2 = find(aTriInGraph.begin(), aTriInGraph.end(), id2);

        if ( (it1 != aTriInGraph.end()) && (it2 != aTriInGraph.end()) )
        {
            //pos1 = (int) (it1 - aTriInGraph.begin());
            //pos2 = (int) (it2 - aTriInGraph.begin());

            pos1 = (int) distance(aTriInGraph.begin(), it1);
            pos2 = (int) distance(aTriInGraph.begin(), it2);

            //energies de l'arete triangle-source et de l'arete triangle-puit
            cTriangle *Tri1 = getTriangle(id1);
            cTriangle *Tri2 = getTriangle(id2);

            E1 = (float)Tri1->computeEnergy(img_idx);
            E2 = (float)Tri2->computeEnergy(img_idx);

            //if (E1 == 0.f)
            //	aGraph.add_tweights( pos1, 0.f, 1.f );
            //else
            //{
                aGraph.add_tweights( pos1, (float)(mLambda*E1), 0.f );
            //}

            //if (E2 == 0.f)
            //	aGraph.add_tweights( pos2, 0.f, 1.f );
            //else
            //{
                aGraph.add_tweights( pos2, (float)(mLambda*E2), 0.f );
            //}

            //energie de l'arete inter-triangles


            //longueur^2 de l'arete coupee par elEdge
            E0 = (float)square_euclid( getVertex( elEdge.v1() ), getVertex( elEdge.v2() ) );

            aGraph.add_edge(pos1, pos2, E0, E0);
            //aGraph.add_edge(pos1, pos2, 1, 1);
        }
    }

#ifdef oldoldold
    for (unsigned int aK=0; aK < aTriInGraph.size(); aK++)
    {
        cTriangle *Tri = getTriangle(aTriInGraph[aK]);

        E = Tri->computeEnergy(img_idx);
        if (E == 0.f)
            aGraph.add_tweights( aK, 1.f, 0.f );
        else
        {
            aGraph.add_tweights( aK, mLambda*E, 0.f );
            //aGraph.add_tweights( aK, 1.f, 0.f );
            /*if (Tri->isInside())
                aGraph.add_tweights( aK, 0.f, 1.f );
            else
                aGraph.add_tweights( aK, 1.f, 0.f );*/
        }
    }
#endif

}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// cZBuf
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cZBuf::cZBuf() :
mSzRes		(1,1),
mImTriIdx   (1,1),
mImMask     (1,1),
mRes        (1,1),
mDataRes    (0),
mDpDef      (0.f),
mIdDef      (0),
mNuage		(0)
{
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cZBuf::~cZBuf(){}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

Im2D_REAL4 cZBuf::BasculerUnMaillage(cMesh const &aMesh)
{
    mRes = Im2D_REAL4(mSzRes.x,mSzRes.y,mDpDef);
    mDataRes = mRes.data();
    mImTriIdx = Im2D_U_INT2(mSzRes.x,mSzRes.y, mIdDef);

    vector <cTriangle> vTriangles;
    aMesh.getTriangles(vTriangles);

    for (unsigned int aK =0; aK<vTriangles.size();++aK)
    {
        BasculerUnTriangle(vTriangles[aK], aMesh);
    }

    return mRes;
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cZBuf::BasculerUnTriangle(cTriangle &aTri, cMesh const &aMesh, bool doMask)
{
    vector <Pt3dr> Sommets;
    aTri.getVertexes(aMesh, Sommets);

    if (Sommets.size() == 3 )
    {
        //Projection du terrain vers l'image
        Pt3dr A3 = mNuage->Euclid2ProfAndIndex(Sommets[0]);
        Pt3dr B3 = mNuage->Euclid2ProfAndIndex(Sommets[1]);
        Pt3dr C3 = mNuage->Euclid2ProfAndIndex(Sommets[2]);

        Pt2dr A2(A3.x, A3.y);
        Pt2dr B2(B3.x, B3.y);
        Pt2dr C2(C3.x, C3.y);

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

        REAL zA = A3.z;
        REAL zB = B3.z;
        REAL zC = C3.z;

        Pt2di aP0 = round_down(Inf(A2,Inf(B2,C2)));
        aP0 = Sup(aP0,Pt2di(0,0));
        Pt2di aP1 = round_up(Sup(A2,Sup(B2,C2)));
        aP1 = Inf(aP1,mSzRes-Pt2di(1,1));

        if (doMask)
        {
            for (INT x=aP0.x ; x<= aP1.x ; x++)
                for (INT y=aP0.y ; y<= aP1.y ; y++)
                {
                    Pt2dr AP = Pt2dr(x,y)-A2;

                    // Coordonnees barycentriques de P(x,y)
                    REAL aPdsB = (AP^AC) / aDet;
                    REAL aPdsC = (AB^AP) / aDet;
                    REAL aPdsA = 1 - aPdsB - aPdsC;
                    if ((aPdsA>-Eps) && (aPdsB>-Eps) && (aPdsC>-Eps))
                    {
                        mImMask.set(x, y, 1);
                    }
                }
        }
        else
        {
            for (INT x=aP0.x ; x<= aP1.x ; x++)
                for (INT y=aP0.y ; y<= aP1.y ; y++)
                {
                    Pt2dr AP = Pt2dr(x,y)-A2;

                    // Coordonnees barycentriques de P(x,y)
                    REAL aPdsB = (AP^AC) / aDet;
                    REAL aPdsC = (AB^AP) / aDet;
                    REAL aPdsA = 1 - aPdsB - aPdsC;
                    if ((aPdsA>-Eps) && (aPdsB>-Eps) && (aPdsC>-Eps))
                    {
                        REAL4 aZ = (float) (zA*aPdsA + zB*aPdsB + zC*aPdsC);
                        if (aZ>mDataRes[y][x])
                        {
                            mDataRes[y][x] = aZ;
                            mImTriIdx.SetI(Pt2di(x,y),aTri.getIdx());
                        }
                    }
                }
        }
    }
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cZBuf::ComputeVisibleTrianglesIndexes()
{
    for (int aK=0; aK < mSzRes.x; aK++)
    {
        for (int bK=0; bK < mSzRes.y; bK++)
        {
            unsigned int Idx = (unsigned int) mImTriIdx.Val(aK,bK);

            if ((Idx != mIdDef) && (find(vTri.begin(), vTri.end(), Idx)==vTri.end())) vTri.push_back(Idx);
        }
    }
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

Im2D_BIN cZBuf::ComputeMask(int img_idx, cMesh &aMesh)
{
    mImMask = Im2D_BIN (mSzRes.x, mSzRes.y, 0);

    #ifdef _DEBUG
        printf ("nb triangles : %d\n", vTri.size());
    #endif

    for (unsigned int aK=0; aK < vTri.size(); aK++)
    {
        cTriangle *aTri = aMesh.getTriangle(vTri[aK]);

        if (aTri->hasAttributes())
        {
            REAL bestAngle = mMaxAngle;
            int  bestImage = -1;
            REAL Angle;

            map<int, vector <REAL> > aMap = aTri->getAttributesMap();

            map<int, vector <REAL> >::const_iterator it;
            for (it = aMap.begin();it != aMap.end(); it++)
            {
                vector <REAL> vAttr = it->second;

                if (vAttr[0] < 0.f) Angle = PI - acos(vAttr[0]);
                else Angle = acos(vAttr[0]);

                if (Angle < bestAngle)
                {
                    bestAngle = Angle;
                    bestImage = it->first;
                }
            }

            if ((bestAngle < mMaxAngle) && (bestImage == img_idx))
            //if (bestAngle < mMaxAngle)
            {
                BasculerUnTriangle(*aTri, aMesh, true);
                aTri->setInside();
            }
        }
    }

    return mImMask;
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

Im2D_BIN cZBuf::ComputeMask(vector <int> const &TriInGraph, RGraph &aGraph, cMesh &aMesh)
{
    mImMask = Im2D_BIN (mSzRes.x, mSzRes.y, 0);

    /*printf(" taille de TriIngraph :  %d", TriInGraph.size());
    printf(" aGraph.get_node_num  :  %d", aGraph.get_node_num());*/

    for (unsigned int aK=0; aK < TriInGraph.size(); aK++)
    {
        if (aGraph.what_segment(aK) == RGraph::SOURCE) BasculerUnTriangle(*(aMesh.getTriangle(TriInGraph[aK])), aMesh, true);
    }

    return mImMask;
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// cEdge
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cEdge::cEdge()
{
    mNode1 = -1;
    mNode2 = -1;
    mV1	   = -1;
    mV2    = -1;
}

cEdge::~cEdge(){}
