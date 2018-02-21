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
// cTextRect
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTextureBox2d::cTextureBox2d(std::vector<int> aTriangles, int idx):
    imgIdx(idx),
    isRotated(false),
    triangles(aTriangles)
{}

void cTextureBox2d::setRect(int aImgIdx, Pt2di aP0, Pt2di aP1)
{
    imgIdx = aImgIdx;
    Box2d<int>::_p0 = aP0;
    Box2d<int>::_p1 = aP1;
}

void cTextureBox2d::setTransfo(const Pt2di &tr, bool rot)
{
    translation = tr;
    isRotated  = rot;
}

bool cTextureBox2d::operator==( cTextureBox2d const & aTR ) const
{
    return  (imgIdx == aTR.imgIdx) &&
            (Box2d<int>::_p0 == aTR._p0) &&
            (Box2d<int>::_p1 == aTR._p1) &&
            (isRotated == aTR.isRotated) &&
            (translation == aTR.translation) &&
            (triangles == aTR.triangles);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// cTriangle
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTriangle::~cTriangle(){}

void cTriangle::addEdge(int idx)
{
    mTriEdges.push_back(idx);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTriangle::cTriangle(cMesh* aMesh, sFace * face, int TriIdx):
    mTriIdx(TriIdx),
    mBestImIdx(mDefImIdx),
    pMesh(aMesh),
    mText0(Pt2dr()),
    mText1(Pt2dr()),
    mText2(Pt2dr())
{
    mTriVertex.push_back(face->verts[0]);
    mTriVertex.push_back(face->verts[1]);
    mTriVertex.push_back(face->verts[2]);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

Pt3dr cTriangle::getNormale(bool normalize) const
{
    vector <Pt3dr> vPts;
    getVertexes(vPts);

    if (normalize)
    {
        Pt3dr p1 = vunit(vPts[1]-vPts[0]);
        Pt3dr p2 = vunit(vPts[2]-vPts[0]);
        return vunit(p1^p2);
    }
    else return (vPts[1]-vPts[0])^(vPts[2]-vPts[0]);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cTriangle::getVertexes(vector <Pt3dr> &vList) const
{
    Pt3dr pt;
    for (unsigned int aK =0; aK < mTriVertex.size(); ++aK)
    {
        pMesh->getVertex(mTriVertex[aK])->getPos(pt);
        vList.push_back(pt);
    }
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cTriangle::getVertexesIndexes(int &v1, int &v2, int &v3)
{
    //ELISE_ASSERT
    v1 = mTriVertex[0];
    v2 = mTriVertex[1];
    v3 = mTriVertex[2];
}


vector<cTriangle *> cTriangle::getNeighbours()
{
    vector <cTriangle*> res;

    for (unsigned int aK=0; aK<mTriEdges.size(); aK++)
    {
        cEdge *edge = pMesh->getEdge(mTriEdges[aK]);

        if (edge->n1() == mTriIdx) res.push_back(pMesh->getTriangle(edge->n2()));
        else if (edge->n2() == mTriIdx) res.push_back(pMesh->getTriangle(edge->n1()));
    }

    return res;
}

vector<int> cTriangle::getNeighbours2()
{
    vector <int> res;

    for (unsigned int aK=0; aK<mTriVertex.size(); aK++)
    {
        cVertex *vertex = pMesh->getVertex(mTriVertex[aK]);

        vector <int> *neighb = vertex->getTriIdx();
        vector<int>::const_iterator it = neighb->begin();
        for (;it!=neighb->end();++it)
            if (*it != mTriIdx) res.push_back(*it);
    }

    return res;
}

void cTriangle::setEdgeIndex(unsigned int pos, int val)
{
    #if _DEBUG
        ELISE_ASSERT(pos < mTriEdges.size(), "cTriangle::setEdgeIndex in cMesh3D.cpp")
    #endif

    mTriEdges[pos] = val;
}

void cTriangle::setVertexIndex(unsigned int pos, int val)
{
    #if _DEBUG
        ELISE_ASSERT(pos < mTriVertex.size(), "cTriangle::setVertexIndex in cMesh3D.cpp")
    #endif

    mTriVertex[pos] = val;
}

void cTriangle::setTextureCoordinates(Pt2dr const &p0, Pt2dr const &p1, Pt2dr const &p2)
{
    mText0 = p0;
    mText1 = p1;
    mText2 = p2;
}

void cTriangle::getTextureCoordinates(Pt2dr &p0, Pt2dr &p1, Pt2dr &p2)
{
    p0 = mText0;
    p1 = mText1;
    p2 = mText2;
}

void cTriangle::removeEdge(int idx)
{
    #if _DEBUG
        ELISE_ASSERT(idx < mTriEdges.size(), "cTriangle::removeEdge in cMesh3D.cpp")
    #endif

    mTriEdges.erase(std::remove(mTriEdges.begin(), mTriEdges.end(), idx), mTriEdges.end());
}

bool cTriangle::operator==( const cTriangle &aTr ) const
{
    return ( (mTriIdx     ==  aTr.mTriIdx)   &&
             (mTriVertex  ==  aTr.mTriVertex) &&
             (mTriEdges   ==  aTr.mTriEdges)  &&
             (mBestImIdx  ==  aTr.mBestImIdx) &&
             (mText0      ==  aTr.mText0) &&
             (mText1      ==  aTr.mText1) &&
             (mText2      ==  aTr.mText2) &&
             (mMapCriter  ==  aTr.mMapCriter)
             );
}

void cTriangle::insertCriter(int aK, float aVal)
{
    mMapCriter.insert(pair <int, float> (aK, aVal));
}

float cTriangle::getCriter(int aK)
{
    map<int,float>::iterator it = mMapCriter.find(aK);

    if (it != mMapCriter.end())
        return it->second;
    else
        return mDefValue;
}

float cTriangle::getBestCriter()
{
    return getCriter(mBestImIdx);
}

void cTriangle::showMap()
{
    cout << "********************* " << mTriIdx << " ************************* " << endl;
    map <int, float>::const_iterator it = mMapCriter.begin();
    for (;it != mMapCriter.end() ;++it)
    {
        cout << "Criter " << (*it).first << " "<< (*it).second << endl;
    }
    cout << endl;
}

/*int cTriangle::getBestImgIndexAfter(int aK)
{
    int res = -1;
    map<int,float>::iterator it = mMapCriter.find(aK);

    if (it != mMapCriter.end())
    {
        float min = FLT_MAX;

        map <int, float>::const_iterator itr = mMapCriter.begin();
        for (;itr != mMapCriter.end() ;++itr)
        {
            if (itr != it)
            {
                if (itr->second < min)
                {
                    min = itr->second;
                    res = itr->first;
                }
            }
        }
    }
    return res;
}*/

Pt3dr cTriangle::meanTexture(CamStenope *aCam, Tiff_Im &aImg)
{
    vector <Pt3dr> Vertex;
    getVertexes(Vertex);

    Pt2dr A2 = aCam->R3toF2(Vertex[0]);             //projection des sommets du triangle
    Pt2dr B2 = aCam->R3toF2(Vertex[1]);
    Pt2dr C2 = aCam->R3toF2(Vertex[2]);

    if (aCam->IsInZoneUtile(A2) && aCam->IsInZoneUtile(B2) && aCam->IsInZoneUtile(C2))
    {
        //float sum = 0.f;
        Pt3dr sum;
        int cptPx = 0;

        /*cout << "img proj = "<< endl;
        cout << A2 << endl;
        cout << B2 << endl;
        cout << C2 << endl;*/

        Pt2dr AB = B2-A2;
        Pt2dr AC = C2-A2;
        REAL aDet = AB^AC;

        if (aDet!=0)
        {
            Pt2di aP0 = round_down(Inf(A2,Inf(B2,C2)));
            aP0 = Sup(aP0,Pt2di(0,0));
            Pt2di aP1 = round_up(Sup(A2,Sup(B2,C2)));
            aP1 = Inf(aP1,aCam->Sz()-Pt2di(1,1));

            if (aImg.nb_chan() == 3)
            {
                Pt2di aSz(aP1-aP0);

                Im2D_U_INT1 aImR(aSz.x,aSz.y,0);
                Im2D_U_INT1 aImG(aSz.x,aSz.y,0);
                Im2D_U_INT1 aImB(aSz.x,aSz.y,0);

                //Crop in aImg from aP0 to aP1
                ELISE_COPY
                (
                    rectangle(Pt2di(0,0), aSz),
                    trans(aImg.in_proj(), aP0),
                    Virgule(aImR.out(),aImG.out(),aImB.out())
                );

                U_INT1 ** aDataR = aImR.data();
                U_INT1 ** aDataG = aImG.data();
                U_INT1 ** aDataB = aImB.data();

                for (int ay=0; ay<aSz.y; ++ay)
                {
                    for (int ax=0; ax<aSz.x; ++ax)
                    {
                        Pt2dr AP = Pt2dr(ax + aP0.x, ay + aP0.y)-A2;

                        // Coordonnees barycentriques de P(x,y)
                        REAL aPdsB = (AP^AC) / aDet;
                        REAL aPdsC = (AB^AP) / aDet;
                        REAL aPdsA = 1 - aPdsB - aPdsC;

                        if ((aPdsA>-Eps) && (aPdsB>-Eps) && (aPdsC>-Eps)) // on est a l'intérieur du triangle
                        {
                            /* unsigned char red   = aDataR[ay][ax];
                            unsigned char green = aDataG[ay][ax];
                            unsigned char blue  = aDataB[ay][ax];

                            cout << "red = " << (int) red << endl;
                            cout << "gre = " << (int) green << endl;
                            cout << "blu = " << (int) blue << endl;
                            cout << "****************" << endl;*/

                            //sum += 0.2126*(float)red + 0.7152*(float)green + 0.0722*(float)blue;
                            sum.x += (float) aDataR[ay][ax];
                            sum.y += (float) aDataG[ay][ax];
                            sum.z += (float) aDataB[ay][ax];
                            cptPx++;
                        }
                    }
                }
            }
        }

        if (cptPx) return sum / (float) cptPx;
    }

    return Pt3dr(1e9,1e9,1e9);
}


//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// cMesh
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cMesh::~cMesh(){}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cMesh::initDefValue(float aVal)
{
    const int nTri = (int)mTriangles.size();
    for (int aK=0; aK < nTri;++aK)
        mTriangles[aK].setDefValue(aVal);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cVertex* cMesh::getVertex(unsigned int idx)
{
    #if _DEBUG
        ELISE_ASSERT(idx < mVertexes.size(), "cMesh3D.cpp cMesh::getVertex, out of vertex array");
    #endif

    return &(mVertexes[idx]);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTriangle* cMesh::getTriangle(unsigned int idx)
{
    #if _DEBUG
        ELISE_ASSERT(idx < mTriangles.size(), "cMesh3D.cpp cMesh::getTriangle, out of faces array");
    #endif

    return &(mTriangles[idx]);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cEdge* cMesh::getEdge(unsigned int idx)
{
    #if _DEBUG
        ELISE_ASSERT(idx < mEdges.size(), "cMesh3D.cpp cMesh::getEdge, out of edges array");
    #endif

    return &(mEdges[idx]);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

static PlyProperty face_props[] =
{
    { "vertex_indices" , PLY_INT , PLY_INT , offsetof(ElPlyFace,vertices) , 1 , PLY_UCHAR, PLY_UCHAR , offsetof(ElPlyFace,nr_vertices) },
};

PlyProperty props[] =
{
    {"x", PLY_FLOAT, PLY_FLOAT, offsetof(sVertex,x), 0, 0, 0, 0},
    {"y", PLY_FLOAT, PLY_FLOAT, offsetof(sVertex,y), 0, 0, 0, 0},
    {"z", PLY_FLOAT, PLY_FLOAT, offsetof(sVertex,z), 0, 0, 0, 0},
};

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cMesh::checkTriangle(int id2, vector<int>::const_iterator it, int aK)
{
    cVertex * vert2 = getVertex(id2);
    vector<int> *tri2  = vert2->getTriIdx();

    if (find(tri2->begin(), tri2->end(), aK) != tri2->end())
    {
        #ifdef _DEBUG
            printf ("found adjacent triangles : %d %d\n", aK, *it);
        #endif

        addEdge(aK, *it);
    }
}

void cMesh::checkEdgesForVertex(int id, int aK)
{
    cVertex * vert = getVertex(id);
    vector<int> * tri = vert->getTriIdx();

    int id0, id1, id2;
    vector<int>::const_iterator it = tri->begin();
    for(;it != tri->end();it++)
    {
         if (*it != aK)
         {
             mTriangles[*it].getVertexesIndexes(id0, id1, id2);
             //cout << "vertex indexes = " << id0 << " " << id1 << " " << id2 << endl;

             if (id != id0) checkTriangle(id0, it, aK);
             if (id != id1) checkTriangle(id1, it, aK);
             if (id != id2) checkTriangle(id2, it, aK);
         }
    }
}

cMesh::cMesh(const std::string & Filename, bool doAdjacence)
{
    PlyFile * thePlyFile;
    int nelems;
    char **elist;
    int file_type;
    float version;
    int nprops;
    int num_elems;
    char *elem_name;

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

            for (int j = 0; j < num_elems; j++)
            {
                sFace *face = (sFace *) malloc (sizeof (sFace));
                ply_get_element (thePlyFile, face);

                //ajout du triangle
                addTriangle(cTriangle(this, face, j));

                getVertex(face->verts[0])->addIdx(j);
                getVertex(face->verts[1])->addIdx(j);
                getVertex(face->verts[2])->addIdx(j);
            }
        }
    }
    ply_close (thePlyFile);

    if (doAdjacence) //Edges
    {
        int id0a, id1a, id2a;

        const int nFaces = (int)mTriangles.size();
        for (int aK = 0; aK < nFaces; ++aK)
        {
            mTriangles[aK].getVertexesIndexes(id0a, id1a, id2a);

            checkEdgesForVertex(id0a, aK);
            checkEdgesForVertex(id1a, aK);
            checkEdgesForVertex(id2a, aK);
        }
    }
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cMesh::addPt(const Pt3dr &aPt)
{
    mVertexes.push_back(cVertex(aPt));
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cMesh::addTriangle(const cTriangle &aTri)
{
    mTriangles.push_back(aTri);
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cMesh::addEdge(int aK, int bK)
{
    if (aK > bK) std::swap(aK, bK); //reordering in growing order

    std::pair<std::set<pair <int,int> >::iterator,bool> ret = mEdgesSet.insert(pair <int,int> (aK,bK));

    if (ret.second)
    {
        int idx = (int)mEdges.size();
        mEdges.push_back(cEdge (aK, bK));

        mTriangles[aK].addEdge(idx);
        mTriangles[bK].addEdge(idx);
    }
}

void cMesh::removeTriangle(int idx, bool doAdjacence)
{
    removeTriangle(mTriangles[idx], doAdjacence);
}

void cMesh::removeTriangle(cTriangle &aTri, bool doAdjacence)
{
    int triIndex = aTri.Idx();

    if (doAdjacence)
    {
        vector <int> *edges = aTri.getEdgesIndex();

       /* cout << "triangle à retirer= " << triIndex << endl;
        cout << "nombre d'edges à retirer =  " << edges.size() << endl;*/

       /* for (unsigned int aK=0; aK< edges.size(); aK++)
        {
            cout << "index des edges à retirer = " << edges[aK] << endl;
            cout << " entre " << mEdges[edges[aK]].n1() << " et " << mEdges[edges[aK]].n2() <<endl;
        }*/

        const int nTriangles = (int)mTriangles.size();
        for (unsigned int aK=0; aK< edges->size(); aK++)
        {
            int edgeIndex = (*edges)[aK];

            cEdge *e = getEdge(edgeIndex);

           // cout << "Edge " << edgeIndex << "between " << e->n1() << " "  << e->n2() << endl;

            int idx = -2;
            if (triIndex == e->n1()) idx = e->n2();
            else if (triIndex == e->n2()) idx = e->n1();

            if (idx != -2)
            {
                mTriangles[idx].removeEdge(edgeIndex);

                int backEdgeIdx = (int)(mEdges.size() - 1);
                mEdges[edgeIndex] = mEdges.back();
                mEdges.pop_back();

                for (int bK=0;bK < nTriangles; bK++ )
                {
                    vector <int> *vIdx = getTriangle(bK)->getEdgesIndex();
                    for(unsigned int cK=0; cK< vIdx->size();++cK)
                    {
                        if ((*vIdx)[cK] == backEdgeIdx) getTriangle(bK)->setEdgeIndex(cK, edgeIndex);
                    }
                }

                for (unsigned int bK=aK+1; bK < edges->size();++bK)
                {
                    if ((*edges)[bK] == backEdgeIdx) (*edges)[bK] = edgeIndex;
                }
                aK--;
            }
        }
    }

    int backTriangleIdx = (int)(mTriangles.size() - 1);
    mTriangles[triIndex] = mTriangles.back();
    mTriangles.pop_back();

    mTriangles[triIndex].setIdx(triIndex);

    if (doAdjacence)
    {
        const int nbEdges = (int)mEdges.size();
        for (int aK=0; aK < nbEdges;++aK)
        {
            cEdge *e = getEdge(aK);
            if (e->n1() == backTriangleIdx) e->setN1(triIndex);
            else if (e->n2() == backTriangleIdx) e->setN2(triIndex);
        }
    }

    const int nbVertex = (int)mVertexes.size();
    for(int aK=0;aK < nbVertex;++aK)
    {
        vector<int> *triIdx = getVertex(aK)->getTriIdx();

        for(unsigned int bK=0;bK < triIdx->size();++bK)
        {
            if ((*triIdx)[bK] == backTriangleIdx) (*triIdx)[bK] = triIndex;
        }
    }
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cMesh::clean()
{
    //searching for triangles to remove
   std::set < int > triangleIdxSet;
   std::set < int, std::greater<int> > toRemove;

   const int nFaces = getFacesNumber();
   for (int aK=0; aK < nFaces;++aK)
   {
       std::vector <int> myList;
       cTriangle * triangle = getTriangle(aK);
       if ((!triangle->isTextured()) && (triangle->getEdgesNumber() <3) && (triangleIdxSet.find(aK) == triangleIdxSet.end()))
           myList.push_back(aK);

       for (unsigned int bK=0; bK < myList.size();++bK)
       {
           cTriangle * Tri = getTriangle(myList[bK]);

           if (!Tri->isTextured())
           {
               vector<int> neighb = Tri->getNeighbours2();

               bool found = false;
               vector<int>::const_iterator it = neighb.begin();
               for(;it!=neighb.end();++it)
               {
                   if ((triangleIdxSet.find(*it) == triangleIdxSet.end()) &&
                           (!getTriangle(*it)->isTextured()))
                   {
                       found = true;
                       myList.push_back(*it);

                       triangleIdxSet.insert(*it);
                   }
               }
               if (found)
                   triangleIdxSet.insert(Tri->Idx());
           }
       }

       vector<int>::const_iterator it = myList.begin();
       for(;it!=myList.end();++it) toRemove.insert(*it);
   }

    cout << "Removing " << toRemove.size() << " / " << nFaces << " faces" <<endl;

    std::set < int, std::greater<int> >::const_iterator itr = toRemove.begin();
    int aCpt = (int)toRemove.size();
    for (; itr != toRemove.end(); ++itr)
    {
         removeTriangle(mTriangles[*itr]);
         aCpt--;
         if ((aCpt%100)==0) std::cout << "removeTriangle still " << aCpt << " to do \n";
    }

    //cout << "Removing isolated vertex" << endl;

    //suppression des points n'appartenant à aucun triangle
    const int nbFaces = getFacesNumber();
    for(int aK=0; aK < getVertexNumber();++aK)
    {
        bool found = false;
        for(int i=0 ; i < nbFaces; i++)
        {
            int vertex1, vertex2, vertex3;
            getTriangle(i)->getVertexesIndexes(vertex1, vertex2, vertex3);

            if ((aK==vertex1) || (aK==vertex2) || (aK==vertex3))
            {
                found = true;
                break;
            }
        }

        if (!found) //remove this point
        {
            //cout << "removing vertex : " << aK << endl;

            int backVertexIdx = (int)(mVertexes.size() - 1);
            mVertexes[aK] = mVertexes.back();
            mVertexes.pop_back();

            for(int i=0 ; i < nbFaces; i++)
            {
                cTriangle * tri= getTriangle(i);
                int vertex1, vertex2, vertex3;
                tri->getVertexesIndexes(vertex1, vertex2, vertex3);

                if (vertex1==backVertexIdx) tri->setVertexIndex(0, aK);
                else if (vertex2==backVertexIdx) tri->setVertexIndex(1, aK);
                else if (vertex3==backVertexIdx) tri->setVertexIndex(2, aK);
            }
            aK--;
        }
    }
}

vector<cTextureBox2d> cMesh::getRegions()
{
    std::set < int > triangleIdxSet;
    std::vector < cTextureBox2d > regions;

    const int nFaces = getFacesNumber();
    for (int aK=0; aK < nFaces;++aK)
    {
        std::vector <int> myList;
        int imgIdx = -1;
        if ((getTriangle(aK)->isTextured()) && (triangleIdxSet.find(aK) == triangleIdxSet.end()))
        {
            imgIdx = getTriangle(aK)->getBestImgIndex();
            myList.push_back(aK);
        }

        for (unsigned int bK=0; bK < myList.size();++bK)
        {
            cTriangle * Tri = getTriangle(myList[bK]);

            if (Tri->isTextured())
            {
                //cout <<"getRegions2" << endl;
                vector<int> neighb = Tri->getNeighbours2(); //CRASH ICI
                //cout <<"getRegions3" << endl;

                bool found = false;
                vector<int>::const_iterator it = neighb.begin();
                for(;it!=neighb.end();++it)
                {
                    if ((triangleIdxSet.find(*it) == triangleIdxSet.end()) &&
                            (getTriangle(*it)->getBestImgIndex() == imgIdx) )
                    {
                        found = true;
                        myList.push_back(*it);

                        triangleIdxSet.insert(*it);
                    }
                }
                if (found)
                    triangleIdxSet.insert(Tri->Idx());
            }
        }

        if (myList.size() >= 1)
        {
            regions.push_back(cTextureBox2d(myList, imgIdx));
        }
    }

    return regions;
}

void cMesh::write(const string & aOut, bool aBin, const string & textureFilename)
{
    bool hasTexture = textureFilename != "";

    string mode = aBin ? "wb" : "w";
    string aBinSpec = MSBF_PROCESSOR() ? "binary_big_endian":"binary_little_endian" ;

    FILE * file = FopenNN(aOut, mode, "cMesh::write");
    fprintf(file,"ply\n");
    fprintf(file,"format %s 1.0\n",aBin?aBinSpec.c_str():"ascii");
    if (hasTexture) fprintf(file,"comment TextureFile %s\n", textureFilename.c_str());
    fprintf(file,"element vertex %i\n", getVertexNumber());
    fprintf(file,"property float x\n");
    fprintf(file,"property float y\n");
    fprintf(file,"property float z\n");
    fprintf(file,"element face %i\n",getFacesNumber());
    fprintf(file,"property list uchar int vertex_indices\n");
    if (hasTexture) fprintf(file,"property list uchar float texcoord\n");
    fprintf(file,"end_header\n");

    Pt3dr pt;
    if (aBin)
    {
        const int nVertex = getVertexNumber();
        for(int aK=0; aK< nVertex; aK++)
        {
            getVertex(aK)->getPos(pt);

            WriteType(file,float(pt.x));
            WriteType(file,float(pt.y));
            WriteType(file,float(pt.z));
        }

        const int nFaces = getFacesNumber();
        for(int aK=0; aK< nFaces; aK++)
        {
            cTriangle * face = &(mTriangles[aK]);

            int t1, t2, t3;
            face->getVertexesIndexes(t1, t2, t3);

            WriteType(file,(unsigned char)3);
            WriteType(file,t1);
            WriteType(file,t2);
            WriteType(file,t3);

            if (hasTexture)
            {
                WriteType(file,(unsigned char)6);

                if (face->isTextured())
                {
                    Pt2dr p1, p2, p3;
                    face->getTextureCoordinates(p1, p2, p3);

                    WriteType(file,(float) p1.x);
                    WriteType(file,(float) p1.y);
                    WriteType(file,(float) p2.x);
                    WriteType(file,(float) p2.y);
                    WriteType(file,(float) p3.x);
                    WriteType(file,(float) p3.y);
                }
                else
                {
                    WriteType(file,0.f);
                    WriteType(file,0.f);
                    WriteType(file,0.f);
                    WriteType(file,0.f);
                    WriteType(file,0.f);
                    WriteType(file,0.f);
                }
            }
        }
    }
    else
    {
        const int nVertex = getVertexNumber();
        for(int aK=0; aK< nVertex; aK++)
        {
            getVertex(aK)->getPos(pt);
            fprintf(file,"%.7f %.7f %.7f\n",pt.x,pt.y,pt.z);
        }

        const int nFaces = getFacesNumber();
        for(int aK=0; aK< nFaces; aK++)
        {
            cTriangle * face = &(mTriangles[aK]);
            int t1, t2, t3;
            face->getVertexesIndexes(t1, t2, t3);

            fprintf(file,"3 %i %i %i ",t1,t2,t3);

            if (hasTexture)
            {
                if (face->isTextured())
                {
                    Pt2dr p1, p2, p3;
                    face->getTextureCoordinates(p1, p2, p3);
                    fprintf(file,"6 %f %f %f %f %f %f\n",p1.x,p1.y,p2.x,p2.y,p3.x,p3.y);
                }
                else
                    fprintf(file,"6 0 0 0 0 0 0\n");
            }
        }
    }
}

void cMesh::Export(string aOut, set<int> const &triangles, bool aBin)
{
    /*
    string mode = "w";  //"a";

    FILE * file = FopenNN(aOut, mode, "cMesh::Export");
    fprintf(file,"ply\n");
    fprintf(file,"format ascii 1.0\n");
    fprintf(file,"element vertex %i\n", (int) triangles.size()*3);
    fprintf(file,"property float x\n");
    fprintf(file,"property float y\n");
    fprintf(file,"property float z\n");
    fprintf(file,"element face %i\n",(int) triangles.size());
    fprintf(file,"property list uchar int vertex_indices\n");
    fprintf(file,"end_header\n");

    Pt3dr pt;

    std::set<int>::const_iterator it = triangles.begin();
    for(;it!=triangles.end();++it)
    {
        cTriangle* face = getTriangle(*it);

        vector <Pt3dr> Pts;
        face->getVertexes(Pts);

        for(unsigned int aK=0; aK<Pts.size();++aK)
        {
            pt = Pts[aK];

            fprintf(file,"%.7f %.7f %.7f\n",pt.x,pt.y,pt.z);
        }
    }

    it = triangles.begin();
    for(int bK=0;it!=triangles.end();++it)
    {
        fprintf(file,"3 %i %i %i\n",bK,bK+1,bK+2);
        bK+=3;
    }
*/
    // ==== test write binary file

    string mode = aBin ? "wb" : "w";
    string aBinSpec = MSBF_PROCESSOR() ? "binary_big_endian":"binary_little_endian" ;

    FILE * file = FopenNN(aOut, mode, "cMesh::Export");
    fprintf(file,"ply\n");
    fprintf(file,"format %s 1.0\n",aBin?aBinSpec.c_str():"ascii");
    fprintf(file,"element vertex %i\n", (int) triangles.size()*3);
    fprintf(file,"property float x\n");
    fprintf(file,"property float y\n");
    fprintf(file,"property float z\n");
    fprintf(file,"element face %i\n",(int) triangles.size());
    fprintf(file,"property list uchar int vertex_indices\n");
    fprintf(file,"end_header\n");

    Pt3dr pt;

    if (aBin)
    {
        std::set<int>::const_iterator it = triangles.begin();
        for(;it!=triangles.end();++it)
        {
            cTriangle* face = getTriangle(*it);

            vector <Pt3dr> Pts;
            face->getVertexes(Pts);

            for(unsigned int aK=0; aK<Pts.size();++aK)
            {
                pt = Pts[aK];
                WriteType(file,float(pt.x));
                WriteType(file,float(pt.y));
                WriteType(file,float(pt.z));
            }
        }
        it = triangles.begin();
        for(int bK=0;it!=triangles.end();++it)
        {
            WriteType(file,(unsigned char)3);
            WriteType(file,bK);
            WriteType(file,bK+1);
            WriteType(file,bK+2);
            bK+=3;
        }
    }
    else
    {
        std::set<int>::const_iterator it = triangles.begin();
        for(;it!=triangles.end();++it)
        {
            cTriangle* face = getTriangle(*it);

            vector <Pt3dr> Pts;
            face->getVertexes(Pts);

            for(unsigned int aK=0; aK<Pts.size();++aK)
            {
                pt = Pts[aK];

                fprintf(file,"%.7f %.7f %.7f\n",pt.x,pt.y,pt.z);
            }
        }

        it = triangles.begin();
        for(int bK=0;it!=triangles.end();++it)
        {
            fprintf(file,"3 %i %i %i\n",bK,bK+1,bK+2);
            bK+=3;
        }
    }
}


//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// cZBuf
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cZBuf::cZBuf(Pt2di sz, float defVal, int aScale) :
mSzRes		(sz),
mImTriIdx   (1,1),
mImMask     (1,1),
mRes        (1,1),
mDataRes    (0),
mDpDef      (defVal),
mIdDef      (INT_MAX),
mNuage		(0),
mScale      (aScale)
{
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cZBuf::~cZBuf(){}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cZBuf::BasculerUnMaillage(cMesh &aMesh)
{
    mRes = Im2D_REAL4(mSzRes.x,mSzRes.y,mDpDef);
    mDataRes = mRes.data();
    mImTriIdx = Im2D_INT4(mSzRes.x,mSzRes.y, mIdDef);

    vector <cTriangle>* vTriangles = aMesh.getTriangles();

    const int nbTriangles = (int)vTriangles->size();
    for (int aK=0; aK<nbTriangles;++aK)
    {
        BasculerUnTriangle((*vTriangles)[aK]);
    }
}

void cZBuf::BasculerUnMaillage(cMesh &aMesh, const CamStenope &aCam)
{
    Pt2di SzRes = mSzRes / mScale;
    mRes = Im2D_REAL4(SzRes.x,SzRes.y,mDpDef);
    mDataRes = mRes.data();
    mImTriIdx = Im2D_INT4(SzRes.x,SzRes.y, mIdDef);

    vector <cTriangle> *vTriangles = aMesh.getTriangles();

    const int nbTriangles = (int)vTriangles->size();
    for (int aK=0; aK<nbTriangles;++aK)
    {
        cTriangle aTri = (*vTriangles)[aK];

        vector <Pt3dr> Sommets;
        aTri.getVertexes(Sommets);

        if (Sommets.size() == 3)
        {
            Pt2dr A2 = aCam.R3toF2(Sommets[0]) / (float) mScale;
            Pt2dr B2 = aCam.R3toF2(Sommets[1]) / (float) mScale;
            Pt2dr C2 = aCam.R3toF2(Sommets[2]) / (float) mScale;

            Pt2dr AB = B2-A2;
            Pt2dr AC = C2-A2;
            REAL aDet = AB^AC;

            if (aDet!=0)
            {
                Pt2di A2i = round_down(A2);
                Pt2di B2i = round_down(B2);
                Pt2di C2i = round_down(C2);

                 //On verifie que le triangle se projete entierement dans l'image
                 //TODO: gerer les triangles de bord
                if (A2i.x >= 0 && B2i.x >= 0 && C2i.x >= 0 && A2i.y >= 0 && B2i.y >= 0 && C2i.y >= 0 && A2i.x < SzRes.x && B2i.x < SzRes.x && C2i.x < SzRes.x && A2i.y < SzRes.y  && B2i.y < SzRes.y  && C2i.y < SzRes.y)
                {
                    Pt3dr center = aCam.OrigineProf();
                    REAL zA = euclid(Sommets[0] - center);  //repris de ElNuage3DMaille ProfOfPtE()
                    REAL zB = euclid(Sommets[1] - center);
                    REAL zC = euclid(Sommets[2] - center);

                    Pt2di aP0 = round_down(Inf(A2,Inf(B2,C2)));
                    aP0 = Sup(aP0,Pt2di(0,0));
                    Pt2di aP1 = round_up(Sup(A2,Sup(B2,C2)));
                    aP1 = Inf(aP1,SzRes-Pt2di(1,1));

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
                                if (aZ<mDataRes[y][x])
                                {
                                    mDataRes[y][x] = aZ;
                                    mImTriIdx.SetI(Pt2di(x,y),aTri.Idx());
                                }
                            }
                        }
                }
            }
        }
    }
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cZBuf::BasculerUnTriangle(cTriangle &aTri, bool doMask)
{
    vector <Pt3dr> Sommets;
    aTri.getVertexes(Sommets);

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
                            mImTriIdx.SetI(Pt2di(x,y),aTri.Idx());
                        }
                    }
                }
        }
    }
}

void cZBuf::getVisibleTrianglesIndexes(set<int> & setIdx)
{
    Pt2di sz = mImTriIdx.sz();
    for (int aK=0; aK < sz.x; aK++)
    {
        for (int bK=0; bK < sz.y; bK++)
        {
            int Idx = mImTriIdx.GetI(Pt2di(aK,bK));

            if (Idx != mIdDef)  setIdx.insert(Idx);
        }
    }
}

void cZBuf::write(string filename)
{
     //conversion du zBuffer en 8 bits
    Pt2di sz = mRes.sz(); //aZBuffer.Sz();
    Im2D_U_INT1 Converted(sz.x, sz.y);
    REAL min = FLT_MAX;
    REAL max = 0.f;
    for (int cK=0; cK < sz.x;++cK)
    {
        for (int bK=0; bK < sz.y;++bK)
        {
            REAL val = mRes.GetR(Pt2di(cK,bK));

            if (val != mDpDef)
            {
                max = ElMax(val, max);
                min = ElMin(val, min);
            }
        }
    }

    printf ("Min, max depth = %4.2f %4.2f\n", min, max );

    for (int cK=0; cK < sz.x;++cK)
        for (int bK=0; bK < sz.y;++bK)
            Converted.SetI(Pt2di(cK,bK),(int)((mRes.GetR(Pt2di(cK,bK))-min) *255.f/(max-min)));

    printf ("Saving %s\n", filename.c_str());
    Tiff_Im::CreateFromIm(Converted, filename);
    printf ("Done\n");
}

void cZBuf::writeImLabel(string filename)
{
    //conversion de l'img de label en 8 bits
    Pt2di sz = mImTriIdx.sz();
    Im2D_U_INT1 LConverted(sz.x, sz.y);
    int lmin = INT_MAX;
    int lmax = 0;
    for (int cK=0; cK < sz.x;++cK)
    {
        for (int bK=0; bK < sz.y;++bK)
        {
            int val = mImTriIdx.GetI(Pt2di(cK,bK));

            if (val != INT_MAX)
            {
                lmax = ElMax(val, lmax);
                lmin = ElMin(val, lmin);
            }
        }
    }

    for (int cK=0; cK < sz.x;++cK)
        for (int bK=0; bK < sz.y;++bK)
            LConverted.SetI(Pt2di(cK,bK),(int)((mImTriIdx.GetI(Pt2di(cK,bK))-lmin) *255.f/(lmax-lmin)));

    printf ("Saving %s\n", filename.c_str());
    Tiff_Im::CreateFromIm(LConverted, filename);
    printf ("Done\n");
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// cEdge
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cEdge::~cEdge(){}

bool cEdge::operator==(const cEdge & e) const
{
    return ((mNode1 == e.mNode1) &&
            (mNode2 == e.mNode2) );
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// cVertex
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------


cVertex::cVertex(const Pt3d<double> &pt):
    mPos(pt)
{}

bool cVertex::operator==(const cVertex & v) const
{
    return ((mPos == v.mPos) &&
            (mTriIdx == v.mTriIdx));
}

cVertex::~cVertex(){}
