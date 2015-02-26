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
#include "../../../CodeExterne/Poisson/include/PlyFile.h"

static const REAL Eps = 1e-7;

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// cTextRect
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cTextRect::cTextRect(std::vector<int> aTriangles):
    imgIdx(-1),
    p0(Pt2di(0,0)),
    p1(Pt2di(0,0)),
    rotation(false),
    translation(Pt2di(0,0)),
    triangles(aTriangles)
{}

void cTextRect::setRect(int aImgIdx, Pt2di aP0, Pt2di aP1)
{
    imgIdx = aImgIdx;
    p0 = aP0;
    p1 = aP1;
}

void cTextRect::setTransfo(const Pt2di &tr, bool rot)
{
    translation = tr;
    rotation  = rot;
}

bool cTextRect::operator==( const cTextRect & aTR ) const
{
    return (imgIdx == aTR.imgIdx) &&
            (p0 == aTR.p0) &&
            (p1 == aTR.p1) &&
            (rotation == aTR.rotation) &&
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
    mInside(false),
    mTriIdx(TriIdx),
    mTextImIdx(mDefTextImIdx),
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
    v1 = mTriVertex[0];
    v2 = mTriVertex[1];
    v3 = mTriVertex[2];
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

void cTriangle::setEdgeIndex(unsigned int pos, int val)
{
    if (mTriEdges.size()>pos)
        mTriEdges[pos] = val;
}

void cTriangle::setVertexIndex(unsigned int pos, int val)
{
    if (mTriVertex.size()>pos)
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
    //bool found = false;

    //TODO: remove
    /*for (unsigned int aK=0; aK < mTriEdges.size();++aK)
    {
        //cout << "Edge =  " << mTriEdges[aK] << endl;
        if (mTriEdges[aK] == idx )
        {
            //cout<< "found ****************************" << endl;
            found = true;
        }
    }

    if (found)
    {*/
        //cout << "removing edge "<< idx << endl;
        mTriEdges.erase(std::remove(mTriEdges.begin(), mTriEdges.end(), idx), mTriEdges.end());
    //}
    /*cout << "new index list= "<<endl;

    for (int aK=0; aK< (int) mTriEdges.size();++aK)
        cout << mTriEdges[aK] << " ";

    cout << endl;*/
}

bool cTriangle::operator==( const cTriangle &aTr ) const
{
    return ( (mInside     ==  aTr.mInside )  &&
             (mTriIdx     ==  aTr.mTriIdx)   &&
             (mTriVertex  ==  aTr.mTriVertex) &&
             (mTriEdges   ==  aTr.mTriEdges)  &&
             (mTextImIdx  ==  aTr.mTextImIdx) &&
             (mAttributes ==  aTr.mAttributes)
             );
}


//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// cMesh
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cMesh::~cMesh(){}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

cVertex* cMesh::getVertex(unsigned int idx)
{
    #if _DEBUG
        ELISE_ASSERT(idx < mVertexes.size(), "cMesh3D.cpp cMesh::getPt, out of vertex array");
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

void cMesh::checkTriangle(int id2, set<int>::const_iterator it, int aK)
{
    cVertex * vert2 = getVertex(id2);
    set<int> tri2 = vert2->getTriIdx();

    if (tri2.find(aK) != tri2.end())
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
    set<int> tri = vert->getTriIdx();

    int id0, id1, id2;
    set<int>::const_iterator it = tri.begin();
    for(;it != tri.end();it++)
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

    if (doAdjacence) //remplissage du graphe d'adjacence
    {
        //int cpt;

        int id0a, id1a, id2a;
        /*int idc0, idc1; //index des sommets communs
        id0a = id1a = id2a = idc0 = idc1 = -1;
        id0b = id1b = id2b = -2;*/

        const int nFaces = mTriangles.size();
        for (int aK = 0; aK < nFaces; ++aK)
        {
            mTriangles[aK].getVertexesIndexes(id0a, id1a, id2a);

            checkEdgesForVertex(id0a, aK);
            checkEdgesForVertex(id1a, aK);
            checkEdgesForVertex(id2a, aK);

           /* for (int bK=aK+1; bK < nFaces; ++bK)
            {
                mTriangles[bK].getVertexesIndexes(id0b, id1b, id2b);

                cpt = 0;
                if((id0b == id0a)||(id1b == id0a)||(id2b == id0a)) {cpt++; idc0 = id0a;}
                if((id0b == id1a)||(id1b == id1a)||(id2b == id1a))
                {
                    if (cpt) idc1 = id1a;
                    else	 idc0 = id1a;

                    cpt++;
                }
                if((id0b == id2a)||(id1b == id2a)||(id2b == id2a))
                {
                    if (cpt) idc1 = id2a;
                    else	 idc0 = id2a;

                    cpt++;
                }

                if (cpt == 2)
                {
                    #ifdef _DEBUG
                        printf ("found adjacent triangles : %d %d - vertex : %d %d\n", aK, bK, idc0, idc1);
                    #endif

                    addEdge(aK, bK, idc0, idc1);
                }
            }*/
        }
    }
}

cMesh::cMesh(const cMesh &aMesh):
    mVertexes(aMesh.mVertexes),
    mTriangles(aMesh.mTriangles),
    mEdges(aMesh.mEdges),
    mLambda(aMesh.mLambda)
{}

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
        int idx = mEdges.size();
        mEdges.push_back(cEdge (aK, bK));

        //cout << "adding edge " << idx << endl;
        mTriangles[aK].addEdge(idx);
        mTriangles[bK].addEdge(idx);
    }
}

void cMesh::removeTriangle(cTriangle &aTri)
{
    vector <int> edges = aTri.getEdgesIndex();
    int index = aTri.getIdx();

   /* cout << "triangle à retirer= " << index << endl;
    cout << "nombre d'edges à retirer =  " << edges.size() << endl;

    for (unsigned int aK=0; aK< edges.size(); aK++)
    {
        cout << "index des edges à retirer = " << edges[aK] << " entre " << mEdges[edges[aK]].n1() << " et " << mEdges[edges[aK]].n2() <<endl;
    }*/

    const int nTriangles = mTriangles.size();
    for (unsigned int aK=0; aK< edges.size(); aK++)
    {
        int edgeIndex = edges[aK];

        cEdge *e = getEdge(edgeIndex);

        //cout << "Edge " << edgeIndex << "between " << e.n1() << " "  << e.n2() << endl;

        int idx = -1;
        if (index == e->n1()) idx = e->n2();
        else if (index == e->n2()) idx = e->n1();

        //cout << "looking for edge " << edgeIndex << " between " << e.n1() << " and " << e.n2() << endl;

        if (idx != -1)
        {

            //cout << "idx = " << idx << " / " << mTriangles.size() << endl;
            //cout << "aK = " << aK  << endl;
            mTriangles[idx].removeEdge(edgeIndex);
            //cout << "ok " << endl;

            for (int bK=0;bK < nTriangles; bK++ )
            {
                vector <int> vIdx = getTriangle(bK)->getEdgesIndex();
                for(unsigned int cK=0; cK< vIdx.size();++cK)
                {
                    if (vIdx[cK] > edgeIndex) getTriangle(bK)->setEdgeIndex(cK, vIdx[cK] - 1);
                }
            }

            for (unsigned int bK=aK+1; bK < edges.size();++bK)
            {
                if (edges[bK] >edgeIndex) edges[bK] = edges[bK] -1;
            }

            mEdges.erase(std::remove(mEdges.begin(), mEdges.end(), *e), mEdges.end());

        }
        else
            cout << "impossible error !!!!!!" << endl;
    }

    mTriangles.erase(std::remove(mTriangles.begin(), mTriangles.end(), aTri), mTriangles.end());

    const int nbTriangles = mTriangles.size();
    for (int aK=index;aK < nbTriangles; aK++ )
    {
        getTriangle(aK)->setIdx(getTriangle(aK)->getIdx()-1);
    }

    for (unsigned int aK=0; aK < mEdges.size();++aK)
    {
        cEdge *e = getEdge(aK);
        if (e->n1() > index) e->setN1(e->n1()-1);
        if (e->n2() > index) e->setN2(e->n2()-1);
    }

}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

//Calcule et stocke l'angle entre Dir et Triangle (appartenant a TriIdx)
void cMesh::setTrianglesAttribute(int img_idx, Pt3dr Dir, set <unsigned int> const &aTriIdx)
{
    set <unsigned int>::const_iterator itri  =aTriIdx.begin();
    for(;itri!=aTriIdx.end();itri++)
    {
        cTriangle *aTri = getTriangle(*itri);

        Pt3dr aNormale = aTri->getNormale(true);

        double cosAngle = scal(Dir, aNormale) / euclid(Dir);

        vector <double> vAttr;
        vAttr.push_back(cosAngle);

        aTri->setAttributes(img_idx, vAttr);
    }
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void cMesh::setGraph(int img_idx, RGraph &aGraph, vector <int> &aTriInGraph, set <unsigned int> const &aVisTriIdx)
{
    int id1, id2, pos1, pos2;
    float E0, E1, E2;

    //parcours des aretes du graphe d'adjacence
    for (unsigned int aK=0; aK < mEdges.size(); aK++)
    {
        id1 = mEdges[aK].n1();
        id2 = mEdges[aK].n2();

        //on recherche id1 et id2 parmi les triangles visibles
        if ((aVisTriIdx.find(id1)!=aVisTriIdx.end()) &&
            (aVisTriIdx.find(id2)!=aVisTriIdx.end()))
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

    //creation des aretes et calcul de leur energie
    for (unsigned int aK=0; aK < mEdges.size(); aK++)
    {
        cEdge elEdge = mEdges[aK];

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
            E0 = 0.;//(float)square_euclid( getVertex( elEdge.v1() ), getVertex( elEdge.v2() ) ); //TODO corriger getVertex

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

void cMesh::clean()
{
    cout << "removing triangle" <<endl;
    int nbFaces = getFacesNumber();
    for(int i=0 ; i < nbFaces; i++)
    {
        cTriangle * Triangle = getTriangle(i);

        if (Triangle->getEdgesNumber() < 3 && !Triangle->isTextured())
        {
            //cout <<"remove triangle " << Triangle->getIdx() << " with " << Triangle->getEdgesNumber() << " edges" << endl;

            //cout <<"sommets = " << Triangle->getVertex(0) << " " << Triangle->getVertex(1) << " " << Triangle->getVertex(2) << endl;

            removeTriangle(*Triangle);
            nbFaces--;
            i--;
        }
        /*else if (!Triangle->isTextured())
            cout << "triangle " << i << " nb edges = " << Triangle->getEdgesNumber() << " textured= " << Triangle->isTextured() << endl;*/
    }

    cout << "removing points" << endl;

    //suppression des points n'appartenant à aucun triangle
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
            mVertexes.erase(std::remove(mVertexes.begin(), mVertexes.end(), mVertexes[aK]), mVertexes.end());
            aK--;

            for(int i=0 ; i < nbFaces; i++)
            {
                cTriangle * tri= getTriangle(i);
                int vertex1, vertex2, vertex3;
                tri->getVertexesIndexes(vertex1, vertex2, vertex3);

                if (vertex1>aK) tri->setVertexIndex(0, vertex1-1);
                if (vertex2>aK) tri->setVertexIndex(1, vertex2-1);
                if (vertex3>aK) tri->setVertexIndex(2, vertex3-1);
            }
        }
    }
}

std::vector<cTextRect> cMesh::getRegions()
{
    std::set < int > triangleIdxSet;
    std::vector < cTextRect > regions;

    const int nFaces = getFacesNumber();
    for (int aK=0; aK < nFaces;++aK)
    {
        std::vector <int> myList;
        if ((getTriangle(aK)->isTextured()) && (triangleIdxSet.find(aK) == triangleIdxSet.end()))
            myList.push_back(aK);

        for (unsigned int bK=0; bK < myList.size();++bK)
        {
            //cout << "triangle " << myList[bK] << endl;
            cTriangle * Tri = getTriangle(myList[bK]);

            if (Tri->isTextured())
            {
                int imgIdx = Tri->getTextureImgIndex();

                vector <cTriangle *> neighb = Tri->getNeighbours(); //TODO: tester le remplacement par les triangles vus par les sommets

                bool found = false;
                for (unsigned int cK=0; cK < neighb.size();++cK)
                {
                    int triIdx = neighb[cK]->getIdx();
                    if ((triangleIdxSet.find(triIdx) == triangleIdxSet.end()) &&
                            (neighb[cK]->getTextureImgIndex() == imgIdx))
                    {
                        found = true;
                        myList.push_back(triIdx);

                        triangleIdxSet.insert(triIdx);
                    }
                }
                if (found)
                    triangleIdxSet.insert(Tri->getIdx());
            }
        }

        //cout << "myList.size() = " << myList.size() << endl;

        if (myList.size() > 1)
        {
            regions.push_back(cTextRect(myList));
        }
    }

    //recherche des triangles isolés (trous dans les regions)

//TODO meilleur bouchage des trous
    for (int aK=0; aK < nFaces;++aK)
    {
        if (triangleIdxSet.find(aK) == triangleIdxSet.end())
        {
            cTriangle * Tri = getTriangle(aK);
            vector <cTriangle *> neighb = Tri->getNeighbours();

            if (neighb.size())
            {
                unsigned int nbNeighb = 0; // number of neighbours with same image index
                int neighbIndex  = -1;
                int textImgIndex = -1;

                neighb.push_back(neighb[0]);

                for (unsigned int cK=0; cK < neighb.size()-1;cK++)
                {
                    if ( (neighb[cK]->getTextureImgIndex() == neighb[cK+1]->getTextureImgIndex()))
                    {
                        nbNeighb++;
                        neighbIndex  = neighb[cK]->getIdx();
                        textImgIndex = neighb[cK]->getTextureImgIndex();
                    }
                }

                /*if (nbNeighb == 0)
                {
                    cout << "BAD CANDIDATE= " << triIdx << endl;
                    neighb.pop_back();
                    for (unsigned int cK=0; cK < neighb.size();cK++)
                    {
                        if ( (neighb[cK]->getTextureImgIndex() == Tri->getTextureImgIndex()))
                        {
                            neighbIndex = neighb[cK]->getIdx();
                            textImgIndex = neighb[cK]->getTextureImgIndex();
                        }
                    }
                    cout << "neighbIndex " << neighbIndex << endl;
                    cout << "textImgIndex " << textImgIndex << endl;
                }*/

                if (/*(Tri->getTextureImgIndex() != textImgIndex) &&*/ (nbNeighb >= 1))
                {
                    //recherche de la region des voisins
                    const int nRegions = regions.size();
                    for(int bK=0; bK < nRegions; ++bK)
                    {
                        std::vector <int> region = regions[bK].triangles;

                        if (find(region.begin(), region.end(), neighbIndex) != region.end())
                        {
                            regions[bK].triangles.push_back(aK);
                            Tri->setTextureImgIndex(textImgIndex);
                        }
                    }
                }

            }
            //else cout << "NO NEIGHBOURS!!!!!!" << endl;
        }
    }
    /*cout << "****************** Resultat *********************" << endl;
    cout << endl;

    for (unsigned int aK=0; aK < regions.size() ; ++aK)
    {
        //first triangle of region aK:
        int triIdx = regions[aK][0];
        cTriangle * Tri = getTriangle(triIdx);

        cout << "one region with " << regions[aK].size() << " triangles, for image " << Tri->getTextureImgIndex() << endl;
    }*/

    return regions;
}

void cMesh::write(const string & aOut, bool aBin, const string & textureFilename)
{
    string mode = aBin ? "wb" : "w";
    string aBinSpec = MSBF_PROCESSOR() ? "binary_big_endian":"binary_little_endian" ;

    FILE * file = FopenNN(aOut, mode, "UV Mapping");         //Ecriture du header
    fprintf(file,"ply\n");
    fprintf(file,"format %s 1.0\n",aBin?aBinSpec.c_str():"ascii");
    fprintf(file,"comment UV Mapping generated\n");
    fprintf(file,"comment TextureFile %s\n", textureFilename.c_str());
    fprintf(file,"element vertex %i\n", getVertexNumber());
    fprintf(file,"property float x\n");
    fprintf(file,"property float y\n");
    fprintf(file,"property float z\n");
    fprintf(file,"element face %i\n",getFacesNumber());
    fprintf(file,"property list uchar int vertex_indices\n");
    fprintf(file,"property list uchar float texcoord\n");
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

            if (face->isTextured())
            {
                Pt2dr p1, p2, p3;
                face->getTextureCoordinates(p1, p2, p3);

                WriteType(file,(unsigned char)6);
                WriteType(file,(float) p1.x);
                WriteType(file,(float) p1.y);
                WriteType(file,(float) p2.x);
                WriteType(file,(float) p2.y);
                WriteType(file,(float) p3.x);
                WriteType(file,(float) p3.y);
            }
            else
                WriteType(file,(unsigned char)0);
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

            Pt2dr p1, p2, p3;
            face->getTextureCoordinates(p1, p2, p3);

            if (face->isTextured())
            {
                Pt2dr p1, p2, p3;
                face->getTextureCoordinates(p1, p2, p3);
                fprintf(file,"6 %f %f %f %f %f %f\n",p1.x,p1.y,p2.x,p2.y,p3.x,p3.y);
            }
            else
                fprintf(file,"0\n");
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

void cZBuf::BasculerUnMaillage(cMesh const &aMesh)
{
    mRes = Im2D_REAL4(mSzRes.x,mSzRes.y,mDpDef);
    mDataRes = mRes.data();
    mImTriIdx = Im2D_INT4(mSzRes.x,mSzRes.y, mIdDef);

    vector <cTriangle> vTriangles;
    aMesh.getTriangles(vTriangles);

    for (unsigned int aK =0; aK<vTriangles.size();++aK)
    {
        BasculerUnTriangle(vTriangles[aK]);
    }
}

void cZBuf::BasculerUnMaillage(const cMesh &aMesh, const CamStenope &aCam)
{
    Pt2di SzRes = mSzRes / mScale;
    mRes = Im2D_REAL4(SzRes.x,SzRes.y,mDpDef);
    mDataRes = mRes.data();
    mImTriIdx = Im2D_INT4(SzRes.x,SzRes.y, mIdDef);

    vector <cTriangle> vTriangles;
    aMesh.getTriangles(vTriangles);

    vector <bool> vTrianglesPartiels;  //0= ok 1=partiellement vu ou caché

    for (unsigned int aK =0; aK<vTriangles.size();++aK)
        vTrianglesPartiels.push_back(false);

    for (unsigned int aK =0; aK<vTriangles.size();++aK)
    {
        cTriangle aTri = vTriangles[aK];

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

               /* Pt2di A2i = round_down(A2);
                Pt2di B2i = round_down(B2);
                Pt2di C2i = round_down(C2);

                 //On verifie que le triangle se projete entierement dans l'image
                 //TODO: gerer les triangles de bord
                if (A2i.x < 0 || B2i.x < 0 || C2i.x < 0 || A2i.y < 0 || B2i.y < 0 || C2i.y < 0 || A2i.x >= mSzRes.x || B2i.x >= mSzRes.x || C2i.x >= mSzRes.x || A2i.y >= mSzRes.y  || B2i.y >= mSzRes.y  || C2i.y >= mSzRes.y)
                     return;*/

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
                                int index = mImTriIdx.GetI(Pt2di(x,y));
                                if (index != mIdDef) vTrianglesPartiels[index] = true;
                                mDataRes[y][x] = aZ;
                                mImTriIdx.SetI(Pt2di(x,y),aTri.getIdx());
                            }
                        }
                    }
            }
        }
    }

    //on enleve les triangles partiellement vus
    for(int aK=0; aK < SzRes.x; aK++)
        for (int bK=0; bK < SzRes.y; bK++)
        {
            int index = mImTriIdx.GetI(Pt2di(aK,bK));

            if ((index != mIdDef) && (vTrianglesPartiels[index]))
            {
                mDataRes[bK][aK] = mDpDef;
                mImTriIdx.SetI(Pt2di(aK,bK),mIdDef);
            }
            else if (index != mIdDef) vTri.insert(index);
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
            int Idx = mImTriIdx.Val(aK,bK);

            if (Idx != mIdDef)  vTri.insert(Idx);
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

    set <unsigned int>::const_iterator itri  =vTri.begin();
    for(;itri!=vTri.end();itri++)
    {
        cTriangle *aTri = aMesh.getTriangle(*itri);

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
                BasculerUnTriangle(*aTri, true);
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
        if (aGraph.what_segment(aK) == RGraph::SOURCE) BasculerUnTriangle(*(aMesh.getTriangle(TriInGraph[aK])), true);
    }

    return mImMask;
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

/*cEdge::cEdge()
{
    mNode1 = -1;
    mNode2 = -1;
    mV1	   = -1;
    mV2    = -1;
}*/

cEdge::~cEdge(){}

bool cEdge::operator==(const cEdge & e) const
{
    return ((mNode1 == e.mNode1) &&
            (mNode2 == e.mNode2)/* &&
            (mV1    == e.mV1)    &&
            (mV2    == e.mV2)*/
            );
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
