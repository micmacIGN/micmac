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

#include "StdAfx.h"
#include "../private/cElNuage3DMaille.h"
#include "../../src/uti_phgrm/GraphCut/MaxFlow/maxflow.h"

class cMesh;
class cVertex;
class cTriangle;
class cEdge;
class cZBuf;

typedef Graph <float,float,float> RGraph;


class cTextureBox2d : public Box2d<int>
{
public:

    cTextureBox2d (std::vector <int> aTriangles, int idx);

    void  setRect(int aImgIdx, Pt2di aP0, Pt2di aP1);
    void  setTransfo(const Pt2di &tr, bool rot);

    int imgIdx;

    bool  isRotated;          // has texture been rotated
    Pt2di translation;  // position of texture in full texture image

    std::vector<int> triangles;

    bool    operator==( const cTextureBox2d & ) const;
};

class cMesh
{
    friend class cTriangle;

    public:
                        cMesh(const string & Filename, bool doAdjacence=true);
                        cMesh(cMesh const &aMesh);

                        ~cMesh();

        void        initDefValue(float aVal);

        int			getVertexNumber() const	{return (int) mVertexes.size();}
        int			getFacesNumber()  const	{return (int) mTriangles.size();}
        int			getEdgesNumber()  const	{return (int) mEdges.size();}

        vector <cTriangle> *	getTriangles() { return &mTriangles; }

        cVertex *   getVertex(unsigned int idx);
        cTriangle *	getTriangle(unsigned int idx);
        cEdge *     getEdge(unsigned int idx);

        void		addPt(const Pt3dr &aPt);
        void		addTriangle(const cTriangle &aTri);
        void		addEdge(int aK, int bK);

        void        removeTriangle(int idx, bool doAdjacence = true);
        void        removeTriangle(cTriangle &aTri, bool doAdjacence = true);

        void        clean();

        vector < cTextureBox2d > getRegions();

        void        write(const string & aOut, bool aBin, const string & textureFilename="");

        void        Export(string aOut, set <int> const &triangles, bool aBin=false);

private:

        void        checkTriangle(int id2, vector<int>::const_iterator it, int aK);
        void        checkEdgesForVertex(int id, int aK);

        vector <cVertex>	mVertexes;
        vector <cTriangle>	mTriangles;
        vector <cEdge>	    mEdges;			//aretes du graphe de voisinage

        set <pair <int,int> > mEdgesSet;
};

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

class cVertex
{
    public:
                    cVertex(const Pt3dr & pt);

                    ~cVertex();

        void		getPos(Pt3dr &pos){ pos = mPos; }
        void		modPos(Pt3dr &pos){ mPos = pos; }
        vector<int> *  getTriIdx() { return &mTriIdx; }
        void        addIdx(int id) { if (find(mTriIdx.begin(), mTriIdx.end(), id) == mTriIdx.end()) mTriIdx.push_back(id); }

        bool    operator==( const cVertex & ) const;

    private:

        Pt3dr          mPos;
        vector<int>    mTriIdx;
};

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

class cTriangle
{
    public:
                cTriangle(cMesh* aMesh, sFace * face, int TriIdx);

                ~cTriangle();

        void    setMesh(cMesh* aMesh) { pMesh = aMesh; }

        void    addEdge(int idx);
        void    removeEdge(int idx);

        Pt3dr	getNormale(bool normalize = false) const;
        void	getVertexes(vector <Pt3dr> &vList) const;

        void	getVertexesIndexes(vector <int> &vList) const {vList = mTriVertex;}
        void	getVertexesIndexes(int &v1, int &v2, int &v3);

        void    setIdx(int id) { mTriIdx = id; }
        int		Idx() const {return mTriIdx;}

        size_t  getEdgesNumber() { return mTriEdges.size(); }

        vector <int>*  getEdgesIndex() { return &mTriEdges; }
        vector<cTriangle *> getNeighbours(); //get 3 neighbors (via edges)
        vector<int> getNeighbours2(); //get neighbors (via vertex)

        void    setEdgeIndex(unsigned int pos, int val);
        void    setVertexIndex(unsigned int pos, int val);

        static int     getDefTextureImgIndex() { return mDefImIdx; }

        void    setBestImgIndex(int val) { mBestImIdx = val; }
        int     getBestImgIndex() { return mBestImIdx; }
       // int     getBestImgIndexAfter(int aK);

        void    setTextureCoordinates(const Pt2dr &p0, const Pt2dr &p1, const Pt2dr &p2);
        void    getTextureCoordinates(Pt2dr &p0, Pt2dr &p1, Pt2dr &p2);

        bool    isTextured() { return mBestImIdx != mDefImIdx; }

        bool    operator==( const cTriangle & ) const;

        void    insertCriter(int aK, float aVal); //set criterion value for index aK
        float   getCriter(int aK);
        float   getBestCriter();
        void    showMap(); //debug only

        Pt3dr   meanTexture(CamStenope *, Tiff_Im &); // mean texture inside triangle

        void    setDefValue(float aVal) { mDefValue = aVal; }

private:

        int							mTriIdx;		// triangle index
        vector <int>				mTriVertex;		// index of vertexes in pMesh->mVertexes
        vector <int>                mTriEdges;      // index of edges in pMesh->Edges
        static const int            mDefImIdx = -1; // default value of image index
        int                         mBestImIdx;     // texture image index

        cMesh       *               pMesh;

        Pt2dr                       mText0;         // Texture Coordinates
        Pt2dr                       mText1;
        Pt2dr                       mText2;

        float                       mDefValue;      // Default value of criterion for choosing best image for texturing (threshold for angle)

        map <int, float>            mMapCriter;     // Map linking image index to criterion
};

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
class cEdge
{
    public:
                 cEdge(int tri1, int tri2){mNode1 = tri1; mNode2 = tri2;}

                ~cEdge();

        int		n1(){return mNode1;}
        int		n2(){return mNode2;}

        void    setN1(int aK) { mNode1 = aK; }
        void    setN2(int aK) { mNode2 = aK; }

        void    decN1() { mNode1--; }
        void    decN2() { mNode2--; }

        bool operator==( const cEdge & ) const;

    private:

        int mNode1; // triangle index
        int mNode2;
};

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

class cZBuf
{
    public:
                cZBuf(Pt2di sz = Pt2di(0,0), float defVal = 0.f, int aScale=1.f);

                ~cZBuf();

        void	BasculerUnMaillage(cMesh &aMesh);			//Projection du maillage dans la geometrie de aNuage, aDef: valeur par defaut de l'image resultante
        void    BasculerUnMaillage(cMesh &aMesh, CamStenope const & aCam);

        void		BasculerUnTriangle(cTriangle &aTri, bool doMask = false); //compute ZBuffer, or Mask (true)

        Im2D_BIN	ComputeMask(int img_idx, cMesh &aMesh);

        void getVisibleTrianglesIndexes(set<int> &setIdx);

        cElNuage3DMaille * &	Nuage() {return mNuage;}

        void					setSelfSz(){mSzRes = mNuage->SzUnique();} //temp

        Pt2di					Sz(){return mSzRes / mScale;}

        Im2D_REAL4*             get() { return &mRes; }

        void                    write(string filename);
        void                    writeImLabel(string filename);

    private:

        Pt2di					mSzRes;			//size result

        Im2D_INT4				mImTriIdx;		//triangle index image (label image)
        Im2D_BIN				mImMask;		//mask image

        Im2D_REAL4				mRes;			//Zbuffer
        float **				mDataRes;

        float					mDpDef;			//default value for depth img (mRes)
        int                     mIdDef;			//default value for label img (mImTriIdx)

        cElNuage3DMaille *		mNuage;

        int                     mScale;         //Downscale factor
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
