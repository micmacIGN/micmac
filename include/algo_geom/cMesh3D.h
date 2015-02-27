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
//#include "general/ptxd.h"
#include "../private/cElNuage3DMaille.h"
#include "../../src/uti_phgrm/MaxFlow/maxflow.h"

class cMesh;
class cVertex;
class cTriangle;
class cEdge;
class cZBuf;

typedef Graph <float,float,float> RGraph;

class cTextRect
{
public:

    cTextRect(std::vector <int> aTriangles);

    void  setRect(int aImgIdx, Pt2di aP0, Pt2di aP1);
    void  setTransfo(Pt2di const &tr, bool rot);
    int   width() { return p1.x - p0.x; }
    Pt2di size()  { return p1 - p0; }

    int imgIdx;
    Pt2di p0; // coin hg
    Pt2di p1; // coin bd

    bool  rotation; //has texture been rotated
    Pt2di translation; //position of texture in full texture image

    std::vector<int> triangles;

    bool    operator==( const cTextRect & ) const;
};

class cMesh
{
    friend class cTriangle;

    public:
                        cMesh(const string & Filename, bool doAdjacence=true);
                        cMesh(cMesh const &aMesh);

                        ~cMesh();

        int			getVertexNumber() const	{return (int) mVertexes.size();}
        int			getFacesNumber()  const	{return (int) mTriangles.size();}
        int			getEdgesNumber()  const	{return (int) mEdges.size();}

        void		getTriangles(vector <cTriangle> &vTriangles) const {vTriangles = mTriangles;}
        vector <cEdge> getEdges() const { return mEdges;}

        cVertex *   getVertex(unsigned int idx);
        cTriangle *	getTriangle(unsigned int idx);
        cEdge *     getEdge(unsigned int idx);

        void		addPt(const Pt3dr &aPt);
        void		addTriangle(const cTriangle &aTri);
        void		addEdge(int aK, int bK);

        void        removeTriangle(cTriangle &aTri);

        void        setTrianglesAttribute(int img_idx, Pt3dr Dir, set <unsigned int> const &aTriIdx);

        void		setGraph(int img_idx, RGraph &aGraph, vector <int> &aTriInGraph, const set<unsigned int> &aTriIdx); //TriInGraph: index of triangles in Graph
        void		setLambda(REAL aL) {mLambda = aL;}

        void        clean();

        vector < cTextRect > getRegions();

        void        write(const string & aOut, bool aBin, const string & textureFilename);

private:

        void        checkTriangle(int id2, set<int>::const_iterator it, int aK);
        void        checkEdgesForVertex(int id, int aK);

        vector <cVertex>	mVertexes;
        vector <cTriangle>	mTriangles;
        vector <cEdge>	    mEdges;			//aretes du graphe de voisinage

        set <pair <int,int> > mEdgesSet;

        REAL				mLambda;
};

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

class cVertex
{
    public:
                    cVertex(const Pt3dr & pt);

                    ~cVertex();

        void		getPos(Pt3dr &pos){ pos = mPos; }
        set<int>    getTriIdx() { return mTriIdx; }
        void        addIdx(int id) { mTriIdx.insert(id); }

        bool    operator==( const cVertex & ) const;

    private:

        Pt3dr		mPos;
        set<int>    mTriIdx;
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

        bool	getAttributes(int image_idx, vector <REAL> &ta) const;
        map <int, vector <REAL> >	getAttributesMap() const {return mAttributes;}

        void    setIdx(int id) { mTriIdx = id; }
        int		getIdx() const {return mTriIdx;}

        void	setAttributes(int image_idx, const vector <REAL> &ta);

        bool	hasAttributes() { return (!mAttributes.empty()); }

        void	setInside() {mInside = true;}

        bool	isInside(){return mInside;}

        REAL	computeEnergy(int img_idx);

        size_t  getEdgesNumber() { return mTriEdges.size(); }

        vector <int>   getEdgesIndex() { return mTriEdges; }
        vector<cTriangle *> getNeighbours(); //renvoie les 3 voisins (par les arêtes)
        set<int> getNeighbours2(); //renvoie les voisins par les sommets

        void    setEdgeIndex(unsigned int pos, int val);
        void    setVertexIndex(unsigned int pos, int val);

        static int     getDefTextureImgIndex() { return mDefTextImIdx; }

        void    setTextureImgIndex(int val) { mTextImIdx = val; }
        int     getTextureImgIndex() { return mTextImIdx; }

        void    setTextureCoordinates(const Pt2dr &p0, const Pt2dr &p1, const Pt2dr &p2);
        void    getTextureCoordinates(Pt2dr &p0, Pt2dr &p1, Pt2dr &p2);

        bool    isTextured() { return mTextImIdx != -1; }

        bool    operator==( const cTriangle & ) const;


private:

        bool						mInside;		// triangle a conserver
        int							mTriIdx;		// triangle index
        vector <int>				mTriVertex;		// index of vertexes in pMesh->mVertexes
        vector <int>                mTriEdges;      // index of edges in pMesh->Edges
        map <int, vector <REAL> >	mAttributes;	// map between image index and triangle attributes
        static const int            mDefTextImIdx = -1;
        int                         mTextImIdx;

        cMesh       *               pMesh;

        Pt2dr                       mText0;         //Texture Coordinates
        Pt2dr                       mText1;
        Pt2dr                       mText2;
};

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
class cEdge
{
    public:
               // cEdge();
                //cEdge(int tri1, int tri2, int v1, int v2){mNode1 = tri1; mNode2 = tri2; mV1 = v1; mV2 = v2;}
                cEdge(int tri1, int tri2){mNode1 = tri1; mNode2 = tri2; /*mV1 = v1; mV2 = v2;*/}

                ~cEdge();

        int		n1(){return mNode1;}
        int		n2(){return mNode2;}
       // int		v1(){return mV1;}
       // int		v2(){return mV2;}

        void    setN1(int aK) { mNode1 = aK; }
        void    setN2(int aK) { mNode2 = aK; }

        bool operator==( const cEdge & ) const;

    private:

        int mNode1; //index du triangle
        int mNode2;

        //int mV1;	//index des sommets communs
        //int mV2;
};

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

class cZBuf
{
    public:
                cZBuf(Pt2di sz = Pt2di(0,0), float defVal = 0.f, int aScale=1.f);

                ~cZBuf();

        void	BasculerUnMaillage(cMesh const &aMesh);			//Projection du maillage dans la geometrie de aNuage, aDef: valeur par defaut de l'image resultante
        void    BasculerUnMaillage(cMesh const &aMesh, CamStenope const & aCam);

        void		BasculerUnTriangle(cTriangle &aTri, bool doMask = false); //soit on calcule le ZBuffer, soit le Masque (true)

        void		ComputeVisibleTrianglesIndexes();
        Im2D_BIN	ComputeMask(int img_idx, cMesh &aMesh);
        Im2D_BIN	ComputeMask(vector <int> const &TriInGraph, RGraph &aGraph, cMesh &aMesh);

        Im2D_INT4				getIndexImage() const {return mImTriIdx;}
        set <unsigned int> *    getVisibleTrianglesIndexes() {return &vTri;}

        cElNuage3DMaille * &	Nuage() {return mNuage;}

        void					setSelfSz(){mSzRes = mNuage->SzUnique();} //temp
        void					setMaxAngle(double aAngle){mMaxAngle = aAngle;}

        Pt2di					Sz(){return mSzRes / mScale;}

        Im2D_REAL4*             get() { return &mRes; }

        void                    write(string filename);
        void                    writeImLabel(string filename);

    private:

        double					mMaxAngle;		//threshold on angle between surface and viewing direction

        Pt2di					mSzRes;			//size result

        Im2D_INT4				mImTriIdx;		//triangle index image (label image)
        Im2D_BIN				mImMask;		//mask image

        Im2D_REAL4				mRes;			//Zbuffer
        float **				mDataRes;

        float					mDpDef;			//default value for depth img (mRes)
        int                     mIdDef;			//default value for label img (mImTriIdx)

        set <unsigned int>      vTri;			//list of visible triangles (contained in the label image)

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
