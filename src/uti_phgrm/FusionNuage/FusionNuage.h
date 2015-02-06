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

#ifndef _ELISE_FUSION_NUAGE_
#define _ELISE_FUSION_NUAGE_

#include "StdAfx.h"

//================== HEADER du HEADER ====================

class cFNuAttrSom;  // Un sommet par image
class cFNuAttrArc;
class cFNuAttrArcSym;

class cAppliFusionNuage;
class cParamFuNu;

typedef ElSom<cFNuAttrSom*,cFNuAttrArc*>  tFNuSom;
typedef ElArc<cFNuAttrSom*,cFNuAttrArc*>  tFNuArc;
typedef ElGraphe<cFNuAttrSom*,cFNuAttrArc*>  tFNuGr;
typedef ElSubGraphe<cFNuAttrSom*,cFNuAttrArc*>  tFNuSubGr;
typedef cSubGrFlagArc<tFNuSubGr>  tFNuSubGrFA;


//=================================================

/*
   Un cLinkPtFuNu est +ou- l'agglomeration de + sieur point (une region); il permet d'avoir
   une description resumee du modele 3D
*/

class cLinkPtFuNu
{
    public :
       Pt2di  Pt() {return Pt2di(mI,mJ);}
       cLinkPtFuNu (INT2,INT2,INT2);
       INT2      mI,mJ,mNb;
};

class cFNuAttrSom
{
    public :
        cFNuAttrSom 
        (
             cElNuage3DMaille *      aN,
             const cImSecOfMaster&   aSecs,
             const std::string &     aNameIm,
             cAppliFusionNuage *,
             Im2D_U_INT1       aImBsH
         );
         bool IsArcValide(cFNuAttrSom *);
         const std::list<cISOM_Vois> & ListVoisInit();

         double   PixelEcartReprojInterpol(cFNuAttrSom * aS2,const Pt2di & aPIm1);

        // cFNuAttrSom ();
    private :
        cFNuAttrSom(const cFNuAttrSom &);  // N.I.

        cAppliFusionNuage *    mAppli;
        cImSecOfMaster         mSecs;  // Structure mere XML
        const cISOM_AllVois &  mVoisInit;
        cElNuage3DMaille *     mStdN;
        std::string            mNameIm;
        std::vector<cLinkPtFuNu>   mPtsTestRec;
        int                    mNbSomTest;
        int                    mSeuilNbSomTest;
        Im2D_Bits<1>           mMasqValid;
};

 // - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *

class cFNuAttrArcSym
{
    public :
    private :
        cFNuAttrArcSym(const cFNuAttrArcSym &); // N.I.

        double mRec;
};

 // - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *

class cFNuAttrArc
{
    public :
          cFNuAttrArc(cFNuAttrArcSym *);
          // cFNuAttrArc();
          cFNuAttrArcSym & ASym() {return *mASym;}
    private :
          cFNuAttrArc(const cFNuAttrArc &); // N.I.

          cFNuAttrArcSym * mASym;
         
};

 // - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
class cParamFuNu
{
   public :
       cParamFuNu();

       int mNbCellCalcGraph;
       double  mPercRecMin;
       int     mSeuilBSH;
};


class cAppliFusionNuage
{
    public :
       std::string NameFileInput(const std::string & aNameIm,const std::string aPost);
       cAppliFusionNuage
       (
            const cParamFuNu & aParam,
            const std::string & aDir,
            const std::string & aPat,
            const std::string & aKeyI2N,
            const std::string & aKeyI2ISec,
            const std::string & aKeyI2BsH
       );
       void NoOp() {}
       const cParamFuNu & Param() {return mParam;}
       tFNuArc * TestNewAndSet(tFNuSom *,tFNuSom *);
       cInterfChantierNameManipulateur* ICNM();
    private :
  
    

        cParamFuNu                       mParam;
        std::string                      mDir;
        std::string                      mPat;
        std::string                      mKeyI2N;
        cInterfChantierNameManipulateur* mICNM;
        tFNuGr                           mGr;                     
        tFNuSubGr                        mAllGr;
        int                              mFlagAIn;
        int                              mFlagATested;
        tFNuSubGrFA                      mGrArcIn;
        tFNuSubGrFA                      mGrArcTested;
        std::map<std::string,tFNuSom *>  mMapSom;
        std::vector<tFNuSom *>           mVSom;
        int                              mNbSom;
        std::set<std::pair<tFNuSom *,tFNuSom *> > mTestedPair;
};


#endif // _ELISE_FUSION_NUAGE_

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
