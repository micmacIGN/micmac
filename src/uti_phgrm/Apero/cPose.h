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

#ifndef _POSE_H_
#define _POSE_H_

#include "Apero.h"

class cPtAVGR;
class cFoncPtOfPtAVGR;
class cAperoVisuGlobRes;
class cTransfo3DIdent;
class cInfoAccumRes;
class cAccumResidu;



class cPtAVGR
{
    public :
        cPtAVGR (const Pt3dr & aP,double aRes);
        Pt3df mPt;
        float mRes;
        float mResFiltr;
        bool  mInQt;
};


typedef enum
{
    eBAVGR_X,
    eBAVGR_Y,
    eBAVGR_Z,
    eBAVGR_Res
} eBoxAVGR;

class cFoncPtOfPtAVGR
{
   public :
       Pt2dr operator () (cPtAVGR * aP) {return  Pt2dr(aP->mPt.x,aP->mPt.y);}
};


class cAperoVisuGlobRes
{
    public :
       void AddResidu(const Pt3dr & aP,double aRes);
       void DoResidu(const std::string & aDir,int aNbMes);

       cAperoVisuGlobRes();

    private :
       Interval  CalculBox(double & aVMil,double & aResol,eBoxAVGR aMode,double PropElim,double Rab);
       Box2dr    CalculBox_XY(double PropElim,double Rab);
       double    ToEcartStd(double anE) const;
       double    FromEcartStd(double anE) const;
       Pt3di     ColOfEcart(double anE);

       typedef ElQT<cPtAVGR *,Pt2dr,cFoncPtOfPtAVGR> tQtTiepT;

       int                  mNbPts;
       std::list<cPtAVGR *> mLpt;
       tQtTiepT *           mQt;
       double               mResol;
       double               mResolX;
       double               mResolY;
       double               mSigRes;
       double               mMoyRes;
       cPlyCloud            mPC;
       cPlyCloud            mPCLeg;  // Legende
       double               mVMilZ;
};

class cTransfo3DIdent : public cTransfo3D
{
     public :
          std::vector<Pt3dr> Src2Cibl(const std::vector<Pt3dr> & aSrc) const {return aSrc;}

};


class cInfoAccumRes
{
     public :
       cInfoAccumRes(const Pt2dr & aPt,double aPds,double aResidu,const Pt2dr & aDir);

       Pt2dr  mPt;
       double mPds;
       double mResidu;
       Pt2dr  mDir;
};


class cAccumResidu
{
    public :
       void Accum(const cInfoAccumRes &);
       cAccumResidu(Pt2di aSz,double aRed,bool OnlySign,int aDegPol);

       const Pt2di & SzRed() {return mSzRed;}

       void Export(const std::string & aDir,const std::string & aName,const cUseExportImageResidu &,FILE * );
       void ExportResXY(TIm2D<REAL4,REAL8>* aTResX,TIm2D<REAL4,REAL8>* aTResY);
       void ExportResXY(const Pt2di&,Pt2dr& aRes);
    private :
       void AccumInImage(const cInfoAccumRes &);

       std::list<cInfoAccumRes> mLIAR;
       int                      mNbInfo;
       double                   mSomPds;
       bool                     mOnlySign;
       double                   mResol;
       Pt2di                    mSz;
       Pt2di                    mSzRed;

       Im2D_REAL4               mPds;
       TIm2D<REAL4,REAL8>       mTPds;
       Im2D_REAL4               mMoySign;
       TIm2D<REAL4,REAL8>       mTMoySign;
       Im2D_REAL4               mMoyAbs;
       TIm2D<REAL4,REAL8>       mTMoyAbs;
       bool                     mInit;
       int                      mDegPol;
       L2SysSurResol *          mSys;
};

#endif //  _POSE_H_


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


