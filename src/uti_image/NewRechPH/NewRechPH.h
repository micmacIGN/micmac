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


#ifndef _NewRechPH_H_
#define _NewRechPH_H_

#include "ExternNewRechPH.h"

typedef float   tElNewRechPH ;
typedef double  tElBufNRPH ;

typedef Im2D<tElNewRechPH,tElBufNRPH>  tImNRPH;
typedef TIm2D<tElNewRechPH,tElBufNRPH> tTImNRPH;


class cAppli_NewRechPH;
class cOneScaleImRechPH;


/************************************************************************/


class cOneScaleImRechPH
{
      public :
          static cOneScaleImRechPH* FromFile (cAppli_NewRechPH &,const double & aS0,const std::string &,const Pt2di & aP0,const Pt2di & aP1);
          static cOneScaleImRechPH* FromScale(cAppli_NewRechPH &,cOneScaleImRechPH &,const double & aSigma);
          tImNRPH Im();

          void CalcPtsCarac();

          void Show(Video_Win* aW);
      private :
          cOneScaleImRechPH(cAppli_NewRechPH &,const Pt2di & aSz,const double & aScale,const int & aNiv);
          bool  SelectVois(const Pt2di & aP,const std::vector<Pt2di> & aVVois,int aValCmp);
          std::list<cPtRemark *>  mLIPM;
   
          cAppli_NewRechPH & mAppli;
          Pt2di     mSz;
          tImNRPH   mIm;
          tTImNRPH  mTIm;
          double    mScale;
          int       mNiv;
           
};
class cAppli_NewRechPH
{
    public :
        cAppli_NewRechPH(int argc,char ** argv,bool ModeTest);

        const double &      DistMinMax() const  {return mDistMinMax;}
        const bool   &      DoMin() const       {return mDoMin;}
        const bool   &      DoMax() const       {return mDoMax;}
        cPlyCloud * PlyC()  const {return mPlyC;}
        const double & DZPlyLay() const {return  mDZPlyLay;}

        bool Inside(const Pt2di & aP) const;
        tPtrPtRemark & PtOfBuf(const Pt2di &);
        tPtrPtRemark  NearestPoint(const Pt2di &);

    private :
        void AddScale(cOneScaleImRechPH *,cOneScaleImRechPH *);
        void Clik();

        std::string mName;
        double      mPowS;
        int         mNbS;
        double      mS0;
        Pt2di       mSzIm;

        std::vector<cOneScaleImRechPH *> mVI1;
        Video_Win  * mW1;
        bool         mModeTest;

        double       mDistMinMax;
        bool         mDoMin;
        bool         mDoMax;
        bool         mDoPly;
        cPlyCloud *  mPlyC;
        double       mDZPlyLay;

        std::vector<std::vector<cPtRemark *> >  mBufLnk;
        std::vector<Pt2di>                      mVoisLnk;
};



#endif //  _NewRechPH_H_


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
aooter-MicMac-eLiSe-25/06/2007*/
