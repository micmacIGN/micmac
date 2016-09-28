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


#ifndef _TiePTri_
#define _TiePTri_

#include "StdAfx.h"

// Header du header
class cAppliTieTri;
class cImTieTri;
class cImMasterTieTri;
class cImSecTieTri;


//  =====================================


class cAppliTieTri
{
      public :

           cAppliTieTri
           (
              cInterfChantierNameManipulateur *,
              const std::string & aDir,  
              const std::string & anOri,  
              const cXml_TriAngulationImMaster &
           );

           void SetSzW(Pt2di , int);


           cInterfChantierNameManipulateur * ICNM();
           const std::string &               Ori() const;
           const std::string &               Dir() const;
           void DoAllTri(const cXml_TriAngulationImMaster &);

           bool  WithW() const;
           Pt2di  SzW() const;
           int    ZoomW() const;
           cImMasterTieTri * Master();
           const std::vector<Pt2di> &   VoisExtr() const;
           bool  & Debug() ;


      private  :
         void DoOneTri(const cXml_Triangle3DForTieP & );

         cInterfChantierNameManipulateur * mICNM;
         std::string                       mDir;
         std::string                       mOri;
         cImMasterTieTri *                 mMasIm;
         std::vector<cImSecTieTri *>       mImSec;
         Pt2di                             mSzW;
         int                               mZoomW;
         bool                              mWithW;

         double                            mDisExtrema;
         std::vector<Pt2di>                mVoisExtr;
         bool                              mDebug;
};


typedef double tElTiepTri ;

class cImTieTri
{
      public :
            friend class cImMasterTieTri;
            friend class cImSecTieTri;

           cImTieTri(cAppliTieTri & ,const std::string& aNameIm);
      protected :
           bool IsExtrema(const TIm2D<tElTiepTri,tElTiepTri> &,Pt2di aP,bool aMax);
           void MakeInterestPoint(const TIm2DBits<1> & aMasq,const TIm2D<tElTiepTri,tElTiepTri> &);

           void LoadTri(const cXml_Triangle3DForTieP & );

           cAppliTieTri & mAppli;
           std::string    mNameIm;
           Tiff_Im        mTif;
           CamStenope *   mCam;
           Pt2dr          mP1Glob;
           Pt2dr          mP2Glob;
           Pt2dr          mP3Glob;

           Pt2dr          mP1Loc;
           Pt2dr          mP2Loc;
           Pt2dr          mP3Loc;
 
           Pt2di          mDecal;
           Pt2di          mSzIm;

           Im2D<tElTiepTri,tElTiepTri>   mImInit;
           TIm2D<tElTiepTri,tElTiepTri>  mTImInit;

           Im2D_Bits<1>                  mMasqTri;
           TIm2DBits<1>                  mTMasqTri;

           int                           mRab;
           Video_Win *                   mW;
};

class cIntTieTriInterest
{
    public :
       Pt2di mPt;
};

class cImMasterTieTri : public cImTieTri
{
    public :
           cImMasterTieTri(cAppliTieTri & ,const std::string& aNameIm);
           void LoadTri(const cXml_Triangle3DForTieP & );

    private :

           std::list<cIntTieTriInterest> mLIP;
           
};

class cImSecTieTri : public cImTieTri
{
    public :
           cImSecTieTri(cAppliTieTri & ,const std::string& aNameIm);
           void LoadTri(const cXml_Triangle3DForTieP & );
    private :
           Im2D<tElTiepTri,tElTiepTri>   mImReech;
           TIm2D<tElTiepTri,tElTiepTri>  mTImReech;
           Pt2di                         mSzReech;
           ElAffin2D                     mAffMas2Sec;
           cImMasterTieTri *             mMaster;
};

#endif //  _TiePTri_


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
