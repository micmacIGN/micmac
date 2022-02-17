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

#if (ELISE_X11)

class cAppli_VCE;
class cCoupe_VCE;

class cCoupe_VCE
{
     public :
         cCoupe_VCE(cAppli_VCE *,const std::string & aName,Video_Win,bool First );
         void LoadIm(Pt2di aTr);
         Tiff_Im Tif() {return mTif;}
         bool First() {return mFirst;}
     private :
         bool                     mFirst;
         Video_Win                mWIm;
         Tiff_Im                  mTif;
         cAppli_VCE *             mAppli;
         Pt2di                    mSzIm;
         int                      mLine;
         Im2D_REAL4               mIm;
};


class cAppli_VCE
{
     public :
         cAppli_VCE(int,char **);
         Pt2di SzIm() {return mSzIm;}
         void ShowLine(float *,bool First);

     private :
         int Px0(bool First) const {return First ? 0 : mPx0;}
         void LoadAll();

         std::string                 mNameIm1;
         std::string                 mNameIm2;
         Video_Win *                 mWGlob;
         Video_Win *                 mWProfil;
         Pt2di                       mSzWIm;
         Pt2di                       mSzIm;
         int                         mSzH;
         int                         mZoom;
         double                      mDyn;
         int                         mPx0;
         Pt2di                       mCurTr;

         std::vector<cCoupe_VCE *>   mCoupes;
};


/**********************************************************/
/*                                                        */
/*                    cCoupe_VCE                          */
/*                                                        */
/**********************************************************/

cCoupe_VCE::cCoupe_VCE
(
        cAppli_VCE * anAppli,
        const std::string & aName,
        Video_Win aW,
        bool First
) :
    mFirst (First),
    mWIm   (aW),
    mTif   (Tiff_Im::StdConv(aName)),
    mAppli (anAppli),
    mSzIm  (mAppli->SzIm()),
    mLine  (mSzIm.y/2),
    mIm    (mSzIm.x,mSzIm.y)
{
}

void cCoupe_VCE::LoadIm(Pt2di aTr)
{
   ELISE_COPY
   (
        mIm.all_pts(),
        trans(mTif.in(0),aTr),
        mIm.out()
   );

   ELISE_COPY(mIm.all_pts(),Max(0,Min(255,mIm.in())),mWIm.ogray());
   mAppli->ShowLine(mIm.data()[mLine],mFirst);
   mWIm.draw_seg
   (
       Pt2dr(0,mLine),
       Pt2dr(mSzIm.x,mLine),
       mWIm.pdisc()(P8COL::blue)
   );
}
/**********************************************************/
/*                                                        */
/*                    cAppli_VCE                          */
/*                                                        */
/**********************************************************/

cAppli_VCE::cAppli_VCE(int argc,char ** argv) :
     mWGlob    (0),
     mWProfil  (0),
     mSzWIm    (500,100),
     mSzH      (255),
     mZoom     (1),
     mDyn      (1.0)
{

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mNameIm1,"Name Image 1", eSAM_IsExistFile)
                    << EAMC(mNameIm2,"Name Image 2", eSAM_IsExistFile),
        LArgMain()
                    << EAM(mSzWIm.x,"SzX",true,"X-Size of Window")
                    << EAM(mSzWIm.y,"SzY",true,"Y-Size of Window")
                    << EAM(mZoom,"Zoom",true,"Zoom of image")
                    << EAM(mPx0,"Px0",true,"Delta Px init")
                    << EAM(mCurTr,"Tr",true,"initial translation")
    );

    if (!MMVisualMode)
    {
        mSzIm = mSzWIm / mZoom;
        int aSzX = mSzWIm.x;

        mWGlob = Video_Win::PtrWStd(Pt2di(aSzX,100));

        Video_Win aWIm1 = Video_Win::WStd(mSzWIm,1.0,mWGlob);
        ELISE_COPY(aWIm1.all_pts(),P8COL::white,aWIm1.odisc());

        mWProfil = new Video_Win(aWIm1,Video_Win::eBasG,Pt2di(aSzX,mSzH));

        Video_Win aWIm2(*mWProfil,Video_Win::eBasG,mSzWIm);
        ELISE_COPY(aWIm2.all_pts(),P8COL::white,aWIm2.odisc());

        mCoupes.push_back(new cCoupe_VCE(this,mNameIm1,aWIm1,true));
        mCoupes.push_back(new cCoupe_VCE(this,mNameIm2,aWIm2,false));

        if (! EAMIsInit(&mCurTr))
           mCurTr = Pt2di(mCoupes.back()->Tif().sz() -mSzIm) / 2;
        LoadAll();

        std::cout << "AAAAAAAAAAAa\n";
        getchar();
    }
}

void cAppli_VCE::LoadAll()
{
  mWProfil->clear();
  for (int aKC=0 ; aKC<int(mCoupes.size()) ; aKC++)
  {
      cCoupe_VCE * aCut =  mCoupes[aKC];
      Pt2di aPx(Px0(aCut->First()),0);
      aCut->LoadIm(mCurTr+aPx);
  }
}

void cAppli_VCE::ShowLine(float * aLine,bool First)
{
    // std::cout << "F " << First << " " << aLine[10] << "\n";
    for (int anX=1 ; anX<mSzIm.x ; anX++)
    {
         Pt2dr aP1((anX-1)*mZoom,(255-aLine[anX-1])*mDyn);
         Pt2dr aP2(anX*mZoom,(255-aLine[anX])*mDyn);
         mWProfil->draw_seg
         (
              aP1,
              aP2,
              mWProfil->pdisc()(First ? P8COL::red : P8COL::green)
         );
    }
}

#endif // ELISE_X11

/**********************************************************/
/*                                                        */
/*                         ::                             */
/*                                                        */
/**********************************************************/

int VisuCoupeEpip_main(int argc,char ** argv)
{
    #if (ELISE_X11)
        cAppli_VCE anAppli(argc,argv);
    #else
        cerr << "X11 is not available" << endl;
    #endif
    return 0;
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/

