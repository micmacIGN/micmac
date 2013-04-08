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
#include "general/all.h"
#include "private/all.h"
#include <algorithm>
#include "im_tpl/image.h"
#include "im_special/hough.h"



static std::string TheGlobName = "/media/MYPASSPORT/Documents/Cours/ENSG-TI-IT2/Interro-2010/Hough/";

double aSeuilBas = 2;
double aSeuilHaut = 5;

class cRecoEchec
{
    public :

        cRecoEchec(const std::string & aPost,double aDyn,bool isSynt) :
            mDyn  (aDyn),
            mPost (aPost),
            mName (CalcNameResult("")),
            mFile (Tiff_Im::StdConv(mName)),
            mSz   (mFile.sz()),
            mIm   (mSz.x,mSz.y),
            mGX   (mSz.x,mSz.y),
            mTGX  (mGX),
            mGY   (mSz.x,mSz.y),
            mTGY  (mGY),
            mGN   (mSz.x,mSz.y),
            mTGN  (mGN),
            mImCont (mSz.x,mSz.y),
            mTImCont (mImCont),
            mImGradHouh   (mSz.x,mSz.y),
            mTImGradHouh  (mImGradHouh),
            mImTetaH    (mSz.x,mSz.y),
            mTImTetaH   (mImTetaH)
        {
             std::cout << "NAME " << mName << "\n";
             ELISE_COPY(mFile.all_pts(),mFile.in(),mIm.out());
             if (isSynt)
                  ELISE_COPY(mIm.all_pts(),mIm.in() + 0.01*FX+0.0123*FY,mIm.out());
        }

        void CalcGrag();
        void MaxLoc();
        void Hough();
        std::string CalcNameResult(const std::string& aStep)
        {
            return  TheGlobName+mPost+ aStep + ".tif";
        }

        double       mDyn;
        std::string  mPost;
        std::string  mName;
        Tiff_Im      mFile;
        Pt2di        mSz;
        Im2D_REAL8   mIm;
        Im2D_REAL8   mGX;
        TIm2D<double,double> mTGX;
        Im2D_REAL8   mGY;
        TIm2D<double,double> mTGY;
        Im2D_REAL8   mGN;
        TIm2D<double,double> mTGN;

        Im2D_U_INT1       mImCont;
        TIm2D<U_INT1,INT> mTImCont;

        Im2D_U_INT1       mImGradHouh;
        TIm2D<U_INT1,INT> mTImGradHouh;

        Im2D_U_INT1       mImTetaH;
        TIm2D<U_INT1,INT> mTImTetaH;
};



void cRecoEchec::CalcGrag()
{
   ELISE_COPY
   (
         mIm.all_pts(),
         deriche(mIm.in_proj(),2.0,10),
         Virgule(mGX.out(),mGY.out())
   );

   ELISE_COPY
   (
       mGN.all_pts(),
       sqrt(Square(mGX.in())+Square(mGY.in())),
       mGN.out()
   );

   ELISE_COPY
   (
       mImTetaH.all_pts(),
       mod(round_ni((256.0/(2*PI)) *polar(Virgule(mGX.in(),mGY.in()),0.0).v1()),256),
       mImTetaH.out()
   );


}

void cRecoEchec::MaxLoc()
{
   Pt2di aP;
    for (aP.x=0 ; aP.x<mSz.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y<mSz.y ; aP.y++)
        {
               mTImCont.oset(aP,0);
               double aN = mTGN.get(aP);
               if (aN > aSeuilBas)
               {
                   Pt2dr aGrad(mTGX.get(aP),mTGY.get(aP)); 
                   Pt2dr aNorm = vunit(aGrad) ;
                   if ((aN>mTGN.getr(Pt2dr(aP)+aNorm,0)) && (aN>=mTGN.getr(Pt2dr(aP)-aNorm,0)))
                   {
                        mTImCont.oset(aP,1);
                        if (aN > aSeuilHaut)
                        {
                              mTImCont.oset(aP,2);
                        }
                   }
               }
        }
    }

   ELISE_COPY
   (
       mImCont.all_pts(),
       Min(255,mGN.in() * (dilat_d4(mImCont.in(0)!=0,1))),
       mImGradHouh.out()
   );

   Tiff_Im::Create8BFromFonc(CalcNameResult("_ContHough"),mSz, mImGradHouh.in());
}


void cRecoEchec::Hough()
{
   ElHough * aHAc = ElHough::NewOne
                    (
                         mSz,
                         1.0,
                         1.0,
                         ElHough::ModeBasic,
                         3,
                         0.1
                    );

std::cout << "BEGIN HOUGH \n";
   Im2D_INT4  aIm = aHAc->PdsAng(mImGradHouh,mImTetaH,0.1,true);
std::cout << "END HOUGH \n";

   int aVMax;
   ELISE_COPY(aIm.all_pts(),aIm.in(),VMax(aVMax));

   Tiff_Im::Create8BFromFonc(CalcNameResult("_HoughAcc"),aIm.sz(), aIm.in() * (255.0/aVMax));


   std::vector<Pt2di> aVMLoc;
   aHAc->CalcMaxLoc(aIm,aVMLoc,3.0,0.05,aVMax/10.0);

  {
   }


/*
   {
     std::string aNameH = CalcNameResult("_Droites");
     Bitm_Win  aBW(aNameH.c_str(),RGB_Gray_GlobPal(),mSz);
     // ELISE_COPY(mIm.all_pts(),mIm.in() * (255.0/aVMax),aBW.ogray());

     ELISE_COPY(mIm.all_pts(),Max(0,Min(255,mIm.in())),aBW.ogray());
     
     for (int aK=0 ; aK< int(aVMLoc.size()) ; aK++)
     {
          Seg2d  aSeg = aHAc->Grid_Hough2Euclid(aVMLoc[aK]);
          aBW.draw_seg(aSeg.p0(),aSeg.p1(),aBW.prgb()(255,0,0));
     }
     aBW.make_gif( aNameH.c_str());
   }
*/


   delete aHAc;



}


void TestReco(const std::string & aName,double aDyn,bool IsSynt)
{
     cRecoEchec aRE(aName,aDyn,IsSynt);
     aRE.CalcGrag();
     aRE.MaxLoc();
     aRE.Hough();
}


int main(int argc,char ** argv)
{

     // TestReco("escargot",3.0,false); 
     // TestReco("Spagehti",3.0,false); 
     // TestReco("Jardin",3.0,false); 
     // TestReco("LignesFuyantes",3.0,false); 
     // TestReco("LignesPar",3.0,false); 
     TestReco("Rayon",3.0,false); 
}



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
