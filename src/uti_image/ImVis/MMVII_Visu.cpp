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

#include "general/sys_dep.h"

#if (ELISE_X11)

#include "Vino.h"

#include "im_tpl/cPtOfCorrel.h"


void  cAppli_Vino::AimeVisu()
{
   ElSimilitude aSim = mScr->to_win();
   double aScale = euclid(aSim.sc());
   if (aScale < 1.0)
      return;

   for (const auto  & aVPC : mAimePCar)
   {
       bool IsMax = aVPC.IsMax();
       // Pt2dr  aFastCrit = FastQuality(aTStd,mAimeCW,*aFCC,!mAimePCar.IsMin(),Pt2dr(0.7,0.8));
       Pt3di  aCoulOk(0,255,0);  // Green
       if (aVPC.NameTypePt() == "Corner")
       {
       }
       else if (aVPC.NameTypePt() == "LaplG")
       {
           aCoulOk = Pt3di(0,0,255);  // blue
       }
       else if (aVPC.NameTypePt() == "OriNorm")
       {
           aCoulOk = Pt3di(0,255,255);  // cyan
       }
       else if (aVPC.NameTypePt() == "Init")
       {
           aCoulOk = Pt3di(128,128,128);  // gray ???
       }
       else
       {
           std::cout <<  "TYPE=[" << aVPC.NameTypePt() << "]\n";
           ELISE_ASSERT(false,"Unknown type");
       }
       for (const auto  & aPC : aVPC.Pts())
       {
           Pt2dr aPI = aPC.PtAff();
           Pt2dr aPW = aSim(aPI);
           if (   (aPW.x>0) && (aPW.y>0) && (aPW.x<SzW().x) && (aPW.y<SzW().y))
           {
               Pt3di aCoul = aCoulOk;
               if (!aPC.SFSelected())
               {
                  aCoul = Pt3di(255,255,0); // yellow
               }
               if (!aPC.OKLP())
               {
                  aCoul = Pt3di(255,128,0); // orange
               }
               if (!aPC.Stable())
               {
                  aCoul = Pt3di(255,0,255); // magenta
               }
               if (!aPC.OKAc())
               {
                  aCoul = Pt3di(255,0,0); // red
               }
           
               if ((aCoul==aCoulOk) || mAimeShowFailed )
               {
                  double aRay = 1*aScale*aPC.ScaleAbs();
                  mW->draw_circle_abs(aPW,aRay,mW->prgb()(aCoul.x,aCoul.y,aCoul.z));
// mW->draw_circle_abs(aPW,aRay+3,mW->prgb()(255,0,0));
                  if (0 && aPC.ChgMaj())
                  {
                      // mW->draw_circle_abs(aPW,aRay+2,mW->pdisc()(aCoul));
                      // mW->draw_circle_abs(aPW,aRay+4,mW->pdisc()(aCoul));
                  }
                  Pt2dr aDir(0,IsMax ? -1 : 1);
                  mW->draw_seg(aPW,aPW+aDir*aRay,mW->prgb()(aCoul.x,aCoul.y,aCoul.z));
                  mW->draw_seg(aPW,aSim(aPC.PtInit()),mW->prgb()(aCoul.x,aCoul.y,aCoul.z));
               }
           }
       }
   }
}

const cXml2007Pt *  cAppli_Vino::AimeGetPC(const Pt2dr & aPU,const cXml2007SetPtOneType** aSet)
{
   *aSet = nullptr;
   const cXml2007Pt * aRes = nullptr;
   double aDMin = 1e10;

   for (const auto  & aVPC : mAimePCar)
   {
       for (const auto  & aPC : aVPC.Pts())
       {
          if (mAimeShowFailed || ( aPC.SFSelected() && aPC.OKLP() && aPC.Stable() && aPC.OKAc()))
          {
              double aD = euclid(aPU,aPC.PtAff());
              if (aD < aDMin)
              {
                 aDMin = aD;
                 aRes = & aPC;
                 *aSet=&(aVPC);
              }
         }
       }
   }
   return aRes;
}

Im2D_REAL4 cAppli_Vino::LoadAimePC(const cXml2007Pt & aPC,const std::string & aNameType,Video_Win * aW)
{
    
   std::string aName =   mDirAime
                       +"STD-Ima-" 
                       +   aNameType
                       +   std::string("-o") + ToString(aPC.NumOct())
                       +   std::string("_i") + ToString(aPC.NumIm())
                       +   "-Tile0_0.tif";

   Tiff_Im aTif(aName.c_str());
   Pt2di aSzTif = aTif.sz();

   int aZoomIm = (1 << aPC.NumOct());
   Pt2dr aP1 = aPC.PtAff() -  Pt2dr( mAimeCW *aZoomIm) ;
   Pt2dr aP2 = aPC.PtAff() +  Pt2dr( mAimeCW * aZoomIm);
   ElSimilitude aU2W = mScr->to_win();
   mW->draw_rect(aU2W(aP1),aU2W(aP2),mW->pdisc()(P8COL::yellow));
   
   Pt2di aPTif = round_ni(aPC.PtAff()) /  aZoomIm;
 


   if (
            (aPTif.x< mAimeCW.x)  
        ||  (aPTif.y< mAimeCW.y)  
        ||  ((mAimeCW.x+ aPTif.x)>= (aSzTif.x)) 
        || ((mAimeCW.y+ aPTif.y)>= (aSzTif.y))
      )
      return Im2D_REAL4(1,1);

   Im2D_REAL4 aRes(mAimeSzW.x,mAimeSzW.y);
   Pt2di aDec = aPTif - mAimeCW;
   ELISE_COPY
   (
       aRes.all_pts(),
       trans(aTif.in(),aDec),
       aRes.out()
   );

   if (aW)
   {
     double aS0,aS1,aS2;
     ELISE_COPY 
     ( 
          aRes.all_pts(), 
          Virgule(1,aRes.in(),Square(aRes.in())),
          Virgule(sigma(aS0),sigma(aS1),sigma(aS2))
     );
     aS1 /= aS0;
     aS2 /= aS0;
     aS2 -= ElSquare(aS1);
     aS2 = sqrt(max(1e-5,aS2));
     ELISE_COPY
     (
        aRes.all_pts(),
        Max(0,Min(255,128 * (1 + (aRes.in()-aS1) / aS2))),
        aW->ogray()
     );

   }

   return aRes;
}

Im2D_REAL4 cAppli_Vino::StdLoadAimePC(const cXml2007Pt & aPC,const cXml2007SetPtOneType* aSet)
{
     return LoadAimePC(aPC,aSet->NameTypePt(),mAimWStd);
}
Im2D_REAL4  cAppli_Vino::I0LoadAimePC(const cXml2007Pt & aPC,const cXml2007SetPtOneType* aSet)
{
     return LoadAimePC(aPC,"Init",mAimWI0);
}
/*
*/

#define TT_SEUIL_AutoCorrel  0.90          // Seuil d'elimination par auto-correlation
#define TT_SEUIL_CutAutoCorrel_INT 0.70    // Seuil d'acceptation rapide par auto correl entiere
#define TT_SEUIL_CutAutoCorrel_REEL 0.80   // Seuil d'acceptation rapide par auto correl reelle
#define TT_Rho0 3.5
#define TT_SzW0 2
#define TT_Rho1 5.0
#define TT_SzW1 3

void cAppli_Vino::AimeShowProfil(Im2D_U_INT1 aILP,int aMode)
{
     int aNbTeta = aILP.sz().x;
     int aNbRho = aILP.sz().y;
     Fonc_Num aF = aILP.in();
     if (aMode==1)
     {
        aF = Abs(aILP.in()-aILP.in()[Virgule((FX+1)%aNbTeta,FY)]);
     }
     if (aMode==2)
     {
        aF = Abs(aILP.in()-aILP.in()[Virgule(FX,FY+1)]);
        aNbRho--;
     }
     Im1D_REAL4 aHist(aNbTeta,0.0);
     ELISE_COPY
     (
        rectangle(Pt2di(0,0),Pt2di(aNbTeta,aNbRho)),
        aF,
        aHist.histo().chc(FX)
     );
}

void  cAppli_Vino::InspectAime(const Pt2dr & aPW_Clik)
{
   // static cFastCriterCompute * aFCC0 = cFastCriterCompute::Circle(3.0);
   // static cFastCriterCompute * aFCC1 = cFastCriterCompute::Circle(5.0);

   ElSimilitude aU2W = mScr->to_win();
   double aScale = euclid(aU2W.sc());
   ElSimilitude aW2U = aU2W.inv();

   Pt2dr aPU_Clik = aW2U(aPW_Clik);

   const cXml2007SetPtOneType* aSet;
   const cXml2007Pt * aPC =   AimeGetPC(aPU_Clik,&aSet);
   Pt2dr aPWCar = aU2W(aPC->PtAff());

   mW->draw_circle_abs(aPWCar,1*aScale*aPC->ScaleAbs()+4,mW->pdisc()(P8COL::white));

   double aSA =  aPC->ScaleAbs() ;
   double aSO =  aPC->ScaleInO() ;
   std::cout << "  SAbs=" << aSA 
             << " SO=" << aSO  
             << " DMov= " << euclid(aPC->PtInit(),aPC->PtAff()) / aPC->ScaleAbs() << "\n";


   Im2D_REAL4 aImStd = StdLoadAimePC(*aPC,aSet);
   if (aImStd.sz().x == 1)
   {
      std::cout << "oooooooooooOUT\n";
      return;
   }
   TIm2D<REAL4,REAL8> aTStd(aImStd);

   Im2D_REAL4 aImInit = I0LoadAimePC(*aPC,aSet);
   TIm2D<REAL4,REAL8> aTInit(aImInit);

   std::cout  << "  PXml   Var=" << aPC->Var()  << " Cor=" << aPC->AutoCor() << "\n";
   std::cout  << "  IDENT=" << aPC->Id()   << " Score=" << aPC->Score() << "\n";

   for (int aK=1 ; aK<2 ; aK++)
   {    
       // cFastCriterCompute * aFCC = (aK==0) ? aFCC0 : aFCC1;
       // Pt2dr  aFastCrit = FastQuality(aTStd,mAimeCW,*aFCC,!mAimePCar.IsMin(),Pt2dr(0.7,0.8));
       // std::cout << "Fast=" << aFastCrit 

       double aRho = (aK==0)  ? TT_Rho0  : TT_Rho1;
       int    aSzW0 = (aK==0)  ? TT_SzW0  : TT_SzW1;
       int    aSzW  = round_ni(aSzW0 * aSO);
       aRho = aRho * (aSzW / double(aSzW0));

/*
       cCutAutoCorrelDir<TIm2D<float,double>> aCutACD(aTInit,Pt2di(0,0),aRho,aSzW);

       bool  aOK = aCutACD.AutoCorrel(mAimeCW,TT_SEUIL_CutAutoCorrel_INT,TT_SEUIL_CutAutoCorrel_REEL,TT_SEUIL_AutoCorrel);
       std::cout  << " AUTOC " <<  aOK  << " Cor=" << aCutACD.mCorOut << " N="<< aCutACD.mNumOut << "Param " << aRho<< " " << aSzW  << "\n";

*/
        if (aK==1)
        {
            // Pt2dr aSc;
            // aCutACD.AutoCorrel(mAimeCW,1,1,1,&aSc);
            mAimWI0->draw_circle_abs(Pt2dr(mAimeCW),3.0,mAimWI0->pdisc()(P8COL::green));
            mAimWI0->draw_circle_loc(Pt2dr(mAimeCW),aRho,mAimWI0->pdisc()(P8COL::green));
            // std::cout << "SC= " << aSc << "\n";
        }
        mAimWStd->draw_circle_loc(Pt2dr(mAimeCW),aRho,mAimWStd->pdisc()(P8COL::green));
   }
   std::cout << "  =================================================\n\n";

   {
       Im2D_U_INT1 aILP = aPC->ImLP();
       Pt2di aSz = aILP.sz();
       if (aSz.x > 1)
       {
           ELISE_COPY
           (
                aILP.all_pts(),
                Max(0,Min(255,128 + 2*(aILP.in()-128))),
                mAimWLP->ogray() // | VMax(aMax) | VMin(aMin)
           );
           ELISE_COPY
           (
                border_rect(Pt2di(-1,-1),aSz+Pt2di(1,1)),
                Virgule(255,128,0),
                mAimWLP->orgb() // | VMax(aMax) | VMin(aMin)
           );
           AimeShowProfil(aILP,0);
           AimeShowProfil(aILP,1);
           AimeShowProfil(aILP,2);
       }
   }


}


#endif



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
