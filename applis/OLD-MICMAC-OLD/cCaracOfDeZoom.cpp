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
#include "MICMAC.h"
#include "im_tpl/image.h"
#include "im_tpl/cPtOfCorrel.h"


namespace NS_ParamMICMAC
{

/******************************************************/
/*                                                    */
/*              cCaracOfDeZoom                        */
/*                                                    */
/******************************************************/

cCaracOfDeZoom::cCaracOfDeZoom
(
     int aDeZoom,
     cCaracOfDeZoom * aDZI,
     cAppliMICMAC &   anAppli
)  :
   mAppli (anAppli),
   mDeZoom (aDeZoom)
{
   mDZI = aDZI;
   int aFreqI = anAppli.FreqPtsInt();
   mRatioPtsInt = ElMin(1.0,double(aDeZoom)/double(aFreqI*mAppli.DeZoomMin()));
   mHasMasqPtsInt = (mRatioPtsInt < 1.0);
   mNameMasqInt =    anAppli.FullDirPyr()
                  +  std::string("Pts_Interets_")
                  +   (   anAppli.ExportForMultiplePointsHomologues().Val() ?
		          StdPrefix(anAppli.PDV1()->Name())                 :
		          anAppli.NameChantier()
		      )
                  +  std::string("_DeZoom")
                  +  ToString(aDeZoom)
                  +  std::string(".tif");

   mSzIm = anAppli.SzOfResol(aDeZoom);

   if (aDZI)
   {
       mEqX.InitByDeZoom(aDZI->mEqX,2,mHasMasqPtsInt ? &mLutX : 0);
       mEqY.InitByDeZoom(aDZI->mEqY,2,mHasMasqPtsInt ? &mLutY : 0);
   }
   else
   {
         cFonc1D_HomTr aFctr(1,0,anAppli.FreqPtsInt());
         mEqX.InitFromFctr(0,mSzIm.x,aFctr);
         mEqY.InitFromFctr(0,mSzIm.y,aFctr);
   }

   if (mHasMasqPtsInt)
   {
       if (! ELISE_fp::exist_file(mNameMasqInt))
       {
              MakeMasqueInteret();
       }
   }
}

int cCaracOfDeZoom::DeZoom() const
{
   return mDeZoom;
}
const cEquiv1D & cCaracOfDeZoom::EqX() const
{
   return mEqX;
}
const cEquiv1D & cCaracOfDeZoom::EqY() const
{
   return mEqY;
}
bool  cCaracOfDeZoom::HasMasqPtsInt() const
{
   return mHasMasqPtsInt;
}
double  cCaracOfDeZoom::RatioPtsInt() const
{
   return mRatioPtsInt;
}
const std::string & cCaracOfDeZoom::NameMasqInt() const
{
  return mNameMasqInt;
}


void cCaracOfDeZoom::MakeMasqueInteret()
{
    Tiff_Im aFileMasq
            (
                mNameMasqInt.c_str(),
                mSzIm,
                GenIm::bits1_msbf,
                Tiff_Im::No_Compr,
                Tiff_Im::BlackIsZero
            );

   ELISE_COPY(aFileMasq.all_pts(),0,aFileMasq.out());

   int aSzBrd = 10 + 2 * round_up(mAppli.FreqPtsInt());
   int aSzMax = round_ni(5e6 / mSzIm.x);
   if (mAppli.ExportForMultiplePointsHomologues().Val())
      aSzMax =  mSzIm.y;

   cDecoupageInterv1D aDec1
                      (
                          cInterv1D<int>(0,mSzIm.y),
                          aSzMax,
                          cInterv1D<int>(-aSzBrd,aSzBrd)
                      );

    for (int aKInt = 0 ; aKInt<aDec1.NbInterv() ; aKInt++)
    {
         MakeMasqueInteret
         (
               aDec1.KthIntervIn(aKInt),
               aDec1.KthIntervOut(aKInt)
         );
    }
}

void cCaracOfDeZoom::MakeMasqueInteret
     (
         const cInterv1D<int> & aIntIn,
         const cInterv1D<int> & aIntOut
     )
{
std::cout << aIntIn.V0() << " - " << aIntIn.V1() << "\n";

   Pt2di aSzM (mSzIm.x,aIntIn.Larg());
   Im2D_Bits<1> aMasq(aSzM.x,aSzM.y,0);
   TIm2DBits<1> aTMasq(aMasq);

   Pt2di aDec(0,aIntIn.V0());
   Pt2di aDecIm =  aDec+round_ni(mAppli.GeomDFPx().P0()/mDeZoom);
   if (mDZI==0)
   {
       cEquiv1D anEqY;
       anEqY.InitByClipAndTr(mEqY,aDec.y,0,aIntIn.Larg());

       const cEchantillonagePtsInterets & anEPI =
               mAppli.EchantillonagePtsInterets().Val();
       eTypeModeEchantPtsI aMode = anEPI.ModeEchantPtsI();
       if (aMode==eModeEchantPtsIntByComandeExterne)
       {
          ELISE_ASSERT
	  (
	       anEPI.KeyCommandeExterneInteret().IsInit() ,
	       "Pas de valeur pour CommandeExterneInteret en mode eModeEchantPtsIntByComandeExterne"
	  );
	  std::string aKeyCom = anEPI.KeyCommandeExterneInteret().Val();
	  std::string aNameIn = mAppli.PDV1()->IMIL()->NameFileOfResol(mDeZoom);
          std::string aNameQuantPtsInt =    mAppli.FullDirPyr()
                                          + std::string("QuantInt_")
		                          + mAppli.NameChantier()
                                          + std::string("_DeZoom")
                                          + ToString(mDeZoom)
                                          + std::string(".tif");

          if (! ELISE_fp::exist_file(aNameQuantPtsInt))
	  {
              std::string aCom = mAppli.ICNM()->Assoc1To2
	                     (
			         aKeyCom,
				 aNameIn,
				 aNameQuantPtsInt,
				 true
			     );
              VoidSystem(aCom.c_str());
         }

         Im2D_REAL4 aImSc = Im2D_REAL4(mSzIm.x,aIntIn.Larg(),0.0);
	 ELISE_COPY
	 (
	    aImSc.all_pts(),
	    trans(Tiff_Im(aNameQuantPtsInt.c_str()).in(),aDecIm),
	    aImSc.out()
	 );
         Im2D_Bits<1> aMasqSc(aImSc.sz().x,aImSc.sz().y);
	 ELISE_COPY(aImSc.all_pts(),aImSc.in()>0,aMasqSc.out());


         int aFreqI = mAppli.FreqPtsInt();
         cRepartPtInteret aRep 
          (
                aImSc,
                aMasqSc,
                mEqX,
                anEqY,
                aFreqI *  anEPI.RepartExclusion().Val(),
                aFreqI *  anEPI.RepartEvitement().Val()
         );
         aMasq = aRep.ItereAndResult();
       }
       else if (aMode==eModeEchantNonAutoCor)
       {
          ELISE_ASSERT
          (
             mAppli.ModeGeomMEC() == eGeomMECIm1,
             "Incoherence in cCaracOfDeZoom::MakeMasqueInteret"
          );
          Im2D_REAL4 aImOri = Im2D_REAL4(mSzIm.x,aIntIn.Larg(),0.0);
          LoadAllImCorrel
          (
              aImOri,
              mAppli.PDV1()->IMIL(),
              mDeZoom,
              aDecIm
          );
          cPtOfCorrel  aPOC(aSzM,aImOri.in(),anEPI.SzVAutoCorrel().Val());
          Im2D_REAL4 aImSc(1,1);
          Im2D_Bits<1> aMasqSc(1,1);

          aPOC.MakeScoreAndMasq
          (
               aImSc,
               anEPI.EstmBrAutoCorrel().Val(), 
               aMasqSc,
               anEPI.SeuilLambdaAutoCorrel().Val(),
               anEPI.SeuilEcartTypeAutoCorrel().Val()
          );

          int aFreqI = mAppli.FreqPtsInt();
          cRepartPtInteret aRep 
          (
                aImSc,
                aMasqSc,
                mEqX,
                anEqY,
                aFreqI *  anEPI.RepartExclusion().Val(),
                aFreqI *  anEPI.RepartEvitement().Val()
          );
          aMasq = aRep.ItereAndResult();
       }
       else
       {
          for (int aClY=0; aClY<anEqY.NbClasses() ;aClY++)
          {
              int aY0,aY1;
              anEqY.ClasseOfNum(aY0,aY1,aClY);
              for (int aClX=0; aClX<mEqX.NbClasses() ;aClX++)
              {
                 int aX0,aX1;
                 mEqX.ClasseOfNum(aX0,aX1,aClX);
                 Pt2di aP(0,0);
                 if (aMode==eModeEchantRegulier)
                 {
                    aP = Pt2di((aX0+aX1)/2,(aY0+aY1)/2);
                 }
                 else if (aMode==eModeEchantAleatoire)
                 {
                    aP.x = aX0 + NRrandom3(aX1-aX0);
                    aP.y = aY0 + NRrandom3(aY1-aY0);
                 }
                 else
                 {
                       ELISE_ASSERT(false,"ModeEchantPtsI");
                 }
                 aTMasq.oset_svp(aP,1);
              }
          }
       }
   }
   else
   {
      Tiff_Im aFM2 (mDZI->mNameMasqInt.c_str());
      Pt2di aDec2 =  aDec *2;
      Pt2di aSzM2 = Inf(aSzM*2,aFM2.sz()-aDec2);
      Im2D_Bits<1> aMasq2(aSzM2.x,aSzM2.y,0);
      ELISE_COPY
      (
           aMasq2.all_pts(),
           trans(aFM2.in(),aDec2),
           aMasq2.out()
      );
      for (int anX=0 ; anX <aSzM2.x ; anX++)
      {
         for (int anY=0 ; anY <aSzM2.y ; anY++)
         {
             if (aMasq2.get(anX,anY))
             {
                 Pt2di aP2  = Pt2di(anX,anY)+aDec2;
                 Pt2di aP = Pt2di(mLutX[aP2.x],mLutY[aP2.y]) -aDec;
                 aTMasq.oset_svp(aP,1);
             }
         }
      }
   }

   Tiff_Im aFileMasq (mNameMasqInt.c_str());

   ELISE_COPY
   (
         rectangle
         (
             Pt2di(0, aIntOut.V0()),
             Pt2di(mSzIm.x, aIntOut.V1())
         ),
         trans(aMasq.in(),Pt2di(0,-aIntIn.V0())),
         aFileMasq.out()
   );
}

};

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
