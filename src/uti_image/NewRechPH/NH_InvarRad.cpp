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


#include "NewRechPH.h"

Pt2di cAppli_NewRechPH::SzInvRad()
{
   return Pt2di(mNbSR, mNbTetaInv);
}

void NormalizeVect(std::vector<double> & aVect)
{
    double aS0  = 0.0;
    double aS1  = 0.0;
    double aS2  = 0.0;
    for (const auto  & aVal : aVect)
    {
         aS0 +=  1.0;
         aS1 +=  aVal;
         aS2 += ElSquare(aVal);
    }
    aS1 /= aS0;
    aS2 /= aS0;
    aS2 -= ElSquare(aS1);
    double aSig = sqrt(ElMax(1e-10,aS2));

    for (auto  & aVal : aVect)
    {
        aVal = (aVal-aS1) / aSig;
    }
}

// Pt2dr aPTBUG (2914.32,1398.2);
Pt2dr aPTBUG (2892.06,891.313);

bool  cAppli_NewRechPH::CalvInvariantRot(cOnePCarac & aPt)
{
   bool BUG= false &&  (euclid(aPt.Pt()+Pt2dr(mP0)-aPTBUG) < 0.02);
   static int aCpt=0;
   aCpt++;
   if (aPt.NivScale() >= mMaxLevR)
   {
      return aPt.OK() = false;
   }

   // Buf[KTeta][KRho]   pour KRho=0, duplication de la valeur centrale
   tImNRPH aImBuf(SzInvRad().x,SzInvRad().y);
   tTImNRPH aTBuf(aImBuf);

   std::vector<cOneScaleImRechPH *>  aVIm;
   // Tableau des distance / au centre pour echantillonner
   std::vector<double>               aVRho;

   int aN0 = aPt.NivScale();
   // aVIm.push_back(mVI1.at(aN0));
   for (int aKRho=0 ; aKRho <mNbSR ; aKRho++)
   {
       aVIm.push_back(mVI1.at(aN0 + aKRho * mDeltaSR));
   }

   double aLastScale = ElSquare(aVIm.at(0)->Scale()) / aVIm.at(1)->Scale();
   double aRho = 0.0;

   for (int aKRho=0 ; aKRho<int(aVIm.size()) ; aKRho++)
   {
      double aCurScale = aVIm.at(aKRho)->Scale();
      double aDRho = ((aCurScale+aLastScale) / 2.0) *  mStepSR;
      aRho += aDRho;

      aVRho.push_back(aRho);
      aLastScale = aCurScale;
   }

   // Calcul de l'image "log/polar"

   for (int aKTeta=0 ; aKTeta<mNbTetaInv; aKTeta++)
   {
      double aTeta = (2*PI*aKTeta)/mNbTetaInv;
      Pt2dr aPTeta = Pt2dr::FromPolar(1.0,aTeta);
      for (int aKRho=0 ; aKRho<int(aVIm.size()) ; aKRho++)
      {
          double aDef = -1e5; 
          Pt2dr aP = aPt.Pt() + aPTeta * aVRho.at(aKRho);
          double aVal = aVIm.at(aKRho)->TIm().getr(aP,aDef);
          if (aVal==aDef)
          {
             return aPt.OK() = false;
          }
          aTBuf.oset(Pt2di(aKRho,aKTeta),aVal);
      }
   }

   if (BUG)
   {
       std::cout << "PTBBUGGGGG 111\n";
       Tiff_Im::CreateFromIm(aImBuf,"NEWHBuf.tif");
   }

   int aKPS4 = mNbTetaInv /4 ;
   int aKPS2 = mNbTetaInv /2 ;
   for (int aKRho=0 ; aKRho<int(aVIm.size()) ; aKRho++)
   {
      bool WithGR = (aKRho!=0);
      double aSomV        =0;
      double aSomV2       =0;
      double aSomGradRadial  =0;
      double aSomGradTeta =0;
      double aSomGradTetaPiS4 =0;
      double aSomGradTetaPiS2 =0;
      for (int aKTeta=0 ; aKTeta<mNbTetaInv; aKTeta++)
      {
          aSomV +=  aTBuf.get(Pt2di(aKRho,aKTeta));
          aSomV2 += ElSquare(aTBuf.get(Pt2di(aKRho,aKTeta)));
          aSomGradTeta +=     ElAbs(aTBuf.get(Pt2di(aKRho,aKTeta)) -aTBuf.get(Pt2di(aKRho,(1    +aKTeta)%mNbTetaInv)));
          aSomGradTetaPiS4 += ElAbs(aTBuf.get(Pt2di(aKRho,aKTeta)) -aTBuf.get(Pt2di(aKRho,(aKPS4+aKTeta)%mNbTetaInv)));
          aSomGradTetaPiS2 += ElAbs(aTBuf.get(Pt2di(aKRho,aKTeta)) -aTBuf.get(Pt2di(aKRho,(aKPS2+aKTeta)%mNbTetaInv)));
          if (WithGR)
          {
             aSomGradRadial += ElAbs(aTBuf.get(Pt2di(aKRho,aKTeta)) -aTBuf.get(Pt2di(aKRho-1,aKTeta)));
          }
      }
      aSomV /=  mNbTetaInv;
      aSomV2 /= mNbTetaInv;
      aSomV2 -= ElSquare(aSomV);
      aSomV2 = sqrt(ElMax(1e-10,aSomV2));

      aPt.CoeffRadiom().push_back(aSomV);
      aPt.CoeffRadiom2().push_back(aSomV2);
      if (WithGR)
      {
         aPt.CoeffGradRadial().push_back(aSomGradRadial);
      }
      aPt.CoeffGradTangent().push_back(aSomGradTeta);
      aPt.CoeffGradTangentPiS4().push_back(aSomGradTetaPiS4);
      aPt.CoeffGradTangentPiS2().push_back(aSomGradTetaPiS2);
      
   }
   NormalizeVect(aPt.CoeffRadiom());
   NormalizeVect(aPt.CoeffRadiom2());
   NormalizeVect(aPt.CoeffGradRadial());
   NormalizeVect(aPt.CoeffGradTangent());
   NormalizeVect(aPt.CoeffGradTangentPiS4());
   NormalizeVect(aPt.CoeffGradTangentPiS2());


   // Sauvegarde
   if (0)
   {
       std::string aDir = "Tmp-NH-InvRad/";
       ELISE_fp::MkDirSvp(aDir);
       std::string aNamePt =   "_Kind-"+ eToString(aPt.Kind()) 
                             + "_Cstr-" + ToString(round_ni(1000 *aPt.ContrasteRel()))
                             + "_Cpt-" + ToString(aCpt)
                           ;

       std::string aName= aDir + "InvRad" + aNamePt +  ".tif";

       L_Arg_Opt_Tiff aLarg;
       aLarg = aLarg + Arg_Tiff(Tiff_Im::AStrip( arrondi_sup(SzInvRad().x,8)));
       Tiff_Im  aSaveBuf
       (
           aName.c_str(),
           SzInvRad(),
           GenIm::real4,
           Tiff_Im::No_Compr,
           Tiff_Im::BlackIsZero,
           aLarg
       );
       ELISE_COPY(aImBuf.all_pts(),aImBuf.in(),aSaveBuf.out());
   }

   

   // Pour l'export xml
   {
      double aS0,aS1,aS2;
      ELISE_COPY
      (
         aImBuf.all_pts(),
         Virgule(1,aImBuf.in(),Square(aImBuf.in())),
         Virgule(sigma(aS0),sigma(aS1),sigma(aS2))
      );
      aS1 /= aS0;
      aS2 /= aS0;
      aS2 -= ElSquare(aS1);
      aS2 = sqrt(ElMax(1e-10,aS2));
      ELISE_COPY(aImBuf.all_pts(),(aImBuf.in()-aS1)/aS2, aImBuf.out());
      aPt.ImRad() = aImBuf;
      aPt.VectRho() = aVRho;
      
      if (BUG)
      {
          std::cout << "PTBBUGGGGG 22\n";
          Tiff_Im::CreateFromIm(aImBuf,"NORM-NEWHBuf.tif");
          getchar();
      }
   }

   return true;
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
aooter-MicMac-eLiSe-25/06/2007*/
