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


/*
double cOneScaleImRechPH::ComputeContrast()
{
   int aSzW = 3;
   for (int aX =aSzW ; aX< mSz.x-aSzW -1; aX+= aSzW)
   {
       for (int aY =aSzW ; aY< mSz.x-aSzW -1; aY+= aSzW)
       {
       }
   }

   double aS1,aSC;
   double aSeuil = 0;

   for (int aK= 0 ; aK<3 ; aK++)
   {
       Symb_FNum aF = Square(mImMod.in());
       Fonc_Num aFPds = 1.0;
       if (aK==1) 
          aFPds = (aF < aSeuil);
       else if (aK>1)
          aFPds =  1 /(1 + aF/ aSeuil);
       
       Symb_FNum aPds = aFPds;
       ELISE_COPY
       (
          mImMod.all_pts(),
          Virgule(aPds,aF * aPds),
          Virgule(sigma(aS1),sigma(aSC))
       );
       aSC  = aSC /aS1;
       aSeuil = aSC * 10;
       std::cout << "cOneScaleImRechPH::ComputeContrast " << aS1 / (mSz.x*mSz.y) << "\n";
   }
   double aRes = sqrt(aSC);
   return aRes;
}
*/


std::vector<double> cOneScaleImRechPH::ShowStatLapl(const std::string & aMes)
{
    std::vector<double>  aVD;
    Pt2di aP;
    double aSomAbs=0;
    for (aP.x=0 ; aP.x<mSz.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y<mSz.y ; aP.y++)
        {
           int aVal = mTImMod.get(aP);
           aVD.push_back(aVal);
           aSomAbs +=  ElAbs(aVal);
        }
    }

    aSomAbs /= aVD.size();

    std::cout   <<  aMes 
                << " 0.1 : "<< KthValProp(aVD,0.1)
                << " 0.2 : "<< KthValProp(aVD,0.2)
                << " 0.5 : "<< KthValProp(aVD,0.5)
                << " 0.8 : "<< KthValProp(aVD,0.8)
                << " 0.9 : "<< KthValProp(aVD,0.9)
                << " MA : " << aSomAbs 
                << "\n";
    return aVD;
}

void cOneScaleImRechPH::SiftMaxLoc(cOneScaleImRechPH* aHR,cOneScaleImRechPH* aLR,cSetPCarac & aSPC,bool FromLR)
{
   double aPowCorrec = 1.0;


   double aCorHR = pow(mScaleAbs/aHR->mScaleAbs,aPowCorrec);
   double aCorLR = pow(mScaleAbs/aLR->mScaleAbs,aPowCorrec);
   std::vector<double> aVMaxH,aVMaxL;
   std::vector<double> aVMinH,aVMinL;


   std::vector<Pt2di> aVoisMinMax0   = SortedVoisinDisk(0.5,2,true);
   std::vector<Pt2di> aVoisMinMaxLHR  = SortedVoisinDisk(-0.1,0.1,true);
   if (DebugNRPH)
   {
      std::cout << "cOneScaleImRechPH::SiftMaxLoc "
                << " HS " << aHR->mScaleAbs
                << "  S " << mScaleAbs
                << " LS " << aLR->mScaleAbs
                << " CH " << aCorHR << " CL " << aCorLR
                << "\n";

      std::vector<double> aVH = aHR->ShowStatLapl("HR");
      std::vector<double> aV0 = ShowStatLapl("S0");
      std::vector<double> aVL = aLR->ShowStatLapl("LR");
      std::cout << "SZmm " << aVoisMinMax0.size() << " " << aVoisMinMax0[0] << aVoisMinMax0.back() << "\n";
      std::cout << "SZmm " << aVoisMinMaxLHR.size() << " " << aVoisMinMaxLHR[0] << aVoisMinMaxLHR.back() << "\n";
      
      // std::vector<double
   }
   // Ajoute l'image de calcul dans le resultat, pour affinage etc ....
   // ELISE_ASSERT(false,"Finire cOneScaleImRechPH::SiftMaxLoc");

   double aRatioPt =  mAppli.RatioDecimLRHR(mNiv,FromLR);


   ELISE_ASSERT(mSifDifMade && aLR->mSifDifMade,"Sift dif pb in cOneScaleImRechPH::SiftMaxLoc");
   // std::vector<Pt2di> aVoisMinMax  = SortedVoisinDisk(0.5,mScale+2,true);
   Im2D_U_INT1 aIFlag = MakeFlagMontant(mImMod);
   TIm2D<U_INT1,INT> aTF(aIFlag);
   Pt2di aP ;
   int aSpaceNbExtr = 0;
   int aSS_NbExtr = 0;
   int aSSCstr = 0;
   bool DoMin = mAppli.DoMin();
   bool DoMax = mAppli.DoMax();

int aNbDebugHRMax =0;
int aNbDebugLRMax =0;
int aNbDebugHRMin =0;
int aNbDebugLRMin =0;

   for (aP.x = 1 ; aP.x <mSz.x-1 ; aP.x++)
   {
// static int aCpt=0; aCpt++; std::cout << "XXXX CPTTTTT= " << aCpt << "\n";
       for (aP.y = 1 ; aP.y <mSz.y-1 ; aP.y++)
       {
           int aFlag = aTF.get(aP);
           eTypePtRemark aLab = eTPR_NoLabel;

// std::cout << "DDDDDDDDd " << mAppli.DistMinMax(Basic) << "\n";

           if (DoMax &&  (aFlag == 0)  && SelectVois(aP,aVoisMinMax0,1))
           {
              // std::cout << "DAxx "<< DoMax << " " << aFlag << "\n";
               aSpaceNbExtr++;
               if (ScaleSelectVois(aHR,aP,aVoisMinMaxLHR,1,aCorHR) && ScaleSelectVois(aLR,aP,aVoisMinMaxLHR,1,aCorLR))
               {
                   aLab = eTPR_LaplMax;
                   aSS_NbExtr ++;
               }
if (DebugNRPH)
{
   if (ScaleSelectVois(aHR,aP,aVoisMinMaxLHR,1,aCorHR)) aNbDebugHRMax++;
   if (ScaleSelectVois(aLR,aP,aVoisMinMaxLHR,1,aCorLR)) aNbDebugLRMax++;
   int aV0 = mTImMod.get(aP);
   if (aV0)
   {
      aVMaxH.push_back(aHR->mTImMod.get(aP) / double(aV0));
      aVMaxL.push_back(aLR->mTImMod.get(aP) / double(aV0));
   }
}
           }
           if (DoMin &&  (aFlag == 255) && SelectVois(aP,aVoisMinMax0,-1))
           {
               // std::cout << "DInnn "<< DoMin << " " << aFlag << "\n";
               aSpaceNbExtr++;
               if (ScaleSelectVois(aHR,aP,aVoisMinMaxLHR,-1,aCorHR) && ScaleSelectVois(aLR,aP,aVoisMinMaxLHR,-1,aCorLR))
               {
                  aLab = eTPR_LaplMin;
                  aSS_NbExtr ++;
               }
if (DebugNRPH)
{
   if (ScaleSelectVois(aHR,aP,aVoisMinMaxLHR,-1,aCorHR)) aNbDebugHRMin++;
   if (ScaleSelectVois(aLR,aP,aVoisMinMaxLHR,-1,aCorLR)) aNbDebugLRMin++;
   int aV0 = mTImMod.get(aP);
   if (aV0)
   {
      aVMinH.push_back(aHR->mTImMod.get(aP) / double(aV0));
      aVMinL.push_back(aLR->mTImMod.get(aP) / double(aV0));
   }
}
           }
          if (aLab != eTPR_NoLabel)
          {
               cOnePCarac aPC;
               aPC.DirMS() = Pt2dr(0,0);
               aPC.Kind() =  aLab;
               aPC.Pt() =  Pt2dr(aP) / aRatioPt ;
               aPC.Scale() = mScaleAbs;
               aPC.NivScale() = mNiv;
               // mAppli.AdaptScaleValide(aPC);
               aPC.ScaleStab() = -1;
               aPC.Id() = mAppli.GetNexIdPts();
               aSPC.OnePCarac().push_back(aPC);
          }
       }
    }

    std::cout << "    LAPL TIFF Scccc= " << mScaleAbs  << " SpaceE=" <<  aSpaceNbExtr << " Scale=" << aSS_NbExtr  << " Ctsr=" << aSSCstr<< "\n";
   if (DebugNRPH)
   {
       // int aNb = mImMod.sz().x  * mImMod.sz().y; Warn unused
       std::cout << "  NbMax ; " 
                 << " Max" << ", H " << aNbDebugHRMax << ", L " << aNbDebugLRMax 
                 << " Min" << ", H " << aNbDebugHRMin << ", L " << aNbDebugLRMin 
                 << "\n";

        std::cout << " Ratio "
                 << " Max" << ", H " << KthValProp(aVMaxH,0.5) << ", L " << KthValProp(aVMaxL,0.5)  
                 << " Min" << ", H " << KthValProp(aVMinH,0.5) << ", L " << KthValProp(aVMinL,0.5)  
                 << "\n";

       // std::cout << " "  << (aSpaceNbExtr *1000.0) / aNb << " " << (aSS_NbExtr *50000.0) / aNb << "\n";
       
       getchar();
   }
}

void cOneScaleImRechPH::SiftMakeDif(cOneScaleImRechPH* aLR)
{
   InitImMod();
   if (mAppli.SaveFileLapl())
   {
      Symb_FNum aDif = (mIm.in()-aLR->mIm.in());
      double aMoy,aAbsMoy;
      double aNb = mSz.x * mSz.y;
      ELISE_COPY
      (
          mImMod.all_pts(),
          Virgule(aDif,Abs(aDif)),
          Virgule(mImMod.out()|sigma(aMoy),sigma(aAbsMoy))
      );
      std::cout << "LAPL TIFF Scccc= " << mScaleAbs  << " M=" << aMoy/aNb << " AM=" << aAbsMoy/aNb << "\n";
      Tiff_Im::CreateFromIm(mImMod,"LAPL-"  + ToString(mNiv) + ".tif");
   }
   else
   {
       for (int aY=0 ; aY<mSz.y ; aY++)
       {
          tElNewRechPH * aLIm  =      mIm.data()[aY];
          tElNewRechPH * aLRIm = aLR->mIm.data()[aY];
          tElNewRechPH * aIMod =   mImMod.data()[aY];
          for (int aX=0 ; aX<mSz.x ; aX++)
          {
              aIMod[aX] = aLIm[aX]-aLRIm[aX];
          }
       }
   }
   mSifDifMade = true;
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
