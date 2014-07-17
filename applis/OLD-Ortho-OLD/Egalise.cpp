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

#include "Ortho.h"


/*************************************************/
/*                                               */
/*          cCC_Appli                            */
/*                                               */
/*************************************************/



void cAppli_Ortho::DoEgalise()
{
   if (mCompMesEg)
   {
       ComputeMesureEgale();
   }

   if (mERPrinc==0)
   {
       mERPrinc = cER_Global::read(mNameFileMesEg);
   }

// std::cout << "XXXXXXXXXXXXXXXXXXXXX " << mCO.Show().Val() << "\n";
// getchar();

   if (mCO.Show().Val())
   {
       mERPrinc->Show1();
   }
   if (mERPhys)
   {
      mERPhys->SolveSys
      (
          mSE->PdsRappelInit().Val(),
          mSE->PdsSingularite().Val()
      );
   }
   mERPrinc->SolveSys
   (
       mSE->PdsRappelInit().Val(),
       mSE->PdsSingularite().Val(),
       mERPhys
   );
//1e-3,1e-6);
}

static void VerifSizeVect(const std::vector<Pt2di> & aV)
{
    ELISE_ASSERT
    (
         (aV.size()>=2),
          "VerifSizeVect"
    );
}

static std::vector<cER_ParamOneSys> ReadParam
                             (
                                  int aNbCh,
                                  const std::vector<Pt2di> & aVPrinc,
                                  const std::vector<Pt2di> & aVSec,
                                  bool  OkEmpty
                             )
{
   VerifSizeVect(aVPrinc);
   std::vector<cER_ParamOneSys> aRes;
   aRes.push_back(cER_ParamOneSys(aVPrinc));
   for (int aK=1 ; aK<aNbCh ; aK++)
   {
       if (aVSec.size()!=0)
       {
          VerifSizeVect(aVSec);
          aRes.push_back(cER_ParamOneSys(aVSec));
       }
       else
          aRes.push_back(cER_ParamOneSys(aVPrinc));

   }
   return aRes;
}

void cAppli_Ortho::ComputeMesureEgale()
{
   int aNbCh =  mSE->EgaliseSomCh() ? 1 : mTF0->nb_chan();
   const cGlobRappInit & aGRI = mSE->GlobRappInit();

   double aPercL1  = mSE->PercCutAdjL1().Val();

   mERPrinc = cER_Global::Alloc
          (
             ReadParam(aNbCh,mSE->DegresEgalVois(),mSE->DegresEgalVoisSec(),false),
             ReadParam(aNbCh,aGRI.Degres(),aGRI.DegresSec(),false),
             mBoxCalc.sz(),
             aGRI.PatternApply().Val(),
             (mSE->AdjL1ByCple().Val()) && (aPercL1<100)
          );
   mERPrinc->PercCutAdjL1() = aPercL1;
   mERPrinc->Show() = mCO.Show().Val();
   if (aGRI.RapelOnEgalPhys().Val())
   {
       std::vector<Pt2di> aVPrinc;
       aVPrinc.push_back(Pt2di(-1,-1));
       aVPrinc.push_back(Pt2di(0,0));
       std::vector<Pt2di> aVSec;
       mERPhys = cER_Global::Alloc
              (
                  ReadParam(aNbCh,aVPrinc,aVSec,false),
                  ReadParam(aNbCh,aVPrinc,aVSec,false),
                  mBoxCalc.sz(),
                  aGRI.PatternApply().Val(),
                  (mSE->AdjL1ByCple().Val()) && (aPercL1<100)
               );
         
       mERPhys->PercCutAdjL1() = aPercL1;
       mERPhys->Show() = mCO.Show().Val();
   }


   for (int aKI=0 ; aKI<int(mVAllOrhtos.size()); aKI++)
   {
       {
          cER_OneIm * anERI = mERPrinc->AddIm
                           (
                               mVAllOrhtos[aKI]->Name(),
                               mVAllOrhtos[aKI]->SzIm()
                           );
          mVAllOrhtos[aKI]->SetERIPrinc(anERI);
       }

       if (mERPhys)
       {
           cER_OneIm * anERI = mERPhys->AddIm
                           (
                               mVAllOrhtos[aKI]->Name(),
                               mVAllOrhtos[aKI]->SzIm()
                           );
           mVAllOrhtos[aKI]->SetERIPhys(anERI);
       }
   }
   MapBoxes(eModeCompMesSeg);
// std::cout << "BEGIN-XXXXX COMPUT L1 PRINC \n";
   mERPrinc->Compute();
// std::cout << "END__XXXXX COMPUT L1 PRINC \n";
   mERPrinc->write(mNameFileMesEg);

   if (mERPhys)
   {
// std::cout << "BEGIN-XXXXX COMPUT L1 PHYS \n";
      mERPhys->Compute();
// std::cout << "END__XXXXX COMPUT L1 PHYS \n";
   }


}



void cAppli_Ortho::RemplitOneStrEgal()
{
  //  Pour la boite courante chargee mCurBoxOut, on dalle avec une dalle de
  //  taille aPer .  On tire au hasard un point dans  la dalle (aP0Glob
  //  qui devient aP0 dans le systeme charge en memoire)
  //  L'ordre dans lesquel est passe les image, est tire aleatoirement,
  //  cela  permettra d'ecrire les equations d'observation par couple successif
  //  sans biaiser 

if (mCO.TestDiff().Val())
{
   mVLI[0]->TestDiff(mVLI[1]);
}

   Pt2di aBr0(0,0);
   int aNbDef = ElMax(1,round_ni(sqrt(mNbPtMoyPerIm/mSE->NbPEqualMoyPerImage().Val())));
   int aPer = mSE->PeriodEchant().ValWithDef(aNbDef);

// std::cout << "PPeeerr " << aNbDef << " " << aPer << "\n";

   cDecoupageInterv2D  aDI2d(mCurBoxOut,Pt2di(aPer,aPer),Box2di(aBr0,aBr0));

   Pt2di aSzGlob = mCurBoxOut.sz();
   Im2D_Bits<1> aImOK(aSzGlob.x,aSzGlob.y,1);
   TIm2DBits<1> aTImOK(aImOK);
   int aSzV = mSE->SzVois();


   for (int aKB=0 ; aKB <aDI2d.NbInterv() ;aKB++)
   {
       Box2di aBox = aDI2d.KthIntervOut(aKB);
       Pt2di aDec = mCurBoxOut._p0;
       Pt2di aP0Glob  = RandomlyGenereInside(aBox)  ;
       Pt2di aP0  = aP0Glob - aDec ;

       // int aSzV = 5;

       std::vector<Pt2di> aVOrd;
       for (int aDx=-aSzV; aDx<=aSzV ; aDx++)
       {
           for (int aDy=-aSzV; aDy<=aSzV ; aDy++)
           {
                Pt2di aQ = aP0 + Pt2di(aDx,aDy);
                aTImOK.oset_svp(aQ,1);
           }
       }

       int aInd0 =  mTImIndex.get(aP0);
       
       if ((aInd0 >=0) && (ValMasqMesure(aP0)))
       {
           cLoadedIm * aImNadir = mVLI[aInd0];
          // double   aCor = mVLI[0]->Correl(aP0,mVLI[1],mVoisCorrel,mSzVC);

           int aNBOk = 0;
           for (int aKI=0 ; aKI<int(mVLI.size()) ; aKI++)
           {

              // if (aImNadir != mVLI[aKI])
              {
                 if (aImNadir->Im2Test() &&  mVLI[aKI]->Im2Test() && (aImNadir != mVLI[aKI]) )
                 {
                     double aCor = aImNadir->Correl(aP0,mVLI[aKI],mVoisCorrel,mSzVC);
                     std::cout << aImNadir->CurOrtho()->Name() << " " 
                               << aP0+aImNadir->DecLoc()  << " " 
                               << mVLI[aKI]->CurOrtho()->Name()  <<  " "
                               << aP0+mVLI[aKI]->DecLoc()  << " " 
                               << mVLI[aKI]->ValeurPC(aP0) << " "
                               << aCor <<  " " << mSeuilCorrel <<  "\n";
                 }

                if (
                         (mVLI[aKI]->ValeurPC(aP0) == 0)
                      && (aImNadir->Correl(aP0,mVLI[aKI],mVoisCorrel,mSzVC) > mSeuilCorrel)
                   )
                {
                   aVOrd.push_back(Pt2di(aKI,NRrandom3(1000000)));
                   for (int aDx=-aSzV; aDx<=aSzV ; aDx++)
                   {
                      for (int aDy=-aSzV; aDy<=aSzV ; aDy++)
                      {
                         Pt2di aQ = aP0 + Pt2di(aDx,aDy);
                         if (mVLI[aKI]->ValeurPC(aQ) != 0)
                         {
                            aTImOK.oset_svp(aQ,0);
                         }
                         else
                         {
                            aNBOk++;
                         }
                      }
                   }
                   aTImOK.oset(aP0,1);
                }
              }
           }
           if (aVOrd.size() >= 2)
           {
              std::sort(aVOrd.begin(),aVOrd.end(),CmpY);

          // double   aCor = mVLI[0]->Correl(aP0,mVLI[1],mVoisCorrel,mSzVC);
/*
*/
              cER_MesureNIm & aMERIPrinc = mERPrinc->NewMesure2DGlob(Pt2dr(aP0Glob));
              cER_MesureNIm * aMERIPhys = 0;
              if (mERPhys)
              {
                  Pt2di aIndBox = aDI2d.IndexOfKBox(aKB);
                  int aSubEchPhy = 2;
                  if ( ((aIndBox.x%aSubEchPhy)==0) && ((aIndBox.y%aSubEchPhy)==0))
                  {
                       aMERIPhys = &(mERPhys->NewMesure2DGlob(Pt2dr(aP0Glob)));
                  }
              }


              for (int aKI=0; aKI<int(aVOrd.size()) ; aKI++)
              {
                  cLoadedIm * aLI = mVLI[aVOrd[aKI].x];
                  cOneImOrhto * anOrt = aLI->CurOrtho();
                  cER_OneIm * anERIPrinc = anOrt->ERIPrinc();
                  cER_OneIm * anERIPhys = aMERIPhys ?  anOrt->ERIPhys(): 0;


                   std::vector<double> aVRes;
                   int aNb=0;
                   for (int aDx=-aSzV; aDx<=aSzV ; aDx++)
                   {
                      for (int aDy=-aSzV; aDy<=aSzV ; aDy++)
                      {
                         Pt2di aQ = aP0 + Pt2di(aDx,aDy);
                         if ( aTImOK.get(aQ,0))
                         {
                            std::vector<double> aVD = aLI->Vals(aQ);
                            if (aNb==0)
                               aVRes = aVD;
                            else
                            {
                                for (int aK=0 ; aK<int(aVRes.size()) ; aK++)
                                    aVRes[aK] += aVD[aK];
                            }
                            aNb++;
                         }
                      }
                   }
                   // std::cout << "NB = " << aNb << "\n";
                   ELISE_ASSERT(aNb!=0,"Incof Nb  in cAppli_Ortho::RemplitOneStrEgal");
                   for (int aK=0 ; aK<int(aVRes.size()) ; aK++)
                       aVRes[aK] /= aNb;

                  // std::vector<double> aVD = aLI->Vals(aP0);

                  if (mSE->EgaliseSomCh())
                  {
                      double aSom=0;
                      for (int aKC=0 ; aKC<int(aVRes.size()) ; aKC++)
                           aSom += aVRes[aKC];
                      aSom /= aVRes.size();
                      aVRes.clear();
                      aVRes.push_back(aSom);
                  }

                  std::vector<float> aVF;
                  for (int aK=0 ; aK<int(aVRes.size()) ; aK++)
                  {  
                      aVF.push_back(aVRes[aK]);
                  }

                  anERIPrinc->AddMesure(aMERIPrinc, aP0+aLI->DecLoc(), aVF);
                  if (anERIPhys)
                     anERIPhys->AddMesure(*aMERIPhys, aP0+aLI->DecLoc(), aVF);
              }
           }
       }
   }

   //mERG->write("toto.dat");
   //cER_Global * anERG =  cER_Global::read("toto.dat");
   // std::cout << anERG << "\n";

   // mERG->Show();
   // anERG->Show();
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
