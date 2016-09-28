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
#include "ReducHom.h"



NS_RHH_BEGIN



/*************************************************/
/*                                               */
/*                  cTestPlIm                    */
/*                                               */
/*************************************************/

double DistNorm(const cElemMepRelCoplan & anEl)
{
   Pt3dr aVert(0,0,1);
   return euclid(aVert-anEl.Norm());
}

double DistNorm(const cTestPlIm & aTpl)
{
   return DistNorm(*(aTpl.mRMCP));
}


void AddAqCohHom2Sys
     (
         L2SysSurResol & aSys,
         RMat_Inertie & aMat,
         double & aSomPds,
         const std::vector<Pt3dr> & aV,
         const cTestPlIm & aPL1,
         const cTestPlIm & aPL2
     )
{
   double aCoeff[4];  // A B TrX TrY
   for (int aK=0 ; aK<int(aV.size()) ; aK++)
   {
       Pt2dr aP(aV[aK].x,aV[aK].y);
       double aPds =aV[aK].z;

       Pt2dr aPt1 = aPL1.mHomI2T.Direct(aP);
       Pt2dr aPt2 = aPL2.mHomI2T.Direct(aP);
       
       aCoeff[0] = aPt1.x;
       aCoeff[1] = -aPt1.y;
       aCoeff[2] = 1;
       aCoeff[3] = 0;
       aSys.AddEquation(aPds,aCoeff,aPt2.x);
       //  
       aCoeff[0] = aPt1.y;
       aCoeff[1] = aPt1.x;
       aCoeff[2] = 0;
       aCoeff[3] = 1;
       aSys.AddEquation(aPds,aCoeff,aPt2.y);
       aSomPds += aPds;

       aMat.add_pt_en_place(aPt2.x,aPt2.y,aPds);
   }
}


// H1on2 permet d'inverser les roles de H1 et H2 ("ou" se fait la comparaison) pour tester la dissymetrie

// TestDiff si active permet de verifier les equation en refaisant le calcul a la main

double TestCohHomogr(const cTestPlIm & aPL1,const cTestPlIm & aPL2,bool H1On2,bool TestDiff = false)
{
  // bool TestDiff= false;

   L2SysSurResol aSys(4,true);
   double aSomPds =0;

   RMat_Inertie aMat;

   if (H1On2 || TestDiff)
   {
      AddAqCohHom2Sys(aSys,aMat,aSomPds,aPL1.mLnk->EchantP1(),aPL1,aPL2);
      if (!TestDiff) 
         AddAqCohHom2Sys(aSys,aMat,aSomPds,aPL2.mLnk->EchantP1(),aPL1,aPL2);
   }
   else
   {
      AddAqCohHom2Sys(aSys,aMat,aSomPds,aPL1.mLnk->EchantP1(),aPL2,aPL1);
      AddAqCohHom2Sys(aSys,aMat,aSomPds,aPL2.mLnk->EchantP1(),aPL2,aPL1);
   }



/*
   double aCoeff[4];  // A B TrX TrY
   for (int aK=0 ; aK<int(aV.size()) ; aK++)
   {
       Pt2dr aP(aV[aK].x,aV[aK].y);
       double aPds =aV[aK].z;

       Pt2dr aPt1 = aPL1.mHomI2T.Direct(aP);
       Pt2dr aPt2 = aPL2.mHomI2T.Direct(aP);
       
       aCoeff[0] = aPt1.x;
       aCoeff[1] = -aPt1.y;
       aCoeff[2] = 1;
       aCoeff[3] = 0;
       aSys.AddEquation(aPds,aCoeff,aPt2.x);
       //  
       aCoeff[0] = aPt1.y;
       aCoeff[1] = aPt1.x;
       aCoeff[2] = 0;
       aCoeff[3] = 1;
       aSys.AddEquation(aPds,aCoeff,aPt2.y);
       aSomPds += aPds;
   }
*/
   bool Ok;
   Im1D_REAL8 aSol = aSys.Solve(&Ok);
   double aResidu = aSys.ResiduOfSol(aSol.data());


   if (TestDiff)
   {
       const std::vector<Pt3dr> & aV =  aPL1.mLnk->EchantP1();
       double * aDS = aSol.data();
       Pt2dr aMul(aDS[0] ,aDS[1]);
       Pt2dr aTR(aDS[2] ,aDS[3]);


       double aSomV = 0;

       for (int aK=0 ; aK<int(aV.size()) ; aK++)
       {
           Pt2dr aP(aV[aK].x,aV[aK].y);
           double aPds =aV[aK].z;

           Pt2dr aPt1 = aPL1.mHomI2T.Direct(aP);
           Pt2dr aPt2 = aPL2.mHomI2T.Direct(aP);

           Pt2dr aDif = aPt1 * aMul + aTR - aPt2;

           std::cout <<"DIF " << aDif  << aPt1  << aP << "\n";

           aSomV += aPds * square_euclid(aDif);
        }

        std::cout << "RSSSS " << aResidu << " " << aSomV << "\n";
   }


   aMat = aMat.normalize();
   double aTrace = sqrt(aMat.s11() + aMat.s22());

   if (TestDiff)
   {
       std::cout << " TRR " << aTrace << " " << aSomPds << "\n";
       getchar();
   }

   aResidu = sqrt(ElMax(0.0,aResidu/aSomPds));
   // aResidu *=  aPL1.mLnk->Srce()->CamC()->Focale();
   aResidu /= aTrace;
   return aResidu;
}

cTestPlIm::cTestPlIm(cLink2Img * aLnk,cElemMepRelCoplan * aRMCP,bool Show, double anEpsRes) :
     mLnk     (aLnk),
     mRMCP    (aRMCP),
     mResiduH (0.),
     mHomI2T  (mRMCP->HomCam2Plan(&mResiduH)),
     mOk      (mResiduH < anEpsRes)
{
/*
            if (Show)
               std::cout << "  PLL " << aLnk->Dest()->Name()
                      << mRMCP.Norm()
                      << mHomI2T.Direct(Pt2dr(0.5,0.5))
                      << "\n";
*/
}


/*************************************************/
/*                                               */
/*                  cImagH                       */
/*                                               */
/*************************************************/

//void cImagH::ShowLnk

void cImagH::TestCplePlan(int aKIm1,int aKIm2)
{
     const std::vector<cLink2Img*> &  aVL = VLink() ;
     cLink2Img * aLnk1   = aVL[aKIm1];
     cLink2Img * aLnk2   = aVL[aKIm2];
     cImagH * aIm1 = aLnk1->Dest();
     cImagH * aIm2 = aLnk2->Dest();

     std::cout << "SZ " << aIm1->mVTPlIm.size() << " " << aIm2->mVTPlIm.size() << "\n";

     printf("XXXXXXXXXX ");
     for (int aKP2=0; aKP2 < int(aIm2->mVTPlIm.size()) ; aKP2++)
          printf("N=%5f ",DistNorm(aIm2->mVTPlIm[aKP2]));
     printf("\n");

     for (int aKP1=0; aKP1 < int(aIm1->mVTPlIm.size()) ; aKP1++)
     {
          printf("N=%5f ",DistNorm(aIm1->mVTPlIm[aKP1]));
          for (int aKP2=0; aKP2 < int(aIm2->mVTPlIm.size()) ; aKP2++)
          {
              double aCost = TestCohHomogr(aIm1->mVTPlIm[aKP1],aIm2->mVTPlIm[aKP2],true);
              printf("C=%5f ",aCost);
          }
          printf("\n");
     }
}

#if (1)

std::string cImagH::EstimatePlan()
{

    // On regarde si on est en mode focus sur une une image (mise au point);
     mPlanEst = false;
     double aAltiCible = mAppli.AltiCible();
/*
     bool FocusOnThisIm = false;

     if (mAppli.HasImFocusPlan())
     {
         if (mName != mAppli.ImFocusPlan())
            return "";
         FocusOnThisIm = true;
     } 
     else
     {
        if (mAppli.SkipPlanDone() && ELISE_fp::exist_file(NameOriHomPlane()))
        {
           mPlanEst = true;
           return "";
        }
     }
*/


     const std::vector<cLink2Img*> &  aVL = VLink() ;
     int aNbL = (int)aVL.size();



     for (int aKL=0 ; aKL<aNbL ; aKL++)
     {
          cLink2Img * aLnk   = aVL[aKL];
          std::cout << "TEPd " << aLnk->Dest()->Name() ;
          // cElHomographie &   aHom = aLnk->Hom12();
          cElHomographie  aHom = aLnk->Dest()->H2ImC().Inverse();
          std::pair<Pt2dr,Pt2dr> aPair(Pt2dr(0,0),Pt2dr(0,0));
          cResMepRelCoplan aRCP = ElPackHomologue::MepRelCoplan(1,aHom,aPair);
          aLnk->Dest()->mVercp = aRCP.VElOk();
          std::vector<cElemMepRelCoplan> &  aVRCP =   aLnk->Dest()->mVercp ;
          for (int aK=0 ; aK<int( aVRCP.size()) ; aK++)
          {
              //    std::cout <<  aLnk->Dest()->Name();
               aVRCP[aK] = aVRCP[aK].ToGivenProf(aAltiCible);
               cTestPlIm  aTPI(aLnk, VData(aVRCP)+aK,false,1e-4*aAltiCible);
               if (aTPI.mOk)
               {
                  if (aK==0)  std::cout << aVRCP[aK].Rot().ImRecAff(Pt3dr(0,0,0))  << aVRCP[aK].Norm() <<aVRCP[aK].ImCam1(Pt2dr(0,0)) ;
                  // std::cout <<  aLnk->Dest()->mVercp[aK].Norm() << aLnk->Dest()->mVercp[aK].DPlan() ;
                  aLnk->Dest()->mVTPlIm.push_back(aTPI);
               }
               else
               {
                 std::cout << "*********";
               }
          }
          std::cout <<  "\n";
          // std::vector<cElemMepRelCoplan>  aVSol = aRCP.VElOk();
     }



      while (1)
      {
          std::cout << "=================\n";
          int aK1,aK2;
          GetLnkKbrd(aK1);
          GetLnkKbrd(aK2);
          TestCplePlan(aK1,aK2);
      }



     for (int aKL=0 ; aKL<aNbL ; aKL++)
     {
         cLink2Img * aLnk   = aVL[aKL];
         aLnk->Dest()->mVercp.clear();
         aLnk->Dest()->mVTPlIm.clear();
     }
     return "";
}



/*************************************************/
/*                                               */
/*                  cImagH                       */
/*                                               */
/*************************************************/
#else






class cCmpNormPlan
{
   public :
      bool operator () (const cElemMepRelCoplan & anEl1,const cElemMepRelCoplan & anEl2)
      {
          // return anEl1.Norm().z > anEl2.Norm().z;
          return DistNorm(anEl1) < DistNorm(anEl2);
      }
};


/*
class cTestPlIm
{
    public :
        cTestPlIm(cLink2Img * aLnk,const cElemMepRelCoplan & aRMCP,bool Show) :
             mLnk     (aLnk),
             mRMCP    (aRMCP),
             mHomI2T  (mRMCP.HomCam2Plan()),
             mOk      (true)
        {
            if (Show)
               std::cout << "  PLL " << aLnk->Dest()->Name()
                      << mRMCP.Norm()
                      << mHomI2T.Direct(Pt2dr(0.5,0.5))
                      << "\n";
        }
       
        cLink2Img *        mLnk;
        cElemMepRelCoplan  mRMCP;
        cElHomographie     mHomI2T;
        bool               mOk;
    private :
        // cTestPlIm(const cTestPlIm&);  // N.I.
};
*/


double CostSolCur(int aK1,const std::vector<cTestPlIm> & aVPlIm,const ElMatrix<double> & aMCost)
{
   double aRes = 0;

   for (int aK2=0 ; aK2<int(aVPlIm.size()); aK2++)
   {
       if (aVPlIm[aK2].mOk)
       {
          aRes += aMCost(aK1,aK2);
       }
   }

   return aRes;
}

int WorstSol(const std::vector<cTestPlIm> & aVPlIm,const ElMatrix<double> & aMCost)
{
   int aRes=-1;
   double aCostMax=-1;

   for (int aK=0; aK<int(aVPlIm.size()); aK++)
   {
       if (aVPlIm[aK].mOk)
       {
           double aCost = CostSolCur(aK,aVPlIm,aMCost);
           if (aCost > aCostMax)
           {
               aCostMax = aCost;
               aRes = aK;
           }
       }
   }
   ELISE_ASSERT(aRes>=0,"WorsSol");

   return aRes;
}



class cTestSolPl
{
    public :
         cTestSolPl(cLink2Img * aLnk,const std::vector<cElemMepRelCoplan> & aVS) :
             mLnk (aLnk),
             mVSols (aVS)
         {
         }

         cLink2Img * mLnk;
         std::vector<cElemMepRelCoplan>  mVSols;
};

void cImagH::TestEstimPlDirect()
{
     const std::vector<cLink2Img*> &  aVL = VLink() ;
     int aNbL = aVL.size();
     std::vector<cTestSolPl> aVS;
     for (int aKL=0 ; aKL<aNbL ; aKL++)
     {
          cLink2Img * aLnk   = aVL[aKL];
          // std::cout << "TEPd " << aLnk->Dest()->Name() << "\n";
          cElHomographie &   aHom = aLnk->Hom12();
          std::pair<Pt2dr,Pt2dr> aPair(Pt2dr(0,0),Pt2dr(0,0));
          cResMepRelCoplan aRCP = ElPackHomologue::MepRelCoplan(1,aHom,aPair);
          std::vector<cElemMepRelCoplan>  aVSol = aRCP.VElOk();
          aVS.push_back(cTestSolPl(aLnk,aVSol));
     }

     for (int aKL=1 ; aKL<int(aVS.size()) ; aKL++)
     {
           cLink2Img * aLnk   = aVL[aKL];
           std::cout << "TEPd " << aLnk->Dest()->Name() ;
           std::vector<cElemMepRelCoplan> & aV1 = aVS[aKL-1].mVSols;
           for (int aN1=0; aN1<int(aV1.size()) ; aN1++)
           {
               std::cout << aV1[aN1].Norm() ;
           }
           std::vector<cElemMepRelCoplan> & aV2 = aVS[aKL].mVSols;

           std::cout << "\n";

           for (int aN1=0; aN1<int(aV1.size()) ; aN1++)
           {
               for (int aN2=0; aN2<int(aV2.size()) ; aN2++)
               {
                    ElRotation3D aR1 = aV1[aN1].Rot().inv();
                    ElRotation3D aR2 = aV2[aN2].Rot().inv();
                    ElMatrix<double> aM1 = aR1.Mat();
                    ElMatrix<double> aM2 = aR2.Mat();

                    printf("%5f ",aM1.L2(aM2));
               }
           }
           std::cout << "\n";
           getchar();
     }

     getchar();
}






std::string cImagH::EstimatePlan()
{
     mPlanEst = false;
     bool FocusOnThisIm = false;
 

     double aAltiCible = mAppli.AltiCible();
     if (mAppli.HasImFocusPlan())
     {
         if (mName != mAppli.ImFocusPlan())
            return "";
          
         // TestEstimPlDirect();
         FocusOnThisIm = true;
     } 
     else
     {
        if (mAppli.SkipPlanDone() && ELISE_fp::exist_file(NameOriHomPlane()))
        {
           mPlanEst = true;
           return "";
        }
     }

     /*
          1-LOOK For the most reliable plane

          1-1  For each image, select the plane form :MepRelCoplan , if sufficient != of other
          1-2  Fill the matrix Mosct[PL1][PL2] whith the fitting cost (difference up to a similitude)
          1-3 Get the the best plane by iteratively suppress the worst and update cost

          2- Generate measure (Im & Ter) in the repair of plane
     */
     std::pair<Pt2dr,Pt2dr> aPair(Pt2dr(0,0),Pt2dr(0,0));
     std::vector<cTestPlIm> aVPlIm;

     if (FocusOnThisIm || mAppli.Show(eShowDetail))
        std::cout << " =========== Begin EstimatePlan " << mName  << " NbL0=" << mLnks.size() << "\n";


     //  1.1 
     // for (tMapName2Link::iterator itL = mLnks.begin(); itL != mLnks.end(); itL++)
     for (int aK=0 ; aK<int(VLink().size()) ; aK++)
     {
          cLink2Img * aLnk   = VLink()[aK];

          // ElPackHomologue & aPack = aLnk->Pack();
          cElHomographie &   aHom = aLnk->Hom12();

          // std::cout << "  " << aLnk->Dest()->Name()  << " Sz " << aPack.size() << " Qual  " << aLnk->QualHom() << " " ;

          cResMepRelCoplan aRCP = ElPackHomologue::MepRelCoplan(1,aHom,aPair);
          std::vector<cElemMepRelCoplan>  aVSol = aRCP.VElOk();
          cCmpNormPlan aCmp;
          std::sort(aVSol.begin(),aVSol.end(),aCmp);

          if (aVSol.size() >=2) 
          {
              double aS0 = DistNorm(aVSol[0]);
              double aS1 = DistNorm(aVSol[1]);

              if (aS0 + mAppli.SeuilDistNorm() < aS1)
              {
                  if (FocusOnThisIm)
                  {
                     ElRotation3D aR1 = aVSol[0].Rot().inv();
                     Pt3dr aP = aVSol[0].ToR1(Pt3dr(0,0,0)) *  (aAltiCible /  aVSol[0].DPlan());
                     std::cout << "   "  << aLnk->Dest()->Name() << " " << aP << " " << aS0 << " >> " << aS1  << " " <<  aVSol[0].ToR1(Pt3dr(0,0,0))  << " " << aVSol[0].DPlan() << "\n";
                  }
                  aVPlIm.push_back(cTestPlIm(aLnk,new cElemMepRelCoplan(aVSol[0]),mAppli.Show(eShowAll)));



              }
                    
          }
     }


      // 1.2
      int aNbCdt = aVPlIm.size();
      ElMatrix<double> aMCost(aNbCdt,aNbCdt,0.0);

      for (int aK1 = 0 ; aK1<int(aVPlIm.size()) ; aK1++)
      {
          for (int aK2 = 0 ; aK2<int(aVPlIm.size()) ; aK2++)
          {
             double aResidu =  (aK1==aK2) ? 0 :  TestCohHomogr(aVPlIm[aK1],aVPlIm[aK2],mAppli.H1On2());
             aMCost(aK1,aK2) += aResidu;
             aMCost(aK2,aK1) += aResidu;
          }
      }

      if (mAppli.Show(eShowAll))
      {
          for (int aK1 = 0 ; aK1<int(aVPlIm.size()) ; aK1++)
          {
              std::cout <<  aVPlIm[aK1].mLnk->Dest()->Name() << " ";
              for (int aK2 = 0 ; aK2<int(aVPlIm.size()) ; aK2++)
              {
                  printf("%5e ",aMCost(aK1,aK2));
              }
              printf(" Som=%5e \n",CostSolCur(aK1,aVPlIm,aMCost));
          }
      }


      // 1.3
      int aKBEst = -1;

       if (aVPlIm.size()==0)
          return "";
       else if (aVPlIm.size()==1)
       {
          aKBEst = 0;
       }
       else
       {
           for (int Elim=2; Elim<int(aVPlIm.size()) ; Elim++)
           {
               int aKWorst = WorstSol(aVPlIm,aMCost);
               aVPlIm[aKWorst].mOk = false;
           }

           std::vector<int> TwoBest;
           for (int aK=0; aK<int(aVPlIm.size()) ; aK++)
           {
               if (aVPlIm[aK].mOk)
               {
                  TwoBest.push_back(aK);
               }
           }
           ELISE_ASSERT(TwoBest.size()==2,"Not Two Best sol !!");
           int aK1 = TwoBest[0];
           int aK2 = TwoBest[1];
           aKBEst = (aVPlIm[aK1].mLnk->NbPts() > aVPlIm[aK2].mLnk->NbPts()) ? aK1 : aK2 ;
       }


       /*  ====      2    =============== */

       if (0) //(mAppli.Show(eShowGlob))
       {
          std::cout << "KBEST " << aVPlIm[aKBEst].mLnk->Dest()->Name() << "\n";
          std::cout << " =========== End  EstimatePlan \n";
       }

       cElemMepRelCoplan  mRMCP = *(aVPlIm[aKBEst].mRMCP) ;
       ElRotation3D aRCam12P = mRMCP.Plan().CoordPlan2Euclid().inv();

       Pt3dr aCentre0 = aRCam12P.ImAff(Pt3dr(0,0,0));
       double aMulXZ =  aAltiCible / aCentre0.z; // Si <0 remet les camera par dessus
       double aMulY  =  ElAbs(aMulXZ);           // Pour que la transfo soit directe

       std::string aDir = mAppli.Dir();

       ElRotation3D aR0(Pt3dr(0,0,0),0,0,0);
       CamStenopeIdeale aCam = CamStenopeIdeale::CameraId(true,aR0);

       
       Pt2dr aPMin(1e10,1e10);
       Pt2dr aPMax(-1e10,-1e10);

       cSetOfMesureAppuisFlottants  aMesureIm;
       cDicoAppuisFlottant          aDAF;
       cMesureAppuiFlottant1Im  aMAF1;
       aMAF1.NameIm() = mName;
       int aCpt=0;

       std::vector<ElPackHomologue>  aVPack;
       std::vector<std::string>  aVNamePack;
       cPatOfName  aPON;
       aPON.AddName(mName);

       for (tMapName2Link::iterator itL = mLnks.begin(); itL != mLnks.end(); itL++)
       {
            cLink2Img * aLnk   = itL->second;
            aPON.AddName(aLnk->Dest()->Name());
            // std::cout << "Plan::Export  " << aLnk->Dest()->Name() << "\n";
            const std::vector<Pt3dr> &  aVPts1 = aLnk->EchantP1();
            cElHomographie aHom =  aLnk->Hom12();

            ElPackHomologue aPack;
            cMesureAppuiFlottant1Im  aMAF2;
            aMAF2.NameIm() = aLnk->Dest()->Name();
            // std::list<Appar23> aL32;
            
            for (int aKP=0 ; aKP<int(aVPts1.size()) ; aKP++)
            {
                Pt3dr aPPds1 = aVPts1[aKP];
                double aPds = aPPds1.z;
                Pt2dr aP1 (aPPds1.x,aPPds1.y);
                Pt2dr aP2 = aHom.Direct(aP1);

                std::string anId = "App_"+ToString(aCpt);
                cOneMesureAF1I aMesIm;
                aMesIm.NamePt() = anId;

                aMesIm.PtIm() = aP1;
                aMAF1.OneMesureAF1I().push_back(aMesIm);
                
                aMesIm.PtIm() = aP2;
                aMAF2.OneMesureAF1I().push_back(aMesIm);

                aPMin.SetInf(aP1);
                aPMin.SetInf(aP2);
                aPMax.SetSup(aP1);
                aPMax.SetSup(aP2);

                aPack.Cple_Add(ElCplePtsHomologues(aP1,aP2,aPds));

                Pt3dr aPTC1 = mRMCP.ImCam1(aP1);
                Pt3dr aPtPl = aRCam12P.ImAff(aPTC1);
                aPtPl = Pt3dr(aPtPl.x*aMulXZ,aPtPl.y*aMulY,aPtPl.z*aMulXZ);

                cOneAppuisDAF anAF;
                anAF.Pt()  = aPtPl;
                anAF.NamePt() = anId;
                double anInc= aAltiCible * 1e-5;
                anAF.Incertitude() = Pt3dr(anInc,anInc,anInc);
                aDAF.OneAppuisDAF().push_back(anAF);
                
                // aL32.push_back(Appar23(aP2,aPtPl));

                
                aCpt++;
            }
            aMesureIm.MesureAppuiFlottant1Im().push_back(aMAF2);

 
            std::string aNameH = mAppli.NameFileHomolH(*aLnk);
            aVPack.push_back(aPack);
            aVNamePack.push_back(aNameH);
       }
 


       aMesureIm.MesureAppuiFlottant1Im().push_back(aMAF1);

//        std::cout << "BOX " << aPMin << " " << aPMax << "\n";

       Pt2dr aSzPh = aPMax-aPMin;
       double aCible = 1000;
       double aRab = 20;

       Pt2dr aPRab(aRab,aRab);
       double aHaut = ElMin(aSzPh.x,aSzPh.y);
       double aFocale =  round_ni( aCible/aHaut);
       Pt2dr aPP = Pt2dr(round_ni(aPRab - aPMin * aFocale));
       Pt2di aSzIm = round_ni(aSzPh*aFocale + aPRab*2);

       for (int aK=0 ; aK<int(aVPack.size()) ; aK++)
       {
           ElPackHomologue & aPack = aVPack[aK];
           for (ElPackHomologue::iterator itH=aPack.begin() ; itH!=aPack.end() ; itH++)
           {
                itH->P1() = aPP + itH->P1()*aFocale;
                itH->P2() = aPP + itH->P2()*aFocale;
           }
           aPack.StdPutInFile(aVNamePack[aK]);
       }

       cCalibrationInternConique aCIC =  StdGetFromPCP(Basic_XML_MM_File("CalibIdentite.xml"),CalibrationInternConique);
       aCIC.SzIm() = aSzIm;
       aCIC.PP() = aPP;
       aCIC.F() = aFocale;
       aCIC.CalibDistortion()[0].ModRad().Val().CDist() = aPP;
       MakeFileXML(aCIC,aDir+  "RHH/" + mName + "-Calib.xml");

       for 
       (
            std::list<cMesureAppuiFlottant1Im>::iterator itMAF=aMesureIm.MesureAppuiFlottant1Im().begin();
            itMAF !=aMesureIm.MesureAppuiFlottant1Im().end();
            itMAF++
       )
       {
            for 
            (
                  std::list<cOneMesureAF1I>::iterator itOM=itMAF->OneMesureAF1I().begin();
                  itOM !=itMAF->OneMesureAF1I().end();
                  itOM++
            )
            {
                itOM->PtIm() = aPP + itOM->PtIm()*aFocale;
            }
       }



       std::string aNameMesureIm = aDir+  "RHH/" + mName + "-Mesure-S2D.xml";
       std::string aNameMesureTer = aDir+ "RHH/" + mName + "-Mesure-S3D.xml";
       MakeFileXML(aMesureIm,aNameMesureIm);
       MakeFileXML(aDAF,aNameMesureTer);


       std::string aCom =   MM3dBinFile("Apero") 
                          + XML_MM_File("Apero-RHH-ByImIndiv.xml") 
                          + " DirectoryChantier=" + mAppli.Dir() 
                          + " +PatternIm=" + QUOTE(aPON.Pattern())
                          + " +MasterIm="  + mName
                          + " +Alti="  + ToString(aAltiCible);


       ///std::cout << "COM = " << aCom << "\n";

        mPlanEst = true;


        return aCom;
       // getchar();
}
#endif


NS_RHH_END


/*
*/
/*
*/




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
