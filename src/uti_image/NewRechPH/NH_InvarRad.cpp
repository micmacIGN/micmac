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

Pt2di cAppli_NewRechPH::SzInvRadUse()
{
   return Pt2di(mNbSR2Use, mNbTetaInv);
}

Pt2di cAppli_NewRechPH::SzInvRadCalc()
{
   return Pt2di(mNbSR2Calc, mNbTetaInv);
}

// Pt2dr aPTBUG (2914.32,1398.2);
Pt2dr aPTBUG (2892.06,891.313);

/*
void NormaliseSigma(double & aMoySom,double & aVarSig,const double & aPds)
{
   aMoySom /=  aPds;
   aVarSig /= aPds;
   aVarSig -= ElSquare(aMoySom);
   aVarSig = sqrt(ElMax(1e-10,aVarSig));
}
*/

static constexpr float DefInvRad = -1e20;

template <class Type,class TypeBuf> double  NormalizeVect(Im2D_INT1  aIout ,Im2D<Type,TypeBuf> aIin,int aK)
{
   int aTx = aIout.tx();
   INT1 * aDOut =  aIout.data()[aK];
   Type * aDIn =  aIin.data()[aK];

   double aS0  = 0.0;
   double aS1  = 0.0;
   double aS2  = 0.0;
   for (int aK=0 ; aK<aTx ; aK++)
   {
      float aVal = aDIn[aK];
      if (aVal != DefInvRad)
      {
      
         aS0 +=  1.0;
         aS1 +=  aVal;
         aS2 += ElSquare(aVal);
      }
   }
   aS1 /= aS0;
   aS2 /= aS0;
   aS2 -= ElSquare(aS1);
   double aSig = sqrt(ElMax(1e-10,aS2));

// static int aCpt=0;
// static int aCptOF=0;
   for (int aK=0 ; aK<aTx ; aK++)
   {
      if (aDIn[aK] != DefInvRad)
      {
         float aVal =  ((aDIn[aK]-aS1) / aSig) ;
         aDOut[aK] =  El_CTypeTraits<INT1>::Tronque(round_ni(aVal*DynU1));

         // aCpt++;
         // if (ElAbs(aVal)>4) aCptOF++;

// std::cout << "VVVVV= " << aVal << "\n";
      }
      else
      {
         aDOut[aK] = 0;
      }
   }

   return aS1;
   // std::cout << "Prop OF " << aCptOF / double(aCpt) << "\n";
}





class cRadInvStat
{
    public :
        cRadInvStat() :
           mComp (false),
           mS0 (0.0),
           mS1 (0.0),
           mS2 (0.0),
           mS3 (0.0)
         {
         }
         void Add(double aVal)
         {
            mLast = aVal;
            mS0++;
            mS1 += aVal;
            mS2 += ElSquare(aVal);
            mS3 += ElSquare(aVal) * aVal;
         }
         void CompStat()
         {
            mS1 /= mS0;
            mS2 /= mS0;
            mS3 /= mS0;
            mS2 -= ElSquare(mS1);
            mS3 -= ElSquare(mS1) *mS1;
            mS2 = sqrt(ElMax(1e-14,mS2));
            mS3 = (mS3>=0) ? pow(mS3,1/3.0) :   -pow(-mS3,1/3.0);
         }
    // private :
        bool mComp;
        double mLast;
        double mS0;
        double mS1;
        double mS2;
        double mS3;
};

class cComputeProfRad
{
    public :
         
         cComputeProfRad(int aNbProf,int aNbTeta) :
             mNbProf  (aNbProf),
             mImProf  (aNbTeta,aNbProf,0.0),
             mTImProf (mImProf),
             mImNorm  (aNbTeta,aNbProf)
         {
         }

         void AddCPR(int aKPr,int aKTeta,const cRadInvStat & aRIS )
         {
             mTImProf.add(Pt2di(aKTeta,aKPr),aRIS.mLast);
         }

         Im2D_INT1 Normalize()
         {
            for (int aKp=0 ; aKp<mNbProf ; aKp++)
                NormalizeVect(mImNorm,mImProf,aKp);
            return mImNorm;
         }

   // void  NormalizeVect(Im2D_INT1  aIout , Im2D_REAL4 aIin, int aK)
         int        mNbProf;
         tImNRPH    mImProf;
         tTImNRPH   mTImProf;
         Im2D_INT1  mImNorm;
};

   // return Pt2di(mNbSR2Use, mNbTetaInv);
double Normalise2D(Im2D_REAL4 aImBuf,Im2D_REAL4 aImOut,int aX0In,int aX1In,int aSzXOut)
{
    int aSzY = aImBuf.sz().y;
    double aS0,aS1,aS2;
    ELISE_COPY
    (
         rectangle(Pt2di(aX0In,0),Pt2di(aX1In,aSzY)),
         Virgule(1,aImBuf.in(),Square(aImBuf.in())),
         Virgule(sigma(aS0),sigma(aS1),sigma(aS2))
    );
    aS1 /= aS0;
    aS2 /= aS0;
    aS2 -= ElSquare(aS1);
    aS2 = sqrt(ElMax(1e-10,aS2));
    ELISE_COPY(rectangle(Pt2di(aX0In,0),Pt2di(aX0In+aSzXOut,aSzY)),(aImBuf.in()-aS1)/aS2, aImOut.out());

    return aS1;
}


void RobustNormalise(Im2D_REAL4 aImBuf,Im2D_REAL4 aImOut,double aProp,double  aMul)
{
   Pt2di aSz = aImBuf.sz();
   std::vector<REAL4> aVV;
   TIm2D<REAL4,REAL8>   aTImBuf(aImBuf);

   Pt2di aP;
   for (aP.x=0 ; aP.x<aSz.x;  aP.x++)
   {
       for (aP.y=0 ; aP.y<aSz.y;  aP.y++)
       {
            aVV.push_back(aTImBuf.get(aP));
       }
   }
   double aV0 =  KthValProp(aVV,aProp);
   double aV1 =  KthValProp(aVV,1-aProp);
   double aMoy = (aV0 + aV1) / 2.0;
   double anEcart = ((aV1-aV0)  / (1-2*aProp)) * aMul;

    
   ELISE_COPY
   (
       aImBuf.all_pts(),
       (aImBuf.in()-aMoy) / ElMax(anEcart,1e-10),
       aImOut.out()
   );

}



Im2D_INT1  MakeImI1(bool isRobust,Im2D_REAL4 aImIn, int aDyn)  
{
   Pt2di aSz = aImIn.sz();
   Im2D_REAL4 aImCor(aSz.x,aSz.y);
   if (isRobust)
       RobustNormalise(aImIn,aImCor,0.8,1.0); // Im2D_REAL4 aImBuf,Im2D_REAL4 aImOut,double aProp,double  aMul)
   else
      Normalise2D(aImIn,aImCor,0,aSz.x,aSz.x);
   // aMoy = Normalise(aImBuf,aImBuf,0,mNbSR2Use,mNbSR2Use);
   Im2D_INT1 aRes(aSz.x,aSz.y);
   ELISE_COPY
   (
        aRes.all_pts(),
        El_CTypeTraits<INT1>::TronqueF(round_ni(aImCor.in()*aDyn)),
        aRes.out()
   );

   return aRes;
}

Im2D_INT1  MakeImI1(bool isRobust,Im2D_REAL4 aImIn)
{
   return MakeImI1(isRobust,aImIn,DynU1);
}

/*
*/

bool  cAppli_NewRechPH::CalvInvariantRot(cOnePCarac & aPt,bool aModeTest)
{
   if (!mTImMasq.get(round_ni(aPt.Pt()),0))
   {
      return aPt.OK() = false;
   }
   bool BUG= false &&  (euclid(aPt.Pt()+Pt2dr(mP0Calc)-aPTBUG) < 0.02);
   static int aCpt=0;
   aCpt++;
   if (aPt.NivScale() >= mMaxLevR)
   {
      static int aCpt=0; aCpt++;
      return aPt.OK() = false;
   }

   // Buf[KTeta][KRho]   pour KRho=0, duplication de la valeur centrale
   Im2D_REAL4 aImBuf(SzInvRadCalc().x,SzInvRadCalc().y);
   TIm2D<REAL4,REAL8> aTBuf(aImBuf);

   std::vector<cOneScaleImRechPH *>  aVIm;
   // Tableau des distance / au centre pour echantillonner
   std::vector<double>               aVRhoAbs;
   std::vector<double>               aVDeltaRadAbs;  // distance entre deux  rho consecutif
   std::vector<double>               aVDeltaTangAbs; // distance entre deux  teta consecutif
   double aStepTeta =  (2*PI)/mNbTetaInv;

   int aN0 = aPt.NivScale();
   // aVIm.push_back(mVI1.at(aN0));
   for (int aKRho=0 ; aKRho <mNbSR2Calc ; aKRho++)
   {
       aVIm.push_back(mVILowR.at(aN0 + aKRho * mDeltaSR));
   }

   double aLastScaleAbs = ElSquare(aVIm.at(0)->ScaleAbs()) / aVIm.at(1)->ScaleAbs();
   double aRhoAbs = 0.0;

   for (int aKRho=0 ; aKRho<int(aVIm.size()) ; aKRho++)
   {
      double aCurScaleAbs = aVIm.at(aKRho)->ScaleAbs();
      double aDRhoAbs = ((aCurScaleAbs+aLastScaleAbs) / 2.0) *  mStepSR;
      aRhoAbs += aDRhoAbs;

      aVRhoAbs.push_back(aRhoAbs);
      aVDeltaRadAbs.push_back(aDRhoAbs);
      aVDeltaTangAbs.push_back(aCurScaleAbs*aStepTeta);
      aLastScaleAbs = aCurScaleAbs;
   }


   // On verifie maintenant que l'on est dedans, pour pouvoir tourner en mode test
   {
      double aRho = aRhoAbs * 1.1  + 5.0;
      Pt2dr aPRho(aRho,aRho);
      Pt2dr aP0 = aPt.Pt() - aPRho;
      Pt2dr aP1 = aPt.Pt() + aPRho;
      if ((aP0.x<0) || (aP0.y<0) || (aP1.x > mSzIm.x) || (aP1.y > mSzIm.y))
      {
         return aPt.OK() = false;
      }
   }
   if (aModeTest)
      return true;

   Pt2di aSzIm(mNbSR2Use,int(eTIR_NoLabel));
   aPt.InvR().ImRad() = Im2D_INT1(aSzIm.x,aSzIm.y,(INT1)0);
   Im2D_REAL4 aBufRad(aSzIm.x,aSzIm.y,DefInvRad);
   REAL4 ** aDataBR = aBufRad.data();

   // Calcul de l'image "log/polar"

   for (int aKTeta=0 ; aKTeta<mNbTetaInv; aKTeta++)
   {
      double aTeta = aKTeta * aStepTeta;
      Pt2dr aPTeta = Pt2dr::FromPolar(1.0,aTeta);
      for (int aKRho=0 ; aKRho<int(aVIm.size()) ; aKRho++)
      {
          double aDef = -1e5; 
          Pt2dr aP = aPt.Pt() + aPTeta * aVRhoAbs.at(aKRho);
          cOneScaleImRechPH & aIm = *(aVIm.at(aKRho));
          double aVal = aIm.TIm().getr(aP/aIm.PowDecim(),aDef);
          // double aVal = aVIm.at(aKRho)->TIm().getr(aP / ,aDef);
          // double aVal = aVIm.at(aKRho)->GetValImPAbs(aP,aDef);
          if (aVal==aDef)
          {
             return aPt.OK() = false;
          }
          aTBuf.oset(Pt2di(aKRho,aKTeta),aVal);
      }
   }

   double aMoy = 0;
   // Normalisation a priori, pour l'instant sans rolling
   if (mRollNorm)
   {
       for (int aX= 0 ; aX<mNbSR2Use ; aX++)
       {
            Normalise2D(aImBuf,aImBuf,aX,aX+mNbSR2Use,1);
       }
   }
   else
   {
      aMoy = Normalise2D(aImBuf,aImBuf,0,mNbSR2Use,mNbSR2Use);
   }
   aPt.MoyLP() = aMoy;

/*
   {
      double aS0,aS1,aS2;
      ELISE_COPY
      (
         rectangle(Pt2di(0,0),SzInvRadUse()),
         Virgule(1,aImBuf.in(),Square(aImBuf.in())),
         Virgule(sigma(aS0),sigma(aS1),sigma(aS2))
      );
      aS1 /= aS0;
      aS2 /= aS0;
      aS2 -= ElSquare(aS1);
      aS2 = sqrt(ElMax(1e-10,aS2));
      ELISE_COPY(aImBuf.all_pts(),(aImBuf.in()-aS1)/aS2, aImBuf.out());
   }
*/


   cComputeProfRad aProfR(5,mNbTetaInv);

   if (BUG)
   {
       std::cout << "PTBBUGGGGG 111\n";
       Tiff_Im::CreateFromIm(aImBuf,"NEWHBuf.tif");
   }

   int aKPS2 = mNbTetaInv /4 ;
   int aKPi  = mNbTetaInv /2 ;
   int aNbGrand = ((int) eTIR_NoLabel) / 3;
   for (int aKRho=0 ; aKRho<mNbSR2Use ; aKRho++)
   {
      double aRealDTeta =  aVDeltaRadAbs[aKRho] / aVDeltaTangAbs[aKRho];
      int aDTeta = ElMax(1,round_ni(aRealDTeta)); // Delta correspondant a 1 rho

      std::vector<cRadInvStat> aVRIS(aNbGrand,cRadInvStat());

      cRadInvStat & aRadiom = aVRIS[0];  // 0
      cRadInvStat & aGradRad = aVRIS[1];   //1
      cRadInvStat & aGradCrois = aVRIS[2]; //1 
      cRadInvStat & aGradTan = aVRIS[3]; //0
      cRadInvStat & aGradTanPiS2 = aVRIS[4]; //0
      cRadInvStat & aGradTanPi = aVRIS[5]; //0
      cRadInvStat & aLaplRad = aVRIS[6];   //2
      cRadInvStat & aLaplTan = aVRIS[7]; // 0
      cRadInvStat & aLaplCrois = aVRIS[8]; // 1
      cRadInvStat & aDiffOpposePi = aVRIS[9]; // 
      cRadInvStat & aDiffOpposePiS2 = aVRIS[10]; //


      bool WithRD1 = (aKRho>=1);
      bool WithRD2 = (aKRho>=2);

      int aKROp = aVIm.size()-1 - aKRho;

      for (int aKTeta=0 ; aKTeta<mNbTetaInv; aKTeta++)
      {
          int aKTetaPlus1 =  (1    +aKTeta)%mNbTetaInv;
          int aKTetaMoins1 = (mNbTetaInv-1    +aKTeta)%mNbTetaInv;
          int aKTetaPiS2  = (aKPS2+aKTeta)%mNbTetaInv;
          int aKTetaPi    = (aKPi+aKTeta)%mNbTetaInv;

          double aVal = aTBuf.get(Pt2di(aKRho,aKTeta));
          double aVTp1 = aTBuf.get(Pt2di(aKRho,aKTetaPlus1));
          double aVTm1 = aTBuf.get(Pt2di(aKRho,aKTetaMoins1));

          aRadiom.Add(aVal);
          aProfR.AddCPR(0,aKTeta,aRadiom);

          aGradTan.Add(ElAbs(aVal-aVTp1));
          aProfR.AddCPR(1,aKTeta,aGradTan);

          aGradTanPiS2.Add(ElAbs(aVal-aTBuf.get(Pt2di(aKRho,aKTetaPiS2))));
          aProfR.AddCPR(2,aKTeta,aGradTanPiS2);

          aGradTanPi.Add(ElAbs(aVal-aTBuf.get(Pt2di(aKRho,aKTetaPi)))); // Pas de profil, ambigu a Pi
          aLaplTan.Add(ElAbs(2*aVal-aVTp1-aVTm1));
          aProfR.AddCPR(3,aKTeta,aLaplTan);

          aDiffOpposePi.Add  (ElAbs(aVal -aTBuf.get(Pt2di(aKROp,aKTetaPi)))); // Pas de profil, ambigu
          aDiffOpposePiS2.Add(ElAbs(aVal -aTBuf.get(Pt2di(aKROp,aKTetaPiS2))));
          aProfR.AddCPR(4,aKTeta,aDiffOpposePiS2);

          if (WithRD1)
          {
             int aKTetaCr =  (aDTeta+aKTeta)%mNbTetaInv;
             double aVT0R1 = aTBuf.get(Pt2di(aKRho-1,aKTeta  ));
             double aVT1R0 = aTBuf.get(Pt2di(aKRho  ,aKTetaCr));
             double aVT1R1 = aTBuf.get(Pt2di(aKRho-1,aKTetaCr));
             aGradRad.Add(ElAbs(aVal-aVT0R1));
             aGradCrois.Add(ElAbs(aVal-aVT1R1));
             aLaplCrois.Add(ElAbs(aVal+aVT1R1 - aVT0R1 - aVT1R0));

             if (WithRD2)
             {
                double aVT0R2 = aTBuf.get(Pt2di(aKRho-2,aKTeta));
                aLaplRad.Add(ElAbs(2*aVT0R1-aVal-aVT0R2));  
             }
          }
      }
      
      for (int aK=0 ; aK< aNbGrand ; aK++)
      {
         cRadInvStat & aRIS = aVRIS[aK];  // 0
         if (aRIS.mS0)
         {
            aRIS.CompStat();
            aDataBR[aK][aKRho] = aRIS.mS1;
            aDataBR[aK+aNbGrand][aKRho] = aRIS.mS2;
            aDataBR[aK+2*aNbGrand][aKRho] = aRIS.mS3;
         }
      }
   }

   for (int aK=0 ; aK<int(eTIR_NoLabel) ; aK++)
   {
       NormalizeVect(aPt.InvR().ImRad(),aBufRad,aK);
   }

   // std::cout << "IMRRRRR " << aPt.InvR().ImRad().sz() << "\n";
   

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
       aLarg = aLarg + Arg_Tiff(Tiff_Im::AStrip( arrondi_sup(SzInvRadCalc().x,8)));
       Tiff_Im  aSaveBuf
       (
           aName.c_str(),
           SzInvRadCalc(),
           GenIm::real4,
           Tiff_Im::No_Compr,
           Tiff_Im::BlackIsZero,
           aLarg
       );
       ELISE_COPY(aImBuf.all_pts(),aImBuf.in(),aSaveBuf.out());
   }

   

   // Pour l'export xml
   {
      if (0)
      {
          double aS0,aS1,aS2;
          ELISE_COPY
          (
             rectangle(Pt2di(0,0),SzInvRadUse()),
             Virgule(1,aImBuf.in(),Square(aImBuf.in())),
             Virgule(sigma(aS0),sigma(aS1),sigma(aS2))
          );
          aS1 /= aS0;
          aS2 /= aS0;
          aS2 -= ElSquare(aS1);
          aS2 = sqrt(ElMax(1e-10,aS2));
          std::cout << "Ssssssss  " << aS1 << " " << aS2 << "\n";
        //   ELISE_COPY(aImBuf.all_pts(),(aImBuf.in()-aS1)/aS2, aImBuf.out());
      }

      aPt.ImLogPol() =  Im2D_INT1(SzInvRadUse().x,SzInvRadUse().y);
      ELISE_COPY(aPt.ImLogPol().all_pts(),El_CTypeTraits<INT1>::TronqueF(round_ni(aImBuf.in()*DynU1)),aPt.ImLogPol().out());
      aPt.VectRho() = aVRhoAbs;
      aPt.ProfR().ImProfil() = aProfR.Normalize();

      // std::cout << "HHHHHHHiiii " << aPt.ProfR().ImProfil().sz() << "\n";

      if (mDoInvarIm)
      {
           static int aCpt = 0; aCpt++;
#if PB_LINK_AUTOCOR
           ELISE_ASSERT(false,"MPD TRICK TO COMPILE : Horrriiiblllleee !!!!!!!");
           cCalcAimeImAutoCorr * aPtraCAIAC=0;
           cCalcAimeImAutoCorr & aCAIAC = *aPtraCAIAC;
#else
           cCalcAimeImAutoCorr aCAIAC(aPt.ImLogPol(),true);
#endif

           aPt.RIAC().IR0() = aCAIAC.mIR0.mImVis;
           aPt.RIAC().IGT() = aCAIAC.mIGT.mImVis;
           aPt.RIAC().IGR() = aCAIAC.mIGR.mImVis;

/*
           aCAIAC.mIR0.MakeTiff("Patch-IR0_Num_"+ToString(aCpt)+".tif");
           aCAIAC.mIGR.MakeTiff("Patch-IGR_Num_"+ToString(aCpt)+".tif");
           aCAIAC.mIGT.MakeTiff("Patch-IGT_Num_"+ToString(aCpt)+".tif");

           if ((aCpt%1000)==0)
              std::cout << "AAAAAAAAAAAAAAaaa " << aCpt  << " : \n";
           // getchar();
*/
      }

      if (BUG)
      {
          std::cout << "PTBBUGGGGG 22\n";
          Tiff_Im::CreateFromIm(aImBuf,"NORM-NEWHBuf.tif");
          getchar();
      }
   }

   return true;
}

//   ====================================================================
//   ====================================================================
//   ====================================================================

class cOnePCarac_HeapParam
{
     public :
        static void SetIndex(tPCPtr   aPCP,int i) { aPCP->HeapInd() = i; }
        static int  Index(tPCPtr     aPCP) { return aPCP->HeapInd(); }
};
class cOnePCarac_HeapCompare
{
    public :
// compare score correl global
        bool operator () (tPCPtr   & aPCP1,tPCPtr   & aPCP2) {return aPCP1->Prio() > aPCP2->Prio();}
        // est ce que objet 1 est meuilleur que 2
};


double ScalePrio(tPCPtr aPCP)
{
   double aSP = aPCP->ScaleStab();
   if(aSP>0) return aSP;
   return aPCP->Scale();
}


void  cAppli_NewRechPH::FilterSPC(cSetPCarac & aSPC,cSetPCarac & aRes,eTypePtRemark aLabel)
{
   mQT->clear();
   cOnePCarac_HeapCompare aCmp; // if aR1 > aR2
   ElHeap<tPCPtr,cOnePCarac_HeapCompare,cOnePCarac_HeapParam> aHeap(aCmp); // He
   for (auto & aPC :  aSPC.OnePCarac())
   {
       if ((aPC.Kind() == aLabel) && aPC.OK())
       {
           aPC.Prio() = ScalePrio(&aPC);
           if (mQT->insert(&aPC,true))
           {
              aHeap.push(&aPC);
           }
       }
   }

   // std::cout <<  " FilterSPCFilllllll  " <<  aHeap.nb() << "\n"; // getchar();
   tPCPtr aPCP;
   int aNb2Add = mNbMaxLabInBox;
   while (aHeap.pop(aPCP) && aNb2Add)
   {
       mQT->remove(aPCP);
       aNb2Add--;
       aRes.OnePCarac().push_back(*aPCP);
       double aDistInfl = mDistStdLab * 3;
       cVecTplResRVoisin<cOnePCarac *> aVRV;
       mQT->RVoisins(aVRV,aPCP->Pt(),aDistInfl);
       std::vector<cOnePCarac *> & aVV = aVRV;
       for (const auto & aVois : aVV)
       {
           if (aVois==aPCP)
           {
           }
           else
           {
               double aDist = euclid(aPCP->Pt()-aVois->Pt());
               double aPds = aDist / aDistInfl;
               aVois->Prio() *= aPds;
               aHeap.MAJ(aVois);  
           }
       }
   }

   // std::cout << "RRRRRr   "<< aRes.OnePCarac().size() << "\n"; getchar();
   // std::cout <<  " FilterSPCFilllllll  " <<  aHeap.nb() << " " << aNb2Add    << "\n"; // getchar();
}


void  cAppli_NewRechPH::FilterSPC(cSetPCarac & aSPC)
{
   cSetPCarac aNew;

   for (int aKLab=0 ; aKLab<int(eTPR_NoLabel) ; aKLab++)
   {
       FilterSPC(aSPC,aNew,eTypePtRemark(aKLab));
   }

   aSPC = aNew;
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
