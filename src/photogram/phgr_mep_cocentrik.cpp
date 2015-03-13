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





/************************************************************/
/*                                                          */
/*                  MEP CO-CENTTRIQUE                       */
/*                                                          */
/************************************************************/

static const double PropCostEcartDist  = 0.95;
static const int    NbCpleBase = 2000;

static int NbForEcart(int aSize)
{
    int aNbVal = round_ni(aSize * PropCostEcartDist);
    aNbVal = ElMax(1,ElMin(aSize-1,aNbVal));

    return aNbVal;
}

double SomEcartDist
       (
           const ElMatrix<REAL> &    aMat,
           const std::vector<Pt3dr>& aVDir1,
           const std::vector<Pt3dr>& aVDir2
       )
{ 
    std::vector<double> aVRes;
    for (int aK=0 ; aK<int(aVDir1.size()) ; aK++)
    {
        aVRes.push_back(euclid(aMat*aVDir1[aK]-aVDir2[aK]));
    }
    // int aNbVal = round_ni(aVRes.size() * PropCostEcartDist);
    // aNbVal = ElMax(1,ElMin(int(aVRes.size())-1,aNbVal));
    return MoyKPPVal(aVRes,NbForEcart(aVRes.size()));
}


ElMatrix<REAL> GlobMepRelCocentrique(double & anEcartMin,const ElPackHomologue & aPack, int aNbRansac,int aNbMaxPts) 
{
   aNbMaxPts = ElMin(aNbMaxPts,aPack.size());

   std::vector<Pt3dr> aVDir1;
   std::vector<Pt3dr> aVDir2;

   cRandNParmiQ aRand(aNbMaxPts,aPack.size());

   for (ElPackHomologue::tCstIter itH=aPack.begin() ; itH!=aPack.end() ; itH++)
   {
      if (aRand.GetNext())
      {
          aVDir1.push_back(vunit(PZ1(itH->P1())));
          aVDir2.push_back(vunit(PZ1(itH->P2())));
      }
   }

   ElMatrix<REAL> aRes(3,3);
   anEcartMin = 1e60;

   while (aNbRansac)
   {
       int aKA = NRrandom3(aVDir1.size());
       int aKB = NRrandom3(aVDir2.size());
       if (aKA!=aKB)
       {
          aNbRansac--;
          ElMatrix<REAL> aMat = ComplemRotation(aVDir1[aKA],aVDir1[aKB],aVDir2[aKA],aVDir2[aKB]);
          double anEc = SomEcartDist(aMat,aVDir1,aVDir2);
          if (anEc<anEcartMin)
          {
              anEcartMin = anEc;
              aRes = aMat;
          }
       }
   }
   return aRes;
}


ElMatrix<REAL> ElPackHomologue::MepRelCocentrique(int aNbRansac,int aNbMaxPts) const
{
    double anEcMin;
    return GlobMepRelCocentrique(anEcMin,*this,aNbRansac,aNbMaxPts);
}
/*
*/



static const int NbRanCoCInit = 200;


class cMEPCoCentrik
{
     public :
        cMEPCoCentrik(const ElPackHomologue & aPack,double aFoc,const ElRotation3D * aRef = 0);
        void OneItereRotPur(ElMatrix<REAL>  & aMat,double & aDist);
        void OneTestMatr(const ElMatrix<REAL>  &,const Pt3dr & aBase,double aCost);

        void ComputePlanBase(const ElMatrix<REAL>  &);
        Pt3dr ComputeDirBase();

        const ElPackHomologue &  mPack;

        const ElRotation3D *     mRef;
        double                   mFoc;
        double                   mCostMin;
        Pt3dr                    mBaseMinIn;
        Pt3dr                    mBaseMinOut;
        std::vector<Pt2dr> mVP1;
        std::vector<Pt2dr> mVP2;
        std::vector<double> mVPds;
        bool                mDoPly;
        std::vector<Pt3dr>  mVOrtBase;  // Des vecteur |_ a la base
        std::vector<Pt3dr>  mPtsPly;  // Des vecteur |_ a la base
        std::vector<Pt3di>  mColPly;  // Des vecteur |_ a la base
};


//  aMat U2 = U1
//  aMat U2 = U1 + U1 ^W 


void cMEPCoCentrik::OneItereRotPur(ElMatrix<REAL>  & aMat,double & anErrStd)
{
    L2SysSurResol aSysLin3(3);
    aSysLin3.GSSR_Reset(false);

    std::vector<double> aVRes;
    double aSomP=0;
    double aSomErr=0;
    for (ElPackHomologue::const_iterator itP=mPack.begin() ; itP!=mPack.end() ; itP++)
    {
         Pt3dr aQ1 = vunit(PZ1(itP->P1()));
         Pt3dr aQ2 =  aMat * vunit(PZ1(itP->P2()));
         double aVQ2[3],aVQ1[3];
         aQ2.to_tab(aVQ2);
         aQ1.to_tab(aVQ1);

         double anEcart = euclid(aQ1-aQ2);
         aVRes.push_back(anEcart);
         double aPds =  itP->Pds() / (1 + ElSquare(anEcart / (2*anErrStd)));

         aSomP += aPds;
         aSomErr += aPds * square_euclid(aQ1-aQ2);;

         ElMatrix<REAL>  aMQ2 =  MatProVect(aQ2);
         for (int aY=0 ; aY< 3 ; aY++)
         {
             double aCoeff[3];
             for (int aX=0 ; aX< 3 ; aX++)
                 aCoeff[aX] = aMQ2(aX,aY);

             aSysLin3.GSSR_AddNewEquation(aPds,aCoeff,aVQ2[aY]-aVQ1[aY],0);
         }
    }
    std::cout << "ERR QUAD " << sqrt(aSomErr/aSomP) * mFoc << "\n";
    anErrStd = MoyKPPVal(aVRes,NbForEcart(aVRes.size()));
    Im1D_REAL8   aSol = aSysLin3.GSSR_Solve (0);
    double * aData = aSol.data();

    ElMatrix<double> aMPV =  MatProVect(Pt3dr(aData[0],aData[1],aData[2]));
   

    aMat  = NearestRotation(aMat * (ElMatrix<double>(3,true) +aMPV));
}

void cMEPCoCentrik::OneTestMatr(const ElMatrix<REAL>  & aMat,const Pt3dr & aBase,double  aCost)
{
    ElRotation3D aRot(aBase,aMat,true);
    double aCostIn =  ExactCostMEP(mPack,aRot,0.1);
    std::cout << "IN COST " <<  aCostIn * mFoc  << " for " << aBase  << "\n";
    for (int aKIter =0 ; aKIter < 10 ; aKIter++)
    {
        cBundleIterLin   aBIL(aRot,2*aCost);

        for (ElPackHomologue::const_iterator itP=mPack.begin() ; itP!=mPack.end() ; itP++)
        {
             aBIL.AddObs(vunit(PZ1(itP->P1())),vunit(PZ1(itP->P2())),itP->Pds());
        }

        for (int aK=2 ; aK<5 ; aK++)
        {
           aBIL.mSysLin5.LVM_Mul(1e-2 ,aK);
        }
/*
*/
         std::cout << "RESIDU " << KthValProp(aBIL.mVRes,0.85) * mFoc  << "\n";

        aRot =  aBIL.CurSol();
    }
    double aCostOut = ExactCostMEP(mPack,aRot,0.1);
    if (aCostOut < mCostMin)
    {
         mCostMin = aCostOut;
         mBaseMinIn = aBase;
         mBaseMinOut = aRot.tr();
    }
   std::cout << "OUT COST " <<  aCostOut * mFoc  << " CMIN " << mCostMin *mFoc << " for " << mBaseMinIn << mBaseMinOut<< "\n\n";
  
}


// Calcul les vecteur qui sont |_  dans un plan contenant la base
void cMEPCoCentrik::ComputePlanBase(const ElMatrix<REAL>  & aMat)
{
    Pt3di aRouge(255,0,0);
    Pt3di aVert(0,255,0);
    std::vector<Pt3dr> aVpt;
    for (int aKP =0 ; aKP< int(mVP1.size()) ; aKP++)
    {
         Pt3dr aQ1 = vunit(PZ1(mVP1[aKP]));
         Pt3dr aQ2 = aMat * vunit(PZ1(mVP2[aKP]));
         Pt3dr anOB = aQ1^aQ2;
         if (euclid(anOB) != 0)
         {
            anOB = vunit(anOB);
            if (mDoPly)
            {
                 mColPly.push_back(aRouge);
                 mPtsPly.push_back(anOB);
            } 
            mVOrtBase.push_back(anOB);
          }
    }


    if (mDoPly)
    {
         for (int aK=0 ; aK<10000  ; aK++)
         {
              int aKA = NRrandom3(mVOrtBase.size());
              int aKB = NRrandom3(mVOrtBase.size());

             
              Pt3dr aBase = mVOrtBase[aKA] ^  mVOrtBase[aKB];
              if (euclid (aBase)>1e-6)
              {
                  mColPly.push_back(aVert);
                  mPtsPly.push_back(vunit(aBase));
              }

         }
    }
}


Pt3dr cMEPCoCentrik::ComputeDirBase()
{
    // Valeur initiale par Ransac
    double aSomMin = 1e30;
    Pt3dr aBestNorm(0,0,0);
    for (int aCpt=0 ; aCpt<NbCpleBase ; )
    {
        int aKA = NRrandom3(mVOrtBase.size());
        int aKB = NRrandom3(mVOrtBase.size());

        if (aKA!=aKB)
        {
            Pt3dr aN = mVOrtBase[aKA] ^  mVOrtBase[aKB];
            if (euclid(aN) !=0)
            {
               aN = vunit(aN);

               double aSom = 0;
               for (int aK=0 ; aK< int(mVOrtBase.size()) ; aK++)
               {
                   aSom += ElAbs(scal(aN,mVOrtBase[aK]));
               }
               aSom /= mVOrtBase.size();
               if (aSom < aSomMin)
               {
                   aSomMin = aSom;
                   aBestNorm = aN;
               }
               aCpt++;
            }
        }
    }
    std::cout  << "SOMMiibnit " << aSomMin << "\n";

    Pt3dr aN0 = aBestNorm;
    // Affinaga par LSQ
    //  Ort . (aBestNor + b B + c C) =0

    for (int aCpt=0 ; aCpt < 3 ; aCpt++)
    {
        Pt3dr aB,aC;
        MakeRONWith1Vect(aBestNorm,aB,aC);
        L2SysSurResol aSysLin2(2);
        aSysLin2.GSSR_Reset(false);

        double aSomP = 0;
        double aSomE = 0;
        for (int aK=0 ; aK< int(mVOrtBase.size()) ; aK++)
        {
             double aCoeff[2];
             Pt3dr aP = mVOrtBase[aK];
             double aCste =  scal(aP,aBestNorm);
             aCoeff[0] = scal(aP,aB);
             aCoeff[1] = scal(aP,aC);
             double aPds = 1 / (1+ElSquare(aCste/aSomMin));
             aSysLin2.GSSR_AddNewEquation(aPds,aCoeff,-aCste,0);
             aSomE += aPds * ElSquare(aCste);
             aSomP += aPds;
        }
        Im1D_REAL8   aSol = aSysLin2.GSSR_Solve (0);
        double * aData = aSol.data();
        aBestNorm = vunit(aBestNorm+aB*aData[0] + aC*aData[1]);

        std::cout << "RESIDU " << sqrt(aSomE/aSomP) << "\n";

    }

    std::cout  << "SOMM " << aSomMin << aN0 << aBestNorm << "\n";

    if (mDoPly)
    {
          Pt3di aBleu(0,0,255);
          int aNb=1000;

          for (int aK=-aNb ; aK<= aNb ; aK++)
          {
              Pt3dr aN = aBestNorm * ((aK*1.2) / aNb);
// std::cout << "AXE " << aN << "\n";
              mPtsPly.push_back(aN);
              mColPly.push_back(aBleu);
          }
          Pt3dr aB,aC;
          MakeRONWith1Vect(aBestNorm,aB,aC);
          aNb=30;
          Pt3di aCyan(0,255,255);
          for (int aKb=-aNb ; aKb<= aNb ; aKb++)
          {
              for (int aKc=-aNb ; aKc<= aNb ; aKc++)
              {
                  Pt3dr aP = aB * (aKb*1.1)/aNb +  aC * (aKc*1.1)/aNb;

                  mPtsPly.push_back(aP);
                  mColPly.push_back(aCyan);
              }
          }
    }
    return aBestNorm;
}
 

extern void TestLinariseAngle(const  ElPackHomologue & aPack,const ElRotation3D &aRot,double aFoc);




cMEPCoCentrik::cMEPCoCentrik(const ElPackHomologue & aPack,double aFoc,const ElRotation3D * aRef) :
    mPack (aPack),
    mRef  (aRef),
    mFoc  (aFoc),
    mCostMin (1e9),
    mDoPly   (true)
{
     InitPackME(mVP1,mVP2,mVPds,aPack);

     ElTimer aChrono;
     double anEcart;
     ElMatrix<REAL> aMat =  GlobMepRelCocentrique(anEcart,mPack,NbRanCoCInit,aPack.size());
     aMat = aMat.transpose() ; // Retour aux convention 2 = > 1


     for (int aK=0 ; aK<6 ; aK++)
     {
         OneItereRotPur(aMat,anEcart);
     }

     // La cest Mat(P2) = P1

     // aMat = aMat.transpose() ;
     std::cout << "Time "    << aChrono.uval() << " Ecart "   << anEcart * mFoc << "\n";

/*
     const std::vector<Pt3di> aVTest = Dir26Cube();

     for (int aKP=0 ; aKP<int(aVTest.size()) ; aKP++)
     {
         OneTestMatr(aMat,vunit(Pt3dr(aVTest[aKP])), anEcart);
     }

     for (int aK= 0 ; aK<100000000 ; aK++)
     {
          Pt3dr aP(NRrandC(),NRrandC(),NRrandC());
          OneTestMatr(aMat,vunit(aP), anEcart );
          std::cout << "ITTTER " << aK << "\n";

          // if ((aK%1000) == 0) getchar();
     }
*/
     double aNoise = 1e-2;
     ElMatrix<double> aMP =  ElMatrix<double>::Rotation(aNoise*NRrandC(),aNoise*NRrandC(),aNoise*NRrandC());
     TestLinariseAngle(aPack,ElRotation3D(aRef->tr(),aRef->Mat(),true),aFoc);

     ComputePlanBase(aMat);
     Pt3dr aNorm = ComputeDirBase();

     std::cout << "END LIN \n";
     getchar();

     ShowMatr("REF/Mat",aRef->Mat()*aMat.transpose());
     // ShowMatr("Cycl",aMat);
     std::cout << "Tr, Ref " << vunit(aRef->tr()) << " " << aNorm << "\n";

     std::cout << "\nTEST REFERNCE \n";
     OneTestMatr(aRef->Mat()*aMP,vunit(aRef->tr()),anEcart);
     std::cout << "\nTEST CALC \n";
     OneTestMatr(aMat,aNorm,anEcart);



     if (mDoPly)
     {
         std::list<std::string> aVCom;
         std::vector<const cElNuage3DMaille *> aVNuage;
         cElNuage3DMaille::PlyPutFile
         (
               "Base.ply",
               aVCom,
               aVNuage,
               &mPtsPly,
               &mColPly,
               true
         );
    }


     getchar();
}


void TestMEPCoCentrik(const ElPackHomologue & aPack,double aFoc,const ElRotation3D * aRef)
{
    cMEPCoCentrik aMC(aPack,aFoc,aRef);
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
