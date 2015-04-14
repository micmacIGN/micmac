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

#include "NewOri.h"



static const double PropStdErDet = 0.75;


static  const double MulLVM = 10;
static  const double InitLVM = 1e-7;
static  const double ValLVMUnused = 1e-7;
static  const double DeltaLVMSteril = 0.2;
bool UseLVM = false;



double VarRel(const double & aNewEr,const double & anOldErr)
{
    return (anOldErr-aNewEr) / anOldErr;
}

/***********************************************************************/
/*                                                                     */
/*               LINEAR                                                */
/*                                                                     */
/***********************************************************************/

/*
double DistRot(const ElRotation3D & aR1,const ElRotation3D & aR2,double aBSurH) 
{
    Pt3dr aB1 = vunit(aR1.tr());
    Pt3dr aB2 = vunit(aR2.tr());
    if (scal(aB1,aB2) <0) aB1 = -aB1;
    double aDB = euclid(aB1-aB2);
    double aDM = sqrt(aR1.Mat().L2(aR2.Mat()));

    // std::cout << " DBase " << aDB << " DRot " << aDM << "\n";

   return aDB*aBSurH  + aDM;
}

*/

double  GenDistRot(const ElRotation3D & aR1,const ElRotation3D & aR2,double aBSurH,bool CorrecSens) 
{
    Pt3dr aB1 = vunit(aR1.tr());
    Pt3dr aB2 = vunit(aR2.tr());
    if (CorrecSens && (scal(aB1,aB2) <0)) aB1 = -aB1;
    double aDB = euclid(aB1-aB2);
    double aDM = sqrt(aR1.Mat().L2(aR2.Mat()));

    // std::cout << " DBase " << aDB << " DRot " << aDM << "\n";

   return aDB*aBSurH  + aDM;
}

double DistRot(const ElRotation3D & aR1,const ElRotation3D & aR2,double aBSurH) 
{
   return GenDistRot(aR1,aR2,aBSurH,true);
}

double SensDepDistRot(const ElRotation3D & aR1,const ElRotation3D & aR2,double aBSurH) 
{
   return GenDistRot(aR1,aR2,aBSurH,false);
}

double DistRot(const ElRotation3D & aR1,const ElRotation3D & aR2) 
{
    return DistRot(aR1,aR2,1.0);
}

cNOCompPair::cNOCompPair(const Pt2dr & aP1,const Pt2dr & aP2,const double & aPds) :
   mP1      (aP1),
   mP2      (aP2),
   mPds     (aPds),
   mQ1      (vunit(PZ1(aP1))),
   mQ2      (vunit(PZ1(aP2)))
{
}



// void AmelioreSolLinear(const ElRotation3D & aRot);
//  Alignement de U1,Base, RU2

double  cNewO_OrInit2Im::CostLinear(const ElRotation3D & aRot,const Pt3dr & aP1,const Pt3dr & aP2,double aTetaMax) const
{
      double aDet = scal(aP1^(aRot.Mat()*aP2),aRot.tr());
      aDet = ElAbs(aDet);

     
      if (aTetaMax<=0) return aDet;
      return  (aDet*aTetaMax) / (aDet + aTetaMax);
}

double cNewO_OrInit2Im::CostLinear(const ElRotation3D & aRot,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax) const
{
    return CostLinear(aRot,vunit(PZ1(aP1)),vunit(PZ1(aP2)),aTetaMax);
}

void cNewO_OrInit2Im::TestCostLinExact(const ElRotation3D & aRot)
{
    for (ElPackHomologue::const_iterator itP=mPackPStd.begin() ; itP!=mPackPStd.end() ; itP++)
    {
         double aCl = CostLinear(aRot,itP->P1(),itP->P2(),-1.0);
         double aCe = ExactCost(aRot,itP->P1(),itP->P2(),-1.0);

         std::cout << "R=" << aCl/aCe << "\n";
    }
}

void cNewO_OrInit2Im::AmelioreSolLinear(ElRotation3D  aRot,const std::string & aMes)
{
  mCurLamndaLVM =  UseLVM ? InitLVM : ValLVMUnused ;
  if (mShow )
  {
       std::cout << "========================= AmelioreSolLinear ======================\n";
       if (mTestC2toC1)
          std::cout << " DistRefInit " <<  DistRot(*mTestC2toC1,aRot) << "\n";
  }
// Si V FIXED PAS DE LEVENBERG SUR ELLE
   ElTimer aChrono;


   cInterfBundle2Image * aBundle =  mQuick ? mBundleIBI150 : mBundleIBI;
   double aNbIterMax = mQuick ? 4 : 9;
   mErStd =   aBundle->ErrInitRobuste(aRot,PropStdErDet);  // Version robuste, sans init
   mErStd =   aBundle->ResiduEq(aRot,mErStd);               // Version moindres carres
   
   


   double aCostIn = PixExactCost(aRot,0.1);

   bool aCont = true;
   bool aSqueeze = false;
   for (double  aNbIter=0 ; aCont ; )
   {
       // std::cout << "ERRR " << mErStd *FocMoy() << "\n";
       double aLastErr = mErStd;
       ElRotation3D aNewR  = aBundle->OneIterEq(aRot,mErStd);
       double anAmelio = aLastErr - mErStd;


       /// std::cout << "AAA " << anAmelio * FocMoy()   << "\n";
       //  std::cout << "AAA " << aErMoy * FocMoy()   << " " << mErStd  * FocMoy()  << " " << PixExactCost(aNewR,0.1) << "\n";
       //  std::cout << "Var ERr " << VarRel(aErMoy,aLastErrMoy) << " :: " << aLastErrMoy << " => " <<  aErMoy << "\n";

       if ((anAmelio >0) || (! UseLVM))
       {
          // Si tres proche de la meilleur sol
          if (mBestSolIsInit)
          {
              if (DistRot(aNewR,mBestSol) < 1e-4)
              {
                 aSqueeze = true;
                 if (!mShow) 
                     return;
              }
          }

          // Si tres proche de la solution precedente
          if (DistRot(aNewR,aRot) < 2e-5) 
          {
             aSqueeze = true;
             if (!mShow) 
                aCont = false;
          }

          aRot = aNewR;
          aNbIter+= 1;
          if (UseLVM)
             mCurLamndaLVM /=  MulLVM;
       }
       else
       {
          aNbIter+= DeltaLVMSteril;
          if (UseLVM) 
             mCurLamndaLVM *= MulLVM;
       }


       if (aNbIter >=  aNbIterMax)
          aCont = false;
   }
   double aCostOut = PixExactCost(aRot,0.1);

   if (aCostOut < mCostBestSol)
   {
      mBestSolIsInit = true;
      mCostBestSol = aCostOut;
      mBestSol = aRot;
      mBestErrStd = mErStd;
      mResidBest = mCurResidu;
   }


   if (mShow)
   {
       std::cout  << "For : " << aMes 
              << " ERStd " << mErStd 
              << " Exact " << aCostIn << " => " << aCostOut 
              << " Time " << aChrono.uval() 
              <<  " Inter " << MedianNuage(mPackStdRed,aRot)
              << " Sqz " << aSqueeze;
         std::cout << "\n";

         if (mTestC2toC1)
         {
            std::cout  << " D/Ref " <<   DistRot(*mTestC2toC1,aRot) << " Trans " << vunit(mTestC2toC1->tr()) << aRot.tr() << "\n";

             // ElMatrix<double> aM = aRot.Mat() * mTestC2toC1->Mat().transpose() ;
/*
             ElMatrix<double> aM =  mTestC2toC1->Mat().transpose() *  aRot.Mat() ;

             for (int aY=0 ; aY<3 ; aY++)
             {
                 printf("     ");
                 for (int aX=0 ; aX<3 ; aX++)
                 {
                      printf("%5f " ,aM(aX,aY));
                 }
                 printf("\n");
             }
             printf("\n");
             ElMatrix<double> aM2 = mTestC2toC1->Mat() ;
             for (int aY=0 ; aY<3 ; aY++)
             {
                 printf("     ");
                 for (int aX=0 ; aX<3 ; aX++)
                 {
                      printf("%5f " ,aM2(aX,aY));
                 }
                 printf("\n");
             }        
             printf("\n");
*/
         }
   }
} 

double PdsLinear(const ElRotation3D & aRot,std::vector<cNOCompPair> & aVP)
{
    double aSomPds=0;
    double aSomError=0;
    ElMatrix<double> aR0 = aRot.Mat();
    Pt3dr aBase = aRot.tr();

    for (int aK=0 ; aK<int(aVP.size()) ; aK++)
    {
        cNOCompPair & aPair = aVP[aK];
        Pt3dr aQp2 = aR0 * aPair.mQ2;
        double aCste =  scal(aBase,aPair.mQ1^aQp2);
        aSomPds += aPair.mLastPdsOfErr;
        aSomError += aPair.mLastPdsOfErr * ElSquare(aCste);
    }
    return aSomError / aSomPds;
}





// Equation pour la base  simple    [U1,Base, R U2] = 0
//      [U1,Base, R0 U2] = 0     
//      [U1,B0 + cC + d D , R0 U2] = 0     
//      B0.(U1^R0U2) +  +c C.[U1,R0U2] + d D. [U1,R0U2] = 0

Pt3dr  cNewO_OrInit2Im::CalcBaseOfRot(ElMatrix<double> aMat,Pt3dr aTr0)
{
    for (int aK=0 ; aK<int(mStCPairs.size()) ; aK++)
    { 
         cNOCompPair & aCP = mStCPairs[aK];
         aCP.mQ2R = aMat * aCP.mQ2;
         aCP.mU1vQ2R = aCP.mQ1 ^aCP.mQ2R;
    }

    ElTimer aChrono;
    double aLastErr = 1e5;
    for (int aK=0 ; aK<5  ; aK++)
    {
        double anErrMoy;
        Pt3dr aNewTr = OneIterCalcBaseOfRot(aMat,aTr0,anErrMoy);

        if (euclid(aNewTr-aTr0) < 1e-5) return aNewTr;


          // std::cout << "DIST-CBR " << euclid(aNewTr-aTr0) << " Err: " << anErrMoy << " VR " << VarRel(anErrMoy,aLastErr) << "\n";

        aTr0 = aNewTr;
        aLastErr = anErrMoy;
        DoNothingButRemoveWarningUnused(aLastErr);
    }
    // std::cout << "Time CalcBaseOfRot " << aChrono.uval() << "\n";

    return aTr0;
}


Pt3dr cNewO_OrInit2Im::OneIterCalcBaseOfRot(ElMatrix<double> aMat,Pt3dr aTr0,double & anErMoy)
{
    cGenSysSurResol & aSys = mSysLin2;
    double aCoef[2];
    aSys.GSSR_Reset(false);
    Pt3dr aC,aD;
    MakeRONWith1Vect(aTr0,aC,aD);

    double aSomErr=0;
    double aSomPds=0;

    for (int aK=0 ; aK<int(mStCPairs.size()) ; aK++)
    {
        double aPds = mBestErrStd-mResidBest[aK];
        if (aPds >0)
        {
            const cNOCompPair & aCP = mStCPairs[aK];
            const Pt3dr aU1vQ2R = aCP.mU1vQ2R ;
            double aCste =  scal(aTr0 ,aU1vQ2R);
            aCoef[0] = scal(aU1vQ2R ,aC);
            aCoef[1] = scal(aU1vQ2R ,aD);
            aSys.GSSR_AddNewEquation(aPds,aCoef,-aCste,0);
            aSomErr += ElAbs(aCste) * aPds;
            aSomPds += aPds;
        }
    }
    anErMoy = aSomErr / aSomPds;

    Im1D_REAL8   aSol = aSys.GSSR_Solve (0);
    double * aData = aSol.data();
    
    Pt3dr aRes =   vunit(aTr0+aC*aData[0] + aD*aData[1]);
    
    return aRes;
}






void  cNewO_OrInit2Im::AddNewInit(const ElRotation3D & aR)
{
}

//  (A^B) . C:




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
