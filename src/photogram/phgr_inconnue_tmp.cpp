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


/*
   Sur le long terme, modifier :

     - la possiblite d'avoir des fonctions multiple (Dim N)
     - associer a un fctr, un cSetEqFormelles
*/

#include "StdAfx.h"
#include <iterator>
#include "ext_stl/numeric.h"



extern bool DebugCamBil;

bool UseProjForFaisParal = true;

static int CptUPL = 0;
bool  BugUPL = false;
bool DBFActiveProf6 = false;
bool DebugPbCondFaisceau  = false;


bool NewBug = false;

bool BugNanFE = false;

/************************************************************/
/*                                                          */
/*                       PROBLEME CHOLESKY                  */
/*                                                          */
/************************************************************/

std::vector<cSsBloc> SomVSBl(const std::vector<cSsBloc> & aV1,const std::vector<cSsBloc> & aV2)
{
   std::vector<cSsBloc> aRes = aV1;
   std::copy(aV2.begin(),aV2.end(),std::back_inserter(aRes));
   return aRes;
}

cTestPbChol::cTestPbChol(const std::string & aName) :
   mName       (aName),
   mMinVP      (10.0),
   mMinSomVNeg (10.0)
{
}

cTestPbChol  theTPC_0("Before");
cTestPbChol  theTPC_1("After");


void cGenSysSurResol::VerifGlob(const std::vector<cSsBloc> & aVSB,bool doCheck,bool doSVD,bool doV0)
{
    double aCoef = CoeffNorm();
    int aNBV = 0 ;
    for (int aK=0 ;aK < int(aVSB.size()) ; aK++)
    {
        aNBV += aVSB[aK].Nb();
    }
    ElMatrix<tSysCho>  aMatr(aNBV,aNBV);
   
    Indexee_EcrireDansMatrWithQuad(aMatr,aVSB,aVSB);

    if (doCheck)
    {
      double aChkS = 0.0 ;
      for (int aX=0 ; aX<aNBV ; aX++)
      {
          double aChkL = 0.0 ;
          for (int aY=0 ; aY<aNBV ; aY++)
          {
                double aV = aMatr(aX,aY) /aCoef;
                double aChK = (aV * (6+sin((double)aY) + cos((double)(aX*aX))) ) ;
                 // aChK = aV;
                aChkL += aChK;
          }
          aChkS += aChkL;
      }
      printf("GLOB CHKS  %.18e \n",aChkS);
    }

   
    int aNb0Av=0;
    if (doV0)
    {
        for (int aX=0 ; aX<aNBV ; aX++)
        {
            for (int aY=0 ; aY<aNBV ; aY++)
            {
                 if (aMatr(aX,aY)==0)
                     aNb0Av ++;
            }
        }
    }
    for (int aX=0 ; aX<aNBV ; aX++)
    {
        for (int aY=aX ; aY<aNBV ; aY++)
        {
                aMatr(aX,aY) = aMatr(aY,aX);
        }
    }

    if (doV0)
    {
        int aNb0=0;
        for (int aX=0 ; aX<aNBV ; aX++)
        {
            for (int aY=0 ; aY<aNBV ; aY++)
            {
                 if (aMatr(aX,aY)==0)
                     aNb0 ++;
            }
        }
        std::cout << " NB0, AVANT " << aNb0Av  << " APRES " << aNb0 << "\n";
    }

    if (doSVD)
    {
        ElMatrix<tSysCho>  aValP(1,aNBV);
        ElMatrix<tSysCho>  aVecP(aNBV,aNBV);
        jacobi(aMatr,aValP,aVecP);

        double aVMIN = 1000 ;
        double aSomVNEG = 0;
        int aNbE2=0;
        int aNbE3=0;
        int aNbE4=0;
        for (int aK=0 ; aK<aNBV ; aK++)
        {
            double aV = aValP(aK,0) / aCoef;
            if (aV< aVMIN) aVMIN = aV;
            if (aV<0)  aSomVNEG += aV;
            if (ElAbs(aV) < 1e-2) aNbE2++;
            if (ElAbs(aV) < 1e-3) aNbE3++;
            if (ElAbs(aV) < 1e-4) aNbE4++;
        }
        std::cout << " VMIN " << aVMIN  << " SOM NEG " << aSomVNEG  << " NBVar " << aNBV << "\n";
        std::cout << " == S2 " << aNbE2 << " S3 " << aNbE3 << " S4 " << aNbE4 <<  "\n";
        if (aVMIN < -1)
           getchar();
    }
}


void cGenSysSurResol::BasicVerifMatPos(const std::vector<cSsBloc> & aVSB,int aCpt)
{
    bool isCroissant = true;
    for (int aK=1 ;aK < int(aVSB.size()) ; aK++)
    {
        if (aVSB[aK-1].I0AbsSolve() >aVSB[aK].I0AbsSolve())
           isCroissant = false;
    }
    if (isCroissant)
       return;

    for (int aKBl = 0 ; aKBl< int(aVSB.size()) ; aKBl++)
    {
        std::cout
                  << "ALLOC [" 
                  << aVSB[aKBl].I0AbsAlloc()   << " : "
                  << aVSB[aKBl].I1AbsAlloc()   << "] " 
                  << "SOLVE [" 
                  << aVSB[aKBl].I0AbsSolve()   << " : "
                  << aVSB[aKBl].I1AbsSolve()   << " ] "
                  << "\n";
    }

    double aCoef = CoeffNorm();
    int aNBV = 0 ;
    for (int aK=0 ;aK < int(aVSB.size()) ; aK++)
    {
        aNBV += aVSB[aK].Nb();
    }
    ElMatrix<tSysCho>  aMatr(aNBV,aNBV);
   
    Indexee_EcrireDansMatrWithQuad(aMatr,aVSB,aVSB);
    double aChkS = 0.0 ;
    for (int aX=0 ; aX<aNBV ; aX++)
    {
        double aChkL = 0.0 ;
        for (int aY=aX ; aY<aNBV ; aY++)
        {
                double aV = aMatr(aX,aY) /aCoef;
                double aChK = aV / (ElSquare(aNBV-aX) + 1 + (aNBV-aY));
                aChkL += aChK;
        }
        aChkS += aChkL;
        // std::cout << "LINE " << aX << " " << aChkL << "\n";
    }

    std::cout << "CPT-upl " << CptUPL << "\n";
    printf("BVMP:: %d %.9e \n",aCpt,aChkS);
     // std::cout  << "BVMP:: " << aCpt << " " << aChkS << "\n";
    getchar();
}


void cGenSysSurResol::VerifMatPos
     (
          const ElMatrix<tSysCho> & aInvLambda,
          const ElMatrix<tSysCho> & aLambda,
          cTestPbChol & aTPC,
          const std::vector<cSsBloc> & aV1,
          const std::vector<cSsBloc> & aV2
     )
{
    VerifMatPos(aInvLambda,aLambda,aTPC,SomVSBl(aV1,aV2));
}
void cGenSysSurResol::VerifMatPos
     (
            ElMatrix<tSysCho>  aInvLambda,
            ElMatrix<tSysCho>  aLambda,
            cTestPbChol & aTPC,
            const std::vector<cSsBloc> & aVSB
     )
{
    
/*
    std::cout << "=========  " <<  aTPC.mName << " =======\n";
    for (int aKBl = 0 ; aKBl< int(aVSB.size()) ; aKBl++)
    {
        std::cout
                  << "ALLOC [" 
                  << aVSB[aKBl].I0AbsAlloc()   << " : "
                  << aVSB[aKBl].I1AbsAlloc()   << "] " 
                  << "SOLVE [" 
                  << aVSB[aKBl].I0AbsSolve()   << " : "
                  << aVSB[aKBl].I1AbsSolve()   << " ] "
                  << "\n";
    }
*/
    double aCoef = CoeffNorm();
    static int aCPT = 0; aCPT++;

    int aNBV = 0 ;
    for (int aK=0 ;aK < int(aVSB.size()) ; aK++)
    {
        aNBV += aVSB[aK].Nb();
    }
    ElMatrix<tSysCho>  aMatr(aNBV,aNBV);
    ElMatrix<tSysCho>  aValP(1,aNBV);
    ElMatrix<tSysCho>  aVecP(aNBV,aNBV);
   
    Indexee_EcrireDansMatrWithQuad(aMatr,aVSB,aVSB);
    double aChkS = 0.0 ;

    for (int aX=0 ; aX<aNBV ; aX++)
    {
        for (int aY=aX ; aY<aNBV ; aY++)
        {
                aMatr(aX,aY) = aMatr(aY,aX);
                aChkS +=  aMatr(aX,aY) /aCoef;
        }
    }

    jacobi(aMatr,aValP,aVecP);

    double aVMIN = 0;
    double aSomVNEG = 0;
    for (int aK=0 ; aK<aNBV ; aK++)
    {
        double aV = aValP(aK,0);
        if (aV< aVMIN) aVMIN = aV;
        if (aV<0)  aSomVNEG += aV;
    }
    if   ((aVMIN<aTPC.mMinVP) || (aSomVNEG<aTPC.mMinSomVNeg))
    {
        std::cout << "=========  " <<  aTPC.mName << " =======\n";
        aTPC.mMinVP = ElMin(aTPC.mMinVP,aVMIN);
        aTPC.mMinSomVNeg = ElMin(aTPC.mMinSomVNeg,aSomVNEG);
        for (int aX=0 ; aX<3 ; aX++)
        {
            for (int aY=aX+1 ; aY<3 ; aY++)
            {
                aLambda(aX,aY) = aLambda(aY,aX);
            }
        }

        ElMatrix<tSysCho>  aLValP(1,3);
        ElMatrix<tSysCho>  aLVecP(3,3);
        jacobi(aLambda,aLValP,aLVecP);

        ElMatrix<tSysCho>  aILValP(1,3);
        ElMatrix<tSysCho>  aILVecP(3,3);
        jacobi(aInvLambda,aILValP,aILVecP);

/*
        for (int aY=0 ; aY<aNBV ; aY++)
        {
            for (int aX=0 ; aX<aNBV ; aX++)
            {
                std::cout  << aMatr(aX,aY) << " ";
            }
            std::cout << "\n";
        }
*/
/*
        std::cout  << "LAMBDA " << aLValP(0,0) << " " 
                                << aLValP(1,0) << " "
                                << aLValP(2,0)  << "\n";
        std::cout  << "InvLAA " << aILValP(0,0) << " " 
                                << aILValP(1,0) << " "
                                << aILValP(2,0)  << "\n";

*/
        std::cout << "CPT-upl " << CptUPL << "VP-MIN " << aTPC.mMinVP/aCoef << " SOMVN MIN " << aTPC.mMinSomVNeg/aCoef << "\n\n";
        // getchar();
    }
}


/************************************************************/
/*                                                          */
/*                       cRapOnZ                            */
/*                                                          */
/************************************************************/

cRapOnZ::cRapOnZ
(
     double aZ,
     double aIncertCompens,
     double aIncertEstim,
     const std::string & aLayerIm
) :
   mZ   (aZ),
   mIC  (aIncertCompens),
   mIE  (aIncertEstim),
   mLayerIm (aLayerIm)
{
}


const std::string & cRapOnZ::LayerIm() const
{
   return mLayerIm;
}



double cRapOnZ::Z() const {return  mZ;}
double cRapOnZ::IncEstim() const {return mIE;}
double cRapOnZ::IncComp() const {return mIC;}
Pt3dr cRapOnZ::PZ() const  
{
   return Pt3dr(0,0,Z());
}


/************************************************************/
/*                                                          */
/*                    cEqf1IncTmp                           */
/*                                                          */
/************************************************************/

cEqf1IncTmp::cEqf1IncTmp
(
     cSetEqFormelles & aSet
) :
   mVal     (0.0) ,
   mAlloc   (aSet.Alloc()),
   mCurInc  (mAlloc.CurInc()),
   mF       (mAlloc.NewF(&mVal))
{
}

void cEqf1IncTmp::SetVal(const double & aVal)
{
   // mVal = aVal;
   mAlloc.SetVar(aVal,mCurInc);
}

Fonc_Num cEqf1IncTmp::F()
{
   return mF;
}

double cEqf1IncTmp::Val() const
{
    return mVal;
}

/************************************************************/
/*                                                          */
/*                    cEqfBlocIncTmp                        */
/*                                                          */
/************************************************************/

cEqfBlocIncTmp::cEqfBlocIncTmp
(
     cSetEqFormelles & aSet,
     const std::string & aName,
     int aNbInc,
     bool Tmp
)  :
   cElemEqFormelle(aSet,Tmp),
   mName          (aName)
   // mFoncRapAffine (0)
{
   for (int aK=0 ; aK<aNbInc ; aK++)
      mIncTmp.push_back(new cEqf1IncTmp(aSet));
}

void cEqfBlocIncTmp::CloseEBI()
{
   IncInterv().SetName(mName);
   CloseEEF();
}

cElemEqFormelle::tContFcteur  cEqfBlocIncTmp::FctrRap(const double * aVal)
{
   return FoncRapp(0,mIncTmp.size(),aVal);
}

/*
cElCompiledFonc *  cEqfBlocIncTmp::FoncRapAffine()
{
   if (mFoncRapAffine==0)
   {
   }
   return mFoncRapAffine;
}
*/


/************************************************************/
/*                                                          */
/*                    cEqfP3dIncTmp                         */
/*                                                          */
/************************************************************/


cEqfP3dIncTmp::cEqfP3dIncTmp
( 
     cSetEqFormelles & aSet,
     const std::string & aName
) :
   cEqfBlocIncTmp(aSet,aName,3,true)
{
}

void cEqfP3dIncTmp::InitVal(const Pt3dr & aP)
{
   mIncTmp[0]->SetVal(aP.x);
   mIncTmp[1]->SetVal(aP.y);
   mIncTmp[2]->SetVal(aP.z);
}

Pt3d<Fonc_Num> cEqfP3dIncTmp::PF()
{
   return Pt3d<Fonc_Num>
          (
	     mIncTmp[0]->F(),
	     mIncTmp[1]->F(),
	     mIncTmp[2]->F()
	  );
	
}

cElemEqFormelle::tContFcteur  cEqfP3dIncTmp::FctrRap(const Pt3dr & aP)
{
   double aV[3];
   aP.to_tab(aV);
   return FoncRapp(0,mIncTmp.size(),aV);
}


/************************************************************/
/*                                                          */
/*                cEqfBlocIncNonTmp                         */
/*                                                          */
/************************************************************/

cEqfBlocIncNonTmp::cEqfBlocIncNonTmp
(
   cSetEqFormelles & aSet,
   const std::string & aName,
   int aNbInc
)  :
   cEqfBlocIncTmp(aSet,aName,aNbInc,false)
{
}

Fonc_Num cEqfBlocIncNonTmp::F(int aK)
{
    return mIncTmp[aK]->F();
}
double   cEqfBlocIncNonTmp::Val(int aK) const
{
    return mIncTmp[aK]->Val();
}

void cEqfBlocIncNonTmp::SetVal(int aK,const double & aVal)
{
   mIncTmp[aK]->SetVal(aVal);
}




/************************************************************/
/*                                                          */
/*                  cBufSubstIncTmp                         */
/*                                                          */
/************************************************************/

cBufSubstIncTmp::cBufSubstIncTmp
(
      cSetEqFormelles * aSet,
      int aNbX,
      int aNbY
)  :

   mSet      (aSet),
   mNbX      (aNbX),
   mNbY      (aNbY),
   mA        (1,aNbX),
   mB        (aNbY,aNbX),
   mBp       (aNbX,aNbY),
   mBpL      (aNbX,aNbY),
   mLambda   (aNbX,aNbX)
{
   // std::cout << " cBufSubstIncTmp  " << aNbX << " " << aNbY << "\n";
}


void cBufSubstIncTmp::Resize(cSetEqFormelles * aSet,int aNbX, int aNbY)
{
   mSet = aSet;
   mNbX = aNbX;
   mNbY = aNbY;
   mA.ResizeInside(1,mNbX);
   mB.ResizeInside(aNbY,aNbX);
   mBp.ResizeInside(aNbX,aNbY);
   mBpL.ResizeInside(aNbX,aNbY);
   mLambda.ResizeInside(aNbX,aNbX);
}


cBufSubstIncTmp * cBufSubstIncTmp::TheBuf()
{
   static cBufSubstIncTmp *  mTheBuf = new cBufSubstIncTmp(0,3,50);
   return mTheBuf;
}


static double aMaxCond = 1e4;
static double aSomCond = 0;
static double aNbCond = 0;
static double aNb100 = 0;

double aSeuilMaxCondSubstFaiseau =  1e60;


static std::vector<double> aVBSurH;
void ReinitStatCondFaisceau()
{
   aVBSurH.clear();
}

void ShowStatCondFaisceau(bool aShowVect)
{
     std::cout << "Cond = " << aMaxCond  
               << "  Moy = " << (aSomCond/aNbCond) 
               << "  SupS = " << (double(aNb100)/double(aNbCond))
               << " CptUPL=" << CptUPL
               << "\n";
      if (false) // (aShowVect && DebugPbCondFaisceau)
      {
          std::sort(aVBSurH.begin(),aVBSurH.end());
          static const int aNbPerc= 7;
          // double aPerc[aNbPerc] = {0.001,0.01,0.1,1.0,5.0,10.0,50.0};
          double aPerc[aNbPerc] = {50.0,90,95,99,99.9,99.99,99.999};
          for (int aKP=0 ; aKP<aNbPerc ; aKP++)
          {
              std::cout << " PERC=" << aPerc[aKP] 
                        << " Val="  << ValPercentile(aVBSurH,aPerc[aKP]) 
                        << "\n";
          }
      }
}


void SymM3(ElMatrix<tSysCho> & aM)
{
   for (int aX=0 ; aX<3 ; aX++)
   {
      for (int aY=0 ; aY<3 ; aY++)
      {
          aM(aX,aY) = aM(aY,aX);
      }
   }
}
void ShowM3(const ElMatrix<tSysCho> & aM,const std::string & aMes)
{
   std::cout << " ============ M3 === : "<<  aMes << "\n";
   for (int aY=0 ; aY<3 ; aY++)
   {
      for (int aX=0 ; aX<3 ; aX++)
      {
          std::cout << aM(aX,aY) << " " ;
      }
      std::cout << "\n";
   }

        ElMatrix<tSysCho>  aValP(1,3);
        ElMatrix<tSysCho>  aVecP(3,3);
        jacobi(aM,aValP,aVecP);
   
      for (int aX=0 ; aX<3 ; aX++)
      {
          std::cout << "   VP=" << aValP(aX,0) << "\n" ;
      }
}


double cBufSubstIncTmp::DoSubst
     (  // X et Y notation de la doc, pas ligne ou colonnes
          cSetEqFormelles * aSet,
          const std::vector<cSsBloc> &  aX_SBlTmp,
          const std::vector<cSsBloc> &  aY_SBlNonTmp,
          const int                     aNbBloc,
          //   const std::vector<int> &  ,//aVX,
          //   const std::vector<int> &  ,// aVY,
          bool                     doRaz
     )
{

if(DebugCamBil)
{
  std::cout << "DoSubst " << aX_SBlTmp.size() << " " << aY_SBlNonTmp.size() << " :: ";
  for (int aK=0 ;aK<int(aY_SBlNonTmp.size()) ; aK++)
  {
      cSsBloc aSBY = aY_SBlNonTmp[aK];
      std::cout << " [" << aSBY.I0AbsAlloc()  <<  " " << aSBY.I1AbsAlloc() << "]";
  }
  std::cout << "\n";
  getchar();
}
   cGenSysSurResol & aSys =  *(aSet->Sys());
/*
*/
  // ELISE_ASSERT(aNbBloc==int(aVY.size()),"Sz inc in cBufSubstIncTmp::DoSubst");
   // Resize(aSet,aVX.size(),aVY.size());
   int aNbX = aX_SBlTmp[0].Nb();
   Resize(aSet,aNbX,aNbBloc);

   bool SSym = aSys.OptSym();
   bool aSz3 = (aNbX==3);
   bool Sym3 = (SSym &&  aSz3);  // Cas en ajustement de faisceau


   if (DebugPbCondFaisceau)
   {
      // aSys.VerifMatPos(mLambda,mLambda,theTPC_1,aY_SBlNonTmp);
      // aSys.BasicVerifMatPos(aY_SBlNonTmp,0);
   }
//==============
   aSys.Indexee_EcrireDansMatrWithQuad(mLambda,aX_SBlTmp,aX_SBlTmp);

   ElMatrix<tSysCho> aSauvL  = mLambda;

   double aCond = mLambda.L2();
   if (Sym3)
   {
       tSysCho ** aDL =mLambda.data();
       cMSymCoffact3x3<tSysCho>   aCf(aDL);
       aCf.CoffSetInv(aDL);
   }
   else
   {
      if (SSym)
      {
          mLambda.SymetriseParleBas();
      }
      self_gaussj(mLambda);
   }


   aCond *=  mLambda.L2();
   aCond = sqrt(aCond) / (mLambda.tx() * mLambda.ty());
   aSomCond += aCond;
   aNbCond++;
   if(aCond> 100)
     aNb100++;
   if (aCond> aMaxCond)
   {       
      aMaxCond = aCond;
      ShowStatCondFaisceau(false);
   }
//   if (DebugPbCondFaisceau) { aVBSurH.push_back(aCond); }



   // EN fait  , c'est deja trop tard .....
   if (true)
   {
       aSys.Indexee_EcrireDansMatrColWithLin(mA,aX_SBlTmp);
       aSys.Indexee_EcrireDansMatrWithQuad(mBp,aX_SBlTmp,aY_SBlNonTmp);
       if (SSym)
       {
            mB.transpose(mBp);
       }
       else
       {
          aSys.Indexee_EcrireDansMatrWithQuad(mB,aY_SBlNonTmp,aX_SBlTmp);
       }

       mBpL.mul(mBp,mLambda);
       if (aSz3)
       {
           aSys.SoutraitProduc3x3
           (
                SSym,
                mBpL,mB,
                &aY_SBlNonTmp
           );
       }
       else
       {
          ELISE_ASSERT(false,"Complete Inconnue Tmp with number != 3");
       }
       ElMatrix<tSysCho> aBpLA =  mBpL * mA;
       aSys.Indexee_SoustraitMatrColInLin(aBpLA,aY_SBlNonTmp);
   }
   if (DebugPbCondFaisceau)
   {
      // aSys.VerifMatPos(mLambda,mLambda,theTPC_1,aY_SBlNonTmp);
       // aSys.BasicVerifMatPos(aY_SBlNonTmp,100);
   }


   if (doRaz)
   {
       aSys.Indexee_LinSet0(aX_SBlTmp);
       aSys.Indexee_QuadSet0(aX_SBlTmp,aY_SBlNonTmp);
       aSys.Indexee_QuadSet0(aX_SBlTmp,aX_SBlTmp);
       aSys.Indexee_QuadSet0(aY_SBlNonTmp,aX_SBlTmp);
   }

   return aCond;
}

void cBufSubstIncTmp::RazNonTmp(cSetEqFormelles * aSet,const std::vector<cSsBloc> &  aY_SBlTmp)
{
     cGenSysSurResol & aSys =  *(aSet->Sys());
     aSys.Indexee_LinSet0(aY_SBlTmp);
     aSys.Indexee_QuadSet0(aY_SBlTmp,aY_SBlTmp);
}



/************************************************************/
/*                                                          */
/*                  cSubstitueBlocIncTmp                    */
/*                                                          */
/************************************************************/

cSubstitueBlocIncTmp::cSubstitueBlocIncTmp(cEqfBlocIncTmp & aBlocTmp) :
   mNbBloc  (0),
   mBlocTmp (aBlocTmp),
   mCond    (-1)
{
   mVSBlTmp.push_back(aBlocTmp.Set()->GetBlocOfI0Alloc(aBlocTmp.IncInterv()));

/*
   for (int anI=mBlocTmp.IncInterv().I0Alloc() ;anI<mBlocTmp.IncInterv().I1Alloc() ; anI++)
       mVIndTmp.push_back(anI);
*/
}

void cSubstitueBlocIncTmp::AddInc(const cIncListInterv & anILI)
{
   const cMapIncInterv & aMap =  anILI.Map();
   for(cMapIncInterv::const_iterator itM=aMap.begin();itM!=aMap.end();itM++)
   {
      cSsBloc aSB = mBlocTmp.Set()->GetBlocOfI0Alloc(*itM);

      bool GotEq = false;

      if  (aSB==mVSBlTmp[0])
      {
            GotEq = true;
      }
      else
      {
          // On a besoin de connaitre l'ordre Tmp/Non Tmp pour la gestion des sym
          ELISE_ASSERT(mVSBlTmp[0].I0AbsAlloc()>=aSB.I1AbsAlloc(),"cSubstitueBlocIncTmp::AddInc recouvrement / TMP ");
      }

      for (int aK=0 ; aK<int(mSBlNonTmp.size()) ; aK++)
      {
         if (aSB==mSBlNonTmp[aK])
         {
            GotEq = true;
         }
         else
         {
               ELISE_ASSERT(aSB.disjoint(mSBlNonTmp[aK]),"cSubstitueBlocIncTmp::AddInc recouvrement");
         }
      }

      if (! GotEq)
      {
         mSBlNonTmp.push_back(aSB);
         mNbBloc += aSB.Nb();
      }

      // std::cout <<  "   "  << itM->I0() << " " << itM->I1() << " " << itM->NumBloc() << "\n";
/*
      for (int anI=itM->I0Alloc() ; anI<itM->I1Alloc() ; anI++)
      {
         if ((!BoolFind(mVIndTmp,anI))&&(!BoolFind(mVINonTmp,anI)))
	     mVINonTmp.push_back(anI);
      }
*/
   }
}


class cCmpSsBloc
{
    public :
     bool operator () (const cSsBloc & aI1,const cSsBloc & aI2)
     {
          return aI1.I0AbsAlloc() < aI2.I0AbsAlloc();
     }
};
/*
*/

void cSubstitueBlocIncTmp::Close()
{
/*
   //   std::sort(mVIndTmp.begin(),mVIndTmp.end());
   //   std::sort(mVINonTmp.begin(),mVINonTmp.end());
*/

   cCmpSsBloc aCmp;
   std::sort(mSBlNonTmp.begin(),mSBlNonTmp.end(),aCmp);

}


void cSubstitueBlocIncTmp::RazNonTmp()
{
   cBufSubstIncTmp::TheBuf()->RazNonTmp(mBlocTmp.Set(),mSBlNonTmp);
}
    

void cSubstitueBlocIncTmp::DoSubst(bool doRaz)
{

   mCond = cBufSubstIncTmp::TheBuf()->DoSubst
           (
              mBlocTmp.Set(),
              mVSBlTmp,mSBlNonTmp,mNbBloc,
              doRaz
           );
}

double  cSubstitueBlocIncTmp::Cond() const
{
    return mCond;
}

/************************************************************/
/*                                                          */
/*                  cParamPtProj                            */
/*                                                          */
/************************************************************/

const double cResiduP3Inc::TheDefBSurH = 1.0;

cParamPtProj::cParamPtProj(double aSeuilBsH,double aSeuilBsHRefut,bool aDebug) :
   mBsH       (cResiduP3Inc::TheDefBSurH),
   mDebug     (aDebug),
   mSeuilBsH  (aSeuilBsH),
   mSeuilBsHRefut  (aSeuilBsHRefut),
   mProjIsInit (false),
   wDist       (true)
{
}

/************************************************************/
/*                                                          */
/*                  cManipPt3TerInc                         */
/*                                                          */
/************************************************************/

cManipPt3TerInc::cManipPt3TerInc
(
        cSetEqFormelles &              aSet,
        cSurfInconnueFormelle *         anEqSurf,
        std::vector<cCameraFormelle *> aVCamVis,
        bool                           aClose
)  :
   mSet        (aSet),
   mP3Inc      (mSet.Pt3dIncTmp()),
   mEqSurf     (anEqSurf),
   mVCamVis    (aVCamVis),
   mSubst      (*mP3Inc),
   mTerIsInit  (false),  // Pour eventuellement eviter le re-calcul
   mPPP        (0.0,0.0,true), // INIT +ou- bidon, refaite par ailleurs
   mMulGlobPds (1.0)

{
    for (int aKC=0; aKC<int(mVCamVis.size()) ; aKC++)
    {
       mSubst.AddInc(mVCamVis[aKC]->IntervAppuisPtsInc());
    }
    if (mEqSurf)
    {
       mSubst.AddInc(mEqSurf->IntervSomInc());
    }

    if (aClose)
       mSubst.Close();
}

void VerifSizeCamPtPds
     (
           std::vector<CamStenope *>    aVCams,
           const cNupletPtsHomologues & aNuple,
           const std::vector<double> &  aVPds,
           std::vector<Pt3dr> *         aVAp
     )
{


   if (
            (aNuple.NbPts()!=int(aVCams.size()))
         || (aNuple.NbPts()!=int(aVPds.size()))
      )
   {
        std::cout << " Dim Pts = " << aNuple.NbPts()
	          << " Nb Cam = " << aVCams.size()
	          << " Nb Pds = " << aVPds.size()
		  << "\n";
        ELISE_ASSERT(false," Dim coherence in CalcPTerInterFaisceauCams");
   }
   double aFNbNN=0;
   for (int aK=0 ; aK<int(aVPds.size()) ; aK++)
       if (aVPds[aK] !=0)
          aFNbNN+=2;
    if (aVAp) 
    {
       Pt3dr aInc = (*aVAp)[1];
       aFNbNN += (aInc.x > 0);
       aFNbNN += (aInc.y > 0);
       aFNbNN += (aInc.z > 0);
    }
    if (aFNbNN<3.0)
    {
        std::cout << "Nb Contraintes Non nuls = " << aFNbNN;
        ELISE_ASSERT(false," Pas assez de poids non nul !");
    }
}

class cCmpX
{
    public :
       bool operator()(const Pt2dr & aP1,const Pt2dr & aP2)
       {
            return aP1.x < aP2.x;
       }
};

Pt3dr CalcPTerIFC_Robuste
      (
           double                       aDistPdsErr,
           std::vector<CamStenope *>    aVCC,
           const cNupletPtsHomologues & aNuple,
           const std::vector<double> &  aVPds
      )
{
   VerifSizeCamPtPds(aVCC,aNuple,aVPds,0);
   ELISE_ASSERT(aVPds[0] !=0 ,"CalcPTerIFC_Robuste Premier poids nul");


   std::vector<Pt2dr> aVAbscPds;

   ElSeg3D aSeg0 = aVCC[0]->F2toRayonR3(aNuple.PK(0));


   int aK;
   for (aK=1 ; aK<int(aVCC.size()) ; aK++)
   {
       if (aVPds[aK] != 0)
       {
          ElSeg3D aSegK = aVCC[aK]->F2toRayonR3(aNuple.PK(aK));
          double aA0,aAK;
          aSeg0.AbscissesPseudoInter(aA0,aAK,aSegK);

          Pt3dr aP0 = aSeg0.PtOfAbsc(aA0);
          Pt3dr aPK = aSegK.PtOfAbsc(aAK);

          Pt2dr aPIm0 = aVCC[0]->R3toF2(aP0);
          Pt2dr aPImK = aVCC[0]->R3toF2(aPK);

          double aDist  =euclid(aPIm0,aPImK);
          double aDistDir = euclid(aSeg0.TgNormee()-aSegK.TgNormee());
          double aPds = aVPds[aK] * aDistDir * (1/(aDist+aDistPdsErr));
          aVAbscPds.push_back(Pt2dr(aA0,aPds));
       }
   }

   cCmpX aCmpX;
   std::sort(aVAbscPds.begin(),aVAbscPds.end(),aCmpX);
   // Cumul des poids
   for (int aKa=1 ; aKa<int(aVAbscPds.size()) ; aKa++)
       aVAbscPds[aKa].y += aVAbscPds[aKa-1].y;

   double aDemiSomPds=aVAbscPds.back().y/2.0;
   
   aK=0;
   while (aVAbscPds[aK].y<aDemiSomPds)
       aK++;

   return aSeg0.PtOfAbsc(aVAbscPds[aK].x);
}

void cManipPt3TerInc::SetTerrainInit(bool isTerInit)
{
  mTerIsInit = isTerInit;
}

void cManipPt3TerInc::SetMulPdsGlob(double aMul)
{
   mMulGlobPds = aMul;
}

std::vector<CamStenope *> cManipPt3TerInc::VCamCur()
{
   std::vector<CamStenope *> aRes;
   for (int aK=0 ; aK<int(mVCamVis.size()) ; aK++)
       aRes.push_back(mVCamVis[aK]->NC_CameraCourante());
   return aRes;
}


Pt3dr cManipPt3TerInc::CalcPTerIFC_Robuste
      (
               double                       aDistPdsErr,
               const cNupletPtsHomologues & aNuple,
               const std::vector<double> &  aVPds
      )
{
   return ::CalcPTerIFC_Robuste(aDistPdsErr,VCamCur(),aNuple,aVPds);
}


/*
inline Pt3dr Pcoord2(const Pt3dr & aP)
{
   return Pt3dr(ElSquare(aP.x),ElSquare(aP.y),ElSquare(aP.z));
}
*/


void InspectInterFaisc
     (
          const std::vector<CamStenope *> &  aVCam,
          const std::vector<double> &  aVPds,
          const cNupletPtsHomologues & aNuple,
          const Pt3dr &                aPTer
     )
{
   for (int aK=0 ; aK<int(aVCam.size()) ; aK++)
   {
        Pt2dr aPIm = aNuple.PK(aK);
        CamStenope & aCam = *(aVCam[aK]);
        ElSeg3D   aSeg = aCam.F2toRayonR3(aPIm);
        Pt2dr aProj = aCam.R3toF2(aPTer);

        std::cout << "SEG[" << aCam.IdCam() << "," << aPIm << "]="   <<   aProj << "  " << aSeg.P0() << " " << aSeg.TgNormee() << " " << aVPds[aK] << "\n";
   }
}

bool OkReproj
     (  
          const std::vector<CamStenope *> &  aVCam,
          const std::vector<double> &  aVPds,
          const Pt3dr &                aPTer,
          int & aKP
    )
{
   aKP = -1;
   for (int aK=0 ; aK<int(aVCam.size()) ; aK++)
   {
       if (aVPds[aK] >0)
       {
           CamStenope & aCam = *(aVCam[aK]);
           Pt2dr aProj = aCam.R3toF2(aPTer);
           if (! aCam.IsInZoneUtile(aProj))
           {
              aKP = aK;
              return false;
           }
        }
   }

   return true;
}

bool ResidualStepByStep = false;

Pt3dr  cManipPt3TerInc::CalcPTerInterFaisceauCams
       (
           const cRapOnZ *      aRAZ,
           bool                         CanUseProjectifP,
           bool                       & OKInter,
           const cNupletPtsHomologues & aNuple,
           const std::vector<double> &  aVPds,
           cParamPtProj &            aParam,
           std::vector<Pt3dr> *      aPAbs,
           std::string *             aMesPb
       )
{

if (0) // (MPD_MM())
{
    std::cout << "====== cManipPt3TerInc::CalcPTerInterFaisceau ====\n";
    for (int aK=0 ; aK< int(aVPds.size()) ; aK++)
       std::cout << "   PDS IM = " <<aVPds[aK] << "\n";;
    if (aPAbs)
       for (int aK=0 ; (2*aK)< int(aPAbs->size()) ; aK+=2)
          std::cout << "   PDS IM = " << (*aPAbs)[2*aK] << (*aPAbs)[2*aK+1] << "\n";
}


   std::vector<CamStenope *>  aVCC = VCamCur();
   VerifSizeCamPtPds(aVCC,aNuple,aVPds,aPAbs);

   std::vector<ElSeg3D> aVS;
   Pt3dr                aSomIntPl(0,0,0);
   double               aSomPds = 0.0;

   static  ElSeg3D aSeg(Pt3dr(0,0,0),Pt3dr(1,1,1));


   static int aCpt = 0; aCpt++;


   for (int aK=0 ; aK<int(aVCC.size()) ; aK++)
   {
        if (aNuple.IsDr(aK))
        {
        }
        else
        {
            Pt2dr aPIm = aNuple.PK(aK);
        // ElSeg3D aSeg ;
            cCamStenopeGrid * aCSG =  mVCamVis[aK]->PIF().CamGrid();
            if (aCSG)
            {
              aSeg = ElSeg3D(aVCC[aK]->PseudoOpticalCenter(),aVCC[aK]->L3toR3(aCSG->F2toDirRayonL3(aPIm)));
            }
            else
            {
              aSeg = aVCC[aK]->F2toRayonR3(aPIm);
            }

        // cCamStenopeGrid * aCSG =  mVCamVis[aK]->PIF().CamGrid();
	    if (mEqSurf)
	    {
	        Pt3dr aIP = mEqSurf->InterSurfCur(aSeg);
	        aSomIntPl = aSomIntPl + aIP* aVPds[aK];
	        aSomPds +=  aVPds[aK];
	    }
	    else 
	    {
               aVS.push_back(aSeg);
            }
         }
   }

       
   // Calcul du B/H par dispersion des direction
   Pt3dr aSomC(0,0,0);
   Pt3dr aSomT(0,0,0);
   Pt3dr aSomT2(0,0,0);
   double aSP = 0;
   double aSP2 = 0;
 // std::cout << "BsssHH " << aParam.mSeuilBsHRefut  << " "  << aParam.mSeuilBsH << " " << CanUseProjectifP<< "\n";
   if (((aParam.mSeuilBsH > 0) || (aParam.mSeuilBsHRefut>0)) && (CanUseProjectifP))
   {
      for (int aK=0 ; aK<int(aVCC.size()) ; aK++)
      {
          double aPds = aVPds[aK] ;
          Pt3dr aT = aVS[aK].TgNormee();
          aSomT = aSomT + aT*aPds;
          aSomT2 = aSomT2 + Pcoord2(aT) *aPds;
          aSP += aPds;
          aSP2 += ElSquare(aPds);
          aSomC = aSomC + aVCC[aK]->PseudoOpticalCenter() * aPds;
      }
      aSomT = aSomT / aSP;
      aSomT2 = aSomT2 / aSP;
      aSomT2 = aSomT2 - Pcoord2(aSomT);
      aSomC = aSomC / aSP;
      // Ce debiaisement est necessaire, par exemple si tous les poids sauf 1 sont
      // presque nuls
      double aDebias = 1 - aSP2/ElSquare(aSP);
      ELISE_ASSERT(aDebias>0,"Singularity in cManipPt3TerInc::CalcPTerInterFaisceauCams ");
      aSomT2 =  aSomT2/ aDebias;
       
      aParam.mEc2 = aSomT2.x+aSomT2.y+aSomT2.z;
      if (aParam.mEc2 <= -1e-7)
      {
           std::cout << "EC2 =" << aParam.mEc2 << "\n";
          ELISE_ASSERT(aParam.mEc2>-1e-7,"Singularity in cManipPt3TerInc::CalcPTerInterFaisceauCams ");
      }
      aParam.mEc2 = sqrt(ElMax(0.0,aParam.mEc2) );
      // Adaptation purement heuristique
      aParam.mBsH  = 1.35 * aParam.mEc2;


      if (aParam.mBsH < aParam.mSeuilBsHRefut)
      {
         OKInter = false;
         if (aMesPb)
         {
              *aMesPb= std::string("BSurH Insuf : ") + ToString(aParam.mBsH);
         }
         return Pt3dr(0,0,0);
      }
   
      if ((aParam.mBsH < aParam.mSeuilBsH) && (! aRAZ))
      {
          aParam.mProjIsInit = true;
      }
      else
      {
          aParam.mProjIsInit = false;
      }
   }
   else
   {
       aParam.mProjIsInit = false;
   }


   if (mEqSurf)
   {
      OKInter = true;
      return aSomIntPl / aSomPds;
   }
   else 
   {
      cResOptInterFaisceaux aROIF;
      bool OK;


      const std::vector<double> *  aPtrVPds = & aVPds;
      std::vector<double> aPMod = aVPds;

      if (aPAbs)
      {
         for (int aK=0 ; aK< int(aVPds.size()) ; aK++)
         {
             aPMod[aK] = aVPds[aK]/aVCC[aK]->ResolSolOfPt((*aPAbs)[0]);
         }
         aPtrVPds = &aPMod;
         
      }

      Pt3dr aRes =  ElSeg3D::L2InterFaisceaux(aPtrVPds,aVS,&OK,aRAZ,aParam.mProjIsInit ? &aROIF : 0,aPAbs);

      if (OK)
      {
         for (int aK=0 ; aK< int(aVPds.size()) ; aK++)
         {
             aPMod[aK] = aVPds[aK]/aVCC[aK]->ResolSolOfPt(aRes);
         }
         aRes =  ElSeg3D::L2InterFaisceaux(&aPMod,aVS,&OK,aRAZ,aParam.mProjIsInit ? &aROIF : 0,aPAbs);
      }


      aParam.mTer = aRes;
      aParam.mHaut = euclid(aParam.mTer-aSomC);

      aParam.mBase = aParam.mBsH * aParam.mHaut;


      int aKPb=-1;
      if (! OkReproj(aVCC,aVPds,aRes,aKPb))
      {
         OKInter = false;
         if (aMesPb)
         {
             *aMesPb = std::string("Mes Out of Im, for Im num : ") + ToString(aKPb);
         }

         return Pt3dr(0,0,0);
      }

      if (!OK)
      {
         OKInter = false;
         if (aMesPb)
         {
             *aMesPb = std::string("Intersection faisceau non definie ???");
         }

         return Pt3dr(0,0,0);
         ELISE_ASSERT(OK,"Pb ElSeg3D::L2InterFaisceaux in cManipPt3TerInc::CalcPTerInterFaisceauCams ");
      }


      if (aParam.mProjIsInit)
      {

          if (aROIF.mVal1 <=0)
          {
              OKInter = false;
              return aRes;
          }
          aParam.mK = aROIF.mVec1 / sqrt(aROIF.mVal1);
          aParam.mI = aROIF.mVec2 / sqrt(aROIF.mVal2);
          aParam.mJ = aROIF.mVec3 / sqrt(aROIF.mVal3);

// Bizzarement, alors que ca augmente le conditionnement
// ca fait converger sur le jeux de donnees Samantha
/*
          aParam.mK = aROIF.mVec1 ;
          aParam.mI = aROIF.mVec2 ;
          aParam.mJ = aROIF.mVec3 ;
*/

          aParam.mP0 = aRes;

      }
      
      OKInter = true;
      return aRes;
   }
}

const std::vector<cCameraFormelle *> &  
       cManipPt3TerInc::VCamVis() const
{
   return mVCamVis;
}



void ShowDebugFaisceau(double aCond,double aBH,int aCpt)
{

    Pt2di aSz(500,500);
    static Video_Win * aW = Video_Win::PtrWStd(aSz);
   
    std::cout << "RAtio " << (aCond*ElSquare(aBH)) << " Cond " <<     aCond  << "BH " << 1/aBH << " Cpt " << aCpt << "\n";
    aCond = log(aCond);
    aBH =  -log(aBH);

    if (aW) 
    {
        aW-> draw_circle_loc(Pt2dr(aCond,aBH)*100.0,1.0,aW->pdisc()(P8COL::red));
    }
}


const cResiduP3Inc& cManipPt3TerInc::UsePointLiaisonGen
                           (
                              double  aLimBsHProj,
                              double  aLimBsHRefut,
                              double  aPdsPl,
                              const   cNupletPtsHomologues & aNuple, 
			      std::vector<double> & aVPdsIm,
			      bool    AddEq,
			      const Pt3dr  * aPtApuis,
			      const Pt3dr  * anIncertApuis,
			      bool           aUseAppAsInit,
                              const cRapOnZ *      aRAZ
                           )
{
   CptUPL++;
   NewBug =   ::DebugPbCondFaisceau   &&
              (
                  // (CptUPL==707005) || (CptUPL==707004)  ||
                  // (CptUPL==707007) || (CptUPL==707006)  ||
                  (CptUPL==707009) || (CptUPL==707008)
              ) ;




   // BugNanFE = (CptUPL>=425770);
   if (BugNanFE)
   {
      std::cout << "CptUPL  " << CptUPL << "\n";
   }

   // if (DebugPbCondFaisceau) { mSubst.RazNonTmp(); }
  
   mResidus.mSomPondEr=0;
   mResidus.mOKRP3I = 1;
   mResidus.mBSurH  = cResiduP3Inc::TheDefBSurH;


   if (MPD_MM()) std::cout << "================== mTerIsInit " <<mTerIsInit << "\n";

   if (!mTerIsInit)
   {
      mPPP = cParamPtProj(aLimBsHProj,aLimBsHRefut,true);


    // if (DebugFaisceau && (aCpt%1000==999)) getchar();

/*
       if (aPtApuis && aUseAppAsInit)
       {
         mResidus.mPTer = *aPtApuis;
         mResidus.mOKRP3I = 1;
       }
*/
       

       {
          bool WithApp = aPtApuis && aUseAppAsInit;
          std::vector<Pt3dr> aVAppui;
          if (aPtApuis && aUseAppAsInit)
          {
               aVAppui.push_back(*aPtApuis);
               aVAppui.push_back(anIncertApuis ? * anIncertApuis : Pt3dr(1,1,1));
          }


          mResidus.mPTer = CalcPTerInterFaisceauCams
                        (
                              aRAZ,
//  GROS BUG ???? AVANT   WithApp && (aRAZ==0),
// GENERE EVIDEMMENT PB EN INTER RASANT COMPREND PAS POURQUOI CA MARCHE AVEC APPUIS ??????
                              (! WithApp) && (aRAZ==0),
                              mResidus.mOKRP3I,
                              aNuple,
                              aVPdsIm,
                              mPPP,
                              (WithApp ? &aVAppui : 0),
                              &mResidus.mMesPb
                         );
   if (MPD_MM()) std::cout << "================== mResidus.mPTer " <<mResidus.mPTer  << " " << mResidus.mBSurH << "\n";
          mResidus.mBSurH  = mPPP.mBsH;
          if (BugNanFE)
          {
             std::cout << mPPP.mTer << "\n";
          }

          if (!mResidus.mOKRP3I) 
          {
// std::cout << "mResidus.mOKRP3I  "  << __LINE__ <<  " " << WithApp << "\n";
               return mResidus;
          }
       }
    }

   static double aBsHMin = 1;
   if (aBsHMin> mPPP.mBsH)
   {
       // std::cout << mPPP.mBsH << "\n";
       aBsHMin = mPPP.mBsH;
  }
// Pour l'instant on bloque avant validation ....
  // mPPP.mProjIsInit = false;

    // if (mPPP.mBsH < 1/30) AddEq=false;
 
    Pt3dr aPTer =   mPPP.mProjIsInit ? Pt3dr(0,0,0) :  mResidus.mPTer;
    // mResidus.mPTer = aPTer;
   if (MPD_MM()) std::cout << "================== aPTer " << aPTer << "\n";
    mP3Inc->InitVal(aPTer);



    // if (AddEq)
    {
        ELISE_ASSERT(aVPdsIm.size()==mVCamVis.size(),"UsePointLiaison");
	int aNbNN =0;
        for (int aK=0 ; aK<int(mVCamVis.size()) ; aK++)
        {
            if (euclid(mResidus.mPTer-mVCamVis[aK]->NC_CameraCourante()->PseudoOpticalCenter())<1e-5)
              aVPdsIm[aK] = 0;
	    if (aVPdsIm[aK]>0) 
            {
	       aNbNN++;
            }
if (MPD_MM())  std::cout  << "==================== Pds " << aVPdsIm[aK] << "\n";
        }
	// if ((aNbNN<2)  && (!mEqSurf) && (! aPtApuis))
        // 	AddEq=0;

        if (    (aNbNN==0)
             || ((aNbNN==1) && (!mEqSurf) && (! aPtApuis))
           )
        {
            mResidus.mOKRP3I = false;
            return mResidus;
        }
    }

    // const std::vector<Pt2dr> &  aResidu = AddEquationProjCam(aNuple,aPds);
    mResidus.mEcIm.clear();
    for (int aK=0 ; aK<int(mVCamVis.size()) ; aK++)
    {
//XXXX
       double aPds= (AddEq?aVPdsIm[aK]:0) * mMulGlobPds;
       if (aVPdsIm[aK]>0)
       {
           Pt2dr anEr = mVCamVis[aK]->AddEqAppuisInc(aNuple.PK(aK),aPds,mPPP,aNuple.IsDr(aK));
           mResidus.mEcIm.push_back(anEr);
if (MPD_MM())  std::cout << "=x=x=x=x=x=x=x=x=x=x=x=x=x " << aNuple.PK(aK) << " " << mMulGlobPds << "\n";

           mResidus.mSomPondEr += aVPdsIm[aK] * mMulGlobPds * square_euclid(anEr);
        }
        else
        {
           // Modif 15/02/2012  avant les residus etait pas superposables aux poses
           mResidus.mEcIm.push_back(Pt2dr(1e5,1e5));
        }
    }

    if (mEqSurf)
    {
        mResidus.mEcSurf = mEqSurf->DoResiduPInc(AddEq?aPdsPl:0);
        mResidus.mSomPondEr += aPdsPl * ElSquare(mResidus.mEcSurf);
//if (aPtApuis) std::cout << "UPLG-CCCCCCC " << mResidus.mSomPondEr << "\n";
    }


    if (aPtApuis  ||  aRAZ ) // && AddEq)
    {
        ELISE_ASSERT(!(aPtApuis&&aRAZ),"Pt apuui et rappel Z simulatanes !! ");
        Pt3dr aPRapel = aPTer;
        if (aPtApuis)
            aPRapel = *aPtApuis;
        if (aRAZ)
           aPRapel = aRAZ->PZ();

        cElemEqFormelle::tContFcteur  aFR = mP3Inc->FctrRap(aPRapel);
        double aVInc[3];
        if (aRAZ)
        {
            aVInc[0] = aVInc[1] = 0;
            aVInc[2] = aRAZ->IncComp();
        }
        else if (aPtApuis)
        {
          anIncertApuis->to_tab(aVInc);
        }
	for (int aK=0 ; aK< 3 ; aK++)
	{
	    if (aVInc[aK] > 0)
	    {
	        double aPds = (1/ElSquare(aVInc[aK])) * mMulGlobPds;
		const std::vector<REAL> &  aV =   AddEq                                     ?
                                                  mSet.VAddEqFonctToSys(aFR[aK],aPds,false) :
                                                  mSet.VResiduSigne(aFR[aK])                ;
/*
		const std::vector<REAL> &  aV =   mSet.VAddEqFonctToSys(aFR[aK],aPds,false) ;
*/
if (MPD_MM())  std::cout  << "y====y===y===yyyyyy " << aPds << " " << aK <<  " " << aFR[aK]  << " " << aV[0] << "\n";
                mResidus.mSomPondEr +=  aPds * ElSquare(aV[0]);
//if (aPtApuis) std::cout << "UPLG-DDDDDDDDDDDD " << mResidus.mSomPondEr << "\n";
	    }
	}
    }


    if (MPD_MM()) std::cout << "HHHHHHHHhhhhhhhhhh " << AddEq << "\n";
    if (AddEq)
    {
       mSubst.DoSubst();
    }

    return mResidus;
}

const cResiduP3Inc& cManipPt3TerInc::UsePointLiaison
                           (
                              double  aLimBsHProj,
                              double  aLimBsH,
                              double  aPdsPl,
                              const   cNupletPtsHomologues & aNuple, 
			      std::vector<double> & aVPdsIm,
			      bool    AddEq,
                              const cRapOnZ *      aRAZ
                           )
{
   return UsePointLiaisonGen(aLimBsHProj,aLimBsH,aPdsPl,aNuple,aVPdsIm,AddEq,0,0,false,aRAZ);
}


const cResiduP3Inc & cManipPt3TerInc::UsePointLiaisonWithConstr
                     (
                          double aLimBsHProj,
                          double  aLimBsH,
                          double aPdsPl,
                          const cNupletPtsHomologues & aNuple,
                          std::vector<double> & aVPdsIm,
                          bool   AddEq , // Si false calcule les residu met ne modifie pas le syst
                          const Pt3dr  & aPtApuis,
                          const Pt3dr  & anIncertApuis,
                          bool           aUseAppAsInit
                      )
{
   return UsePointLiaisonGen
          (
               aLimBsHProj,
               aLimBsH,
               aPdsPl,aNuple,aVPdsIm,AddEq,
               &aPtApuis,&anIncertApuis,aUseAppAsInit,
               0
          );
}


/*
const std::vector<Pt2dr> & cManipPt3TerInc::ResiduPointLiaison
                           (
                              const cNupletPtsHomologues & aNuple,
			      Pt3dr * aResPTer 
                           )
{
    Pt3dr aPTer = CalcPTerInterFaisceauCams(aNuple);
    if (aResPTer)
       *aResPTer = aPTer;
    SetValueInitialPTer(aPTer);
    const std::vector<Pt2dr> &  aResidu = AddEquationProjCam(aNuple,0);

    return aResidu;
}
*/


/************************************************************/
/*                                                          */
/*                  cSetEqFormelles                         */
/*                                                          */
/************************************************************/

cEqfP3dIncTmp * cSetEqFormelles::Pt3dIncTmp()
{
   if (mPt3dIncTmp==0)
   {
      mPt3dIncTmp = new cEqfP3dIncTmp(*this,"Tmp_PTer");
      mPt3dIncTmp->CloseEBI();
      AddObj2Kill(mPt3dIncTmp);

      mLBlocTmp.push_back(mPt3dIncTmp);
   }

   return mPt3dIncTmp;
}


/*
cBufSubstIncTmp * cSetEqFormelles::GetBuff(int aNbTmp,int aNbNonTmp)
{
   cBufSubstIncTmp *  &aBuf =  mDicBufSubst[Pt2di(aNbTmp,aNbNonTmp)];
   if (aBuf==0)
       aBuf = new cBufSubstIncTmp(*this,aNbTmp,aNbNonTmp);

   return aBuf;
}
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
