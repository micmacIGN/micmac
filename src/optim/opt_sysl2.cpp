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

cVectMatMul::~cVectMatMul()
{
}

cVectPreCond::~cVectPreCond()
{
}

cParamCalcVarUnkEl * NullPCVU = 0;

/****************************************************/
/*                                                  */
/*            cGenSysSurResol                       */
/*                                                  */
/****************************************************/

cGenSysSurResol::~cGenSysSurResol()
{
}

bool  cGenSysSurResol::ResiduIsComputedAfterSolve() {return false;}
tSysCho   cGenSysSurResol:: ResiduAfterSol() const
{
    ELISE_ASSERT(false,"cGenSysSurResol::ResiduAfterSol");
    return 0;
}

bool  cGenSysSurResol::InverseIsComputedAfterSolve() {return false;}
tSysCho   cGenSysSurResol::GetElemInverseQuad(int i,int j) const 
{
    ELISE_ASSERT(false,"cGenSysSurResol::GetElemInverseQuad");
    return 0;
}

bool  cGenSysSurResol::CanCalculVariance() const {return false;}
bool  cGenSysSurResol::IsCalculingVariance() const {return false;}
void  cGenSysSurResol::SetCalculVariance(bool)
{
    ELISE_ASSERT(false,"cGenSysSurResol::SetCalculVariance");
}
double  cGenSysSurResol::Redundancy() const
{
    ELISE_ASSERT(false,"cGenSysSurResol::Redundancy");
    return 0.0;
}

void cGenSysSurResol::Show() const
{
}

double  cGenSysSurResol::R2Pond() const
{
    ELISE_ASSERT(false,"cGenSysSurResol::R2Pond");
    return 0.0;
}


double  cGenSysSurResol::Variance(int aK)
{
    ELISE_ASSERT(false,"cGenSysSurResol::Variance");
    return 0.0;
}
double * cGenSysSurResol::CoVariance(int aK1,int aK2)
{
    ELISE_ASSERT(false,"cGenSysSurResol::CoVariance");
    return 0;
}
bool    cGenSysSurResol::IsTmp(int aK) const
{
    ELISE_ASSERT(false,"cGenSysSurResol::IsTmp");
   return false;
}

void cGenSysSurResol::SetTmp(const std::vector<cSsBloc> &  aVx,const std::vector<cSsBloc> &  aVy,bool ForSet)
{
    ELISE_ASSERT(false,"cGenSysSurResol::SetTmp");
}
int    cGenSysSurResol::NumTmp(int aK) const
{
    ELISE_ASSERT(false,"cGenSysSurResol::NumTmp");
   return -1;
}
int    cGenSysSurResol::NumNonTmp(int aK) const
{
    ELISE_ASSERT(false,"cGenSysSurResol::NumNonTmp");
   return -1;
}
int    cGenSysSurResol::InvNumNonTmp(int aK) const
{
    ELISE_ASSERT(false,"cGenSysSurResol::InvNumNonTmp");
   return -1;
}


cGenSysSurResol::cGenSysSurResol
(
     bool CstrAssumed,
     bool OptSym,
     bool GereNonSym,
     bool GereBloc
) :
        mCstrAssumed  (CstrAssumed),
        mOptSym       (OptSym),
        mGereNonSym   (GereNonSym),
        mGereBloc     (GereBloc),
    // mSetEqEmpty(true),
    mPhaseContrainte(true),
    mNbContrainte(0),
    mC(1,1),
    mE(1,1),
    mtL(1,1),
    mtLC(1,1),
    mtLCE(1,1),
    mSol(1,1),
    mCSol(1,1),
        mGP  (1,1),
        mNewCstrIsInit (false),
        mNewCstrIsTraitee (false),
        mUseSpeciCstrUniVar (true),
        // mUseSpeciCstrUniVar (false),
        mValCstr        (1,0.0),
        mDValCstr       (0),
        mIsCstr         (1,0),
        mDIsCstr       (0)
{
}




void cGenSysSurResol::GSSR_Reset(bool WithCstr)
{
// std::cout    << "RESET cGenSysSurResol::GSSR_Reset\n";

  mIsCstr.raz();

  mPhaseContrainte = WithCstr;
  mFirstEquation   = true;
  // mSetEqEmpty = true;
  mNbContrainte = 0;
  mNewCstrIsTraitee = false;
  V_GSSR_Reset();
}


bool cGenSysSurResol::IsCstrUniv(int anX,double & aVal)
{
   if (mNewCstrIsInit && mDIsCstr[anX])
   {
       aVal = mDValCstr[anX];
       return true;
   }
   return false;
}


void  cGenSysSurResol::AssertPhaseContrainte()
{
    ELISE_ASSERT
    (
        mPhaseContrainte,
    "cGenSysSurResol::AssertPhaseContrainte "
    );
}

void  cGenSysSurResol::AssertPhaseEquation()
{
    ELISE_ASSERT
    (
        ! mPhaseContrainte,
    "cGenSysSurResol::AssertPhaseEquation"
    );
}

void cGenSysSurResol::SetPhaseEquation(const std::vector<int> * aVA2S)
{
    AssertPhaseContrainte();
    mPhaseContrainte = false;
    TraitementContrainteUniVar(aVA2S);
}



void cGenSysSurResol::GSSR_AddContrainteIndexee
     (
         const std::vector<int> & aVI,
         REAL * aC,
         REAL aE
      )
{
    AssertPhaseContrainte();

 // Gestion specifique des contraintes univariees
   if ((! mNewCstrIsInit) && mUseSpeciCstrUniVar)
   {
       mIsCstr = Im1D_U_INT1(NbVar(),0);
       mDIsCstr = mIsCstr.data();
       mValCstr = Im1D_REAL8(NbVar(),0.0);
       mDValCstr = mValCstr.data();
       mNewCstrIsInit = true;
   }
   else
   {
       ELISE_ASSERT(mValCstr.tx()==NbVar(),"Sz - Incoherence in GSSR_AddContrainte");
   }
   int aNbVarNN = 0;
   int aKNN = -1;
   for (int y=0 ; y<int(aVI.size()) ; y++)
   {
       if (aC[y] !=0)
       {
            aNbVarNN++;
            aKNN = y;
       }
   }


   if ((aNbVarNN==1) && mUseSpeciCstrUniVar)
   {
        mDIsCstr[aVI[aKNN]] = 1;
        mValCstr.data()[aVI[aKNN]] = aE/aC[aKNN];
       // std::cout << "xrt-CSTR[" << aVI[aKNN] <<"] = "  << aE/aC[aKNN]   << " C=" << aC[aKNN] << "\n";
        return;
   }


 // Gestion des contrainte l'ancienne si pas univariee

    mNbContrainte++;
    // Premiere contrainte
    if (mNbContrainte == 1)
    {
       ELISE_ASSERT(AcceptContrainteNonUniV(),"Ce systeme n'accepte que les contraintes uni var");
       mC.set_to_size(NbVar(),NbVar());
       mE.set_to_size(1,NbVar());
       // a priori inutile, mais pour initialiser toujours
       for (int y=0 ; y<NbVar() ; y++)
           mE(0,y) = 0;
       mtL.set_to_size(NbVar(),1);
       mtLC.set_to_size(NbVar(),1);
       mSol.set_to_size(1,NbVar());
       mCSol.set_to_size(1,NbVar());
    }
    mLineCC = NbVar() -mNbContrainte;
    ELISE_ASSERT(mLineCC>=0,"Too much contrainte in cGenSysSurResol");

    for (INT aV=0; aV<NbVar(); aV++)
    {
        mC(aV,mLineCC) = 0;
    }
    for (int y=0 ; y<int(aVI.size()) ; y++)
    {
        mC(aVI[y],mLineCC) = aC[y];
    }
    mE(0,mLineCC) = aE;
}


bool  cGenSysSurResol::AcceptContrainteNonUniV() const
{
   return true;
}


void cGenSysSurResol::TraitementContrainteUniVar(const std::vector<int> * aVA2S)
{
    if (! mUseSpeciCstrUniVar)
       return;
    if (! mNewCstrIsInit)
       return;
    if (mNewCstrIsTraitee)
       return;
    mNewCstrIsTraitee = true;


    INT NBV = NbVar();

    // La premiere fois on rajoute les equations qui feront que  le systeme ne sera pas
    // degenere
    for (int aKV=0 ; aKV< NBV; aKV++)
    {
        if (mDIsCstr[aKV])
        {
              // std::cout << " YYYYYYYYYYYY : " << aKV << "\n";
              std::vector<INT> aVInd;
              int aKI = aKV;
/*
              if (aVA2S && (!PermutIndex))
                 aKI = (*aVA2S)[aKV];
*/

// std::cout << "KVVVV " << aKV << " KI " << aKI << "\n";
              aVInd.push_back(aKI);
              double aCoef1 = 1.0;
              Basic_GSSR_AddNewEquation_Indexe(0,0,0,aVInd,1.0,&aCoef1,mValCstr.data()[aKV],NullPCVU);
// std::cout << "PB QUand on passe par indexee \n";
        }
    }
// std::cout << "cGenSysSurResol::TraitementContrainteUniVar \n" ;  getchar();

}

void cGenSysSurResol::GSSR_AddNewEquation
     (REAL aPds,REAL * aL,REAL aB,double * aCoordCur)
{
    AssertPhaseEquation();
    // ELISE_ASSERT
    INT NBV = NbVar();


    if (mUseSpeciCstrUniVar && mNewCstrIsInit)
    {
       for (INT x=0 ;  x<NBV ; x++)
       {
           if (mDIsCstr[x])
           {
// BUG aB += aL[x] * mDValCstr[x];  quasi indetectable car  mDValCstr vaut quasi
// toujours 0, car fait en delta; La magouille faite XYklTY de phgr_formel.cpp
//  montre que c'est bien un signe - qu'il faut !
               aB -= aL[x] * mDValCstr[x];
               if (aCoordCur)
               {
                   aB -= aL[x] * aCoordCur[x];
               }
               aL[x] = 0;
           }
       }
    }


    if ((mNbContrainte==0) || mCstrAssumed)
    {
       // mSetEqEmpty = false;
       V_GSSR_AddNewEquation(aPds,aL,aB);
       return;
    }

    if (mFirstEquation)
    {
        ComplBaseParLeHaut(mC,mNbContrainte);
    mFirstEquation = false;

        mGP.SelfSetMatrixInverse(mC,3);


    // On rajoute des contrainte nulle sur les
    // N derniere variable
       for (INT y= mLineCC ; y<NBV ; y++)
           {
               for (INT x=0 ;  x<NBV ; x++)
           {
                    mtL(x,0) = (x==y);
           }
               V_GSSR_AddNewEquation(1.0,mtL.data()[0],0.0);
           }
    }

    for (INT x=0 ;  x<NBV ; x++)
    {
        mtL(x,0) = aL[x];
    }
    mtLC.mul(mtL,mC);
    mtLCE.mul(mtLC,mE);
    for (INT x=mLineCC ;  x<NBV ; x++)
       mtLC(x,0)=0;

    V_GSSR_AddNewEquation(aPds, mtLC.data()[0],aB-mtLCE(0,0));
}


Im1D_REAL8  cGenSysSurResol::GSSR_Solve(bool * aResOk)
{

    Im1D_REAL8 aSol = V_GSSR_Solve(aResOk);



    if ((mNbContrainte==0) || mCstrAssumed)
       return aSol;


    for (INT y=0 ;  y<NbVar() ; y++)
        mSol(0,y) = aSol.data()[y];
    mSol += mE;
    mCSol.mul(mC,mSol);

    for (INT y=0 ;  y<NbVar() ; y++)
    {
        aSol.data()[y] = mCSol(0,y);
    }

    return aSol;
}
void cGenSysSurResol::Basic_GSSR_AddNewEquation_Indexe
     (
         const std::vector<cSsBloc> * aVSB,
         double * aFullC,int aNbTot,
         const std::vector<INT> & aVInd ,
         REAL aPds,REAL * aCoeff,REAL aB,
         cParamCalcVarUnkEl *    aPCVU
     )
{
/*
     static int aCpt=0 ;
     if (BugUPL) aCpt++;
*/

     ELISE_ASSERT
     (
         (mNbContrainte==0) || mCstrAssumed,
         "Old contrainte in AddNewEquation_Indexe"
     );

     AssertPhaseEquation();
    // mSetEqEmpty = false;
     V_GSSR_AddNewEquation_Indexe(aVSB,aFullC,aNbTot,aVInd,aPds,aCoeff,aB,aPCVU);

}
void cGenSysSurResol::GSSR_AddNewEquation_Indexe
     (
         const std::vector<cSsBloc> * aVSB,
         double * aFullC,int aNbTot,
         const std::vector<INT> & aVInd ,
         REAL aPds,REAL * aCoeff,REAL aB,
         cParamCalcVarUnkEl * aPCVU
     )
{
     // Done in Basic_GSSR_AddNewEquation_Indexe  -> AssertPhaseEquation();
     Basic_GSSR_AddNewEquation_Indexe(aVSB,aFullC,aNbTot,aVInd,aPds,aCoeff,aB,aPCVU);
}





void cGenSysSurResol::AddOneBloc(const cSsBloc &,const cSsBloc &, REAL aPds,REAL * aCoeff)
{
    ELISE_ASSERT(false,"No cGenSysSurResol::AddOneBloc");
}
void cGenSysSurResol::AddOneBlocDiag(const cSsBloc &, REAL aPds,REAL * aCoeff)
{
    ELISE_ASSERT(false,"No cGenSysSurResol::AddOneBlocDiag");
}
void cGenSysSurResol::AddOneBlocCste(const cSsBloc &, REAL aPds,REAL * aCoeff,REAL aB)
{
    ELISE_ASSERT(false,"No cGenSysSurResol::AddOneBlocCste");
}



void cGenSysSurResol::V_GSSR_AddNewEquation_Indexe
     (
         const std::vector<cSsBloc> * aVSB,
         double *,int aNb,
         const std::vector<INT> & aVInd ,
         REAL aPds,REAL * aCoeff,REAL aB,
         cParamCalcVarUnkEl *
     )
{
    ELISE_ASSERT(false,"No V_GSSR_AddNewEquation_Indexe");
}


bool cGenSysSurResol::GSSR_UseEqMatIndexee()
{
    return false;
}


void cGenSysSurResol::V_GSSR_EqMatIndexee
     (
         const std::vector<INT> & aVInd,
         REAL aPds,REAL ** aMat,
         REAL * aVect,REAL aCste
     )
{
   ELISE_ASSERT(false,"No cGenSysSurResol::GSSR_EqMatIndexee");
}

void cGenSysSurResol::GSSR_EqMatIndexee
     (
         const std::vector<INT> & aVInd,
         REAL aPds,REAL ** aMat,
         REAL * aVect,REAL aCste
     )
{
     V_GSSR_EqMatIndexee(aVInd,aPds,aMat,aVect,aCste);
}







void cGenSysSurResol::GSSR_Add_EqFitDroite(REAL aXi,REAL aYi,REAL aPds)
{
   REAL aCoeff[2];
   aCoeff[0] = aXi;
   aCoeff[1] = 1;
   GSSR_AddNewEquation(aPds,aCoeff,aYi,0);
}

void cGenSysSurResol::GSSR_Add_EqFitPlan(REAL aXi,REAL aYi,REAL aZi,REAL aPds)
{
   REAL aCoeff[3];
   aCoeff[0] = aXi;
   aCoeff[1] = aYi;
   aCoeff[2] = 1;
   GSSR_AddNewEquation(aPds,aCoeff,aZi,0);
}


void cGenSysSurResol::GSSR_SolveEqFitDroite(REAL & aAx,REAL &aB,bool * aOk)
{
     Im1D_REAL8  aSol = GSSR_Solve(aOk);
     if (aOk && (! *aOk))
        return;
     ELISE_ASSERT(aSol.tx()==2,"cGenSysSurResol::GSSR_SolveEqFitDroite");
     aAx = aSol.data()[0];
     aB = aSol.data()[1];
}

void cGenSysSurResol::GSSR_SolveEqFitPlan(REAL & aAx,REAL &aBy,REAL & aC,bool * aOk)
{
     Im1D_REAL8  aSol = GSSR_Solve(aOk);
     if (aOk && (! *aOk))
        return;
     ELISE_ASSERT(aSol.tx()==3,"cGenSysSurResol::GSSR_SolveEqFiPlan");
     aAx = aSol.data()[0];
     aBy = aSol.data()[1];
     aC  = aSol.data()[2];
}

void L2SysSurResol::GSSR_Add_EqInterPlan3D(const Pt3dr& aDirOrtho,const Pt3dr& aP0,double aPds)
{
   static std::vector<int> aVInd;
   if (aVInd.empty())
   {
      aVInd.push_back(0);
      aVInd.push_back(1);
      aVInd.push_back(2);
   }
   double aCoeff[3];
   // Pt3dr aD = vunit(aDirOrtho);
   // std::cout << euclid(aDirOrtho) << "\n";
   aCoeff[0] =  aDirOrtho.x;
   aCoeff[1] =  aDirOrtho.y;
   aCoeff[2] =  aDirOrtho.z;

   //GSSR_AddNewEquation(aPds,aCoeff,scal(aDirOrtho,aP0),0);
    L2SysSurResol::V_GSSR_AddNewEquation_Indexe(0,0,0,aVInd,aPds,&(aCoeff[0]),scal(aDirOrtho,aP0),NullPCVU);
}

void L2SysSurResol::GSSR_AddEquationFitOneVar(int aNumVar,double aVal,double aPds)
{
    if (aPds<=0) return;
    std::vector<int> aVInd;
    aVInd.push_back(aNumVar);

    double aCoef1 = 1.0;

    L2SysSurResol::V_GSSR_AddNewEquation_Indexe(0,0,0,aVInd,aPds,&aCoef1,aVal,NullPCVU);
}

void   L2SysSurResol::GSSR_AddEquationPoint3D(const Pt3dr & aP,const Pt3dr &  anInc)
{
    if (anInc.x > 0)
       GSSR_AddEquationFitOneVar(0,aP.x,ElSquare(1/anInc.x));
    if (anInc.y > 0)
       GSSR_AddEquationFitOneVar(1,aP.y,ElSquare(1/anInc.y));
    if (anInc.z > 0)
       GSSR_AddEquationFitOneVar(2,aP.z,ElSquare(1/anInc.z));
}



bool  L2SysSurResol::InverseIsComputedAfterSolve() {return true;}
tSysCho   L2SysSurResol::GetElemInverseQuad(int i,int j) const 
{
    return mDataInvtLi_Li[j][i];
}





Im2D_REAL8   L2SysSurResol::tLi_Li()
{
   return mtLi_Li;
}



void L2SysSurResol::GSSR_Add_EqInterDroite3D(const Pt3dr& aDirDroite,const Pt3dr& aP0,double aPds)
{
   Pt3dr aU = OneDirOrtho(aDirDroite);
   GSSR_Add_EqInterPlan3D(aU,aP0,aPds);
   GSSR_Add_EqInterPlan3D(aDirDroite^aU,aP0,aPds);
}

Pt3dr cGenSysSurResol::Pt3dSolInter(bool * aOk)
{
     Im1D_REAL8  aSol = GSSR_Solve(aOk);
     if (aOk && (! *aOk))
        return Pt3dr(1e33,-1e44,0);
     ELISE_ASSERT(aSol.tx()==3,"cGenSysSurResol::SolInter");

     return Pt3dr(aSol.data()[0],aSol.data()[1],aSol.data()[2]);
}



void cResOptInterFaisceaux::Init(const ElMatrix<double> & aMat)
{
   ElMatrix<double> aVecP(3,3),aValP(1,3);
   std::vector<int> aInd = jacobi(aMat,aValP,aVecP);

    mVal1 = aValP(aInd[0],0);
    mVal2 = aValP(aInd[1],0);
    mVal3 = aValP(aInd[2],0);

    aVecP.GetCol(aInd[0],mVec1);
    aVecP.GetCol(aInd[1],mVec2);
    aVecP.GetCol(aInd[2],mVec3);
}

/*
Pt3dr  tCho2double(const Pt3d<tSysCho> & aP)
{
   return Pt3dr(aP.x,aP.y,aP.z);
}
*/


Pt3dr ElSeg3D::L2InterFaisceaux
      (
           const std::vector<double> * aVPds,
           const std::vector<ElSeg3D> & aVS,
           bool * aOK,
           const cRapOnZ *      aRAZ ,
           cResOptInterFaisceaux * aROIF,
           const std::vector<Pt3dr> *  aVPts
      )
{
   if (! aVPts)
   {
       ELISE_ASSERT(int(aVS.size())>=2,"ElSeg3D::L2InterFaisceaux pas assez de pts");
   }
   static L2SysSurResol aSys(3);
   aSys.GSSR_Reset(false);
   int aNbEq=0;

   for (int aKS=0; aKS<int(aVS.size()); aKS++)
   {
      double aPds = ((aVPds!=0) ? (*aVPds)[aKS] : 1.0);
      if (aPds > 0)
      {
         aSys.GSSR_Add_EqInterDroite3D (aVS[aKS].TgNormee(),aVS[aKS].P0(),aPds);
         aNbEq+=2;
      }
   }

   if (aRAZ)
   {
       double aCoeff[3]={0,0,1};
       aSys.AddEquation(1/ElSquare(aRAZ->IncEstim()),aCoeff,aRAZ->Z());
       aNbEq+=1;
   }

   if (aVPts)
   {
       int aNb = (int)aVPts->size();
       ELISE_ASSERT((aNb%2)==0,"ElSeg3D::L2InterFaisceaux aVPts impair !!!");
       for (int aK=0 ; aK<aNb ; aK+=2)
       {
            aSys.GSSR_AddEquationPoint3D((*aVPts)[aK],(*aVPts)[aK+1]);
            aNbEq +=3;
       }
   }
   ELISE_ASSERT(aNbEq>=3,"Not Enouh Equation in ElSeg3D::L2InterFaisceaux");

   if (aROIF)
   {
      aROIF->Init(aSys.tLi_Li().ToMatrix());
   }


   return  aSys.Solve3x3Sym(aOK);

/*
   Pt3dr aP0 = aSys.Solve3x3Sym(aOK);
   Pt3dr aP1 =  aSys.Pt3dSolInter(aOK);
   ELISE_ASSERT(euclid(aP0-aP1)<1e-5,"aSys.Solve3x3Sym/aSys.Pt3dSolInter");
   return aSys.Pt3dSolInter(aOK);
*/
}

/****************************************************/
/*                                                   */
/*            L2SysSurResol                         */
/*                                                  */
/****************************************************/


     //  == Interface to cGenSysSurResol

Im1D_REAL8  L2SysSurResol::V_GSSR_Solve(bool * aResOk)
{

   return Solve(aResOk);
}

INT  L2SysSurResol::NbVar() const
{
    return mNbVar;
}

void L2SysSurResol::V_GSSR_Reset()
{
    Reset();
}

void L2SysSurResol::V_GSSR_AddNewEquation(REAL aPds,REAL * aCoeff,REAL aB)
{
    AddEquation(aPds,aCoeff,aB);
}



     // ================================



bool L2SYM = false;
// DebugPbCondFaisceau

L2SysSurResol::L2SysSurResol(INT aNbVar,bool IsSym) :
    cGenSysSurResol
    (
         !DebugPbCondFaisceau, // true,
         // L2SYM, !L2SYM,
   // false,false,
         IsSym, (! IsSym),     // MPD 05-04_2015 ; avant L2SYM, sous optimale pour subsitution, voit pas pourquoi cela serait bien ???
        true
    ),
    mNbVar             (aNbVar),
    mtLi_Li            (mNbVar,mNbVar,0.0),
    mDatatLi_Li        (mtLi_Li.data()),
    mInvtLi_Li         (mNbVar,mNbVar,0.0),
    mDataInvtLi_Li     (mInvtLi_Li.data()),
    mbi_Li             (aNbVar,0.0),
    mDatabi_Li         (mbi_Li.data()),
    mBibi              (0.0),
    mSolL2             (aNbVar),
    mDataSolL2         (mSolL2.data()),
    mNbEq              (0),
    mNbIncReel         (mNbVar),
    mRedundancy        (1.0),
    mMaxBibi           (0),
    mNumTmp            (aNbVar,-1),
    // mDIsTmp
    mDNumTmp           (mNumTmp.data()),
    mNumNonTmp         (aNbVar,-1),
    mDNumNonTmp        (mNumNonTmp.data()),
    mInvNumNonTmp      (aNbVar,-1),
    mInvDNumNonTmp     (mInvNumNonTmp.data()),
    mDoCalculVariance  (false),
    mCoVariance        (aNbVar,aNbVar,0.0),
    mDCoVar            (mCoVariance.data()),
    mSomPds            (0.0),
    mSomR2Pds          (0.0)
{
    // std::cout << "L2SysSurResol::L2SysSurResol " << IsSym << "\n";
}


void L2SysSurResol::SetNum(INT4 * aDataInv,INT4 * aData,const std::vector<cSsBloc> &  aVx,bool ForSet)
{
    int aNum=0;
    for (int aKx=0 ; aKx <int(aVx.size()) ; aKx++)
    {
        const cSsBloc & aBlX = aVx[aKx];
        int aI0x = aBlX.I0AbsSolve();
        int aI1x = aBlX.I1AbsSolve();
        for (int aX=aI0x ; aX<aI1x ; aX++)
        {
             aData[aX] = ForSet ? aNum : -1 ;
             if (aDataInv)
             {
                aDataInv[aNum]  = ForSet  ? aX : -1;
             }
             aNum++;
        }
    }
}

void L2SysSurResol::SetTmp(const std::vector<cSsBloc> &  aVx,const std::vector<cSsBloc> &  aVy,bool ForSet)
{
// std::cout << "XxxxL2SysSurResol::SetTm\n";
    SetNum(0,mDNumTmp,aVx,ForSet);
    SetNum(mInvDNumNonTmp,mDNumNonTmp,aVy,ForSet);
}

bool    L2SysSurResol::IsTmp(int aK) const
{
   return mDNumTmp[aK]!=-1;
}
int    L2SysSurResol::NumTmp(int aK) const
{
   return mDNumTmp[aK];
}
int    L2SysSurResol::NumNonTmp(int aK) const
{
   return mDNumNonTmp[aK];
}
int    L2SysSurResol::InvNumNonTmp(int aK) const
{
   return mInvDNumNonTmp[aK];
}


void L2SysSurResol::SetCalculVariance(bool aDo)
{
   mDoCalculVariance = aDo;
}
bool L2SysSurResol::CanCalculVariance() const
{
  return true;
}
double  L2SysSurResol::Variance(int aK)
{
   return mDCoVar[aK][aK];
}
double * L2SysSurResol::CoVariance(int aK1,int aK2)
{
   return &mDCoVar[aK1][aK2];
}

bool L2SysSurResol::IsCalculingVariance() const
{
   return mDoCalculVariance;
}

double   L2SysSurResol::Redundancy() const
{
    return mRedundancy;
}

void L2SysSurResol::Show() const
{
   std::cout << "NbEq =" << mNbEq << "\n";
   std::cout  << "R2P=" << mSomR2Pds << "SP=" << mSomPds << "\n";
   std::cout << " RAS=" << ResiduAfterSol() << "\n";
}

double   L2SysSurResol::R2Pond() const
{
    // return mSomR2Pds / mSomPds;
    return (mSomR2Pds / mNbEq) / mRedundancy ;
}

void L2SysSurResol::SetSize(INT aNbVar)
{
    mNbVar = aNbVar;
    mtLi_Li =  mtLi_Li.AugmentSizeTo(Pt2di(mNbVar,mNbVar),0.0);
    mDatatLi_Li = mtLi_Li.data();


    mInvtLi_Li =  mInvtLi_Li.AugmentSizeTo(Pt2di(mNbVar,mNbVar),0.0);
    mDataInvtLi_Li = mInvtLi_Li.data();

    mbi_Li =  mbi_Li.AugmentSizeTo(mNbVar,0.0);
    mDatabi_Li = mbi_Li.data();
    mSolL2 =mSolL2.AugmentSizeTo(mNbVar,0.0);;
    mDataSolL2 = mSolL2.data();

    mNumTmp = mNumTmp.AugmentSizeTo(mNbVar,-1);
    mDNumTmp = mNumTmp.data();
    mNumNonTmp = mNumNonTmp.AugmentSizeTo(mNbVar,-1);
    mDNumNonTmp = mNumNonTmp.data();
    mInvNumNonTmp = mInvNumNonTmp.AugmentSizeTo(mNbVar,-1);
    mInvDNumNonTmp = mInvNumNonTmp.data();

    mCoVariance = mCoVariance.AugmentSizeTo(Pt2di(aNbVar,aNbVar),0.0); 
    mDCoVar     = mCoVariance.data();

}

void L2SysSurResol::Reset()
{
   mtLi_Li.raz();
   mbi_Li.raz();
   mCoVariance.raz();
   ELISE_COPY(mNumTmp.all_pts(),-1,mNumTmp.out());
   ELISE_COPY(mNumNonTmp.all_pts(),-1,mNumNonTmp.out());
   ELISE_COPY(mInvNumNonTmp.all_pts(),-1,mInvNumNonTmp.out());
   mBibi = 0.0;
   mRedundancy = 1.0 - mNbIncReel / double(mNbEq);
   mNbEq = 0;
   mNbIncReel = mNbVar;

   mMaxBibi = 0;
   mSomPds = 0;
   mSomR2Pds = 0;
}

void L2SysSurResol::AddEquation(REAL aPds,REAL * aCoeff,REAL aB)
{
     static std::vector<int> VInd;
     static std::vector<REAL> VALS;
     VInd.clear();
     VALS.clear();

     for (INT iVar1=0 ; iVar1<mNbVar ; iVar1++)
     {
         if (aCoeff[iVar1] != 0.0)  // Acceleration pour les formes creuses
         {
             VInd.push_back(iVar1);
             VALS.push_back(aCoeff[iVar1]);
         }
     }
     L2SysSurResol::V_GSSR_AddNewEquation_Indexe(0,0,0,VInd,aPds,&VALS[0],aB,NullPCVU);
}


void L2SysSurResol::Indexee_EcrireDansMatrColWithLin
              (ElMatrix<double> &aMatr,const std::vector<cSsBloc> &  aVy) const
{

   // cGenSysSurResol::Indexee_EcrireDansMatrColWithLin(aMatr,aVy);

    double ** aDOut = aMatr.data();
    int yout = 0;
    for (int aKy=0 ; aKy <int(aVy.size()) ; aKy++)
    {
        int aI0y = aVy[aKy].I0AbsSolve();
        int aI1y = aVy[aKy].I1AbsSolve();
        for (int yin = aI0y; yin < aI1y ; yin++)
        {
            aDOut[yout][0] = mDatabi_Li[yin];
            yout ++;
        }
    }

}

double  L2SysSurResol::ResiduOfSol(const double * aData)
{
   double aRes = mBibi;
   for (int aKy=0 ; aKy< mNbVar ; aKy++)
   {
       aRes -= 2* mDatabi_Li[aKy] * aData[aKy];
       for (int aKx=0 ; aKx< mNbVar ; aKx++)
       {
            aRes += mDatatLi_Li[aKy][aKx] * aData[aKy] * aData[aKx];
       }
   }
   // ELISE_ASSERT(false,"cGenSysSurResol::ResiduOfSol");
   return aRes;
}


 

void L2SysSurResol::Indexee_EcrireDansMatrWithQuad
     (
            ElMatrix<double> &aMatr,
            const std::vector<cSsBloc> &  aVx,
            const std::vector<cSsBloc> &  aVy
     )  const
{
   // cGenSysSurResol::Indexee_EcrireDansMatrWithQuad(aMatr,aVx,aVy);

    double ** aDOut = aMatr.data();
    int xout=0;
    for (int aKx=0 ; aKx <int(aVx.size()) ; aKx++)
    {
        const cSsBloc & aBlX = aVx[aKx];
        int aI0x = aBlX.I0AbsSolve();
        int aNbX = aBlX.Nb();
        int yout=0;
        for (int aKy=0 ; aKy <int(aVy.size()) ; aKy++)
        {
            const cSsBloc & aBlY = aVy[aKy];
            int aI0y = aVy[aKy].I0AbsSolve();
            if ((aI0x>=aI0y) || (!mOptSym))
            {
                int aI1y = aBlY.I1AbsSolve();
                for (int aYIn=aI0y ; aYIn<aI1y ; aYIn++)
                {
                    memcpy
                    (
                        aDOut[yout]+xout,
                        mDatatLi_Li[aYIn]+aI0x,
                        aNbX*sizeof(double)
                    );
                    yout++;
                }
            }
            else
            {
                yout += aBlY.Nb();
            }
        }
        xout += aNbX;
    }

}

void L2SysSurResol::Indexee_LinSet0  (const std::vector<cSsBloc> & aVy)
{
   // cGenSysSurResol::Indexee_LinSet0(aVInd);
    for (int aKy=0 ; aKy <int(aVy.size()) ; aKy++)
    {
        int aI0y = aVy[aKy].I0AbsSolve();
        int aI1y = aVy[aKy].I1AbsSolve();
        for (int yin = aI0y; yin < aI1y ; yin++)
        {
             mDatabi_Li[yin] = 0;
        }
    }
/*
*/
}

void  L2SysSurResol::Indexee_QuadSet0
      (
             const std::vector<cSsBloc> & aVx,
             const std::vector<cSsBloc> & aVy
      )
{
 //  cGenSysSurResol::Indexee_QuadSet0(aVIndx,aVIndy);
    for (int aKx=0 ; aKx <int(aVx.size()) ; aKx++)
    {
        const cSsBloc & aBlX = aVx[aKx];
        int aI0x = aBlX.I0AbsSolve();
        int aNbX = aBlX.Nb();
        for (int aKy=0 ; aKy <int(aVy.size()) ; aKy++)
        {
            const cSsBloc & aBlY = aVy[aKy];
            int aI0y = aVy[aKy].I0AbsSolve();
            if ((aI0x>=aI0y) || (!mOptSym))
            {
                int aI1y = aBlY.I1AbsSolve();
                for (int aYIn=aI0y ; aYIn<aI1y ; aYIn++)
                {
                    MEM_RAZ(mDatatLi_Li[aYIn]+aI0x,aNbX);
                }
            }
        }
    }
}


void  L2SysSurResol::SoutraitProduc3x3
     (
                          bool                   Sym,
                          ElMatrix<double> &aM1,
                          ElMatrix<double> &aM2,
                          const std::vector<cSsBloc> * aYVSB

     )
{

   mNbIncReel += 3;
   double ** aData1 = aM1.data();
   double ** aData2 = aM2.data();

   double * aL2A = aData2[0];
   double * aL2B = aData2[1];
   double * aL2C = aData2[2];
   int aNbBl = (int)aYVSB->size();
   int Y0InBloc = 0;
   for (int aKBly=0 ; aKBly <aNbBl ; aKBly++)
   {
        const cSsBloc & aBlY = (*aYVSB)[aKBly];
        const cIncIntervale * aIntY  = aBlY.Intervale();
        int aNumBlocIntervY    = aIntY->NumBlocSolve();

        int aI0y = aBlY.I0AbsSolve();
        int aI1y = aBlY.I1AbsSolve();

        int X0InBloc =0;
        for (int aKBlx=0 ; aKBlx <aNbBl ; aKBlx++)
        {
            const cSsBloc & aBlX = (*aYVSB)[aKBlx];
            const cIncIntervale * aIntX  = aBlX.Intervale();
            int aNumBlocIntervX    = aIntX->NumBlocSolve();
            if ((!mOptSym) || (aNumBlocIntervX>=aNumBlocIntervY))
            {
               int aI0x = aBlX.I0AbsSolve();
               int aI1x = aBlX.I1AbsSolve();
               int yin = Y0InBloc;

               for (int aYOut = aI0y; aYOut<aI1y ; aYOut++,yin++)
               {
                   double * aLineOut = mDatatLi_Li[aYOut];
                   double A1 = aData1[yin][0];
                   double B1 = aData1[yin][1];
                   double C1 = aData1[yin][2];

                   int aDebX  = (mOptSym && (aNumBlocIntervX==aNumBlocIntervY)) ? aYOut : aI0x;

                   int xin = X0InBloc + aDebX-aI0x;
                   for (int XOut = aDebX ; XOut<aI1x ; XOut++)
                   {
                        aLineOut[XOut] -= A1 * aL2A[xin] + B1 * aL2B[xin] + C1 *aL2C[xin];
                       xin++;
                   }
               }

            }
            X0InBloc += aBlX.Nb();
        }

        Y0InBloc += aBlY.Nb();
   }
}



// Classe permettant de faire communiquer V_GSSR_AddNewEquation_Indexe et SoutraitProduc3x3 (ou DoSubst ?)
// pour le calcul de variance covariance en cas d'elimination d'inconnues (complement de Schurr)
// En effet lorsque l'on effectue la substitution  on a, pour l'instant, perdu trace du detail des
// obsevation qui ont amene au calcul




void L2SysSurResol::V_GSSR_AddNewEquation_Indexe
     (
           const std::vector<cSsBloc> * aVSB,
           double * aFullCoef, int aNbTot,
           const std::vector<INT> & aVInd ,
           REAL aPds,
           REAL * aCoeff,
           REAL aB,
           cParamCalcVarUnkEl *aCalcUKn
     )
{

// ==================================

/*
tPCVU aBufCalcUKn;
tPCVU * aCalcUKn = &aBufCalcUKn;
aCalcUKn = 0;
*/
 
// ==================================

     mVarCurResidu = 0.0;
     mVarCurSomLjAp = 0;
     // std::vector<int> 

     std::vector<int> aVarIndGlob;
     std::vector<int> aVarIndCoeff;

     if  (aVSB ) //  && (int(aVInd.size())==aNbTot))
     {
         double aPB =  aPds *aB;
         int aNbBl = (int)aVSB->size();

         int Y0InBloc =0;
         for (int aKBly=0 ; aKBly <aNbBl ; aKBly++)
         {
              const cSsBloc & aBlY = (*aVSB)[aKBly];
              const cIncIntervale * aIntY  = aBlY.Intervale();
              int aNumBlocIntervY    = aIntY->NumBlocSolve();

              int aI0y = aBlY.I0AbsSolve();
              int aI1y = aBlY.I1AbsSolve();

              int X0InBloc =0;
              for (int aKBlx=0 ; aKBlx <aNbBl ; aKBlx++)
              {
                  const cSsBloc & aBlX = (*aVSB)[aKBlx];
                  const cIncIntervale * aIntX  = aBlX.Intervale();
                  int aNumBlocIntervX    = aIntX->NumBlocSolve();
                  if ((!mOptSym) || (aNumBlocIntervX>=aNumBlocIntervY))
                  {
                     int aI0x = aBlX.I0AbsSolve();
                     int aI1x = aBlX.I1AbsSolve();
                     int yin = Y0InBloc;

                     for (int aYOut = aI0y; aYOut<aI1y ; aYOut++)
                     {
                         double aPCV1 =  aPds * aFullCoef[yin++];
                         int aDebX  = (mOptSym && (aNumBlocIntervX==aNumBlocIntervY)) ? aYOut : aI0x;

                         int xin = X0InBloc + aDebX-aI0x;
                         double * aLineOut = mDatatLi_Li[aYOut];
                         for (int XOut = aDebX ; XOut<aI1x ; XOut++)
                         {
                             aLineOut[XOut] += aPCV1 * aFullCoef[xin++];
                         }
                     }

                  }
                  X0InBloc += aBlX.Nb();
              }

              int yin = Y0InBloc;

              for (int YOut=aI0y ; YOut<aI1y ; YOut++)
              {
                  if (mDoCalculVariance)
                  {
                     aVarIndGlob.push_back(YOut);
                     aVarIndCoeff.push_back(yin);
                  }
                  mDatabi_Li [YOut] +=  aPB * aFullCoef[yin++];
              }
              Y0InBloc += aBlY.Nb();
         }
     }
     else
     {
        INT NbInd = (INT) aVInd.size();
        for (INT Ind1 =0 ; Ind1<NbInd ; Ind1++)
        {
            int iVar1 = aVInd[Ind1];
            double aPCV1 =   aPds * aCoeff[Ind1];
            mDatabi_Li [iVar1] +=  aB * aPCV1;
            if (mDoCalculVariance)
            {
               aVarIndGlob.push_back(iVar1);
               aVarIndCoeff.push_back(Ind1);
            }

            // Si mOptSym o n remplit la partie telle que  Ind2 >= Ind1
            // donc x >= y, donc  partie "superieure"
            int aDebInd2 = mOptSym  ? Ind1 : 0;
            for (INT Ind2 =aDebInd2 ; Ind2<NbInd ; Ind2++)
            {
                 int iVar2 = aVInd[Ind2];
                 mDatatLi_Li[iVar1][iVar2] +=  aPCV1 * aCoeff[Ind2];
            }
        }
     }


     if (mDoCalculVariance)
     {
         if (aCalcUKn)
         {
            aCalcUKn->NewEl(aB,aPds);
         }

         double * aLocCoeff = (aVSB ? aFullCoef : aCoeff);
         double aResidual = -aB;
         std::vector<double> aVSomAPl(aVarIndCoeff.size());

         for (int aK=0 ; aK<int(aVarIndCoeff.size()) ; aK++ )
         {
             aResidual+=  aLocCoeff[aVarIndCoeff[aK]]  * mDataSolL2[aVarIndGlob[aK]];
             if (aCalcUKn)
             {
                 aCalcUKn->AddVal(aLocCoeff[aVarIndCoeff[aK]],aVarIndGlob[aK]);
             }
         }
         mSomR2Pds += ElSquare(aResidual) * aPds;
         mSomPds   += aPds;

         if (aCalcUKn)
         {
              aCalcUKn->SetResidu(aResidual);
         }
         else
         {
            for (int aKI=0 ; aKI<int(aVarIndCoeff.size()) ; aKI++ )
            {
                double aSomApL = 0;
                for (int aKJ=0 ; aKJ<int(aVarIndCoeff.size()) ; aKJ++ )
                {
                    aSomApL += aLocCoeff[aVarIndCoeff[aKJ]] *  mDataInvtLi_Li[aVarIndGlob[aKI]][aVarIndGlob[aKJ]];
                }
                aVSomAPl[aKI] = aSomApL;
            }

            for (int aKI=0 ; aKI<int(aVarIndCoeff.size()) ; aKI++ )
            {
                for (int aKJ=0 ; aKJ<int(aVarIndCoeff.size()) ; aKJ++ )
                {
                    mDCoVar[aVarIndGlob[aKI]][aVarIndGlob[aKJ]] +=  ElSquare(aPds * aResidual) * aVSomAPl[aKI] * aVSomAPl[aKJ];
                }
            }
         }
     }

// std::cout << "L2SR:VG " << mDatabi_Li[0] << "\n";

     mBibi += aPds * ElSquare(aB);
     mMaxBibi = ElMax(mMaxBibi,aPds * ElSquare(aB));
     mNbEq ++;
}

void L2SysSurResol::V_GSSR_EqMatIndexee
     (
          const std::vector<INT> & aVInd,
          REAL aPds,REAL ** aMat,
          REAL * aVect,REAL aCste
     )
{
// ELISE_ASSERT(false,"L2SysSurResol::V_GSSR_EqMatIndexee");
    mBibi +=  aPds * aCste;
    INT aNbInd = (INT) aVInd.size();
    for (INT aKX=0 ; aKX<aNbInd ; aKX++)
    {
         mDatabi_Li[aVInd[aKX]] -=  aPds * aVect[aKX];
         for (INT aKY=0 ; aKY<aNbInd ; aKY++)
             mDatatLi_Li[aVInd[aKY]][aVInd[aKX]] +=  aPds *aMat[aKY][aKX];
    }
}


tSysCho  L2SysSurResol::GetElemQuad(int x,int y) const
{
   return mDatatLi_Li[y][x];
}
void  L2SysSurResol::SetElemQuad(int x,int y,const tSysCho& aV )
{
   mDatatLi_Li[y][x] = aV;
}

tSysCho   L2SysSurResol::GetElemLin(int y) const
{
   return mDatabi_Li[y];
}
void L2SysSurResol:: SetElemLin(int y,const tSysCho& aV )
{
   mDatabi_Li[y] = aV;
}



bool L2SysSurResol::GSSR_UseEqMatIndexee()
{
   return true;
   // return true;
}




void L2SysSurResol::AddTermLineaire(INT aK,REAL aVal)
{
    mDatabi_Li [aK] -= aVal/2.0;
}

void L2SysSurResol::AddTermQuad(INT aK1,INT aK2,REAL aVal)
{
    mDatatLi_Li[aK1][aK2] += aVal;
}


Im1D_REAL8  L2SysSurResol::Solve(bool * aResOk)
{
   if (mOptSym)
   {
      ELISE_ASSERT(mNbContrainte==0,"L2SysSurResol::Solve");
      for (int aK1=0 ; aK1 <mNbVar ; aK1++)
      {
          for (int aK2=0 ; aK2 <aK1 ; aK2++)
              mDatatLi_Li[aK1][aK2] =   mDatatLi_Li[aK2][aK1];
      }
   }
// std::cout << "L2SysSurResol " << mDatatLi_Li[0][1] << " " << mDatatLi_Li[1][0]  << "\n";
// getchar();

   if (mNbContrainte)
   {

    INT NbVarTot = mNbVar + mNbContrainte;
        GaussjPrec aGP(NbVarTot,1);
        ElMatrix<REAL> & M  = aGP.M();
        ElMatrix<REAL> & b  = aGP.b();
        ElMatrix<REAL> & x  = aGP.x();

        for (INT ky=0;ky <NbVarTot ; ky++)
        {
		   if (ky < mNbVar)
			b(0,ky) = mbi_Li.data()[ky];
		   else
			b(0,ky) = mE(0,mLineCC+ky-mNbVar);

			   for (INT kx=0;kx <NbVarTot ; kx++)
			   {
				   if ((kx<mNbVar) && (ky<mNbVar))
				   {
					   M(kx,ky) = mtLi_Li.data()[kx][ky];
				   }
				   else if ((kx>=mNbVar) && (ky>=mNbVar))
				   {
					   M(kx,ky) = 0;
				   }
				   else
				   {
					   INT X = ElMin(kx,ky);
					   INT Y = ElMax(kx,ky);
					   M(kx,ky) =  mC(X,mLineCC+Y-mNbVar);
				   }
			   }
        }
        bool Ok = aGP.init_rec();
    if ( aResOk)
           *aResOk = Ok;

    if (Ok)
    {
       for (INT k=0; k<6; k++)
       {
            aGP.amelior_sol();
       }
       for (INT kx=0;kx <mNbVar ; kx++)
            mSolL2.data()[kx] = x(0,kx);
    }
    else
    {
           ELISE_ASSERT(aResOk,"Singular Matrix in  L2SysSurResol::Solve");
    }

        return mSolL2;
   }


    GaussjPrec aGP(mNbVar,1);
    ElMatrix<REAL> & M  = aGP.M();
    ElMatrix<REAL> & b  = aGP.b();
    ElMatrix<REAL> & x  = aGP.x();


    for (INT ky=0;ky <mNbVar ; ky++)
    {
           b(0,ky) = mbi_Li.data()[ky];
           for (INT kx=0;kx <mNbVar ; kx++)
           {
               M(kx,ky) = mtLi_Li.data()[kx][ky];
           }
    }



    bool Ok = aGP.init_rec();



    ELISE_ASSERT(mInvtLi_Li.sz()==Pt2di(mNbVar,mNbVar),"Incohe in L2:Solve");
    ElMatrix<REAL> & MInv  = aGP.Minv();
    for (INT ky=0;ky <mNbVar ; ky++)
    {
           for (INT kx=0;kx <mNbVar ; kx++)
           {
               mDataInvtLi_Li[kx][ky]=  MInv(kx,ky);
               // M(kx,ky) = mtLi_Li.data()[kx][ky];
           }
    }

 

    if (aResOk)
       *aResOk = Ok;
    if (Ok)
    {
        for (INT k=0; k<6; k++)
        aGP.amelior_sol();

         for (INT kx=0;kx <mNbVar ; kx++)
         mSolL2.data()[kx] = x(0,kx);
    }
    mResiduAfterSol  =  ResiduOfSol(mSolL2.data());
    return mSolL2;
}

tSysCho    L2SysSurResol::ResiduAfterSol() const
{
    return mResiduAfterSol;
}

bool  L2SysSurResol::ResiduIsComputedAfterSolve() 
{
   return true;
}




/*********************************************************************/
/*                                                                   */
/*        Classe optimise pour la decomposition en valeur singuliere */
/*   des matrices 3x3                                                */
/*                                                                   */
/*********************************************************************/

template <class Type> class cSVD3x3
{
     public :
           cSVD3x3 (ElMatrix<double> & aMat);

        // Soit A les matrice a SVD


       // Contient la matrice A
           Type m00, m10, m20;
           Type m01, m11, m21;
           Type m02, m12, m22;

        // Contient la matric  A tA
/*
   Avec les notatiion  "habituelle"

   a  b  c     a  b  c
   b  e  f     d  e  f
   c  f  i     g  h  i
*/
           Type a;
           Type b;
           Type c;
           Type e;
           Type f;
           Type i;

   //  produit qui reviennent plusieurs fois
           Type ae;
           Type ei;
           Type ia;
           Type b2;
           Type c2;
           Type f2;

    // Polynome caracteristique  de A tA   L^3 + K2 L^2 + K1 L + K0 = 0

            Type  K0;
            Type  K1;
            Type  K2;
    // Polynome reduit      Z^3 +  pZ + q , en posant L = Z - K2/3
            Type K2Div3;
            Type p;
            Type q;

            Type Discr;


            Type PolRed(Type aSol) {return aSol*aSol*aSol + p*aSol + q;}
            void TestSolRed(Type aSol)
            {
                 std::cout << "PolRed " << aSol <<  " => " << PolRed(aSol) << "\n";
            }
            Type PolInit(Type aSol) {return aSol*aSol*aSol + K2*aSol*aSol + K1*aSol + K0;}
            void TestSolInit(Type aSol)
            {
                 std::cout << "PolInit " << aSol <<  " => " << PolInit(aSol) << "\n";
            }

     // ValP1 et VecP1
            Type R1;
            Type x1, y1, z1;

      //  Matrice  A tA + R1 Id

            Type  aR1;
            Type  eR1;
            Type  iR1;
            void TestSolAR(Type x,Type y,Type z,Type R)
            {
                std::cout  << " Rxyz="<<  ElAbs((a+R)*x +  b*y +   c*z)
                                        + ElAbs(b*x +  (e+R)*y +   f*z)
                                        + ElAbs(c*x +    f*y + (i+R)*z)
                           << "\n";
            }
            void TestSolAR1() {TestSolAR(x1,y1,z1,R1);}
            void TestSolAR2() {TestSolAR(x2,y2,z2,mVP2);}
            void TestSolAR3() {TestSolAR(x3,y3,z3,mVP3);}

            void TestSolVP1()
            {
                std::cout  << " VPRes="<<  ElAbs(a*x1+b*y1+ c*z1 + R1*x1)
                                         + ElAbs(b*x1+e*y1+f*z1 +R1*y1)
                                         + ElAbs(c*x1+f*y1+i*z1 +R1*z1)
                           << "\n";
            }

     // Orthog a x1,y1,z1
            Type x2O,y2O,z2O;
            Type x3O,y3O,z3O;

     // Image des prec par AtA
            Type Ax2O,Ay2O,Az2O;
            Type Ax3O,Ay3O,Az3O;
            Type mVP2;
            Type mVP3;

            Type x2, y2, z2;
            Type x3, y3, z3;

            void MulAtA(Type & xo,Type &yo,Type &zo,const Type & xi,const Type &yi,const Type &zi)
            {
                xo = a*xi + b*yi + c*zi;
                yo = b*xi + e*yi + f*zi;
                zo = c*xi + f*yi + i*zi;
            }


     //   aR1  b    c       X      0
     //   b    eR1  f       Y  =   0
     //   c    f    iR1     1      0

     void MakeNorm(Type &x,Type & y, Type &z)
     {
       Type aNorm = sqrt(x*x + y*y + z * z);
       x /= aNorm;
       y /= aNorm;
       z /= aNorm;
     }

      void TestRON()
      {
             std::cout
                       << " N1 " << (x1*x1+y1*y1+z1*z1)
                       << " N2 " << (x2O*x2O+y2O*y2O+z2O*z2O)
                       << " N3 " << (x3O*x3O+y3O*y3O+z3O*z3O)
                       << " S12 " << (x1*x2O+y1*y2O+z1*z2O)
                       << " S13 " << (x1*x3O+y1*y3O+z1*z3O)
                       << " S23 " << (x2O*x3O+y2O*y3O+z2O*z3O)
                       << "\n";
      }

};

/*
   Det =  a (ei -f2) + i(ae-b2) + e (c2-ai)

   mDet = ae*i + 2 *b*f*c - (a*fh) - e * cg - i *bd;
*/

template <class Type> cSVD3x3<Type>::cSVD3x3 (ElMatrix<double> & aMat)
{
    double ** aDM = aMat.data();
    double * aL0 = aDM[0];
    double * aL1 = aDM[1];
    double * aL2 = aDM[2];

   // Memorise la matrice pour un acces direct non indexe
    m00 = aL0[0];    m10 = aL0[1];    m20 = aL0[2];
    m01 = aL1[0];    m11 = aL1[1];    m21 = aL1[2];
    m02 = aL2[0];    m12 = aL2[1];    m22 = aL2[2];

   // Calcul A tA
    a = m00*m00  + m10*m10 + m20*m20 ;  // L0 . L0
    b = m00*m01  + m10*m11 + m20*m21 ;  // L0 . L1
    c = m00*m02  + m10*m12 + m20*m22 ;  // L0 . L2
    e = m01*m01  + m11*m11 + m21*m21 ;  // L1 . L1
    f = m01*m02  + m11*m12 + m21*m22 ;  // L1 . L2
    i = m02*m02  + m12*m12 + m22*m22 ;  // L2 . L2


   // Calcul de produit qui reviennent plusieurs fois
    ae = a * e;
    ei = e * i;
    ia = a * i;
    b2 = b * b;
    c2 = c * c;
    f2 = f * f;

    // Polynome caracteristique   L^3 + K2 L^2 + K1 L + K0 = 0
    K0 =  ae * i + 2 * b * f * c - a*f2 - e*c2 - i * b2; // Det
    K1 =  ae + ei + ia - b2  -c2 - f2;  // Somme des 3 det 2x2 diag
    K2 =  a + e + i;                    // trace

    // En posant  L' = L - K2 /3

    K2Div3 = K2 /3.0;
    p =  K1 - K2* K2Div3;
    q =  K0 + (K2/27 ) * (2*K2*K2 -9*K1);


    // Discr tjs > 0  car mat diag
    // Discr = q * q + (4*p*p*p)/27;
    // Discr = - (4 *p*p*p +9 * q * q);

    // Je comprends plus trop PK, mais experim p est tjs < 0, donc OK ....
    // std::cout << "P= " << p << "\n"; ELISE_ASSERT(p<=0,"JpppPPppp");
    if (p>0) p=0;


    Type ppp = p * p * p;
    Type ro = sqrt(-ppp/27.0);
    Type om = acos(-q/(2*ro));
    R1  = 2*sqrt(-p/3)*cos(om/3.0) - K2Div3;


   //  ==== TEST si PB, verifie que R1 est bien racine du polynome carac
   // TestSolInit(R1);

/*
    Code initial du Cardan d'apres P Julien :
    Type y = sqrt(-p/3)*cos(om/3.0);
    Type z = sqrt(-p)*sin(om/3.0);
    Type R1 = 2*y ;
    Type R2  = -y + z;
    Type R3  = -y - z;
*/

     // A tA + R1 Id
     aR1 = a+R1;
     eR1 = e+R1;
     iR1 = i+R1;

     // Pour trouver le vecteur propre si par ex
     // Si Z = 1; le systeme devient
     //
     //   aR1  b    c       X      0
     //   b    eR1  f       Y  =   0
     //   c    f    iR1     1      0


     // On cherche le systeme des 3 le  stable
/*
     {
        Type deltaZ = aR1 * eR1 - b2;    Type AbsDZ = ElAbs(deltaZ);
        Type deltaX = eR1 * iR1 - f2;    Type AbsDX = ElAbs(deltaX);
        Type deltaY = aR1 * iR1 - c2;    Type AbsDY = ElAbs(deltaY);

        if ((AbsDZ>AbsDX) &&  (AbsDZ>AbsDY))
        {
             x1 = ( -eR1 * c  +b *f   ) / deltaZ;
             y1 = (b*c       - aR1 * f) / deltaZ;
             z1 = 1.0;
        }
        else if ((AbsDX>AbsDY) && (AbsDX>AbsDZ))
        {
              x1 = 1.0;
              y1 =  (-iR1*b +f*c)/deltaX;
              z1 =  (f*b - eR1*c)/deltaX;
        }
        else
        {
              x1 =  ( -iR1 * b  +c *f   ) /  deltaY;
              y1 =  1;
              z1 =  (c*b - aR1*f)/deltaY;
        }
     }

     MakeNorm(x1,y1,z1);
     TestSolAR1();
*/


/*
     {
          double aCoef[3] ;
          L2SysSurResol aSys(3);
          aCoef[0] = aR1; aCoef[1] = b  ; aCoef[2] = c;  aSys.AddEquation(1,aCoef,0);
          aCoef[0] = b  ; aCoef[1] = eR1; aCoef[2] = f;  aSys.AddEquation(1,aCoef,0);
          aCoef[0] = c  ; aCoef[1] = f; aCoef[2] = iR1;  aSys.AddEquation(1,aCoef,0);

          Type aG2 = aR1*aR1 + eR1*eR1 + iR1*iR1 + b2 + c2 +f2 ;
          Type aDiv = 1e-15;
          aG2 = ElMax(aG2,Type(aDiv)) * aDiv;
          // aG2 = 1;

          for (int aK=0 ; aK<3 ; aK++)
          {
               for (int aK2=0 ; aK2<3 ; aK2++)
                   aCoef[aK2] = (aK2==aK);
               aSys.AddEquation(aG2,aCoef,1);
               // aSys.AddEquation(aG2,aCoef,-1);
          }

          bool Ok;
          Im1D_REAL8 aSol = aSys.Solve(&Ok);
          x1 = aSol.data()[0];
          y1 = aSol.data()[1];
          z1 = aSol.data()[2];
 std::cout << "XYZ " << x1 << " "<< y1 << " " << z1 << "\n";
          MakeNorm(x1,y1,z1);
          TestSolAR1();
     }
*/

     /*  Calcul robusrte ? du noyau */
     Type aDiv = 1e-15;
     {
          cMSymCoffact3x3<Type> aCof;

          Type aG2 = aR1*aR1 + eR1*eR1 + iR1*iR1 + b2 + c2 +f2 ;
          aG2 = ElMax(aG2,Type(aDiv)) * aDiv;

          aCof.a = aG2+ aR1 * aR1 + b2 + c2;
          aCof.b =     (aR1 +eR1)*b  + c*f;
          aCof.c =     b*f +c*(aR1+iR1);
          aCof.e = aG2+ b2 + eR1*eR1+f2;
          aCof.f =      c*b + (eR1 + iR1)* f;
          aCof.i = aG2+ c2 +f2 + iR1*iR1;
          aCof.FinishCoFact();


          Type aV[3] = {1,1,1};
          Pt3d<Type>  aP = aCof.CoffMul(aV);
          x1 = aP.x;
          y1 = aP.y;
          z1 = aP.z;
          MakeNorm(x1,y1,z1);
          //  std::cout << "XYZ " << x1 << " "<< y1 << " " << z1 << "\n";
          // TestSolAR1();
     }

     //   aR1  b    c       X      0    aR1 |aR1 b  c #  b     b eR1 f #  c      c  f iR1
     //   b    eR1  f       Y  =   0    b   |         #  eR1A          #  f
     //   c    f    iR1     1      0    c   |         #  f             #  iR1
/*
*/

     //TestSolVP1();

     // Calcul d'un vecteur orthog
     {
         Type AX1 =ElAbs(x1);
         Type AY1 =ElAbs(y1);
         Type AZ1 =ElAbs(z1);
         if ( (AX1>AZ1) || (AY1>AZ1))
         {
               x2O = -y1;
               y2O =  x1;
               z2O = 0.0;
         }
         else
         {
              x2O = -z1;
              y2O = 0.0;
              z2O = x1;
         }
     }
     MakeNorm(x2O,y2O,z2O);

     // Calcul de l'autre vecteur
     x3O = y1 * z2O - z1 * y2O;
     y3O = z1 * x2O - x1 * z2O;
     z3O = x1 * y2O - y1 * x2O;

     //TestRON();

     // Analyse de la matrice dans la base x2O .. x3O

     {
         // Image par AtA de x2O ..
         MulAtA(Ax2O,Ay2O,Az2O,x2O,y2O,z2O);
         MulAtA(Ax3O,Ay3O,Az3O,x3O,y3O,z3O);

         // Image par AtA dans le repere x2O ...
         Type aS22 = x2O*Ax2O +  y2O*Ay2O + z2O*Az2O;
         Type aS23 = x2O*Ax3O +  y2O*Ay3O + z2O*Az3O;
         Type aS33 = x3O*Ax3O +  y3O*Ay3O + z3O*Az3O;

         // Det(M + VP1 Id)  = 0
         Type aDiscr   = sqrt(ElSquare(aS22-aS33) + 4*ElSquare(aS23));
         mVP2 = -(aS22+aS33 + aDiscr) /2.0;
         mVP3 = -(aS22+aS33 -aDiscr)/ 2.0;


         if (0)
         {
            std::cout << "DET " << (aS22+mVP2)*(aS33+mVP2) - ElSquare(aS23) << "\n";
            std::cout << "VERIF VP/ DET" << (mVP2 * mVP3 * R1  / K0)+1 << "\n";
            std::cout << "VERIF VP/ TRACE" << (mVP2 + mVP3 + R1 ) /  K2+1 << "\n";

            Type aS32 = x3O*Ax2O +  y3O*Ay2O + z3O*Az2O;
            std::cout << "SSs " << aS23 << " " << aS32 << "\n";
            std::cout << aS22+ aS33 << "\n"; // => Warn
         }


         Type aS22VP = aS22 + mVP2;
         Type aS33VP = aS33 + mVP2;

         // Calcul deu noyau de la matrice reduite

             //          aS22VP aS23       |           aS23   aS33VP
             //   aS22VP                   |     aS23
             //   aS23                     |     aS33VP

         Type  aG2 =  aS22VP*aS22VP + aS33VP*aS33VP +  aS23*aS23;
         aG2 = ElMax(aG2,Type(aDiv)) * aDiv;

         Type  aMA =  aG2 + ElSquare(aS22VP) + ElSquare(aS23);
         Type  aMB =  aS23 * (aS22VP + aS33VP);
         Type  aMC =  aG2  + ElSquare(aS23) + ElSquare(aS33VP);

          Type aDet = aMA * aMC - ElSquare(aMB);

          // Image de (1,1) par l'inverse
          Type aVpX = (aS33VP-aS23) / aDet;
          Type aVpY = (aS22VP-aS23) / aDet;

          Type aNorm = sqrt(aVpX*aVpX + aVpY*aVpY);
          aVpX /= aNorm;
          aVpY /= aNorm;
          if (0)
          {
              Type aNX  = aS22VP * aVpX + aS23   * aVpY;
              Type aNY  = aS23   * aVpX + aS33VP * aVpY;
              std::cout << "VP " << (aVpX*aVpX+aVpY*aVpY-1)  << " " << aNX << " " << aNY  << "\n";
          }

          x2 =  aVpX*x2O  + aVpY*x3O;
          y2 =  aVpX*y2O  + aVpY*y3O;
          z2 =  aVpX*z2O  + aVpY*z3O;

          //TestSolAR2();
     }

     x3 = y1 * z2 - z1 * y2;
     y3 = z1 * x2 - x1 * z2;
     z3 = x1 * y2 - y1 * x2;
     //TestSolAR3();


     if (0)
     {
          ElMatrix<double> MtM = aMat * aMat.transpose();

          ElMatrix<double> aDiag(3,3);
          aDiag(0,0) = -R1;
          aDiag(1,1) = -mVP2;
          aDiag(2,2) = -mVP3;

          Pt3dr aV1(x1,y1,z1);
          Pt3dr aV2(x2,y2,z2);
          Pt3dr aV3(x3,y3,z3);

          ElMatrix<double> aRot(3,3);
          SetCol(aRot,0,aV1);
          SetCol(aRot,1,aV2);
          SetCol(aRot,2,aV3);

           ElMatrix<double> aDif = MtM - aRot*aDiag*aRot.transpose();

           std::cout << "Chek MtM = R D tR " << aDif.L2() << "\n";

          ElMatrix<double> aSqrtDiag(3,3);
          aSqrtDiag(0,0) = 1/ sqrt(ElAbs(R1));
          aSqrtDiag(1,1) = 1/ sqrt(ElAbs(mVP2));
          aSqrtDiag(2,2) = 1/ sqrt(ElAbs(mVP3));

           ElMatrix<double>  aR2 =  aSqrtDiag *  aRot.transpose() * aMat;

           ElMatrix<double> aDifDec = aR2*aR2.transpose() -ElMatrix<double>(3,true);
           std::cout << "Chek dec " << aDifDec.L2() << "\n";
           // ShowMatr("RR22",aR2*aR2.transpose() -ElMatrix<double>(3,true));

          // ShowMatr("Diff",aDif);
          // ShowMatr("xxx",MtM);
          // ShowMatr("ddd",aRot*aDiag*aRot.transpose());
          // std::cout << euclid(aV1) << " " << euclid(aV2) << " " << euclid(aV3) << "\n";
     }

}

ElMatrix<double> RanM33()
{
   ElMatrix<double> aM(3,3);

   for (int anX=0 ; anX<3; anX++)
       for (int anY=0 ; anY<3; anY++)
           aM(anX,anY) = NRrandC();
   return aM;
}

//   Time 3x3 : 0.0318749
//   Time svdcmp : 0.199912
//   MatEssToMulipleRot : 1.88438

extern std::list<ElRotation3D>  MatEssToMulipleRot(const  ElMatrix<REAL> & aMEss,double LBase);


void TestSVD3x3()
{
    //double aDrMin = 1e100;
    std::vector<ElMatrix<double> > aVM;


    for (int aK=0 ; aK< 1000 ; aK++)
    {
         ElMatrix<double> aM = RanM33();
         aVM.push_back(aM);
    }

    int aNb = 100;
    ElTimer aChro3x3;
    for (int aKt=0 ; aKt<aNb ; aKt++)
    {
        for (int aKM=0 ; aKM<int(aVM.size()) ; aKM++)
        {
             cSVD3x3<double> aS2(aVM[aKM]);
        }
    }
    std::cout << "Time 3x3 : " << aChro3x3.uval() << "\n";

    ElTimer aChroMM;
    for (int aKt=0 ; aKt<aNb ; aKt++)
    {
        for (int aKM=0 ; aKM<int(aVM.size()) ; aKM++)
        {
              aVM[aKM]* aVM[aKM];
              aVM[aKM].transpose();
              ElMatrix<REAL>::Rotation(0,0,1.0);
              ElRotation3D(Pt3dr(0,0,0),aVM[aKM],true);
        }
    }
    std::cout << "Time mm : " << aChroMM.uval() << "\n";


    ElTimer aChroSVD;
    for (int aKt=0 ; aKt<aNb ; aKt++)
    {
        for (int aKM=0 ; aKM<int(aVM.size()) ; aKM++)
        {
              ElMatrix<REAL>  aU(3,3),aLineDiag(1,3),aV(3,3);
              svdcmp_diag(aVM[aKM],aU,aLineDiag,aV,true);
              //MatEssToMulipleRot(aVM[aKM],1.0);
        }
    }
    std::cout << "Time svd : " << aChroSVD.uval() << "\n";




}





template class cSVD3x3<double>;
template class cSVD3x3<REAL16>;


/***********************************************************/

template <class Type> cMSymCoffact3x3<Type>::cMSymCoffact3x3()
{
}

template <class Type> cMSymCoffact3x3<Type>::cMSymCoffact3x3(Type ** aMat)
{
// Pour notations voir
// http://www.philipperey.net/maths3-4/algebreLineaireVecteurs/algebreLineaire/inverseMatrice.php
//
// Methode basee sur les cofacteurs

   Type * L0 = aMat[0];

   a =  L0[0];
   b = L0[1];  // d
   c = L0[2];  // g
   e =  aMat[1][1];
   f =  aMat[1][2];  //h
   i  =  aMat[2][2];
   FinishCoFact();
}

template <class Type> void cMSymCoffact3x3<Type>::FinishCoFact()
{
/*

   a  b c     a  b  c
   b  e f     d  e  f
   c  f i     g  h  i

   double * aL2 =  aMat[2];
   double a =  aMat[0][0];
   double b = aMat[1][0];
   double e = aMat[1][1];
   double c = aL2[0];
   double f = aL2[1];
   double i = aL2[2];
*/

   Type fh = f*f;
   Type cg = c*c;
   Type bd = b*b;

   Type ae = a*e;

   mDet = ae*i + 2 *b*f*c - (a*fh) - e * cg - i *bd;

   mA = e*i - fh;
   mE = a*i - cg;
   mI = ae  - bd;

   mB = c*f - b*i;
   mC = b*f -c*e;
   mF = c*b -a*f;
}


template <class Type> Pt3d<Type> cMSymCoffact3x3<Type>::CoffVecInv(const Type * aVect) const
{
    Type U = aVect[0];
    Type V = aVect[1];
    Type W = aVect[2];

 /*
     A  B  C    U
     B  E  F    V
     C  F  I    W
*/
    Type x = mA*U + mB*V + mC*W;
    Type y = mB*U + mE*V + mF*W;
    Type z = mC*U + mF*V + mI*W;

    return Pt3d<Type>(x/mDet,y/mDet,z/mDet);
}

template <class Type> Pt3d<Type> cMSymCoffact3x3<Type>::CoffMul(const Type * aVect) const
{
    Type U = aVect[0];
    Type V = aVect[1];
    Type W = aVect[2];

    Type x = mA*U + mB*V + mC*W;
    Type y = mB*U + mE*V + mF*W;
    Type z = mC*U + mF*V + mI*W;

    return Pt3d<Type>(x,y,z);
}





template <class Type> void cMSymCoffact3x3<Type>::CoffSetInv(Type ** aMat)
{

   aMat[0][0] = mA /mDet;
   aMat[1][1] = mE /mDet;
   aMat[2][2] = mI /mDet;

   aMat[0][1] = aMat[1][0]  = mB / mDet;
   aMat[0][2] = aMat[2][0]  = mC / mDet;
   aMat[1][2] = aMat[2][1]  = mF / mDet;

   /*
   static const int  aNbIter = 0;
   if (aNbIter)
   {
      static Type  aErMax = 0.0;
      Type aEr[aNbIter];

      for (int aK=0 ; aK< aNbIter ; aK++)
      {
        // aV00 ... : erreur / a l'identite
          Type aEr00 = a *aMat[0][0] + b *  aMat[0][1]  + c * aMat[0][2] -1.0;
          Type aEr10 = a *aMat[1][0] + b *  aMat[1][1]  + c * aMat[1][2] ;
          Type aEr20 = a *aMat[2][0] + b *  aMat[2][1]  + c * aMat[2][2] ;


          Type aEr01 = b *aMat[0][0] + e *  aMat[0][1]  + f * aMat[0][2] ;
          Type aEr11 = b *aMat[1][0] + e *  aMat[1][1]  + f * aMat[1][2] -1.0 ;
          Type aEr21 = b *aMat[2][0] + e *  aMat[2][1]  + f * aMat[2][2] ;


          Type aEr02 = c *aMat[0][0] + f *  aMat[0][1]  + i * aMat[0][2] ;
          Type aEr12 = c *aMat[1][0] + f *  aMat[1][1]  + i * aMat[1][2] ;
          Type aEr22 = c *aMat[2][0] + f *  aMat[2][1]  + i * aMat[2][2] -1.0 ;



          aEr[aK] =      ElAbs(aEr00) + ElAbs(aEr10) + ElAbs(aEr20)
                       + ElAbs(aEr01) + ElAbs(aEr11) + ElAbs(aEr21)
                       + ElAbs(aEr02) + ElAbs(aEr12) + ElAbs(aEr22)
                    ;

        // aC00 ... : terme correctif
          Type aC00 =  aMat[0][0]*aEr00 + aMat[1][0]*aEr01 +  aMat[2][0]*aEr02;
          Type aC10 =  aMat[0][0]*aEr10 + aMat[1][0]*aEr11 +  aMat[2][0]*aEr12;
          Type aC20 =  aMat[0][0]*aEr20 + aMat[1][0]*aEr21 +  aMat[2][0]*aEr22;

          // Type aC01 =  aMat[0][1]*aEr00 + aMat[1][1]*aEr01 +  aMat[2][1]*aEr02;
          Type aC11 =  aMat[0][1]*aEr10 + aMat[1][1]*aEr11 +  aMat[2][1]*aEr12;
          Type aC21 =  aMat[0][1]*aEr20 + aMat[1][1]*aEr21 +  aMat[2][1]*aEr22;

          // Type aC02 =  aMat[0][2]*aEr00 + aMat[1][2]*aEr01 +  aMat[2][2]*aEr02;
          // Type aC12 =  aMat[0][2]*aEr10 + aMat[1][2]*aEr11 +  aMat[2][2]*aEr12;
          Type aC22 =  aMat[0][2]*aEr20 + aMat[1][2]*aEr21 +  aMat[2][2]*aEr22;


          aMat[0][0] -= aC00;
          aMat[1][1] -= aC11;
          aMat[2][2] -= aC22;

          aMat[0][1] -= aC10;
          aMat[0][2] -= aC20;
          aMat[1][2] -= aC21;

          aMat[1][0] = aMat[0][1];
          aMat[2][0] = aMat[0][2];
          aMat[2][1] = aMat[1][2];
      }
      if (0)//(aEr[aNbIter-1] > aErMax)
      {
         aErMax = aEr[aNbIter-1];
         std::cout << "#######  ##### Err= " << aEr[0]
                   << " " << aEr[1]
                   << " " << aEr[aNbIter-1] << "\n";

      }
   }
   */
}

template class cMSymCoffact3x3<double>;
template class cMSymCoffact3x3<REAL16>;



Pt3d<double>  L2SysSurResol::Solve3x3Sym(bool * OK)
{
  cMSymCoffact3x3<double> aMCS(mDatatLi_Li);

   if (ElAbs(aMCS.mDet<1e-30))
   {
       if (OK) *OK = false;
       return Pt3dr(1e40,1e50,1e70);
   }
   if  (OK) *OK =true;

   return aMCS.CoffVecInv(mDatabi_Li);
}



/* Footer-MicMac-eLiSe-25/06/2007

   Ce logiciel est un programme informatique servant a  la mise en
   correspondances d'images pour la reconstruction du relief.

   Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
   respectant les principes de diffusion des logiciels libres. Vous pouvez
   utiliser, modifier et/ou redistribuer ce programme sous les conditions
   de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
   sur le site "http://www.cecill.info".

   En contrepartie de l'accessibilite au code source et des droits de copie,
   de modification et de redistribution accordes par cette licence, il n'est
   offert aux utilisateurs qu'une garantie limitee.  Pour les mêmes raisons,
   seule une responsabilite restreinte pese sur l'auteur du programme,  le
   titulaire des droits patrimoniaux et les concedants successifs.

   A cet egard  l'attention de l'utilisateur est attiree sur les risques
   associes au chargement,  a  l'utilisation,  a  la modification et/ou au
   developpement et a  la reproduction du logiciel par l'utilisateur etant
   donne sa specificite de logiciel libre, qui peut le rendre complexe a
   manipuler et qui le reserve donc a  des developpeurs et des professionnels
   avertis possedant  des  connaissances  informatiques approfondies.  Les
   utilisateurs sont donc invites a  charger  et  tester  l'adequation  du
   logiciel a  leurs besoins dans des conditions permettant d'assurer la
   securite de leurs systemes et ou de leurs donnees et, plus generalement,
   a l'utiliser et l'exploiter dans les memes conditions de securite.

   Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
   pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
   termes.
   Footer-MicMac-eLiSe-25/06/2007/*/
