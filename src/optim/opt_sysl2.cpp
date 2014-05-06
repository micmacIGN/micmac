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

/****************************************************/
/*                                                  */
/*            cGenSysSurResol                       */
/*                                                  */
/****************************************************/

cGenSysSurResol::~cGenSysSurResol()
{
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
              Basic_GSSR_AddNewEquation_Indexe(0,0,0,aVInd,1.0,&aCoef1,mValCstr.data()[aKV]);
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

if (true && (NbVar() >8))
{
/*
ElMatrix<double> aM2(NbVar(),NbVar());
ElMatrix<double> aL2(1,NbVar());

for (int aJ=0; aJ< NbVar() ; aJ++)
{
aL2(0,aJ) = GetElemLin(aJ);
for (int aI=0; aI< NbVar() ; aI++)
  aM2(aI,aJ) =  GetElemQuad(aI,aJ);
}
ElMatrix<double> aS2 = gaussj(aM2) * aL2;



std::cout << "NBV " << NbVar() << "NB CONTRAINTE " << mNbContrainte << "  Assumed : " << ContraintesAssumed() << "\n";
for (int aK=0 ; aK<NbVar()  ; aK++)
   std::cout << "*************jjkk--- " << aK << " " << aSol.data()[aK] << " " << aS2(0,aK)  << " M2 " << aM2(aK,aK) << "\n";

getchar();
*/


if (0)
{

for (int aJ=0; aJ< NbVar() ; aJ++)
{

double aS0=0;
for (int aK=0; aK< NbVar() ; aK++)
   aS0 += aSol.data()[aK] * GetElemQuad(aJ,aK);



   float aV = (float)GetElemLin(aJ);
   printf("%d %f %f %f :: ",aJ,aSol.data()[aJ],aV,aS0);
for (int aK=0 ; aK< NbVar() ; aK++)
{
   float aV = (float)GetElemQuad(aJ,aK);
   printf("%f ",aV);
}
printf("\n");
}
}
}


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
         REAL aPds,REAL * aCoeff,REAL aB
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
     V_GSSR_AddNewEquation_Indexe(aVSB,aFullC,aNbTot,aVInd,aPds,aCoeff,aB);

/*
    if (BugUPL )
    {
         if (aCpt==7)//  || (aCpt==8))
         {
            for (int aKB=0 ; aKB<(int)aVSB->size() ; aKB++)
            {
                std::cout  << aVSB->size() << " "
                          << (*aVSB)[aKB].I0AbsAlloc() << " " 
                          << (*aVSB)[aKB].I1AbsAlloc() << " " 
                          << (*aVSB)[aKB].I0AbsSolve() << " " 
                          << (*aVSB)[aKB].I1AbsSolve() << " " 
                          << " BugUPL in cGenSysSurResol::BGAI " << aPds << "\n";
            }
            BasicVerifMatPos(*aVSB,aCpt);
         }
    }
*/
}
void cGenSysSurResol::GSSR_AddNewEquation_Indexe
     (
         const std::vector<cSsBloc> * aVSB,
         double * aFullC,int aNbTot,
         const std::vector<INT> & aVInd ,
         REAL aPds,REAL * aCoeff,REAL aB
     )
{
     // Done in Basic_GSSR_AddNewEquation_Indexe  -> AssertPhaseEquation();
     Basic_GSSR_AddNewEquation_Indexe(aVSB,aFullC,aNbTot,aVInd,aPds,aCoeff,aB);
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
         REAL aPds,REAL * aCoeff,REAL aB
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
if (MPD_MM())
{
// std::cout << "PLLANNN " << aPds << "\n";
// getchar();
}
    L2SysSurResol::V_GSSR_AddNewEquation_Indexe(0,0,0,aVInd,aPds,&(aCoeff[0]),scal(aDirOrtho,aP0));
}

void L2SysSurResol::GSSR_AddEquationFitOneVar(int aNumVar,double aVal,double aPds)
{
    if (aPds<=0) return;
    std::vector<int> aVInd;
    aVInd.push_back(aNumVar);
    
    double aCoef1 = 1.0;

    L2SysSurResol::V_GSSR_AddNewEquation_Indexe(0,0,0,aVInd,aPds,&aCoef1,aVal);
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

   for (int aKS=0; aKS<int(aVS.size()); aKS++)
   {
      double aPds = ((aVPds!=0) ? (*aVPds)[aKS] : 1.0);
      if (aPds > 0)
      {
         aSys.GSSR_Add_EqInterDroite3D (aVS[aKS].TgNormee(),aVS[aKS].P0(),aPds);
      }
   }

   if (aRAZ)
   {
       double aCoeff[3]={0,0,1};
       aSys.AddEquation(1/ElSquare(aRAZ->IncEstim()),aCoeff,aRAZ->Z());
   }

   if (aVPts)
   {
       int aNb = aVPts->size();
       ELISE_ASSERT((aNb%2)==0,"ElSeg3D::L2InterFaisceaux aVPts impair !!!");
       for (int aK=0 ; aK<aNb ; aK+=2)
       {
            aSys.GSSR_AddEquationPoint3D((*aVPts)[aK],(*aVPts)[aK+1]);
       }
   }

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
         L2SYM, !L2SYM,
   // false,false,
//         IsSym, (! IsSym),
        true
    ),
    mNbVar           (aNbVar),
    mtLi_Li          (mNbVar,mNbVar,0.0),
    mDatatLi_Li      (mtLi_Li.data()),
    mInvtLi_Li       (mNbVar,mNbVar,0.0),
    mDataInvtLi_Li   (mInvtLi_Li.data()),
    mbi_Li           (aNbVar,0.0),
    mDatabi_Li       (mbi_Li.data()),
    mBibi            (0.0),
    mSolL2           (aNbVar),
    mDataSolL2       (mSolL2.data()),
    mNbEq            (0),
    mMaxBibi         (0)
{
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
}

void L2SysSurResol::Reset()
{
   mtLi_Li.raz();
   mbi_Li.raz();
   mBibi = 0.0;
   mNbEq = 0;
   mMaxBibi = 0;
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
     L2SysSurResol::V_GSSR_AddNewEquation_Indexe(0,0,0,VInd,aPds,&VALS[0],aB);
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
   double ** aData1 = aM1.data();
   double ** aData2 = aM2.data();

   double * aL2A = aData2[0];
   double * aL2B = aData2[1];
   double * aL2C = aData2[2];
   int aNbBl = aYVSB->size();
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


void L2SysSurResol::V_GSSR_AddNewEquation_Indexe
     (
           const std::vector<cSsBloc> * aVSB,
           double * aFullCoef, int aNbTot,
           const std::vector<INT> & aVInd ,
           REAL aPds,
           REAL * aCoeff,
           REAL aB
     )
{
// static int aCpt=0; aCpt++;
//std::cout << "L2:VANI "  << aCpt << "\n";
// bool aBug = (aCpt==44984);
/*
if ((aNbTot>10) && (aCpt%97==0))
std::cout <<  aNbTot << " " << aVInd.size()  << "\n";
*/



     if  (aVSB ) //  && (int(aVInd.size())==aNbTot))
     {
         double aPB =  aPds *aB;
         int aNbBl = aVSB->size();

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
                     mDatabi_Li [YOut] +=  aPB * aFullCoef[yin++];
              }
              Y0InBloc += aBlY.Nb();
         }

/*
         int yin =0;
         int aNbBl = aVSB->size();
         for (int aKBly=0 ; aKBly <aNbBl ; aKBly++)
         {
             const cSsBloc & aBlY = (*aVSB)[aKBly];
             int aI0y = aBlY.I0AbsSolve();
             int aI1y = aBlY.I1AbsSolve();

             int aDebBlx =  mOptSym  ? aKBly : 0;
             int X0InBloc = 0;
             for (int aKBlx=0  ; aKBlx <aDebBlx ; aKBlx++)
                 X0InBloc += (*aVSB)[aKBlx].Nb();
             

             for (int YOut=aI0y ; YOut<aI1y ; YOut++)
             {
                 double aPCV1 =   aPds * aFullCoef[yin++];
                 if (aPCV1)
                 {
                     mDatabi_Li [YOut] +=  aB * aPCV1;
                     double * aLineOut = mDatatLi_Li[YOut];

                     int xin = X0InBloc;

                     for (int aKBlx=aDebBlx  ; aKBlx <aNbBl ; aKBlx++)
                     {
                         const cSsBloc & aBlX = (*aVSB)[aKBlx];
                         int aI0x = aBlX.I0AbsSolve();

                         int aDebX  = (mOptSym && (aKBlx==aKBly)) ? YOut : aI0x;
                         int aI1X = aBlX.I1AbsSolve();
                         xin += aDebX-aI0x;

                         for (int XOut = aDebX ; XOut<aI1X ; XOut++)
                         {
                                aLineOut[XOut] += aPCV1 * aFullCoef[xin++];
                         }
                     }
                 }
             }
         }
*/
     }
     else
     {
        INT NbInd = (INT) aVInd.size();
        for (INT Ind1 =0 ; Ind1<NbInd ; Ind1++)
        {
            int iVar1 = aVInd[Ind1];
            double aPCV1 =   aPds * aCoeff[Ind1];
            mDatabi_Li [iVar1] +=  aB * aPCV1;

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


/*
void L2SysSurResol::GetMatr(ElMatrix<REAL> & M,ElMatrix<REAL> & tB)
{
    M.set_to_size(mNbVar,mNbVar);
    tB.set_to_size(mNbVar,1);

    for (INT kx=0;kx <mNbVar ; kx++)
    {
        tB(kx,0) = mbi_Li.data()[kx];
        for (INT ky=0;ky <mNbVar ; ky++)
        {
            M(kx,ky) = mtLi_Li.data()[kx][ky];
        }
    }
}
*/

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
	        aGP.amelior_sol();
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

    if (aResOk)
       *aResOk = Ok;
/*
    cout << "L2SysSurResol::Solve GP " << Ok << "\n";
    if (! Ok)
    {
         for (INT ky=0;ky <mNbVar ; ky++)
         {
             for (INT kx=0;kx <mNbVar ; kx++)
             {
                 cout <<  mtLi_Li.data()[kx][ky] << " ";
             }
             cout << "\n";
         }
         cout << "----------------\n";
    }
*/
    if (Ok)
    {
        for (INT k=0; k<6; k++)
	    aGP.amelior_sol();

         for (INT kx=0;kx <mNbVar ; kx++)
	     mSolL2.data()[kx] = x(0,kx);
    }
     return mSolL2;
}

/*
struct cMSymCoffact3x3
{
    double mA;
    double mE;
    double mI;
    double mB;
    double mC;
    double mF;
    double mDet;

    cMSymCoffact3x3(double ** aMat);
    Pt3dr Inv(const double *) const;
};
*/


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
/*

   a  b c
   b  e f
   c  f i 

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
