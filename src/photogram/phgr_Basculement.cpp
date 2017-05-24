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

/*******************************************/
/*                                         */
/*     cL2EqObsBascult                     */
/*                                         */
/*******************************************/

cL2EqObsBascult::cL2EqObsBascult
(
     cSetEqFormelles &       aSet,
     const cSolBasculeRig &  aV0,
     bool                    Code2Gen
) :
  mSet      (&aSet),
  mRF       (mSet->NewRotation(eRotLibre,ElRotation3D(aV0.Tr(),aV0.Rot(),true))),
  mBlocL    (aSet,"Lambda",1),
//  mIntL     (false,"Lambda",aSet,1),
//  mLambda   (aV0.Lambda()),
//  mLF       (mSet->Alloc().NewF(&mLambda)),
  mN1       ("P1"),
  mN2       ("P2"),
  mResidu   (mRF->C2M(mN1.PtF()*mBlocL.F(0))-mN2.PtF()),
  mNameType ("EqObsBaclt")
{
     mBlocL.SetVal(0,aV0.Lambda());
     mBlocL.CloseEBI();
     mRF->IncInterv().SetName("Orient");
     mLInterv.AddInterv(mRF->IncInterv());
     mLInterv.AddInterv(mBlocL.IncInterv());

     mFoncEqResidu = cElCompiledFonc::AllocFromName(mNameType);
     
     if (Code2Gen)
     {
        GenCode();
        return;
     }
 
     if (mFoncEqResidu==0)
     {
        ELISE_ASSERT(false,"Can Get Code Comp for cCameraFormelle::cEqAppui");    
        mFoncEqResidu = cElCompiledFonc::DynamicAlloc(mLInterv,Fonc_Num(0));
     }

     mFoncEqResidu->SetMappingCur(mLInterv,mSet);


     mN1.InitAdr(*mFoncEqResidu);
     mN2.InitAdr(*mFoncEqResidu);

     mSet->AddFonct(mFoncEqResidu);
}
 
void cL2EqObsBascult::GenCode()
{
   std::vector<Fonc_Num> aV;
   aV.push_back(mResidu.x);
   aV.push_back(mResidu.y);
   aV.push_back(mResidu.z);

   cElCompileFN::DoEverything
   (
        std::string("CodeGenere")+ELISE_CAR_DIR+"photogram"+ELISE_CAR_DIR, 
        mNameType,  
        aV, 
        mLInterv 
   );
}
void  cL2EqObsBascult::AddObservation(Pt3dr aP1,Pt3dr aP2,double aPds,bool WD2)
{
    mN1.SetEtat(aP1);
    mN2.SetEtat(aP2);

    mSet->AddEqFonctToSys(mFoncEqResidu,aPds,WD2);
}

cL2EqObsBascult::~cL2EqObsBascult()
{
}

cSolBasculeRig cL2EqObsBascult::CurSol() const
{
   ElRotation3D aRot = mRF->CurRot();
   return cSolBasculeRig
          (
              Pt3dr(0,0,0),
              aRot.tr(),
              aRot.Mat(),
              mBlocL.Val(0)
          );
}

cL2EqObsBascult  * cSetEqFormelles::NewEqObsBascult
                   (
                        const cSolBasculeRig &  aV0,
                        bool                    Cod2Gen
                   )
{
   cL2EqObsBascult * aRes = new cL2EqObsBascult(*this,aV0,Cod2Gen);
   AddObj2Kill(aRes);
   

   return aRes;
}


/*******************************************/
/*                                         */
/*    cSolBasculeRig                       */
/*                                         */
/*******************************************/

const Pt3dr & cSolBasculeRig::Tr() const {return mTr;}
const ElMatrix<double> & cSolBasculeRig::Rot() const {return mMatR;}
double cSolBasculeRig::Lambda() const {return mL;}


Pt3dr cSolBasculeRig::operator()(const Pt3dr & aP) const
{
    return mTr + mMatR * aP * mL;
}

ElRotation3D cSolBasculeRig::TransformOriC2M(const ElRotation3D & aRot) const
{
   return ElRotation3D
          (
             (*this)(aRot.tr()),
             mMatR * aRot.Mat(),
             true
          );
}

cSolBasculeRig::cSolBasculeRig
(
           const Pt3dr & aPAvant,
           const Pt3dr & aPApres,
           const ElMatrix<double> & aMatR,
           double aLambda
) : 
  mMatR (aMatR),
  mL    (aLambda),
  mTr   (aPApres-aMatR*aPAvant*aLambda)
{
}

cSolBasculeRig cSolBasculeRig::SBRFromElems(const Pt3dr & aTr,const ElMatrix<double> & aRot,const double & aLambda )
{
    return cSolBasculeRig(Pt3dr(0,0,0),aTr,aRot,aLambda);
}


cSolBasculeRig cSolBasculeRig::Inv() const
{
    ElMatrix<double> aMInv = mMatR.transpose();
    return SBRFromElems(  -(aMInv*(mTr/mL)),aMInv,1/mL);
}


cSolBasculeRig  cSolBasculeRig::Id()
{
   return cSolBasculeRig
          (
              Pt3dr(0,0,0),
              Pt3dr(0,0,0),
              ElMatrix<double>::Rotation(0,0,0),
              1.0
          );
}

cSolBasculeRig cSolBasculeRig::StdSolFromPts
               (
                    const std::vector<Pt3dr> & aV1,
                    const std::vector<Pt3dr> & aV2,
                    const std::vector<double> * aVPds, // si 0 ts les pds valent 1
                    int   aNbRansac             ,
                    int   aNbL2                
                )
{
   int aNbPts= aV1.size();
   ELISE_ASSERT(aNbPts>=3,"Not enough point in cSolBasculeRig::StdSolFromPts");
   ELISE_ASSERT(int(aV2.size())==aNbPts,"Incoherent size in cSolBasculeRig::StdSolFromPts");
   if (aVPds)
   {
       ELISE_ASSERT(int(aVPds->size())==aNbPts,"Incoherent size in cSolBasculeRig::StdSolFromPts");
   }

   cRansacBasculementRigide aBasc(false); // false  => Pas de vitesse
   for (int aKP=0 ; aKP<aNbPts ; aKP++)
   {
       aBasc.AddExemple(aV1[aKP],aV2[aKP],0,"");
   }
   aBasc.CloseWithTrGlob(false);

   aBasc.ExploreAllRansac(aNbRansac);
   cSolBasculeRig  aSBR = aBasc.BestSol();



   if (aNbL2)
   {
       cSetEqFormelles aSetEq (cNameSpaceEqF::eSysPlein);
       cL2EqObsBascult  * aL2Basc = aSetEq.NewEqObsBascult(aSBR,false);
       aSetEq.SetClosed();

       for (int aKEt=0 ; aKEt<aNbL2 ; aKEt++)
       {
           aSetEq.SetPhaseEquation();
           for (int aKP=0 ; aKP<int(aV1.size()) ; aKP++)
           {
               aL2Basc->AddObservation(aV1[aKP],aV2[aKP],(aVPds?(*aVPds)[aKP]:1.0));
           }
           aSetEq.SolveResetUpdate();
           aSBR=aL2Basc->CurSol();
       }

   }
   return aSBR;
}
/*
*/


std::vector<Pt3dr> cSolBasculeRig::Src2Cibl(const std::vector<Pt3dr> & aInput) const 
{
    std::vector<Pt3dr> aRes;

    for (int aK=0 ; aK<int(aInput.size()) ; aK++)
        aRes.push_back((*this)(aInput[aK]));

    return aRes;
}


         //===========================
         //  cRansacBasculementRigide 
         //===========================

/*******************************************/
/*                                         */
/*    cBasculementRigide                   */
/*                                         */
/*******************************************/


double cRansacBasculementRigide::CostSol(const cSolBasculeRig & aSBR) const
{
   double aSom=0;
   for (int aK= 0 ; aK<int(mAvant.size()) ; aK++)
   {
       aSom += square_euclid(mApres[aK]-aSBR(mAvant[aK]));
   }
   return aSom;
}


void  cRansacBasculementRigide::TestNewSol(const cSolBasculeRig & aSBR) 
{
    double aCost = CostSol(aSBR);
    // std::cout << "SBR-COST " << aCost << "\n";
    if (aCost <mCostBestSol)
    {
       mCostBestSol = aCost;
       mBestSol = aSBR;
    }
}

double cRansacBasculementRigide::EstimLambda() const
{
    double aSAvant = 0;
    double aSApres = 0;
    for (int aK= 0 ; aK<int(mAvant.size()) ; aK++)
    {
        aSAvant += square_euclid(mAvant[aK]-mP0Avant);
        aSApres += square_euclid(mApres[aK]-mP0Apres);
    }
    ELISE_ASSERT(aSAvant!=0,"cRansacBasculementRigide::EstimLambda");
    return sqrt(aSApres/aSAvant);
}

const std::vector<Pt3dr>  & cRansacBasculementRigide::PAvant() const
{
   return mAvant;
}

const std::vector<Pt3dr> &  cRansacBasculementRigide::PApres() const
{
   return mApres;
}

const  std::vector<std::string> & cRansacBasculementRigide::Names()  const
{
   return mNames;
}


void cRansacBasculementRigide::AssertOpened() const
{
   ELISE_ASSERT(!mClosed,"cRansacBasculementRigide::AssertOpened");
}

void cRansacBasculementRigide::AssertKValide(int aK) const
{
   ELISE_ASSERT
   (
     (aK>=0) && (aK<int(mAvant.size())),
     "cRansacBasculementRigide::AssertKValide"
   );
}

bool  cRansacBasculementRigide::Close(bool aSvp)
{
   AssertOpened();
   if ( mAvant.size() < 3)
   {
       ELISE_ASSERT
       (
          aSvp,
          "Not enough samples (Min 3) in cRansacBasculementRigide"
       );
       return false;
   }
   mClosed = true;
   mLambda = EstimLambda();

   return true;
}

int   cRansacBasculementRigide::CurK() const
{
  return (int)(mAvant.size() - 1);
}

#define TheCostDef 1e50 

cRansacBasculementRigide::cRansacBasculementRigide(bool aUseV) :
   mClosed      (false),
   mUseV        (aUseV),
   mP0Avant     (0.0,0.0,0.0),
   mP0Apres     (0.0,0.0,0.0),
   mLambda      (-1),
   mCostBestSol (TheCostDef),
   mBestSol     (cSolBasculeRig::Id())
{
}

bool cRansacBasculementRigide::SolIsInit() const
{
   return mCostBestSol != TheCostDef;
}
void cRansacBasculementRigide::AssertSolInit() const
{
   ELISE_ASSERT(SolIsInit(),"cRansacBasculementRigide::AssertSolInit");
}


const cSolBasculeRig & cRansacBasculementRigide::BestSol() const
{
    AssertSolInit();
    return mBestSol;
}



cRansacBasculementRigide::~cRansacBasculementRigide()
{
}

void cRansacBasculementRigide::AddExemple
     (
            const Pt3dr & aAvant,
            const Pt3dr & aApres,
            const Pt3dr * aSpeedApres,
            const std::string & aName
     )
{
   ELISE_ASSERT((aSpeedApres!=0)==(mUseV),"Incoherence UseV cRansacBasculementRigide::AddExemple");
   AssertOpened();
   mAvant.push_back(aAvant);
   mApres.push_back(aApres);
   mNames.push_back(aName);
   mP0Avant  = mP0Avant + aAvant;
   mP0Apres  = mP0Apres + aApres;

   if (aSpeedApres)
   {
         mSpeedApres.push_back(*aSpeedApres);
   }
}

bool cRansacBasculementRigide::CloseWithTrOnK(int aK,bool aSvp)
{
  AssertKValide(aK);
  
  mP0Avant = mAvant[aK];
  mP0Apres = mApres[aK];

  return Close(aSvp);
}


bool cRansacBasculementRigide::CloseWithTrGlob(bool aSvp)
{
  mP0Avant = mP0Avant / double(mAvant.size());
  mP0Apres = mP0Apres / double(mApres.size());

  return Close(aSvp);
}


cSolBasculeRig    cRansacBasculementRigide::SolOfK1K2(int aK1,int aK2,bool & OkSol) const
{
   AssertKValide(aK1);
   AssertKValide(aK2);

   ELISE_ASSERT(aK1!=aK2,"cRansacBasculementRigide::SolOfK1K2");

   Pt3dr aV1Avant = mAvant[aK1] - mP0Avant;
   Pt3dr aV2Avant = mAvant[aK2] - mP0Avant;

   Pt3dr aV1Apres = mApres[aK1] - mP0Apres;
   Pt3dr aV2Apres = mApres[aK2] - mP0Apres;

   if ( 
            (euclid(aV1Avant^aV2Avant) < 1e-5)
        ||  (euclid(aV1Apres^aV2Apres) < 1e-5)
      )
   {
      OkSol = false;
      return cSolBasculeRig::Id();
   }

   OkSol = true;
   ElMatrix<REAL> aMat = ComplemRotation(aV1Avant,aV2Avant,aV1Apres,aV2Apres);

   return cSolBasculeRig(mP0Avant,mP0Apres,aMat,mLambda);
}

//
//   R Av + Tr = Apr + D V
//
//    C0 C1 C2  Av.x
//    C3 C4 C5  Av.y
//    C3 C4 C5  Av.z
//    

void Reset13(double *aCoef,int aL,Pt3dr aP,double aV)
{
     for (int aK=0 ; aK<13 ;aK++)
         aCoef[aK] = 0;

      aCoef[3*aL+0] = aP.x;
      aCoef[3*aL+1] = aP.y;
      aCoef[3*aL+2] = aP.z;
      aCoef[9+aL]   = 1;   // Tr
      aCoef[12]     = -aV;  // Vx Vy Vz
}
void cRansacBasculementRigide::EstimateDelay() 
{
    ELISE_ASSERT(mAvant.size() >=5,"Need at least 5 sample to use speed");

    L2SysSurResol aSys(13);
    aSys.GSSR_Reset(false);

     for (int aK=0 ; aK<int(mAvant.size()) ; aK++)
     {
         double aCoeff[13];

         Reset13(aCoeff,0,mAvant[aK],mSpeedApres[aK].x);
         aSys.GSSR_AddNewEquation(1.0,aCoeff,mApres[aK].x,0);

         Reset13(aCoeff,1,mAvant[aK],mSpeedApres[aK].y);
         aSys.GSSR_AddNewEquation(1.0,aCoeff,mApres[aK].y,0);

         Reset13(aCoeff,2,mAvant[aK],mSpeedApres[aK].z);
         aSys.GSSR_AddNewEquation(1.0,aCoeff,mApres[aK].z,0);

     }
    
     Im1D_REAL8  aSol = aSys.GSSR_Solve(0);
     mDelay = aSol.data()[12];

     mApresInit =  mApres;

     std::cout << "delay init :::    " << mDelay  << "\n";
     for (int aK=0 ; aK<int(mAvant.size()) ; aK++)
     {
         mApres[aK] =  mApres[aK] +  mSpeedApres[aK] * mDelay;
     }
}

double cRansacBasculementRigide::Delay() const
{
   ELISE_ASSERT(mUseV,"No vitesse for delay un  cRansacBasculementRigide");
   return mDelay;
}

void cRansacBasculementRigide::ExploreAllRansac(int aNbMax) 
{
    if (mUseV)
    {
       EstimateDelay();
    }

   int aNbCple = (mAvant.size() * (mAvant.size()-1)) / 2;

   cRandNParmiQ aSel(aNbMax,aNbCple);

   // std::cout << "cRandNParmiQcRandNParmiQ " << aNbMax << " " << aNbCple << "\n";


    for (int aK1=0 ; aK1<int(mAvant.size()) ; aK1++)
    {
       for (int aK2=aK1+1 ; aK2<int(mAvant.size()) ; aK2++)
       {
           if (aSel.GetNext())
           {
               bool Ok;
               cSolBasculeRig  aSBR= SolOfK1K2(aK1,aK2,Ok);
               // std::cout << "SBR K1-K2 " << aK1 << " " << aK2 << " " << Ok << " " << mCostBestSol << "\n";
               if (Ok)
               {
                  TestNewSol(aSBR);
               }
           }
           else
           {
           }
       }
    }
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
