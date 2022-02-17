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
#include "ext_stl/numeric.h"


static bool Debug = true;


void AssertNoNan(const double&  x,const int & aLine,const std::string & aFile)
{
    if (std_isnan(x))
    {
       std::cout << "At Line " << aLine << " of file " << aFile << "\n";
       ELISE_ASSERT(false,"Unexpected Nan Number")
    }
}

void AssertNoNan(const Pt2dr & aP ,const int & aLine,const std::string & aFile)
{
   AssertNoNan(aP.x,aLine,aFile);
   AssertNoNan(aP.y,aLine,aFile);
}

#define ASSERT_NO_NAN(anObj) AssertNoNan(anObj,__LINE__,__FILE__)

/***********************************************/
/*                                             */
/*              cElemGrapheIm                  */
/*                                             */
/***********************************************/

cElemGrapheIm::cElemGrapheIm() :
   mOk    (false),
   mNb    (0),
   mPMin  (1e10,1e10),
   mPMax  (-1e10,-1e10),
   mHom   (cElHomographie::Id())
{
}

void cElemGrapheIm::AddPt(const Pt2df & aPt1,const Pt2df & aPt2)
{
    mNb++;
    mPMin.SetInf(aPt1);
    mPMax.SetSup(aPt1);
    mInert1.add_pt_en_place(aPt1.x,aPt1.y);
}

const int      cElemGrapheIm::Nb() const      {return mNb;}
const Pt2df &  cElemGrapheIm::PMin() const    {return mPMin;}
const Pt2df &  cElemGrapheIm::PMax() const    {return mPMax;}
bool   cElemGrapheIm::Ok() const       {return mOk;}
const RMat_Inertie &  cElemGrapheIm::Inert1() const {return mInert1;}

void cElemGrapheIm::CloseOK()
{
    mOk = true;
    mInert1 = mInert1.normalize();

    mCdg1 =MatCdg(mInert1);

   Seg2d aSeg= seg_mean_square(mInert1);
   Pt2dr aU1 = aSeg.v01() ;
   Pt2dr aU2 = aU1 * Pt2dr(0,1);
   double aL1 = sqrt(ValQuad(mInert1,aU1));
   double aL2 = sqrt(ValQuad(mInert1,aU2));

    mUL1  = aU1 * aL1;
    mUL2  = aU2 * aL2;
}

void cElemGrapheIm::SetPackHom(const ElPackHomologue & aPckH)
{
    mHom = cElHomographie(aPckH,true);
}

  // aCpl.SetHomographie(cElHomographie(aPack,true));

Pt2dr cElemGrapheIm::FromCNorm(const Pt2dr & aP)
{
    return mCdg1 + mUL1 * aP.x + mUL2 * aP.y;
}

void  cElemGrapheIm::SetParamL1(double aK0,double aKx,double aKy)
{
    mK0 = aK0;
    mKx = aKx;
    mKy = aKy;
}


Pt2dr cElemGrapheIm::P1to2(const Pt2dr & aP)
{
   return mHom.Direct(aP);
}


const std::vector <Pt2dr> & TabNorm ()
{
   static std::vector<Pt2dr> aRes;
   if (aRes.empty())
   {
       Pt2dr aP;

       int aRay = 2;
       for (aP.x=-aRay ; aP.x <= aRay; aP.x++)
       {
          for (aP.y=-aRay ; aP.y <= aRay; aP.y++)
          {
              if (euclid(aP) <= aRay+0.1)
              {
                  aRes.push_back(aP);
              }
          }
       }
  
   }
   return aRes;
}

double  cElemGrapheIm::FactCorrec1to2(const Pt2dr & aP)
{
     return mK0 + mKx * aP.x + mKy * aP.y;
}




/***********************************************/
/*                                             */
/*              cER_ParamOneSys                */
/*                                             */
/***********************************************/

cER_ParamOneSys::cER_ParamOneSys(const std::vector<Pt2di> & aDeg) :
    mDeg    (aDeg),
    mNbVarTot (0)
{
   for (int aK=0 ; aK<int(mDeg.size()) ; aK++)
       mNbVarTot += NbVarOfDeg(mDeg[aK]);
}

int cER_ParamOneSys::NbVarOfDegI(int aDeg)
{
    return ((1+aDeg) * (2+aDeg)) / 2;
}

int  cER_ParamOneSys::NbVarOfDeg(Pt2di  aDeg)
{
  int aD1 = ElMax(aDeg.x,aDeg.y);
  int aD0 = ElMin(aDeg.x,aDeg.y);

  return NbVarOfDegI(aD1) - NbVarOfDegI(aD1-aD0-1);
}

/*
*/

int cER_ParamOneSys::NbVarTot() const
{
    return mNbVarTot;
}

int  cER_ParamOneSys::DegMaxRadiom() const
{
   return (int)mDeg.size();
}

Pt2di  cER_ParamOneSys::DegXYOfDegRadiom(int aDeg) const
{
   return mDeg.at(aDeg);
}

void  cER_ParamOneSys::write(ELISE_fp & aFP) const
{
  aFP.write(mDeg);
}

cER_ParamOneSys cER_ParamOneSys::read(ELISE_fp & aFP)
{
   return cER_ParamOneSys( aFP.read((std::vector<Pt2di> *)0));
}


/***********************************************/
/*                                             */
/*              cER_MesureOneIm                */
/*                                             */
/***********************************************/

cER_MesureOneIm::cER_MesureOneIm
(
    int aKIm,
    const Pt2df & aPt,
    const std::vector<float> &         aV
) :
  mKIm  (aKIm),
  mPtIm (aPt),
  mVal  (aV)
{
}




void cER_MesureOneIm::AddMoyVal(std::vector<double>  & aSomV,std::vector<double>  & aSom1)
{
   for (int aK=0 ; aK<int(mVal.size()); aK++)
   {
      aSomV[aK] += mVal[aK];
      aSom1[aK] ++;
   }
}

double cER_MesureOneIm::SomVal() const
{
   double aSom = 0;

   for (int aK=0 ; aK<int(mVal.size()); aK++)
       aSom += mVal[aK];

   return aSom;
}



void cER_MesureOneIm::write(ELISE_fp & aFp) const
{
   aFp.write_INT4(mKIm);
   aFp.write(mPtIm);
   aFp.write_INT4((int)mVal.size());

   for (int aK=0 ; aK<int(mVal.size()) ; aK++)
   {
       aFp.write_REAL4(mVal[aK]);
   }
}

cER_MesureOneIm cER_MesureOneIm::read(ELISE_fp & aFp)
{
    int aK =  aFp.read_INT4();
    Pt2df aPt; aFp.read(&aPt);

    std::vector<float> aVal;
    int aNb = aFp.read_INT4();
    for (int aK=0 ; aK<aNb; aK++)
        aVal.push_back(aFp.read_REAL4());
    //  float aVal = aFp.read_REAL4();

    return cER_MesureOneIm(aK,aPt,aVal);
}

int cER_MesureOneIm::NBV() const
{
   return (int)mVal.size();
}

int cER_MesureOneIm::KIm() const
{
   return mKIm;
}

const Pt2df & cER_MesureOneIm::Pt() const
{
   // return Pt2dr(mPtIm.x,mPtIm.y);
   return mPtIm;
}

Pt2dr cER_MesureOneIm::RPt() const
{
   return Pt2dr(mPtIm.x,mPtIm.y);
}


float cER_MesureOneIm::KthVal(int aK) const
{
   return mVal[aK];
}



/***********************************************/
/*                                             */
/*               cER_MesureNIm                 */
/*                                             */
/***********************************************/

cER_MesureNIm::cER_MesureNIm(eTypeERGMode aMode,const Pt3df& aPAbs,const Pt3df & aPNorm) :
   mMode   (eTypeERGMode(aMode)),
   mPAbs   (aPAbs),
   mPNorm  (aPNorm)
{
}

void cER_MesureNIm::AddAMD(cAMD_Interf * anAMD)
{
   for (int aK1=0 ; aK1<int(mMes.size()) ; aK1++)
   {
      for (int aK2=0 ; aK2<int(mMes.size()) ; aK2++)
      {
         anAMD->AddArc(mMes[aK1].KIm(),mMes[aK2].KIm());
      }
  }
}

void cER_MesureNIm::AddMoyVal(std::vector<double> & aSomV,std::vector<double> & aSom1)
{
   for (int aK=0; aK<int(mMes.size()) ; aK++)
   {
       mMes[aK].AddMoyVal(aSomV,aSom1);
   }
}


void cER_MesureNIm::AddMesure(const cER_MesureOneIm & aMes)
{
   mMes.push_back(aMes);
}

void cER_MesureNIm::write(ELISE_fp & aFp) const
{
   aFp.write_U_INT1(mMode);
   if (mMode >= eERG_2D)
   {
      aFp.write_REAL4(mPAbs.x);
      aFp.write_REAL4(mPAbs.y);
   }
   if (mMode >= eERG_3D)
   {
      aFp.write_REAL4(mPAbs.z);
   }
   if (mMode >= eERG_3DNorm)
   {
      aFp.write(mPNorm);
   }


   int aNb = (int)mMes.size();
   aFp.write_INT4(aNb);
   for (int aK=0 ; aK<aNb ; aK++)
       mMes[aK].write(aFp);
}

void cER_MesureNIm::read(cER_MesureNIm & aMNIm,ELISE_fp & aFp)
{
   aMNIm.mMode = eTypeERGMode(aFp.read_U_INT1());

   aMNIm.mPAbs = Pt3df(0,0,0);
   aMNIm.mPNorm = Pt3df(0,0,0);
   if (aMNIm.mMode >= eERG_2D)
   {
      aMNIm.mPAbs.x = aFp.read_REAL4();
      aMNIm.mPAbs.y = aFp.read_REAL4();
   }
   if (aMNIm.mMode >= eERG_3D)
   {
      aMNIm.mPAbs.z = aFp.read_REAL4();
   }
   if (aMNIm.mMode >= eERG_3DNorm)
   {
      aMNIm.mPNorm = aFp.read((Pt3df *)0);
   }
    int aNb= aFp.read_INT4();
    for (int aK=0; aK<aNb ; aK++)
       aMNIm.AddMesure(cER_MesureOneIm::read(aFp));
}


int cER_MesureNIm::NBV() const
{
   int aRes =0 ;

   for (int aK=0 ; aK<int(mMes.size()) ; aK++)
      aRes += mMes[aK].NBV();

   return aRes;
}

int cER_MesureNIm::NbMes() const
{
   return (int)mMes.size();
}


const cER_MesureOneIm & cER_MesureNIm::KthMes(int aK) const
{
   return mMes[aK];
}

Pt3dr  cER_MesureNIm::PAbs() const
{
    return Pt3dr(mPAbs.x,mPAbs.y,mPAbs.z);
}

/***********************************************/
/*                                             */
/*                cER_SolOnePower              */
/*                                             */
/***********************************************/
static double Scal(const std::vector<double> & aV1,const std::vector<double> & aV2)
{
   ELISE_ASSERT(aV1.size()==aV2.size(),"Incoh in Scal Vec Vec");
   double aRes=0;
   for (int aK=0 ; aK<int(aV1.size()) ; aK++)
       aRes += aV1[aK] * aV2[aK];

   return aRes;
}

cER_SolOnePower::cER_SolOnePower
(
        const cER_SolOneCh &  aSolCh,
        Pt2di aDeg,
        const std::vector<double> & aSol
) :
  mSolCh (aSolCh),
  mIm    (mSolCh.Im()),
  mNbDisc(2 + ElSquare(1+2*ElMax(aDeg.x,aDeg.y))),
  mGrid  (
             true,
             Pt2dr(-2,-2),
             Pt2dr(mIm.SzIm()+Pt2di(2,2)),
             Pt2dr( mIm.SzIm())/double(mNbDisc),
             "toto"
         )
{
   Pt2di aSzGr = mGrid.SzGrid();
   Pt2di aPGr;
   for (aPGr.x=0 ; aPGr.x<aSzGr.x ; aPGr.x++)
   {
      for (aPGr.y=0 ; aPGr.y<aSzGr.y ; aPGr.y++)
      {
           Pt2dr aPIm = mGrid.ToReal(Pt2dr(aPGr));
           Pt2dr aPN = mIm.ToPNorm(Pt2d<float>::RP2ToThisT(aPIm));
           std::vector<double> aCoeff;
           cER_SysResol::AddCoeff(1.0,aPN,aDeg,aCoeff);
           double aV = Scal(aSol,aCoeff);
           mGrid.SetValueGrid(aPGr,aV);
      }
   }
}

double cER_SolOnePower::Value(const Pt2dr & aP) const
{
   return mGrid.Value(aP);
}

#if (0)
#endif


/***********************************************/
/*                                             */
/*                cER_SolOneCh                 */
/*                                             */
/***********************************************/

cER_SolOneCh::cER_SolOneCh
(
    int aKCh,
    const cER_OneIm & anIm,
    Im1D_REAL8 aSolGlob
) :
   mIm (anIm)
{
   const cER_ParamOneSys & aParam = anIm.ERG()->ParamKCh(aKCh);
   cIncIntervale * anII = anIm.II(aKCh);

   int aD0 = anII->I0Alloc();
   for (int aDR=0 ; aDR<aParam.DegMaxRadiom() ; aDR++)
   {
       Pt2di aDXY = aParam.DegXYOfDegRadiom(aDR);
       int aD1 =  aD0 + cER_ParamOneSys::NbVarOfDeg(aDXY);

       std::vector<double> aVals;
       for (int aK=aD0 ; aK<aD1; aK++)
       {
           aVals.push_back(aSolGlob.At(aK));
           if (Debug)  std::cout << " " << aVals.back();
       }
       mSols.push_back(new cER_SolOnePower(*this,aDXY,aVals));



       aD0  = aD1;
   }
   ELISE_ASSERT(aD0==anII->I1Alloc(),"Incoh in  cER_SolOneCh::cER_SolOneCh");
}


const cER_OneIm & cER_SolOneCh::Im() const { return mIm; }

cER_SolOneCh::~cER_SolOneCh()
{
   DeleteAndClear(mSols);
}

double cER_SolOneCh::Value(const Pt2dr & aP,const std::vector<double> & aVV) const
{
    double aRes = 0;
    for (int aK=0 ; aK<int(aVV.size()) ; aK++)
    {
        aRes += mSols[aK]->Value(aP) * aVV[aK];
    }

    return aRes;
}

/***********************************************/
/*                                             */
/*                cER_OneIm                    */
/*                                             */
/***********************************************/

cER_OneIm::cER_OneIm
(
     cER_Global * anERG,
     int aKIm,
     Pt2di aSz,
     const std::string & aName
) :
   mERG   (anERG),
   mKIm   (aKIm),
   mSz    (aSz),
   mName  (aName),
   m2UseFCG (anERG->UseImForCG(aName)),
   mL1Init  (false),
   mL1K0    (0),
   mL1Kx    (0),
   mL1Ky    (0),
   mNbMesValidL1 (0)
{
   for (int aK=0 ;aK<mERG->NbCh() ; aK++)
   {
       int aNbVar = mERG->ParamKCh(aK).NbVarTot();
       mII.push_back(new cIncIntervale(mName,aNbVar*aKIm,aNbVar*(1+aKIm)));
       mSols.push_back(0);
       // mSB.push_back(new 
   }
}

void cER_OneIm::SetParamL1(double aK0,double aKx,double aKy)
{
   mL1Init = true;
   mL1K0   = aK0;
   mL1Kx   = aKx;
   mL1Ky   = aKy;
}

double  cER_OneIm::FactCorrectL1(const Pt2dr & aP) const
{
   if (! mL1Init) return 1.0;
   return exp(mL1K0 + mL1Kx*aP.x + mL1Ky * aP.y);
}


void cER_OneIm::SetSeuilL1Predict(double aSeuil)
{
    mSeuilPL1 = aSeuil;
}
double  cER_OneIm::SeuilPL1() const
{
   return mSeuilPL1;
}


const std::string & cER_OneIm::Name() const {return mName;}
const Pt2di & cER_OneIm::SzIm() const {return mSz;}
cIncIntervale * cER_OneIm::II(const int & aK) const { return mII.at(aK); }
cER_Global * cER_OneIm::ERG() const  {return mERG;}

bool cER_OneIm::UseForCompenseGlob() const
{
   return m2UseFCG;
}
 
Pt2dr   cER_OneIm::ToPNorm(const Pt2df & aPIm) const
{
   return Pt2dr
          (
              aPIm.x / double(mSz.x),
              aPIm.y / double(mSz.y)
          );
}

void cER_OneIm::AddMesure
     (
          cER_MesureNIm & aMNIm,
          const Pt2di& aPt, 
          const std::vector<float> & aV
     )
{
    ELISE_ASSERT
    (
         (int(aV.size()) == mERG->NbCh()),
         "Size Incoh in cER_OneIm::AddMesure"
    );
    aMNIm.AddMesure(cER_MesureOneIm(mKIm,Pt2d<float>::IP2ToThisT(aPt),aV));
}


void cER_OneIm::write(ELISE_fp & aFP) const
{
    aFP.write(mKIm);
    aFP.write(mSz);
    aFP.write(mName);
}

cER_OneIm * cER_OneIm::read(ELISE_fp & aFP,cER_Global * anERG) 
{
   int aKIm = aFP.read_INT4() ;
   Pt2di aPt = aFP.read((Pt2di  *)0);
   std::string aName = aFP.read((std::string *)0);

   cER_OneIm * aRes =  new cER_OneIm(anERG,aKIm,aPt,aName);

    return aRes;
}

inline int cER_OneIm::KIm() const
{
   return mKIm;
}

int & cER_OneIm::NbMesValidL1() {return mNbMesValidL1;}

void cER_OneIm::MakeVecVals
     (
            std::vector<double> & aVVals,
            const Pt2df & aPI,
            double aV0,
            const cER_ParamOneSys & aParam
     )
{
    aVVals.clear();
    double aPowV0=1.0;
    for (int aD=0 ; aD<aParam.DegMaxRadiom() ; aD++)
    {
        aVVals.push_back(aPowV0);
        aPowV0 *= aV0;
    }
}

int cER_OneIm::InitObs(const Pt2df & aPI,double aV0,const cER_ParamOneSys & aParam)
{
    Pt2dr aPN = ToPNorm(aPI);
    mCoeff.clear();
    int aRes=-1;
    std::vector<double> aVVals;
    MakeVecVals(aVVals,aPI,aV0,aParam);
 
    for (int aD=0 ; aD<aParam.DegMaxRadiom() ; aD++)
    {
        cER_SysResol::AddCoeff(aVVals[aD],aPN,aParam.DegXYOfDegRadiom(aD),mCoeff);
        if (aD==0)
        {
             aRes = (int)mCoeff.size();
        }
    }
    return aRes;
}

void cER_OneIm::PrevSingul
     (
          int aKR1XY0,
          double aPds,
          int aKCh,
          double aVal,
          cGenSysSurResol * aSys
     )
{
   std::vector<double> aVC = mCoeff;
   for (int aK1=0 ; aK1<int(mCoeff.size()); aK1++)
   {
       for (int aK2=0 ; aK2<int(mCoeff.size()); aK2++)
       {
            mCoeff[aK2] = (aK1==aK2) ? aVC[aK2] : 0;
       }
       AddRappelValInit(aPds,aKCh,(aK1==aKR1XY0) ? aVal : 0.0,aSys);
   }
   mCoeff = aVC;
}

void cER_OneIm::AddRappelValInit(double aPds,int aKCh,double aVal,cGenSysSurResol * aSys)
{
  std::vector<cSsBloc> aSBl;
  aSBl.push_back(mII[aKCh]->SsBlocComplet());
  std::vector<int> aVInd;

  aSys->GSSR_AddNewEquation_Indexe
  (
          &aSBl, &mCoeff[0], aSBl[0].Nb(),
          aVInd, aPds,(double *) 0 ,aVal,
          NullPCVU
  );

}

void cER_OneIm::PushCoeff(std::vector<double> & aV,double aSign)
{
   for (int aK=0; aK<int(mCoeff.size()) ; aK++)
      aV.push_back(aSign*mCoeff[aK]);
}

void cER_OneIm::AddEqValsEgal(double aPds,int aKCh,cER_OneIm &  anI2,cGenSysSurResol * aSys)
{
  std::vector<cSsBloc> aSBl;
  aSBl.push_back(mII[aKCh]->SsBlocComplet());
  aSBl.push_back(anI2.mII[aKCh]->SsBlocComplet());
  std::vector<int> aVInd;

  std::vector<double> aCoefs;
  PushCoeff(aCoefs,1.0);
  anI2.PushCoeff(aCoefs,-1.0);

  aSys->GSSR_AddNewEquation_Indexe
  (
          &aSBl, &aCoefs[0], aSBl[0].Nb(),
          aVInd, aPds,(double *) 0 ,0.0,
          NullPCVU
  );
}

void cER_OneIm::SetSol(int aKCh,Im1D_REAL8 aSolGlob)
{
    if (Debug)
    {
       std::cout << "     " << (m2UseFCG?"+":"-") << "NAME-IM : " << mName  << " SOLS " ;
    }
    delete mSols.at(aKCh);
    mSols.at(aKCh) = new cER_SolOneCh(aKCh,*this,aSolGlob);

    if (Debug)
    {
       std::cout << "\n";
    }
}



double  cER_OneIm::Value(const Pt2dr& aP,double anInput,int aKParam)
{
   std::vector<double> aVV;
   MakeVecVals(aVV,Pt2d<float>::RP2ToThisT(aP),anInput,mERG->ParamKCh(aKParam));
   return mSols[aKParam]->Value(aP,aVV);
}
/*
*/



void cER_OneIm::ValueLoc(const Pt2dr& aP,std::vector<double> & aRes,const std::vector<double> & aInPut)
{
   bool OneCh = (mSols.size() == 1);
   if (! OneCh)
   {
       ELISE_ASSERT(aInPut.size()==mSols.size(),"Size Incoh in cER_OneIm::Value");
   }
   //std::vector<double> aVV;
   aRes.clear();
   for (int aKRes=0 ; aKRes<int(aInPut.size())  ; aKRes++)
   {
       int aKParam  = OneCh ? 0 : aKRes;
       aRes.push_back(Value(aP,aInPut.at(aKRes),aKParam));

/*
       MakeVecVals(aVV,aP,aInPut.at(aKRes),mERG->ParamKCh(aKParam));
       aRes.push_back(mSols[aKParam]->Value(aP,aVV));
*/
   }

}

// double  cER_OneIm::Value(const Pt2dr& aP,double anInput,int aKParam)

double cER_OneIm::ValueGlobBrute
       (
            const Pt2dr& aP,
            double aVal,
            int aKParam,
            const Pt3dr& aPGlob
       )
{
   aVal = Value(aP, aVal,aKParam);
   aVal = mERG->ImG0()->Value(Pt2dr(aPGlob.x,aPGlob.y),aVal,aKParam);

   return aVal;
}


void cER_OneIm::ValueGlobBrute
     (
            const Pt2dr& aP,
            std::vector<double> & aRes,
            const std::vector<double> & aInPut,
            const Pt3dr& aPGlob
     )
{
   std::vector<double> aTmp;
   ValueLoc(aP,aTmp,aInPut);
   // aRes = aTmp;
   mERG->ImG0()->ValueLoc(Pt2dr(aPGlob.x,aPGlob.y),aRes,aTmp);
}

void cER_OneIm::ValueGlobCorr
     (
            const Pt2dr& aP,
            std::vector<double> & aRes,
            const std::vector<double> & aInPut,
            const Pt3dr& aPGlob
      )
{
    ValueGlobBrute(aP,aRes,aInPut,aPGlob);
    mERG->CorrecVal(aRes);
}

double  cER_OneIm::ValueGlobCorr
     (
            const Pt2dr& aP,
            double aVal ,
            int aKParam,
            const Pt3dr& aPGlob
      )
{
    aVal =  ValueGlobBrute(aP,aVal,aKParam,aPGlob);

    return mERG->CorrecVal(aVal,aKParam);
}



/***********************************************/
/*                                             */
/*                cER_SysResol                 */
/*                                             */
/***********************************************/

cER_SysResol::cER_SysResol
(
     const cER_ParamOneSys & aParam,
     cER_Global & anERG
) :
   mParam (aParam),
   mERG   (anERG),
   mSys   (0)
{
}

cER_SysResol::~cER_SysResol()
{
    delete mSys;
}

void cER_SysResol::Reset()
{
    mSys->GSSR_Reset(false);
}

cGenSysSurResol * cER_SysResol::Sys()
{
  return mSys;
}


void cER_SysResol::InitSys(const std::vector<int> & aRnk)
{
   if (mSys) return;

   ELISE_ASSERT(mBlocsIncAlloc.size()==aRnk.size(),"Incoh in cER_SysResol::InitSys");

   for (int aK=0 ; aK<int(mBlocsIncAlloc.size()) ; aK++)
   {
       mBlocsIncAlloc[aK]->SetOrder(aRnk[aK]);
   }

   mMOI.Init(mBlocsIncAlloc);
   if ( mERG.Show())
   {
       mL2Sys = new L2SysSurResol(mERG.NbIm() * mParam.NbVarTot());
       mSys = mL2Sys;
   }
   else
   {
       cElMatCreuseGen * aMatCr = cElMatCreuseGen::StdBlocSym(mMOI.BlocsIncSolve(),mMOI.I02NblSolve());
       mSys = new cFormQuadCreuse(mERG.NbIm() * mParam.NbVarTot() ,aMatCr);
   }

}

void cER_SysResol::AddCoeff
     (
         double aV0,
         Pt2dr  aP,
         Pt2di aDegreMax,
         std::vector<double> & aVC
     )
{

bool aTest = (aDegreMax.x!=aDegreMax.y);
aTest = false;
if (aTest)
{
std::cout << "V0 " << aV0 << " DM " << aDegreMax << "\n";
aV0=1.0;
aP = Pt2dr (2,10);
}

   if (aDegreMax.y>aDegreMax.x)
   {
       ElSwap(aDegreMax.y,aDegreMax.x);
       ElSwap(aP.y,aP.x);
   }
   double aVPowY = aV0;
   for (int aDY=0 ; aDY<= aDegreMax.y ; aDY++)
   {
        double aVPowXY = aVPowY;
        for (int aDX=aDY ; aDX<= aDegreMax.x ; aDX++)
        {
if (aTest)
std::cout << "aVPowXY " << aVPowXY << "\n";
            aVC.push_back(aVPowXY);
            aVPowXY *= aP.x;
        }

        aVPowY *= aP.y;
   }

if (aTest)
   getchar();


#if (0)
bool aTest = (aDegreMax.x!=aDegreMax.y);

if (aTest)
{
std::cout << "V0 " << aV0 << " DM " << aDegreMax << "\n";
aV0=1.0;
aP = Pt2dr (2,10);
}

   aVC.push_back(aV0);
if (aTest)
std::cout << aVC.back() << "\n";
   // 1
   // X Y
   // X2 XY Y2
   // ....
   if (aDegreMax.y>aDegreMax.x)
   {
       ElSwap(aDegreMax.y,aDegreMax.x);
       ElSwap(aP.y,aP.x);
   }

   int aKXn = 0;
   for (int aDegX=1 ; aDegX<= aDegreMax.x ; aDegX++)
   {
       int aNextaKXn = aVC.size();
       aVC.push_back(aVC[aKXn]*aP.x);


       int aNb = aVC.size();
       int aDegY = ElMin(aDegX,aDegreMax.y);


       int aK = aNb-aDegY;  // K est l'indexe du monome  X ^ aDeg-1
       aVC.push_back(aVC[aK]*aP.x);
std::cout << aVC.back() << "\n";
       for (; aK<aNb; aK++)
       {
          aVC.push_back(aVC[aK]*aP.y);
if (aTest)
std::cout << aVC.back() << "\n";
       }
   }
if (aTest)
   getchar();
#endif
}

void cER_SysResol::AddBlocInc(cIncIntervale * anII)
{
    mBlocsIncAlloc.push_back(anII);
}

Im1D_REAL8   cER_SysResol::GetSol()
{
    
     Im1D_REAL8  aSol = mSys->GSSR_Solve(0);;

     return mMOI.ReordonneSol(aSol);
}


/***********************************************/
/*                                             */
/*             cER_Global                      */
/*                                             */
/***********************************************/

static const int MAJIC = 145259383;
#define NUMV_Appar_ComputL1Cple 1000001
static const int NUMV= NUMV_Appar_ComputL1Cple;

cER_Global::cER_Global
(
    const std::vector<cER_ParamOneSys> & aParam,
    const std::vector<cER_ParamOneSys> & aParamGlob,
    Pt2di                                aSzGlob,
    const std::string &                  aPatAdjustGlob,
    bool                                 isTop,
    bool                                 isGlob,
    int                                 ComputL1Cple
) :
    mPatSelAdjGlob  (isTop ? new cElRegex(aPatAdjustGlob,10) : 0),
    mNamePSAG       (aPatAdjustGlob),
    mErgTop    (0),
    mErgG      (0),
    mIsTop     (isTop),
    mIsGlob    (isGlob),
    mSzG       (aSzGlob),
    mNumV      (NUMV),
    mParam     (aParam),
    mParamGlob (aParamGlob),
    mNbCh      ((int)aParam.size()),
    mImG0      (0),
    mMoy       (mNbCh),
    mNbEch     (mNbCh),
    mAMD       (0),
    mComputed  (! mIsTop),
    mComputL1Cple (ComputL1Cple),
    mPercCutAdjL1 (70.0),
    mShow         (false)
{
   for (int aKP=0 ; aKP<mNbCh ; aKP++)
        mSys.push_back(new cER_SysResol(mParam[aKP],*this));

   if (mIsTop)
   {
       mErgG = new cER_Global(aParamGlob,aParamGlob,aSzGlob,aPatAdjustGlob,false,true,false);
       mErgG->mErgTop = this;
   }
   
}

double & cER_Global::PercCutAdjL1()
{
   return mPercCutAdjL1;
}

bool & cER_Global::Show()
{
    return mShow;
}

void cER_Global::AssertComputed()
{
   ELISE_ASSERT(mComputed,"cER_Global::AssertComputed");
}

void cER_Global::Compute()
{
   if (mComputed)  return;
      
   std::cout << "Begin L1 computation\n";
   mComputed = true;

  if (mComputL1Cple)
     DoComputeL1Cple();

   std::cout << "END L1 computation\n";
}

void cER_Global::DoComputeL1Cple()
{
   mGrIm  = std::map<Pt2di,cElemGrapheIm>();
   for (std::list<cER_MesureNIm>::const_iterator itM= mMes.begin(); itM!=mMes.end(); itM++)
   {
        const cER_MesureNIm & aMN =  *itM;
        int aNbM = aMN.NbMes();

        for (int aKM1=0 ; aKM1<aNbM ; aKM1++)
        {
            for (int aKM2=aKM1+1 ; aKM2<aNbM ; aKM2++)
            {
                 const cER_MesureOneIm * aM1 = & (aMN.KthMes(aKM1));
                 const cER_MesureOneIm * aM2 = & (aMN.KthMes(aKM2));
                 if (aM1->KIm() > aM2->KIm())
                 {
                    ElSwap(aM1,aM2);
                 }
                 Pt2di anInd(aM1->KIm(),aM2->KIm());
                 mGrIm[anInd].AddPt(aM1->Pt(),aM2->Pt());
            }
        }
   }
    

   mNbCplOk = 0;
   mPdsTot = 0;
   
   int aNbRest = (int)mGrIm.size();
   for (std::map<Pt2di,cElemGrapheIm>::iterator itD =mGrIm.begin(); itD!=mGrIm.end() ; itD++)
   {
       cER_OneIm * aI1 = mVecIm[itD->first.x];
       cER_OneIm * aI2 = mVecIm[itD->first.y];
       cElemGrapheIm &  aCpl = itD->second;

       int aNb = aCpl.Nb();
       Pt2df aPMin = aCpl.PMin();
       Pt2df aPMax = aCpl.PMax();
       Pt2df aSz  =  aPMax-aPMin;

       Pt2di aSzIm = Sup(aI1->SzIm(),aI2->SzIm());

       // std::cout << " HHHH " << aNb << " " << (aSz.x>0.05*aSzIm.x) << " " << ( aSz.y>0.05*aSzIm.y)  << "\n";
       // if ((aNb > 200) && (aSz.x>0.1*aSzIm.x) && ( aSz.y>0.1*aSzIm.y))
       if ((aNb > 10) && (aSz.x>0.05*aSzIm.x) && ( aSz.y>0.05*aSzIm.y))
       {
          DoComputeL1Cple(aI1,aI2,aCpl);
       }
       aNbRest--;
       if ((aNbRest%10) == 0)
          std::cout <<  "     paire restante " << aNbRest << "\n";
   }

   //   Pour chaque image i on a une fonction Fi a trois param  Ki, Ai Bi
   //        Fi(x,y) = e(Ki+Aix + Biy)   t.q Ri Fi soit normalisee    RiFi = Rj Fj
   //
   //    On a les fonctions de transfert Tij calculees par moindre L1 precedentes telles que
   //   
   //         Ri(x,y) Tij(x,y) = Rj(x,y)
   //       
   //         Rj / Ri = Tij = Fi/Fj
   //      (1)    Log(Tij) = Ki + Ai x + BiY - Kj - Aj x - Bj y
   //      L'equation (1) doit etre verifiee pour les points du domaine de calcul de Tij,
   //      ici on prend 5 points definis par l'ellipse d'inertie
   //      Pour que le systeme soit defini, on rajoute une contrainte Sigma(Ki) = 0
   //      On rajoute aussi Sigma(Ai) = 0, Sigma(Bi) = 0 car sinon indet dans le cas habituel
   //      ou x1 et X2 sont en tranlation



   int aNbIm  = (int)mVecIm.size();
   int aNbInc = 3*aNbIm;
   Im1D_REAL8 aVec (aNbInc);
   double * aDataInc = aVec.data();
   SystLinSurResolu aSys(aNbInc,mNbCplOk * (int)(TabNorm().size() + 1));


   for (int aOffs=0 ; aOffs < 3 ; aOffs++)
   {
      aVec.raz();
      for (int aKIm = 0 ; aKIm < aNbIm ; aKIm++)
          aDataInc[3*aKIm+aOffs] = 1;
      aSys.PushEquation(aDataInc,0,mPdsTot);
   }



   for (std::map<Pt2di,cElemGrapheIm>::iterator itD =mGrIm.begin(); itD!=mGrIm.end() ; itD++)
   {
       cElemGrapheIm &  aCpl = itD->second;
       if (aCpl.Ok())
       {
            aVec.raz();
            int aIndIm1 = itD->first.x;
            int aIndIm2 = itD->first.y;
            //  R1 (K0 + Kx P1.x + Ky P2.y) = R2
            for (int aK = 0 ; aK<int(TabNorm().size()) ; aK++)
            {
                 Pt2dr aP1 = aCpl.FromCNorm(TabNorm()[aK]);
                 Pt2dr aP2 = aCpl.P1to2(aP1);
                 double aFact = aCpl.FactCorrec1to2(aP1); 
                 if ((aFact>0.1) &&  (aFact < 10.0))
                 {
                     //   ASSERT_NO_NAN(aP1);
                     //   ASSERT_NO_NAN(aP2);
                     //   ASSERT_NO_NAN(aFact);
                      aDataInc[3*aIndIm1]   =   1.0;
                      aDataInc[3*aIndIm1+1] =   aP1.x;
                      aDataInc[3*aIndIm1+2] =   aP1.y;
                      aDataInc[3*aIndIm2]   =  -1.0;
                      aDataInc[3*aIndIm2+1] =  -aP2.x;
                      aDataInc[3*aIndIm2+2] =  -aP2.y;

                      aSys.PushEquation(aDataInc,log(aFact),aCpl.Nb());



                      aDataInc[3*aIndIm1  ]   =0;
                      aDataInc[3*aIndIm1+1]   =0;
                      aDataInc[3*aIndIm1+2]   =0;
                      aDataInc[3*aIndIm2  ]   =0;
                      aDataInc[3*aIndIm2+1]   =0;
                      aDataInc[3*aIndIm2+2]   =0;
                 }
            }
       }
   }
   Im1D_REAL8  aSol = aSys.L1Solve();
   double aSK0 = 0;
   for (int aKIm = 0 ; aKIm < aNbIm ; aKIm++)
   {
        double aK0 = aSol.data()[3*aKIm];
        aSK0 += aK0;
        double aKx = aSol.data()[3*aKIm +1];
        double aKy = aSol.data()[3*aKIm +2];
         
        mVecIm[aKIm]->SetParamL1(aK0,aKx,aKy);
   }


   // Ca teste que les  FactCorrectL1 modelisent bien les  aCpl.FactCorrec1to2
   if (0)
   {
     for (std::map<Pt2di,cElemGrapheIm>::iterator itD =mGrIm.begin(); itD!=mGrIm.end() ; itD++)
     {
       cER_OneIm * aI1 = mVecIm[itD->first.x];
       cER_OneIm * aI2 = mVecIm[itD->first.y];
       cElemGrapheIm &  aCpl = itD->second;
       if (aCpl.Ok())
       {
          TestModelL1ByCple(aI1,aI2,aCpl);
       }
     }
     getchar();
   }

   mGotOnePbL1 = false;
   for (int aKIm=0 ; aKIm<int(mVecIm.size()) ; aKIm++)
   {
       MakeStatL1ByIm(mVecIm[aKIm]);
   }

   if (mGotOnePbL1)
   {
       std::cout << "=================LIST OF IMAGES WITH NO MEASURE====================\n";
       for (int aKIm=0 ; aKIm<int(mVecIm.size()) ; aKIm++)
       {
           if (mVecIm[aKIm]->NbMesValidL1() == 0)
           {
                   std::cout << "   No Valide Measures for " << mVecIm[aKIm]->Name() << "\n";
           }
       }
       ELISE_ASSERT(false,"There were images with no validated measures");
   }
}


double cER_Global::DifL1Normal(const cER_MesureOneIm * aM1,const cER_MesureOneIm * aM2) const
{
    Pt2dr  aP1 = aM1->RPt() ;
    double aR1 = aM1->SomVal();
    double aFact1 =  mVecIm[aM1->KIm()]->FactCorrectL1(aP1);

    Pt2dr  aP2 = aM2->RPt() ;
    double aR2 = aM2->SomVal();
    double aFact2 =  mVecIm[aM2->KIm()]->FactCorrectL1(aP2);


    double aL = sqrt(aFact1/aFact2);
    return  aR1*aL - aR2/aL;
}


void  cER_Global::MakeStatL1ByIm(cER_OneIm * anIm)
{
   static int aCpt=0; aCpt++;

   int aKIm = anIm->KIm();
   std::vector<double> aVDif;


   for (std::list<cER_MesureNIm>::const_iterator itM= mMes.begin(); itM!=mMes.end(); itM++)
   {
        const cER_MesureNIm & aMN =  *itM;
        int aNbM = aMN.NbMes();
 
        const cER_MesureOneIm * theM =0;
        for (int aKM=0 ; aKM<aNbM ; aKM++)
        {
            const cER_MesureOneIm * aM = & (aMN.KthMes(aKM));
            if (aM->KIm() == aKIm)
               theM = aM;
        }
        if (theM !=0)
        {

           for (int aKM=0 ; aKM<aNbM ; aKM++)
           {
              const cER_MesureOneIm * aM = & (aMN.KthMes(aKM));
              if (aM->KIm() != aKIm)
              {
                   double aDif = DifL1Normal(theM,aM);
                   aVDif.push_back(ElAbs(aDif));
              }
           }
        }
   }
   anIm->NbMesValidL1() = (int)aVDif.size();
   std::sort(aVDif.begin(),aVDif.end());

   std::cout << " L1 COMPUT  " <<  anIm->Name()  << " Nb Mes Valid " << aVDif.size() << " CPT " << aCpt << "\n";
   if (aVDif.size() == 0)
   {
       mGotOnePbL1 = true;
   }
   else
   {
      anIm->SetSeuilL1Predict( ValPercentile(aVDif,mPercCutAdjL1));
   }
}


void cER_Global::TestModelL1ByCple(cER_OneIm * aI1,cER_OneIm * aI2,cElemGrapheIm & aCpl)
{
   int aKI1 = aI1->KIm();
   int aKI2 = aI2->KIm();

   std::vector<double> aVDif;

   for (std::list<cER_MesureNIm>::const_iterator itM= mMes.begin(); itM!=mMes.end(); itM++)
   {
        const cER_MesureNIm & aMN =  *itM;
        int aNbM = aMN.NbMes();
 
        const cER_MesureOneIm * aM1 = 0;
        const cER_MesureOneIm * aM2 = 0;
        for (int aKM=0 ; aKM<aNbM ; aKM++)
        {
            const cER_MesureOneIm * aM = & (aMN.KthMes(aKM));
            if (aM->KIm() == aKI1)
               aM1 = aM;
            if (aM->KIm() == aKI2)
               aM2 = aM;
        }
        if (aM1 && aM2)
        {
            // double aR1 = aM1->SomVal();
            Pt2dr  aP1 = aM1->RPt() ;
            // double aR2 = aM2->SomVal();
            Pt2dr  aP2 = aM2->RPt() ;

            double aF1 =  aI1->FactCorrectL1(aP1);
            double aF2 =  aI2->FactCorrectL1(aP2);
            double aR12 =  aCpl.FactCorrec1to2(aP1);
            double aDif = ElAbs(aR12-aF1/aF2);

            aVDif.push_back(aDif);
            
        }
    }

   std::sort(aVDif.begin(),aVDif.end());

/*
   std::cout << "L1  [" << aI1->Name() << "," << aI2->Name() << "] " << aCpl.Nb() 
             << " MED " << ValPercentile(aVDif,50)
             << " V70 " << ValPercentile(aVDif,70)
             << " V90 " << ValPercentile(aVDif,90)
             << "\n";
*/

}




void cER_Global::DoComputeL1Cple(cER_OneIm * aI1,cER_OneIm * aI2,cElemGrapheIm & aCpl)
{
   mPdsTot += aCpl.Nb();
   mNbCplOk ++;
   aCpl.CloseOK();
   int aKI1 = aI1->KIm();
   int aKI2 = aI2->KIm();
   int aNbOk = 0;

   SystLinSurResolu aSys(3,aCpl.Nb());
   double aTab[3];

   std::vector<const cER_MesureOneIm *> aVM1;
   std::vector<const cER_MesureOneIm *> aVM2;

// bool Swap = true;
/*
   Pt2dr aCdg1 (MatCdg(aCpl.Inert1()));

   Seg2d aSeg= seg_mean_square(aCpl.Inert1());
   Pt2dr aU1 = aSeg.v01() ;
   Pt2dr aU2 = aU1 * Pt2dr(0,1);
   double aL1 = sqrt(ValQuad(aCpl.Inert1(),aU1));
   double aL2 = sqrt(ValQuad(aCpl.Inert1(),aU2));
*/

 
//   Pt2dr aCdg2 (MatCdg(aCpl.Inert2()));
// if (Swap)  ElSwap(aCdg1,aCdg2);

   ElPackHomologue aPack;
   for (std::list<cER_MesureNIm>::const_iterator itM= mMes.begin(); itM!=mMes.end(); itM++)
   {
        const cER_MesureNIm & aMN =  *itM;
        int aNbM = aMN.NbMes();
 
        const cER_MesureOneIm * aM1 = 0;
        const cER_MesureOneIm * aM2 = 0;
        for (int aKM=0 ; aKM<aNbM ; aKM++)
        {
            const cER_MesureOneIm * aM = & (aMN.KthMes(aKM));
            if (aM->KIm() == aKI1)
               aM1 = aM;
            if (aM->KIm() == aKI2)
               aM2 = aM;
        }
        if (aM1 && aM2)
        {
// if (Swap) ElSwap(aM1,aM2);
            //  R1 (K0 + K1 P1.x + K2 P2.y) = R2
            aVM1.push_back(aM1);
            aVM2.push_back(aM2);
            aNbOk++;
            double aR1 = aM1->SomVal();
            Pt2dr  aP1 = aM1->RPt() ;
            double aR2 = aM2->SomVal();

            aTab[0] =   aR1;
            aTab[1] =   aR1 * aP1.x;
            aTab[2] =   aR1 * aP1.y;

            aSys.PushEquation(aTab,aR2,1.0);
            aPack.Cple_Add(ElCplePtsHomologues(aM1->RPt(),aM2->RPt()));
        }
   }
   aCpl.SetPackHom(aPack);
   Im1D_REAL8  aSol = aSys.L1Solve();
   aCpl.SetParamL1(aSol.data()[0],aSol.data()[1],aSol.data()[2]);

   std::vector<double> aVDif;


   for (int aK=0 ; aK<int(aVM1.size()) ; aK++)
   {
       const cER_MesureOneIm * aM1 =  aVM1[aK];
       const cER_MesureOneIm * aM2 =  aVM2[aK];

       double aR1 = aM1->SomVal();
       Pt2dr  aP1 = aM1->RPt();
       double aR2 = aM2->SomVal();

       double aDif = aR2 - aR1 * aCpl.FactCorrec1to2(aP1);
       aVDif.push_back(ElAbs(aDif));
       // std::cout << "DIF " << aDif << " " << aR1 << " " << aR2 << "\n";
   }

   std::sort(aVDif.begin(),aVDif.end());

/*
   std::cout << "GrIm[" << aI1->Name() << "," << aI2->Name() << "] " << aCpl.Nb() 
             << " MED " << ValPercentile(aVDif,50)
             << " V70 " << ValPercentile(aVDif,70)
             << "\n";

             << " LAMBDA " << aL1 << "/" << aL2  
             << " SZ  " <<  aCpl.PMax() - aCpl.PMin()
             << " V60 " << ValPercentile(aVDif,60)
             << " V70 " << ValPercentile(aVDif,70)
             << " V80 " << ValPercentile(aVDif,80)
             << " V90 " << ValPercentile(aVDif,90)

             << " " << aSol.data()[0]
             << " " << aSol.data()[1]
             << " " << aSol.data()[2]
*/

}



cER_Global * cER_Global::Alloc
           (
               const std::vector<cER_ParamOneSys> & aParam,
               const std::vector<cER_ParamOneSys> & aParamGlob,
               Pt2di aSzGlob,
               const std::string & aPatAdjustGlob,
               bool                ComputL1Cple
           )
{
   return new cER_Global(aParam,aParamGlob,aSzGlob,aPatAdjustGlob,true,false,ComputL1Cple);
}

cER_Global::~cER_Global()
{
   delete mErgG;
   DeleteAndClear(mVecIm);
   DeleteAndClear(mSys);
}



int  cER_Global::NbIm() const
{
   return (int)mVecIm.size();
}



void  cER_Global::PrevSingul(cER_OneIm * anIm,int aKCh,double aPds)
{
     int aKR1XY0 = anIm->InitObs
                   (
                      Pt2d<float>((float)(anIm->SzIm().x/2.0),(float)(anIm->SzIm().y/2.0)),
                      RadMoy(aKCh),
                      mParam[aKCh]
                   );
      anIm->PrevSingul(aKR1XY0,aPds,aKCh,RadMoy(aKCh),mSys[aKCh]->Sys());
}


bool cER_Global::ValideMesure(const cER_MesureOneIm & aMes1,const cER_MesureOneIm & aMes2)
{
    if (!mComputL1Cple) 
       return true;

    double aDif =  ElAbs(DifL1Normal(&aMes1,&aMes2));
    double aSeuil1  = mVecIm[aMes1.KIm()]->SeuilPL1();
    double aSeuil2  = mVecIm[aMes2.KIm()]->SeuilPL1();


    ASSERT_NO_NAN(aDif);
    ASSERT_NO_NAN(aSeuil1);
    ASSERT_NO_NAN(aSeuil2);

    return    (2 * aDif) < (aSeuil1 + aSeuil2);
}

double aNbValide = 0;
double aNbTot = 0;

void cER_Global::AddOneObsTop
     (
         int aKCh,
         double aPdsValInit,
         const cER_MesureNIm & anERMNI
     )
{
    //  initialisation des coefficiente de polygone
    for (int aKM=0; aKM<anERMNI.NbMes() ; aKM++)
    {
        const cER_MesureOneIm & aMes = anERMNI.KthMes(aKM);
        cER_OneIm * anERI = mVecIm[aMes.KIm()];
        double aVal = aMes.KthVal(aKCh);
        anERI->InitObs(Pt2d<float>::RP2ToThisT(aMes.RPt()),aVal,mParam[aKCh]);

        anERI->AddRappelValInit(aPdsValInit,aKCh,aVal,mSys[aKCh]->Sys());
    }

    for (int aKM1=0; aKM1<anERMNI.NbMes() ; aKM1++)
    {
       const cER_MesureOneIm & aMes1 = anERMNI.KthMes(aKM1);
       cER_OneIm * anERI1 = mVecIm[aMes1.KIm()];
       for (int aKM2=aKM1 +1; aKM2<anERMNI.NbMes() ; aKM2++)
       {
           const cER_MesureOneIm & aMes2 = anERMNI.KthMes(aKM2);
aNbTot++;
           if (ValideMesure(aMes1,aMes2))
           {
aNbValide++;
              cER_OneIm * anERI2 = mVecIm[aMes2.KIm()];
              anERI1->AddEqValsEgal(1.0/anERMNI.NbMes(),aKCh,*anERI2,mSys[aKCh]->Sys());
           }
       }
    }

}

void cER_Global::AddOneObsGlob
     (
         int aKCh,
         double aPdsValInit,
         const cER_MesureNIm & anERMNI,
         cER_Global * CompensRapGlob
     )
{
    ELISE_ASSERT(mVecIm.size()==1,"cER_Global::AddOneObsGlob");
    cER_OneIm * anERI0 = mVecIm[0];
    Pt3dr aPAbs = anERMNI.PAbs();        
    Pt2dr aP2a(aPAbs.x,aPAbs.y);


    for (int aKM=0; aKM<anERMNI.NbMes() ; aKM++)
    {
        const cER_MesureOneIm & aMes = anERMNI.KthMes(aKM);
        cER_OneIm * anERI = mErgTop->mVecIm[aMes.KIm()];
        if (anERI->UseForCompenseGlob())
        {
            double aValInit = aMes.KthVal(aKCh);
            double aValCor  = anERI->Value(aMes.RPt(),aValInit,aKCh);

            double aVCible = aValInit;
            if (CompensRapGlob)
            {
                  cER_OneIm * anERIComp = CompensRapGlob->mVecIm[aMes.KIm()];
                  aVCible =   anERIComp->ValueGlobCorr(aMes.RPt(),aVCible,aKCh,aPAbs);
 
                  // std::cout << "XXAddOneObsGlob " << aVCible << " " << aValInit << "\n";
            }
/*
*/

// if (aKCh==0) std::cout << aValInit << " " << aValCor << "\n";
// double  cER_OneIm::Value(const Pt2dr& aP,double anInput,int aKParam)
            anERI0->InitObs(Pt2d<float>::RP2ToThisT(aP2a),aValCor,mParam[aKCh]);
            anERI0->AddRappelValInit(1.0,aKCh,aVCible,mSys[aKCh]->Sys());
        }
    }
}



void cER_Global::InitSys()
{
   if (mAMD) return;

   mAMD = new cAMD_Interf((int)mVecIm.size());
   for (std::list<cER_MesureNIm>::iterator itM=mMes.begin();itM!=mMes.end();itM++)
   {
       itM-> AddAMD(mAMD);
   }
   for (int aKIm=0 ; aKIm<int(mVecIm.size()) ; aKIm++)
   {
      mAMD->AddArc(aKIm,aKIm);
   }
   mRnk = mAMD->DoRank();

   for (std::list<cER_MesureNIm>::iterator itM=mMes.begin();itM!=mMes.end();itM++)
       itM->AddMoyVal(mMoy,mNbEch);

}


void cER_Global::OneItereOneChSys(int aKCh,double aPdsValInit,double aPdsSingul,cER_Global * CompensRapGlob)
{
    InitSys();
    mSys[aKCh]->InitSys(mRnk);
    mSys[aKCh]->Reset();

    if (aPdsSingul)
    {
       for (int aKIm=0 ; aKIm<int(mVecIm.size()) ; aKIm++)
       {
            PrevSingul(mVecIm[aKIm],aKCh,aPdsSingul);
       }
    }

    const std::list<cER_MesureNIm> & aLMes = mIsTop ?
                                             mMes   :
                                             mErgTop->mMes;
    for 
    (
       std::list<cER_MesureNIm>::const_iterator itM=aLMes.begin();
       itM!=aLMes.end();
       itM++
    )
    {
       if (mIsTop)
       {
          AddOneObsTop(aKCh,aPdsValInit,*itM);
       }
       if (mIsGlob)
       {
          AddOneObsGlob(aKCh,aPdsValInit,*itM,CompensRapGlob);
       }
    }

    Im1D_REAL8  aSol  =  mSys[aKCh]->GetSol();

    if (Debug) 
       std::cout << " -- cER_Global::OneItereOneChSys NbInc " << aSol.tx() << " KCH " <<  aKCh << "\n";
    for (int aKIm=0 ; aKIm<int(mVecIm.size()) ; aKIm++)
    {
        mVecIm[aKIm]->SetSol(aKCh,aSol);
    }
    if (Debug) 
       std::cout << " OK SET SOL \n";

}


void cER_Global::OneItereSys(double aPdsInit,double aPdsSingul,cER_Global * CompensRG)
{
    for (int aKCh=0 ; aKCh<mNbCh ; aKCh++)
    {
       std::cout << "DO CANAL " << aKCh << "\n";
       OneItereOneChSys(aKCh,aPdsInit,aPdsSingul,CompensRG);
    }
}


void  cER_Global::SolveSys(double aPdsInit,double aPdsSingul,cER_Global * CompensRG)
{
    if (CompensRG && mIsTop)
    {
        ELISE_ASSERT(mVecIm.size()==CompensRG->mVecIm.size(),"Size Coherence in cER_Global::SolveSys");
        for (int aKIm=0 ; aKIm<int(mVecIm.size()) ; aKIm++)
        {
            const std::string & aN1 =  mVecIm[aKIm]->Name() ; 
            const std::string & aN2 =  CompensRG->mVecIm[aKIm]->Name() ; 
            ELISE_ASSERT(aN1==aN2,"Name Coherence in cER_Global::SolveSys");
        }
    }


    AssertComputed();
    OneItereSys(aPdsInit,aPdsSingul,CompensRG);
    if (mIsTop)
    {
        mImG0 = mErgG->AddIm("Global",mSzG);

        mErgG->mMoy = mMoy;
        mErgG->mNbEch = mNbEch;
        mErgG->SolveSys(aPdsInit,aPdsSingul,CompensRG);


        mSomStat =  std::vector<double>(mNbCh,0);
        mMoyAv   =  std::vector<double>(mNbCh,0);
        mMoyApr  =  std::vector<double>(mNbCh,0);
        mMoyGrad =  std::vector<double>(mNbCh,0);

        for 
        (
            std::list<cER_MesureNIm>::const_iterator itM = mMes.begin();
            itM != mMes.end();
            itM++
        )
        {
             AddStat(*itM);
        }
        for (int aKC=0 ; aKC<mNbCh ; aKC++)
        {
            mMoyAv[aKC]    /= mSomStat[aKC];
            mMoyApr[aKC]   /= mSomStat[aKC];
            mMoyGrad[aKC]  /= mSomStat[aKC];
            mMoyGrad[aKC]  = 1/mMoyGrad[aKC];
            mMoyAv[aKC]  = mMoyAv[aKC] - mMoyApr[aKC]* mMoyGrad[aKC];
        }

        // mImG0->ModifStat(mMoyAv,mMoyApr,mMoyGrad);
    }
}

double cER_Global::CorrecVal(double aV,int aKC)
{
   return mMoyAv[aKC] + aV *  mMoyGrad[aKC];
}


void  cER_Global::CorrecVal(std::vector<double> & aV)
{
  for (int aKC=0 ; aKC<mNbCh ; aKC++)
  {
      aV[aKC] = CorrecVal(aV[aKC],aKC);
      // aV[aKC] =mMoyAv[aKC] + aV[aKC] *  mMoyGrad[aKC];
  }
}

void cER_Global::AddStat(const cER_MesureNIm & aMesNIm)
{
    Pt3dr aPG = aMesNIm.PAbs();
    for (int aKM=0 ; aKM<aMesNIm.NbMes() ; aKM++)
    {
         const cER_MesureOneIm &  aMes = aMesNIm.KthMes(aKM);
         Pt2dr aP2 = aMes.RPt();
         std::vector<double> aV0Av;
         std::vector<double> aV1Av;
         for (int aKC=0 ; aKC<mNbCh ; aKC++)
         {
             aV0Av.push_back(aMes.KthVal(aKC));
             aV1Av.push_back(aV0Av.back()+1.0);
         }
         cER_OneIm * anERI = mVecIm[aMes.KIm()];
         std::vector<double> aV0Apr;
         std::vector<double> aV1Apr;
         anERI->ValueGlobBrute(aP2,aV0Apr,aV0Av,aPG);
         anERI->ValueGlobBrute(aP2,aV1Apr,aV1Av,aPG);

         for (int aKC=0 ; aKC<mNbCh ; aKC++)
         {
             mSomStat[aKC]  += 1;
             mMoyAv[aKC]    += aV0Av[aKC];
             mMoyApr[aKC]   += aV0Apr[aKC];
             mMoyGrad[aKC]  += aV1Apr[aKC]-aV0Apr[aKC];
         }
    }
}


bool  cER_Global::UseImForCG(const std::string & aName) const
{
  return  (mPatSelAdjGlob==0) || (mPatSelAdjGlob->Match(aName));
}



cER_OneIm * cER_Global::AddIm(const std::string & aName,const Pt2di & aSz)
{
   cER_OneIm * anIm = new cER_OneIm
                          (
                              this,
                              (int)mVecIm.size(),
                              aSz,
                              aName
                          );

    ELISE_ASSERT
    (
        mDicoIm[aName]==0,
        "Name confict in cER_Global::AddIm"
    );
    AddIm(anIm);

    return anIm;
}

void cER_Global::AddIm(cER_OneIm * anIm)
{
    mDicoIm[anIm->Name()]= anIm;
    mVecIm.push_back(anIm);
    for (int aKCh=0; aKCh<mNbCh ; aKCh++)
    {
       mSys[aKCh]->AddBlocInc(anIm->II(aKCh));
    }
}

cER_MesureNIm & cER_Global::NewMesure2DPure()
{
    mMes.push_back(cER_MesureNIm(eERG_2D_Pure,Pt3df(0,0,0),Pt3df(0,0,0)));
    return mMes.back();
}

cER_MesureNIm & cER_Global::NewMesure2DGlob(const Pt2dr & aP)
{
    mMes.push_back(cER_MesureNIm(eERG_2D_Pure,Pt3df((float)aP.x,(float)aP.y,0.f),Pt3df(0.f,0.f,0.f)));
    return mMes.back();
}


cER_OneIm * cER_Global::ImG0()
{
   return mImG0;
}

/*
void cER_Global::SuprLastMesure()
{
   ELISE_ASSERT(!mMes.empty(),"cER_Global::SuprLastMesure");
   mMes.pop_back();
}
*/


void cER_Global::write(ELISE_fp & aFP) const
{
    aFP.write_INT4(MAJIC);
    aFP.write_INT4(mNumV);
    aFP.write_INT4(mComputL1Cple);
    aFP.write(mSzG);
    aFP.write(mNamePSAG);
    
    aFP.write_INT4(mNbCh);
    for (int aKCh=0 ; aKCh<mNbCh ; aKCh++)
    {
        mParam[aKCh].write(aFP);
        mParamGlob[aKCh].write(aFP);
    }

    aFP.write_INT4((INT4)mVecIm.size());
    for (int aK=0 ; aK<int(mVecIm.size()) ; aK++)
    {
          mVecIm[aK]->write(aFP);
    }

    aFP.write_INT4((INT4)mMes.size());
    for 
    (
        std::list<cER_MesureNIm>::const_iterator itM=mMes.begin();
        itM!=mMes.end();
        itM++
    )
    {
       itM->write(aFP);
    }
}

void cER_Global::write(const std::string & aName) const
{
   ELISE_fp aFP(aName.c_str(),ELISE_fp::WRITE);
   write(aFP);
   aFP.close();
}

const cER_ParamOneSys & cER_Global::ParamKCh(int aK) const
{
  return mParam.at(aK);
}

const int & cER_Global::NbCh() const
{
   return mNbCh;
}

cER_Global * cER_Global::read(ELISE_fp & aFP)
{
   int aMaj =  aFP.read_INT4();
   ELISE_ASSERT(aMaj==MAJIC,"Uncorrect majic number for cER_Global"); 
   int aNumV  = aFP.read_INT4();

   int aComputL1Cple = 0;
   if (aNumV >= NUMV_Appar_ComputL1Cple)
      aComputL1Cple = aFP.read_INT4();

   Pt2di aSzGlob = aFP.read((Pt2di *)0);
   std::string  aNamePSAG = aFP.read((std::string *)0);


   int aNbCh =  aFP.read_INT4();
   std::vector<cER_ParamOneSys> aVPOS;
   std::vector<cER_ParamOneSys> aVGlob;

   for (int aKCh=0 ; aKCh<aNbCh ; aKCh++)
   {
       aVPOS.push_back(cER_ParamOneSys::read(aFP));
       aVGlob.push_back(cER_ParamOneSys::read(aFP));
   }
   // cER_ParamOneSys aPOS = cER_ParamOneSys::read(aFP);



   cER_Global * anERG = cER_Global::Alloc( aVPOS, aVGlob, aSzGlob, aNamePSAG, (aComputL1Cple!=0) );
   anERG->mNumV = aNumV;

   int aNbIm =  aFP.read_INT4();
   for (int aK=0; aK<aNbIm ; aK++)
   {
        cER_OneIm * anI = cER_OneIm::read(aFP,anERG);
        anERG->AddIm(anI);
   }

   int aNbMes =  aFP.read_INT4();
   for (int aK=0; aK<aNbMes ; aK++)
   {
      cER_MesureNIm aMes(eERG_2D_Pure,Pt3df(0,0,0),Pt3df(0,0,0));
      anERG->mMes.push_back(aMes);
      cER_MesureNIm::read(anERG->mMes.back(),aFP);
   }
   

    anERG->Compute();
    
   return anERG;
}


cER_Global *  cER_Global::read(const std::string & aName)
{
   ELISE_fp aFP(aName.c_str(),ELISE_fp::READ);
   cER_Global * aRes = read(aFP);
   aFP.close();

   return aRes;
}

int cER_Global::NBV() const
{
   int aRes = 0;
   for 
   (
       std::list<cER_MesureNIm>::const_iterator itM=mMes.begin();
       itM!=mMes.end();
       itM++
   )
   {
      aRes += itM->NBV();
   }

   return aRes;
}

double cER_Global::RadMoy(int aKCh) const
{
   return mMoy.at(aKCh) / mNbEch.at(aKCh);
}


void cER_Global::Show1() const
{
   std::cout << "NB IM " << mVecIm.size() << "\n";
   std::cout << " IM " << mVecIm.front()->Name() << " " <<  mVecIm.back()->Name() << "\n";
   std::cout << "  NBV = " << NBV() << "\n";
/*
   for 
   (
       std::list<cER_MesureNIm>::const_iterator itM=mMes.begin();
       itM!=mMes.end();
       itM++
   )
   {
      std::cout << "  -- " << itM->NBV() << "\n";
   }
*/
   std::cout << " Nb Mes " << mMes.size() << "\n";
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe �  
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
