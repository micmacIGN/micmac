#include "SymbDer/SymbDer_Common.h"
#include "MMVII_PCSens.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Geom2D.h"
// #include <set>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 CamStenope * Std_Cal_From_File
             (
                 const std::string & aNameFile,
                 const std::string &  aNameTag = "CalibrationInternConique"
             );
*/

/**
   \file cCentralPerspCam.cpp

   \brief implementation  of functionnality for intrincic calibration of 
*/

using namespace NS_SymbolicDerivative;

namespace MMVII
{


/* ******************************************************* */
/*                                                         */
/*             cDataPerspCamIntrCalib                      */
/*                                                         */
/* ******************************************************* */

cDataPerspCamIntrCalib:: cDataPerspCamIntrCalib() :
    mTypeProj       (eProjPC::eStenope), // need an init for AddData input
    mCSPerfect      (1.0,cPt2dr(0,0)),
    mDataPixDomain  (cPt2di(1,1))
{}

cDataPerspCamIntrCalib:: cDataPerspCamIntrCalib
(
     const std::string & aName,
     eProjPC        aTypeProj,           ///< type of projection 
     const cPt3di & aDegDir,             ///< degrees of distorstion  Rad/Dec/Univ
     const std::vector<double>&  aVParams,  ///< vector of constants, or void
     const cCalibStenPerfect & aCSP,           ///< Calib w/o dist
     const  cDataPixelDomain  & aDataPixDomain,              ///< sz, domaine of validity in pixel
     const cPt3di & aDegPseudoInv,       ///< degree of inverse approx by least square
     int aSzBuf                          ///< sz of buffers in computation
)  :
    mName            (aName),
    mTypeProj        (aTypeProj),
    mDir_Degr        (aDegDir),
    mDir_VDesc       (DescDist(aDegDir)),
    mVTmpCopyParams  (aVParams),
    mCSPerfect       (aCSP),
    mDataPixDomain   (aDataPixDomain),
    mInv_Degr        (aDegPseudoInv),
    mSzBuf           (aSzBuf)
{
    // correct vect param, when first use, parameter can be empty meaning all 0  
    if (mVTmpCopyParams.size() != mDir_VDesc.size())
    {
       MMVII_INTERNAL_ASSERT_strong(mVTmpCopyParams.empty(),"cPerspCamIntrCalib Bad size for params");
       mVTmpCopyParams.resize(mDir_VDesc.size(),0.0);
    }
}

cDataPerspCamIntrCalib::cDataPerspCamIntrCalib
(
    const std::string & aName,
    eProjPC aTypeProj,
    const cPt3di & aDeg,
    double aFoc,
    cPt2di & aNbPix,
    int aSzBuf
) :
    cDataPerspCamIntrCalib
    (
	         aName,
	         aTypeProj, 
	         aDeg,
	         std::vector<double>(),
                 cCalibStenPerfect(aFoc,ToR(aNbPix)/2.0),
                 cDataPixelDomain(aNbPix),
	         aDeg,
	         aSzBuf
    )
{
}

    
void cDataPerspCamIntrCalib::AddData(const cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("Name",anAux),mName);
    mDataPixDomain.AddData(cAuxAr2007("PixelFrame",anAux));

    {
        MMVII::EnumAddData(anAux,mTypeProj,"Projection");
	    /*  MODIF MPD, has "rediscover" the "EnumAddData"  function ...
        std::string aNameProj= E2Str(mTypeProj);
        MMVII::AddData(cAuxAr2007("Projection",anAux),aNameProj);
        if (anAux.Input())
	   mTypeProj = Str2E<eProjPC>(aNameProj);
	   */
    }
    {
           cAuxAr2007 aAuxAux("Auxiliary",anAux);
           MMVII::AddData(cAuxAr2007("SzBuffer",aAuxAux),mSzBuf);
           MMVII::AddData(cAuxAr2007("Degree",aAuxAux),mDir_Degr);
           MMVII::AddData(cAuxAr2007("DegreeApproxInv",aAuxAux),mInv_Degr);
    }
    if (anAux.Input())
    {
       mDir_VDesc = DescDist(mDir_Degr);
       mVTmpCopyParams.resize(mDir_VDesc.size());
    }

    {
           cAuxAr2007 aAuxSten("PerfectProj",anAux);
           MMVII::AddData(cAuxAr2007("F",aAuxSten),mCSPerfect.F());
           MMVII::AddData(cAuxAr2007("PP",aAuxSten),mCSPerfect.PP());
    }

    std::vector<std::string>  aTypeDist={"Radial","Decentric","Polynomial"};
    {
       cAuxAr2007 aAuDist("Distorsion",anAux);
       for (int aKTypeDist=0 ; aKTypeDist<3 ; aKTypeDist++)
       {
           cAuxAr2007 aAuxTypeDist(aTypeDist.at(aKTypeDist),aAuDist);
	   for (size_t aKD=0 ; aKD<mDir_VDesc.size() ; aKD++)
	   {
               // tricky make assumption on int equiv : eRad, eDecX, eDecY, eMonX, eMonY,
               //   eRad(0) =>0  ::   eDecX(1), eDecY(2) => 1   ::   eMonX(3), eMonY(4) => 2
               bool DoAtThisStep = ((int(mDir_VDesc[aKD].mType)+1)/2)==aKTypeDist;
	       if (DoAtThisStep)
	       {
                   MMVII::AddData(cAuxAr2007(mDir_VDesc[aKD].mName,aAuxTypeDist),mVTmpCopyParams.at(aKD));
		   AddComment(anAux.Ar(),mDir_VDesc[aKD].mLongName);
	       }
           }
       }
    }

    MMVII::AddData(cAuxAr2007("Informations",anAux),mVectInfo);

    if (!anAux.Input())
        mVTmpCopyParams.clear();
}

void cDataPerspCamIntrCalib::PushInformation(const std::string & anInfo)
{
   mVectInfo.push_back(anInfo);
}
std::vector<std::string> & cDataPerspCamIntrCalib::VecInfo() {return mVectInfo;}



/* ******************************************************* */
/*                                                         */
/*                 cPerspCamIntrCalib                      */
/*                                                         */
/* ******************************************************* */


	// constructor destuctor  "big" modifier

	
cPerspCamIntrCalib::cPerspCamIntrCalib(const cDataPerspCamIntrCalib & aData) :
    cDataPerspCamIntrCalib (aData),
    mDefProj            (cDefProjPerspC::ProjOfType(mTypeProj)),
    mPixDomain          (&mDataPixDomain),
	// ------------ direct -------------
    mDir_Proj           ( new  cDataMapCalcSymbDer<tREAL8,3,2>
                               (
                                    EqCPProjDir(mTypeProj,false,mSzBuf),   // equatio, w/o derivative
                                    EqCPProjDir(mTypeProj,true,mSzBuf),    // equation with derivatives
			            std::vector<double>(),                 // parameters, empty here
			            true                                   // equations are "adopted" (i.e will be deleted in destuctor)
                               )
                        ),
    mDir_Dist           (NewMapOfDist(mDir_Degr,mVTmpCopyParams,mSzBuf)),
	// ------------ inverse -------------
    mInv_CSP            (mCSPerfect.MapInverse()),
    mPhgrDomain         (new cDataMappedBoundedSet<tREAL8,2>(&mPixDomain,&mInv_CSP,false,false)),
    mInv_VDesc          (DescDist(mInv_Degr)),
    mInv_Params         (mInv_VDesc.size(),0.0),
    mInvApproxLSQ_Dist  (NewMapOfDist(mInv_Degr,mInv_Params,mSzBuf)),
    mInv_BaseFDist      (EqBaseFuncDist(mInv_Degr,mSzBuf)),
    mInv_CalcLSQ        (new cLeastSqCompMapCalcSymb<tREAL8,2,2>(mInv_BaseFDist)),
    mThresholdPhgrAccInv (1e-9),
    mThresholdPixAccInv  (mThresholdPhgrAccInv * F()),
    mNbIterInv           (10),
    mThreshJacPI         (0.5),
    mDist_DirInvertible  (new cDataIIMFromMap<tREAL8,2> (mDir_Dist,mInvApproxLSQ_Dist,mThresholdPhgrAccInv,mNbIterInv,false,false)),
    mInv_Proj            ( new  cDataMapCalcSymbDer<tREAL8,2,3>
                               (
                                    EqCPProjInv(mTypeProj,false,mSzBuf),   // equatio, w/o derivative
                                    EqCPProjInv(mTypeProj,true,mSzBuf),    // equation with derivatives
			            std::vector<double>(),                 // parameters, empty here
			            true                                   // equations are "adopted" (i.e will be deleted in destuctor)
                               )
		         ),
    mInvIsUpToDate       (false)
{
    mVTmpCopyParams.clear();
}


cPerspCamIntrCalib * cPerspCamIntrCalib::Alloc(const cDataPerspCamIntrCalib & aData)
{
     cPerspCamIntrCalib * aRes =  new cPerspCamIntrCalib(aData);

     return aRes;
}

	//  ==================  read/write 2 files  ====================

void AddData(const cAuxAr2007 & anAux,cDataPerspCamIntrCalib & aPCIC)
{
    aPCIC.AddData(anAux);
}

void  cPerspCamIntrCalib::ToFile(const std::string & aNameFile ) const
{
    //  make a local copy to have a own for constness
    mVTmpCopyParams = VParamDist();
    SaveInFile(static_cast<const cDataPerspCamIntrCalib&>(*this),aNameFile);
}

void  cPerspCamIntrCalib::ToFileIfFirstime(const std::string & aNameFile ) const
{
     MMVII::ToFileIfFirstime(*this,aNameFile);
}


cPerspCamIntrCalib * cPerspCamIntrCalib::FromFile(const std::string & aName)
{
    return RemanentObjectFromFile<cPerspCamIntrCalib,cDataPerspCamIntrCalib>(aName);
}

std::string cPerspCamIntrCalib::PrefixName() {return "Calib-" + cSensorCamPC::PrefixName() + "-";}

	//  ==================  destuctor  "big" modifier ====================

cPerspCamIntrCalib::~cPerspCamIntrCalib()
{
     delete mPhgrDomain;	
     delete mDir_Dist;
     delete mDir_Proj;

     delete mInvApproxLSQ_Dist;
     delete mInv_BaseFDist;
     delete mInv_CalcLSQ;
     delete mDist_DirInvertible;
     delete mInv_Proj;
}

void cPerspCamIntrCalib::UpdateLSQDistInv()
{
    // allocate obect, just need to be done once
    /*
    if (mInvApproxLSQ_Dist==nullptr)
    {
        mInvApproxLSQ_Dist  = NewMapOfDist(mInv_Degr,mInv_Params,mSzBuf);
        mInv_BaseFDist = EqBaseFuncDist(mInv_Degr,mSzBuf);
        mInv_CalcLSQ   = new cLeastSqCompMapCalcSymb<tREAL8,2,2>(mInv_BaseFDist);
	mDist_DirInvertible = new   cDataIIMFromMap<tREAL8,2> (mDir_Dist,mInvApproxLSQ_Dist,mThresholdPhgrAccInv,mNbIterInv,false,false);
    }
    */
	/*
    if (mDist_DirInvertible==nullptr)
    {
	 mDist_DirInvertible = new   cDataIIMFromMap<tREAL8,2> (mDir_Dist,mInvApproxLSQ_Dist,mThresholdPhgrAccInv,mNbIterInv,false,false);
    }
    */

    // create structure for map inversion

    //  Approximate number of point (in the biggest size), here +or- less square of minimum
    //  but also take care of having always a minimum, else generate problem in pixelization during growing
    size_t aNbMin = std::max(size_t(10),2+mInv_VDesc.size());

    cComputeMapInverse<double,2> aCMI
    (
       mThreshJacPI,         ///< Threshold on jacobian to ensure inversability
       PtSeedInv(),          ///< Seed point, in input space
       aNbMin,    ///< Approximate number of point (in the biggest size), here +or- less square of minimum
       (*mPhgrDomain),       ///< Set of validity, in output space
       (*mDir_Dist),         ///< Maping to invert : InputSpace -> OutputSpace
        mInv_CalcLSQ,        ///< Structure for computing the invert on base of function using least square
       false                 ///< Not in  Test
   );
   aCMI.DoAll(mInv_Params); // compute the parameters
   mInvApproxLSQ_Dist->SetObs(mInv_Params); // set these parameters in approx inverse

   mInvIsUpToDate = true;
}

void cPerspCamIntrCalib::UpdateLSQDistIfRequired() const
{
     if (! mInvIsUpToDate)
     {
         const_cast<cPerspCamIntrCalib*>(this)->UpdateLSQDistInv();
     }
}

/*
 *         Pi             Dist          Perf
 *    P3  ----->  PProj ------> PDist   -----> Pixel = middle of sensor
 */

cPt2dr cPerspCamIntrCalib::PtSeedInv() const
{
     cPt2dr aPMiddle = ToR(SzPix()) / 2.0;
     cPt2dr  aPDist = mInv_CSP.Value(aPMiddle);

     cMappingIdentity<tREAL8,2> aMapId;
     cDataIIMFromMap<tREAL8,2>  aDRI(mDir_Dist,&aMapId,1e-2/F(),100,false,false);

     cPt2dr  aPProj = aDRI.Inverse(aPDist);
     return aPProj;
}


std::vector<cPt2dr>  cPerspCamIntrCalib::PtsSampledOnSensor(int aNbByDim) const
{

	/*
     cPt2dr aPMil = ToR(SzPix()) / 2.0;
     cPt2dr  aP1 = mInv_CSP.Value(aPMil);
     cMappingIdentity<tREAL8,2> aMapId;
     cDataIIMFromMap<tREAL8,2>  aDRI(mDir_Dist,&aMapId,1e-2/F(),100,false,false);

     cPt2dr  aP2 = aDRI.Inverse(aP1);
     cPt2dr  aP3 =  mDir_Dist->Value(aP2);
     // cPt2dr aP2 = aP1;
     StdOut() <<  "SSSSSS " << aPMil  << aP1<< aP2 <<  aP3 << "\n";
     */


    // UpdateLSQDistIfRequired();

    cComputeMapInverse<double,2> aCMI
    (
       mThreshJacPI,         ///< Threshold on jacobian to ensure inversability
       PtSeedInv(),          ///< Seed point, in input space
       aNbByDim,    ///< Approximate number of point (in the biggest size), here +or- less square of minimum
       (*mPhgrDomain),       ///< Set of validity, in output space
       (*mDir_Dist),         ///< Maping to invert : InputSpace -> OutputSpace
        nullptr,
       false                 ///< Not in  Test
   );
   aCMI.DoPts();

    std::vector<cPt2dr> aRes = aCMI.GetPtsOut();
    aRes = static_cast<const cDataInvertibleMapping<tREAL8,2>&>(mCSPerfect).Values(aRes);

    return aRes;
}



	     // ==================  geometric manips =====================

const  std::vector<cPt2dr> &  cPerspCamIntrCalib::Values(tVecOut & aV3 ,const tVecIn & aV0 ) const 
{
     static tVecOut aV1,aV2;
     mDir_Proj->Values(aV1,aV0);
     mDir_Dist->Values(aV2,aV1);
     mCSPerfect.Values(aV3,aV2);
     
     return aV3;
}


double cPerspCamIntrCalib::VisibilityOnImFrame(const cPt2dr & aP) const
{
   return mPixDomain.Insideness(aP);
}


double cPerspCamIntrCalib::Visibility(const cPt3dr & aP) const
{
     double MaxCalc = 100.0;

     UpdateLSQDistIfRequired();
     
     cPt2dr aPphgr = mDir_Proj->Value(aP);
     cPt2dr aPDist  = mDir_Dist->Value(aPphgr);

     //  For domain where dist is inversible this should be sufficient
     cPt2dr aPIm   = mCSPerfect.Value(aPDist);
     double aRes1 = mPixDomain.InsidenessWithBox(aPIm);
     // dont want to do inversion too far it may overflow ...
     if (aRes1<-MaxCalc)
        return aRes1;

     // If the point come faraway because dist non invertible, we must refute it
     cPt2dr aPPhgrBack = mDist_DirInvertible->Inverse(aPDist);
     // multiply by focal to have pixels, subr
     double aRes2  = 1e-2 - Norm2(aPphgr-aPPhgrBack) * F();

     // if we are inside,  aRes2~0 and Res1 is meaningfull for insideness
     if ((aRes1>=0) && (aRes2>=0))  
         return aRes1;

     return std::min(aRes1,aRes2);
}


const  std::vector<cPt3dr> &  cPerspCamIntrCalib::Inverses(tVecIn & aV3 ,const tVecOut & aV0 ) const 
{
     UpdateLSQDistIfRequired();

     static tVecOut aV1,aV2;
     mInv_CSP.Values(aV1,aV0);
     mDist_DirInvertible->Inverses(aV2,aV1);
     mInv_Proj->Values(aV3,aV2);
     
     return aV3;
}

cPt3dr  cPerspCamIntrCalib::Inverse(const tPtOut & aPt) const 
{
     std::vector<tPtIn>  aVecIn;
     std::vector<tPtOut> aVecOut{aPt};
     const  std::vector<tPtIn> & aVRes = Inverses(aVecIn,aVecOut);

     cPt3dr aRes =  aVRes.at(0);
     return aRes;
}

tREAL8  cPerspCamIntrCalib::InvProjIsDef(const tPtOut & aPix ) const
{
    return mDefProj.P2DIsDef(mDist_DirInvertible->Inverse(mInv_CSP.Value(aPix)));
}

      //   ----  object in unknown system (bundle adj ...) ----------------
     
void cPerspCamIntrCalib::UpdateCSP() 
{
    mInv_CSP       = mCSPerfect.MapInverse();
}
void cPerspCamIntrCalib::OnUpdate() 
{
   // The inverst for dist and csp must be recomputed
    mInv_CSP       = mCSPerfect.MapInverse();

    // if we are here, great proba modif has been done, so force Update
    UpdateLSQDistInv  ();
}

void cPerspCamIntrCalib::PutUknowsInSetInterval() 
{
    // Unknown have escpaped and will probably be modified
    mInvIsUpToDate = false;

    mSetInterv->AddOneInterv(mCSPerfect.F());
    mSetInterv->AddOneInterv(mCSPerfect.PP());
    mSetInterv->AddOneInterv(VParamDist());
}

cCalculator<double> * cPerspCamIntrCalib::EqColinearity(bool WithDerives,int aSzBuf)
{
    return EqColinearityCamPPC(mTypeProj,mDir_Degr,WithDerives,aSzBuf);
}
      //   ----  Accessor  to distorsion ----------------

const std::vector<double> & cPerspCamIntrCalib::VParamDist() const 
{
    return mDir_Dist->VObs(); 
}

std::vector<double> & cPerspCamIntrCalib::VParamDist() 
{ 
    mInvIsUpToDate = false;  // caution, but now param are exfiltred ..
    return mDir_Dist->VObs(); 
}

const   std::vector<cDescOneFuncDist> &  cPerspCamIntrCalib::VDescDist() const { return this->mDir_VDesc; }

int cPerspCamIntrCalib::IndParamDistFromName(const std::string& aName,bool SVP) const
{
    for (size_t aK=0 ; aK<this->mDir_VDesc.size() ; aK++)
        if (this->mDir_VDesc[aK].mName == aName)
		return aK;
    if (! SVP)
    {
        MMVII_UnclasseUsEr("Invalid  distorsion param :"+aName);
    }
    return -1;
}

double  cPerspCamIntrCalib::ParamDist(const std::string & aName) const 
{
   return VParamDist().at(IndParamDistFromName(aName));
}
void  cPerspCamIntrCalib::SetParamDist(const std::string & aName,const double&aVal) 
{
   mInvIsUpToDate = false;
   VParamDist().at(IndParamDistFromName(aName)) = aVal;
}
bool    cPerspCamIntrCalib::IsNameParamDist(const std::string & aName) const 
{
    return VParamDist().at(IndParamDistFromName(aName,true)) >=0;
}



      //   ----  Accessor  -  modifier ----------------

void cPerspCamIntrCalib::SetThresholdPhgrAccInv(double aThr)
{
    mThresholdPhgrAccInv = aThr;
    mThresholdPixAccInv = aThr * F();

    mDist_DirInvertible->SetDTolInv(mThresholdPhgrAccInv);
}

void cPerspCamIntrCalib::SetThresholdPixAccInv(double aThr)
{
     SetThresholdPhgrAccInv(aThr/F());
}

const std::string & cPerspCamIntrCalib::Name()   const {return mName ;}
const cPt3di & cPerspCamIntrCalib::DegDir() const {return mDir_Degr;}

const double & cPerspCamIntrCalib::F()      const {return mCSPerfect.F() ;}
const cPt2dr & cPerspCamIntrCalib::PP()     const {return mCSPerfect.PP();}

const cPt2di & cPerspCamIntrCalib::SzPix() const {return mPixDomain.Sz();}

const cDataMapping<tREAL8,3,2>* cPerspCamIntrCalib::Dir_Proj() const { return mDir_Proj;}
const cDataMapping<tREAL8,2,2>* cPerspCamIntrCalib::Dir_Dist() const { return mDir_Dist;}

const cDataMapping<tREAL8,2,3>* cPerspCamIntrCalib::Inv_Proj() const
{
    // UpdateLSQDistIfRequired();

    return mInv_Proj;
}

const cDataInvertibleMapping<tREAL8,2>* cPerspCamIntrCalib::Dir_DistInvertible() const
{
    UpdateLSQDistIfRequired();

    return mDist_DirInvertible;
}



       /* ================================================================== */
       /*                 BENCH VERIF PART                                   */
       /* ================================================================== */

void cPerspCamIntrCalib::TestInvInit(double aTolApprox,double aTolAccurate)
{
	// test inversion for the distorsion
     {
         // generate 2d-point ine photogram coordinate , after distorsion
         double aRhoMax =  mPhgrDomain->Box().DistMax2Corners(cPt2dr(0,0));
         std::vector<cPt2dr>  aVPt1;
         mPhgrDomain->GridPointInsideAtStep(aVPt1,aRhoMax/10.0);

	 //  undist them by approx-lsq invers
         std::vector<cPt2dr>  aVPt2; // undist
         mInvApproxLSQ_Dist->Values(aVPt2,aVPt1);

	 // distord them back, should have  aVPt3 ~ aVPt1
         std::vector<cPt2dr>  aVPt3;
         mDir_Dist->Values(aVPt3,aVPt2);

	 //  undist them more accurately with predictive + iterative
         std::vector<cPt2dr>  aVPt4; // undist
         mDist_DirInvertible->Inverses(aVPt4,aVPt1);

	 // distord them back, should have  aVPt5 ~ aVPt1 (more accurateky)
         std::vector<cPt2dr>  aVPt5;
         mDir_Dist->Values(aVPt5,aVPt4);

         double aSD12=0;  // som dist  V1/V2 to have a referebce
         double aSD23=0;  // som dist  V1/V2 to have another referebce
         double aSD13=0;  //  V1/V3 should be low
         double aSD15=0;  //  V1/V5 should be very low
         for (size_t aKPt=0 ; aKPt<aVPt1.size() ; aKPt++)
         {
		 //  add all that, use square dist for efficiency
              aSD12 +=  SqN2(aVPt1.at(aKPt)-aVPt2.at(aKPt));
              aSD23 +=  SqN2(aVPt2.at(aKPt)-aVPt3.at(aKPt));
              aSD13 +=  SqN2(aVPt1.at(aKPt)-aVPt3.at(aKPt));
              aSD15 +=  SqN2(aVPt1.at(aKPt)-aVPt5.at(aKPt));

         }
	     // transform sum of square dist  an averager of distance
         aSD12 = std::sqrt(aSD12/aVPt1.size());
         aSD23 = std::sqrt(aSD23/aVPt1.size());
         aSD13 = std::sqrt(aSD13/aVPt1.size());
         aSD15 = std::sqrt(aSD15/aVPt1.size());

         MMVII_INTERNAL_ASSERT_bench((aSD13/aSD12<aTolApprox),"Test approx inv");
         MMVII_INTERNAL_ASSERT_bench((aSD15/aSD12<aTolAccurate),"Test approx inv");
     }

     // test global inversion
     {
         // generate 2D point on grid
         std::vector<cPt2dr>  aVPt0;
         mPixDomain.GridPointInsideAtStep(aVPt0,Norm2(mDataPixDomain.Sz())/20.0);

	 // filter them because witj some projection point to far are not invetrible
         std::vector<cPt2dr>  aVPt1;
	 for (const auto & aPt0 : aVPt0)
             if (InvProjIsDef(aPt0) >= 1e-2)
                aVPt1.push_back(aPt0);

	 // compute direction of bundles
         std::vector<cPt3dr>  aVPt2;
	 Inverses(aVPt2,aVPt1);

	 // back project on images
         std::vector<cPt2dr>  aVPt3;
	 Values(aVPt3,aVPt2);

         // aSD13 -> som dist between initial points and their back proj
         double aSD13=0;  
         for (size_t aKPt=0 ; aKPt<aVPt1.size() ; aKPt++)
         {
              double aD =  SqN2(aVPt1.at(aKPt)-aVPt3.at(aKPt));
	      MMVII_INTERNAL_ASSERT_tiny(ValidFloatValue(aD),"Bad value in TestInvInit");
              aSD13 += aD;
	 }

         aSD13 = std::sqrt(aSD13/aVPt1.size())  / mCSPerfect.F()  ;
         MMVII_INTERNAL_ASSERT_bench(aSD13<1e-8,"Test approx inv");
     }
}

void cPerspCamIntrCalib::InitRandom(double aAmpl)
{
     double aRhoMax =  mPhgrDomain->Box().DistMax2Corners(cPt2dr(0,0));

     cRandInvertibleDist  aParamRID ( mDir_Degr, aRhoMax, RandUnif_0_1(), aAmpl);
     mDir_Dist->SetObs(aParamRID.VParam());
     UpdateLSQDistInv ();
}
 

void BenchCentralePerspective(cParamExeBench & aParam,eProjPC aTypeProj)
{
    tREAL8 aDiag = 1000 * (1+10*RandUnif_0_1());
    cPt2di aSz (aDiag*(1+RandUnif_0_1()),aDiag*(1+RandUnif_0_1()));
    cPt2dr aPP(   aSz.x()*(0.5+0.1*RandUnif_C())  , aSz.y()*(0.5+0.1*RandUnif_C())  );
    tREAL8  aFoc =  aDiag * (0.2 + 3.0*RandUnif_0_1());

    for (int aK=0 ; aK<2 ; aK++)
    {
       cPerspCamIntrCalib* aCam = cPerspCamIntrCalib::Alloc
                                  (
	                                 cDataPerspCamIntrCalib
	                                 (
	                                       "BenchCam",
                                                aTypeProj,
	                                        (aK==0) ? cPt3di(2,0,0) : cPt3di(3,1,1),
	                                        std::vector<double>(),
	                                        cCalibStenPerfect(aFoc,aPP),
	                                        cDataPixelDomain(aSz),
	                                        (aK==0) ? cPt3di(5,1,1) :cPt3di(7,2,5),
	                                        100
	                                 )
                                  );
       aCam->SetThresholdPhgrAccInv(1e-9);

       aCam->InitRandom(0.1);
       /// aCam->UpdateLSQDistIfRequired();

       aCam->TestInvInit((aK==0) ? 1e-3 : 1e-2, 1e-4);

       delete aCam;
    }
}


void BenchCentralePerspective(cParamExeBench & aParam)
{
    if (! aParam.NewBench("CentralPersp")) return;

    cCalibStenPerfect aCS(1,cPt2dr(0,0));
    // in fact this is not necessary , btw maintain just in case and see if the test fail
    MMVII_INTERNAL_ASSERT_bench(&(aCS.F())+1 == &(aCS.PP().x()) ,"Assertion cCalibStenPerfect memory model");
    MMVII_INTERNAL_ASSERT_bench(&(aCS.F())+2 == &(aCS.PP().y()) ,"Assertion cCalibStenPerfect memory model");

    BenchCentralePerspective_ImportV1(aParam);


    int aNbTime = std::min(20,3+aParam.Level());
    for (int aTime=0 ; aTime<aNbTime ; aTime++)
    {
        for (int aKEnum=0 ; aKEnum<int(eProjPC::eNbVals) ; aKEnum++)
        {
            BenchCentralePerspective(aParam,eProjPC(aKEnum));
        }
    }



    aParam.EndBench();
}



/* ******************************************************* */
/*                                                         */
/*                 cCalibStenPerfect                       */
/*                                                         */
/* ******************************************************* */

cCalibStenPerfect::cCalibStenPerfect(tScal aFoc,const tPt & aPP) :
    mF   (aFoc),
    mPP  (aPP)
{
}
cCalibStenPerfect::cCalibStenPerfect(const cCalibStenPerfect & aCS) :
    cCalibStenPerfect(aCS.mF,aCS.mPP)
{
}


cCalibStenPerfect cCalibStenPerfect::MapInverse() const
{
    //  aQ= PP+ aP * F  ;  aP = (aQ-PP) /aF
    return  cCalibStenPerfect(  1.0/mF  ,  -mPP/mF  );
}

const double & cCalibStenPerfect::F()  const {return mF ;}
const cPt2dr & cCalibStenPerfect::PP() const {return mPP;}
double & cCalibStenPerfect::F()  {return mF ;}
cPt2dr & cCalibStenPerfect::PP() {return mPP;}

const  typename cCalibStenPerfect::tVecPt &  cCalibStenPerfect::Values(tVecPt & aVOut,const tVecPt & aVIn) const
{
     const size_t aNbIn = aVIn.size();
     aVOut.resize(aNbIn);

#ifdef _OPENMP
#pragma omp parallel for
#endif
     for (size_t aK=0; aK < aNbIn; aK++) 
     {
	     aVOut[aK] = mPP + aVIn[aK] * mF;
     }
     return aVOut;
}

const  typename cCalibStenPerfect::tVecPt &  cCalibStenPerfect::Inverses(tVecPt & aVOut,const tVecPt & aVIn) const
{
     const size_t aNbIn = aVIn.size();
     aVOut.resize(aNbIn);

#ifdef _OPENMP
#pragma omp parallel for
#endif
     for (size_t aK=0; aK < aNbIn; aK++) 
     {
	     aVOut[aK] = (aVIn[aK] - mPP) / mF;
     }
     return aVOut;
}




}; // MMVII

