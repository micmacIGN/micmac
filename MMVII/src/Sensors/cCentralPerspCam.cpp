#include "SymbDer/SymbDer_Common.h"
#include "MMVII_PCSens.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Geom2D.h"
// #include <set>

#ifdef _OPENMP
#include <omp.h>
#endif


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
    mMapPProj2Im    (1.0,cPt2dr(0,0)),
    mDataPixDomain  (cPt2di(1,1))
{}

cDataPerspCamIntrCalib:: cDataPerspCamIntrCalib
(
     const std::string & aName,
     eProjPC        aTypeProj,           ///< type of projection 
     const cPt3di & aDegDir,             ///< degrees of distorstion  Rad/Dec/Univ
     const std::vector<double>&  aVParams,  ///< vector of constants, or void
     const cMapPProj2Im & aCSP,           ///< Calib w/o dist
     const  cDataPixelDomain  & aDataPixDomain,              ///< sz, domaine of validity in pixel
     const cPt3di & aDegPseudoInv,       ///< degree of inverse approx by least square
     int aSzBuf,                          ///< sz of buffers in computation
     bool  isFraserModel
)  :
    mName            (aName),
    mIsFraserMode    (isFraserModel),
    mTypeProj        (aTypeProj),
    mDir_Degr        (aDegDir),
    mDir_VDesc       (DescDist(aDegDir,mIsFraserMode)),
    mVTmpCopyParams  (aVParams),
    mMapPProj2Im       (aCSP),
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
    const cPt2di & aNbPix,
    bool           PPIsRel,
    const cPt2dr & aPP,
    int aSzBuf,
    bool isFraserModel
) :
    cDataPerspCamIntrCalib
    (
	         aName,
	         aTypeProj, 
	         aDeg,
	         std::vector<double>(),
                 cMapPProj2Im(aFoc, PPIsRel ? MulCByC(ToR(aNbPix), aPP) : aPP),
                 cDataPixelDomain(aNbPix),
	         aDeg,
	         aSzBuf,
                 isFraserModel
    )
{

}

    
void cDataPerspCamIntrCalib::AddData(const cAuxAr2007 & anAux0)
{
    cAuxAr2007 anAux("InternalCalibration",anAux0);
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
           MMVII::AddData(cAuxAr2007("IsFraserMode",aAuxAux),mIsFraserMode);
    }
    if (anAux.Input())
    {
       mDir_VDesc = DescDist(mDir_Degr,mIsFraserMode);
       mVTmpCopyParams.resize(mDir_VDesc.size());
    }

    {
           cAuxAr2007 aAuxSten("PerfectProj",anAux);
           MMVII::AddData(cAuxAr2007("F",aAuxSten),mMapPProj2Im.F());
           MMVII::AddData(cAuxAr2007("PP",aAuxSten),mMapPProj2Im.PP());

	   // Just in case redo a coherent object
	   if (anAux.Input())
              mMapPProj2Im = cMapPProj2Im(mMapPProj2Im.F(),mMapPProj2Im.PP());
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
		   anAux.Ar().AddComment(mDir_VDesc[aKD].mLongName);
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
    mVoidDist           (mDir_Degr==cPt3di(0,0,0)),
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
    mDir_Dist           (NewMapOfDist(mDir_Degr,mVTmpCopyParams,mSzBuf,mIsFraserMode)),
	// ------------ inverse -------------
    mMapIm2PProj            (mMapPProj2Im.MapInverse()),
    mPhgrDomain         (new cDataMappedBoundedSet<tREAL8,2>(&mPixDomain,&mMapIm2PProj,false,false)),
    mInv_VDesc          (DescDist(mInv_Degr,mIsFraserMode)),
    mInv_Params         (mInv_VDesc.size(),0.0),
    mInvApproxLSQ_Dist  (NewMapOfDist(mInv_Degr,mInv_Params,mSzBuf,mIsFraserMode)),
    mInv_BaseFDist      (EqBaseFuncDist(mInv_Degr,mSzBuf,mIsFraserMode)),
    mInv_CalcLSQ        (mVoidDist ? nullptr : new cLeastSqCompMapCalcSymb<tREAL8,2,2>(mInv_BaseFDist)),
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
    //  make a local copy to have a own for constness, BtW Param dist are own by distorsion
    mVTmpCopyParams = VParamDist();
    SaveInFile(static_cast<const cDataPerspCamIntrCalib&>(*this),aNameFile);
}

void  cPerspCamIntrCalib::ToFileIfFirstime(const std::string & aNameFile ) const
{
     MMVII::ToFileIfFirstime(*this,aNameFile);
}


cPerspCamIntrCalib * cPerspCamIntrCalib::FromFile(const std::string & aName,bool Remanent)
{
    cPerspCamIntrCalib * aCalib = nullptr;
    if (Remanent) 
       aCalib = RemanentObjectFromFile<cPerspCamIntrCalib,cDataPerspCamIntrCalib>(aName);
    else
       aCalib =  ObjectFromFile<cPerspCamIntrCalib,cDataPerspCamIntrCalib>(aName);

    std::string aNameWithFile = LastPrefix( FileOfPath(aName)) ;
    if (aNameWithFile != aCalib->Name())
    {
        StdOut() << "File:" << aNameWithFile << " Tag:" << aCalib->Name() << "\n";
        StdOut() << "Fullname:" << aName << "\n";
	MMVII_UnclasseUsEr("Name of calib differs between file and tags");
    }

    return aCalib;
}

std::string cPerspCamIntrCalib::PrefixName() {return "Calib-" + cSensorCamPC::PrefixName() + "-";}
std::string cPerspCamIntrCalib::SharedCalibPrefixName() {return "CalibIntr" ;}

	//  ==================  destuctor  "big" modifier ====================

cPerspCamIntrCalib::~cPerspCamIntrCalib()
{
     delete mDefProj;
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

    if (!mVoidDist)
    {
         size_t aNbMin = std::max(size_t(10),2+mInv_VDesc.size());
         std::unique_ptr<cComputeMapInverse<double,2>> aPCMI  
         (
               cComputeMapInverse<double,2>::Alloc
               (
                    mThreshJacPI,         ///< Threshold on jacobian to ensure inversability
                    PtSeedInv(),          ///< Seed point, in input space
                    aNbMin,    ///< Approximate number of point (in the biggest size), here +or- less square of minimum
                    mDefProj,
                    (*mPhgrDomain),       ///< Set of validity, in output space
                    (*mDir_Dist),         ///< Maping to invert : InputSpace -> OutputSpace
                     mInv_CalcLSQ,        ///< Structure for computing the invert on base of function using least square
                    false                 ///< Not in  Test
              )
         );
          aPCMI->DoAll(mInv_Params); // compute the parameters

       mInvApproxLSQ_Dist->SetObs(mInv_Params); // set these parameters in approx inverse
    }

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
// static int aCpt=0 ; aCpt++;
     /**  Regarding the seed :
      *      - if we put at PP we have problem if PP is outside image (like with shift lense camera)
      *      - if we put it in midle with have problem with orthographik camera if midle is outside definition
      *      
      *      However the case of thes orthographik camera, with short F and PP far from middle is very unrealistic
      *      due to simulation , so we maintain it at midle for now , maybe more sophisticated later ...
     */

     // cPt2dr aSeedPix = PP();
     cPt2dr aSeedPix = ToR(SzPix()) / 2.0;
     cPt2dr  aPDist = mMapIm2PProj.Value(aSeedPix);


     cMappingIdentity<tREAL8,2> aMapId;
     cDataIIMFromMap<tREAL8,2>  aDRI(mDir_Dist,&aMapId,1e-2/F(),100,false,false);

     cPt2dr  aPProj = aDRI.Inverse(aPDist);

     // StdOut() << "PSEED "  << aPProj<< " " << aCpt << std::endl;
     return aPProj;
}

static cPt2dr  PBug(233.2,2576.5);


std::vector<cPt2dr>  cPerspCamIntrCalib::PtsSampledOnSensor(int aNbByDim,bool InPixel) const
{
    // UpdateLSQDistIfRequired();
    std::unique_ptr<cComputeMapInverse<double,2>> aPCMI  
    (
          cComputeMapInverse<double,2>::Alloc
          (
             mThreshJacPI,         ///< Threshold on jacobian to ensure inversability
             PtSeedInv(),          ///< Seed point, in input space
             aNbByDim,    ///< Approximate number of point (in the biggest size), here +or- less square of minimum
             mDefProj,
             (*mPhgrDomain),       ///< Set of validity, in output space
             (*mDir_Dist),         ///< Maping to invert : InputSpace -> OutputSpace
              nullptr,
             false                 ///< Not in  Test
         )
    );
    aPCMI->DoPts();

    std::vector<cPt2dr> aRes = aPCMI->GetPtsOut();
    if (InPixel)
    {
       // aRes = (mMapPProj2Im).Values(aRes);
       aRes = static_cast<const cDataInvertibleMapping<tREAL8,2>&>(mMapPProj2Im).Values(aRes);
    }

    return aRes;
}



	     // ==================  geometric manips =====================

const  std::vector<cPt2dr> &  cPerspCamIntrCalib::Values(tVecOut & aV3 ,const tVecIn & aV0 ) const 
{
     static tVecOut aV1,aV2;
     mDir_Proj->Values(aV1,aV0);
     mDir_Dist->Values(aV2,aV1);
     mMapPProj2Im.Values(aV3,aV2);
     
     return aV3;
}


double cPerspCamIntrCalib::DegreeVisibilityOnImFrame(const cPt2dr & aPIm) const
{
     double MaxCalc = 10.0;
     //  For domain where dist is inversible this should be sufficient
     double aRes1 = mPixDomain.InsidenessWithBox(aPIm);
     // dont want to do inversion too far it may overflow ...
     if (aRes1<-MaxCalc)
        return aRes1;

     UpdateLSQDistIfRequired();

     cPt2dr  aPDist   =  mMapIm2PProj.Value(aPIm);  //  point with dist
     cPt2dr  aPUndist =  mDist_DirInvertible->Inverse(aPDist);   //  point w/o dist
     cPt2dr aPDistBack  = mDir_Dist->Value(aPUndist);   // dist again, should go back to aPDist is we are invertible

     double aRes2  = 1e-2 - Norm2(aPDist-aPDistBack) * F();  // did invertion work well
     double aRes3 =  mDefProj->P2DIsDef(aPUndist) * F();     // must be in domain of projection

     // if we are inside,  aRes2~0 and Res1 is meaningfull for insideness
     if ((aRes1>=0) && (aRes2>=0)   && (aRes3>=0))
         return aRes1;

     return std::min(aRes3,std::min(aRes1,aRes2));
}


double cPerspCamIntrCalib::DegreeVisibility(const cPt3dr & aP) const
{
     // if the point is not OK for projector dont try any more ...
     tREAL8 aRes3 =  mDefProj->P3DIsDef(aP) *F();

     if (aRes3 <=0)
        return aRes3 ;

     double MaxCalc = 10.0;

     UpdateLSQDistIfRequired();
     
     cPt2dr aPphgr = mDir_Proj->Value(aP); //  Compute  projected point  (as x/z y/z for example)
     cPt2dr aPDist  = mDir_Dist->Value(aPphgr);   // compute distorted point

     //  For domain where dist is inversible this should be sufficient
     cPt2dr aPIm   = mMapPProj2Im.Value(aPDist);
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

     return std::min(aRes3,std::min(aRes1,aRes2));
}


const  std::vector<cPt3dr> &  cPerspCamIntrCalib::DirBundles(tVecIn & aV3 ,const tVecOut & aV0 ) const 
{
     UpdateLSQDistIfRequired();

     CheckBeforeInverse(aV0);

     static tVecOut aV1,aV2;
     mMapIm2PProj.Values(aV1,aV0);
     mDist_DirInvertible->Inverses(aV2,aV1);
     mInv_Proj->Values(aV3,aV2);
     
     return aV3;
}

cPt3dr  cPerspCamIntrCalib::DirBundle(const tPtOut & aPt) const 
{
     std::vector<tPtIn>  aVecIn;
     std::vector<tPtOut> aVecOut{aPt};
     const  std::vector<tPtIn> & aVRes = DirBundles(aVecIn,aVecOut);

     cPt3dr aRes =  aVRes.at(0);
     return aRes;
}

cPt2dr  cPerspCamIntrCalib::Undist(const tPtOut & aP0) const
{
    cPt3dr aPt = DirBundle(aP0);

    cPt2dr aP1 = Proj(aPt) / aPt.z();

    return mMapPProj2Im.Value(aP1);
}

tREAL8  cPerspCamIntrCalib::InvProjIsDef(const tPtOut & aPix ) const
{
    return mDefProj->P2DIsDef(mDist_DirInvertible->Inverse(mMapIm2PProj.Value(aPix)));
}

      //   ----  object in unknown system (bundle adj ...) ----------------
     
void cPerspCamIntrCalib::UpdateMapProj2Im() 
{
    mMapIm2PProj       = mMapPProj2Im.MapInverse();
}
void cPerspCamIntrCalib::OnUpdate() 
{
   // The inverst for dist and csp must be recomputed
    mMapIm2PProj       = mMapPProj2Im.MapInverse();

    // if we are here, great proba modif has been done, so force Update
    UpdateLSQDistInv  ();
}

void cPerspCamIntrCalib::PutUknowsInSetInterval() 
{
    // Unknown have escpaped and will probably be modified
    mInvIsUpToDate = false;

    mSetInterv->AddOneInterv(mMapPProj2Im.F());
    mSetInterv->AddOneInterv(mMapPProj2Im.PP());
    mSetInterv->AddOneInterv(VParamDist());
}

void  cPerspCamIntrCalib::GetAdrInfoParam(cGetAdrInfoParam<tREAL8> & aGAIP)
{
   aGAIP.TestParam(this,&(mMapPProj2Im.F()),"F");
   aGAIP.TestParam(this,&(mMapPProj2Im.PP().x()),"PPx");
   aGAIP.TestParam(this,&(mMapPProj2Im.PP().y()),"PPy");

   for (size_t aK=0 ; aK<this->mDir_VDesc.size() ; aK++)
   {
       aGAIP.TestParam
       (
	  this,
          &( VParamDist().at(aK) )    , 
	  this->mDir_VDesc[aK].mName
       );
   }
}



cCalculator<double> * cPerspCamIntrCalib::EqColinearity(bool WithDerives,int aSzBuf,bool ReUse)
{
    return EqColinearityCamPPC(mTypeProj,mDir_Degr,WithDerives,aSzBuf,ReUse,mIsFraserMode);
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

void cPerspCamIntrCalib::SetName(const std::string &  aNewName) {  mName= aNewName ;}
const std::string & cPerspCamIntrCalib::Name()   const {return mName ;}
const cPt3di & cPerspCamIntrCalib::DegDir() const {return mDir_Degr;}

const double & cPerspCamIntrCalib::F()      const {return mMapPProj2Im.F() ;}
const cPt2dr & cPerspCamIntrCalib::PP()     const {return mMapPProj2Im.PP();}

const cPt2di & cPerspCamIntrCalib::SzPix() const {return mPixDomain.Sz();}

const cDataMapping<tREAL8,3,2>* cPerspCamIntrCalib::Dir_Proj() const { return mDir_Proj;}
const cDataMapping<tREAL8,2,2>* cPerspCamIntrCalib::Dir_Dist() const { return mDir_Dist;}
const cPixelDomain & cPerspCamIntrCalib::PixelDomain() const {return mPixDomain;}

eProjPC cPerspCamIntrCalib::TypeProj() const {return mTypeProj;}

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


#if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny )
void cPerspCamIntrCalib::CheckBeforeInverse(const tVecOut & aVecOut) const
{
    for (const auto & aPOut : aVecOut)
	    CheckBeforeInverse(aPOut);
}
void cPerspCamIntrCalib::CheckBeforeInverse(const tPtOut & aPOut ) const
{
    tREAL8 aThreshHigh = -0.2;
    tREAL8 aThreshLow =  -0.5;
    tREAL8 aOutness = mPixDomain.InsidenessWithBox(aPOut) / F();

    if (aOutness<aThreshHigh)
    {
        std::string aMesg = "Suspicious point for inversion, Pt=" + ToStr(aPOut) + " SzIma=" + ToStr(SzPix()) ;
        if (aOutness<aThreshLow)
	{
           MMVII_UnclasseUsEr(aMesg);
	}
	else
	{
           MMVII_DEV_WARNING(aMesg);
	}
    }
}
#else
void cPerspCamIntrCalib::CheckBeforeInverse(const tVecOut &) const { }
void cPerspCamIntrCalib::CheckBeforeInverse(const tPtOut & ) const { }
#endif



       /* ================================================================== */
       /*                 BENCH VERIF PART                                   */
       /* ================================================================== */

void cPerspCamIntrCalib::TestInvInit(double aTolApprox,double aTolAccurate)
{
	// test inversion for the distorsion
     {
         // generate 2d-point ine photogram coordinate , after distorsion
         // double aRhoMax =  mPhgrDomain->Box().DistMax2Corners(cPt2dr(0,0));
         std::vector<cPt2dr>  aVPt0 = PtsSampledOnSensor(10,false);

	 /// mPhgrDomain->GridPointInsideAtStep(aVPt0,aRhoMax/10.0);

	 //   Filters them because due to mDefProj , not all belong to fiting domain
         std::vector<cPt2dr>  aVPt1;
	 for (const auto & aP0 : aVPt0)
	 {
             cPt2dr aP0Ud = mDist_DirInvertible->Inverse(aP0);
             // cPt2dr aP0Rd = mDir_Dist->Value(aP0Ud);

             if (mDefProj->InsideWithBox(aP0Ud))
	     {
                aVPt1.push_back(aP0);
		// StdOut() << "P000 =" << aP0 << " " << aP0Ud << std::endl;
	     }
	 }

        /*StdOut() << "TestInvInitTestInvInit " <<  E2Str(mTypeProj)  
		 << "SZZZZ " << aVPt0.size() <<  " " << aVPt1.size() 
		  << " RM " << mDefProj->mRhoMax<< "\n";
        */


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

	 // StdOut() << "SDDDDD " << aSD12 << " " << aSD23 << " " << aSD13 <<  " " << aSD15 << std::endl;

         MMVII_INTERNAL_ASSERT_bench((aSD13==0) || (aSD13/aSD12<aTolApprox),"Test approx inv");
         MMVII_INTERNAL_ASSERT_bench((aSD15==0) || (aSD15/aSD12<aTolAccurate),"Test approx inv");
     }

     // test global inversion
     {
	 // filter them because witj some projection point to far are not invetrible
         std::vector<cPt2dr>  aVPt1 = PtsSampledOnSensor(20,true);

	 // compute direction of bundles
         std::vector<cPt3dr>  aVPt2;
	 DirBundles(aVPt2,aVPt1);

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

         aSD13 = std::sqrt(aSD13/aVPt1.size())  / mMapPProj2Im.F()  ;
         MMVII_INTERNAL_ASSERT_bench(aSD13<1e-8,"Test approx inv");
     }
}

void cPerspCamIntrCalib::InitRandom(double aAmpl)
{
     bool isFraserMode = true;
     double aRhoMax =  mPhgrDomain->Box().DistMax2Corners(cPt2dr(0,0));

     cRandInvertibleDist  aParamRID ( mDir_Degr, aRhoMax, RandUnif_0_1(), aAmpl,isFraserMode);
     mDir_Dist->SetObs(aParamRID.VParam());
     UpdateLSQDistInv ();
}
 

cPerspCamIntrCalib * cPerspCamIntrCalib::RandomCalib(eProjPC aTypeProj,int aKDeg)
{

    tREAL8 aDiag = 1000 * (1+10*RandUnif_0_1());
    cPt2di aSz (aDiag*(1+RandUnif_0_1()),aDiag*(1+RandUnif_0_1()));
    cPt2dr aMidle = ToR(aSz)/2.0;
    cPt2dr aPP(   aSz.x()*(0.5+0.1*RandUnif_C())  , aSz.y()*(0.5+0.1*RandUnif_C())  );
    tREAL8  aFoc =  aDiag * (0.2 + 3.0*RandUnif_0_1());

    UpdateMax(aFoc,2* Norm2(aPP-aMidle));

    std::vector<cPt3di>  aDegDir{{0,0,0},{2,0,0},{3,1,1}};
    std::vector<cPt3di>  aDegInv{{0,0,0},{5,1,1},{7,2,5}};

    cPerspCamIntrCalib* aCam = cPerspCamIntrCalib::Alloc
                                (
	                                 cDataPerspCamIntrCalib
	                                 (
	                                       "BenchCam",
                                                aTypeProj,
						aDegDir.at(aKDeg),
	                                        std::vector<double>(),
	                                        cMapPProj2Im(aFoc,aPP),
	                                        cDataPixelDomain(aSz),
						aDegInv.at(aKDeg),
	                                        100
	                                 )
                                );

       aCam->SetThresholdPhgrAccInv(1e-9);

       if (BUGCAL)
           StdOut() << "RrrrAtio="  << Norm2(aPP-aMidle) / aFoc << " TTTt=" << E2Str(aTypeProj)  << std::endl;

       aCam->InitRandom(0.1);

       if (BUGCAL)
           StdOut() << "Kkkkkkkkkkkkkkkkk" << std::endl;

       return aCam;
}


void BenchCentralePerspective(cParamExeBench & aParam,eProjPC aTypeProj)
{
    for (size_t aK=0 ; aK<3 ; aK++)
    {
       cPerspCamIntrCalib * aCam = cPerspCamIntrCalib::RandomCalib(aTypeProj,aK);
       aCam->TestInvInit((aK==0) ? 1e-3 : 1e-2, 1e-4);

       cSensorCamPC::BenchOneCalib(aCam);

       delete aCam;
    }
}


void BenchCentralePerspective(cParamExeBench & aParam)
{
    BenchCentralePerspective(aParam,eProjPC::eOrthoGraphik);

    if (! aParam.NewBench("CentralPersp")) return;

    cMapPProj2Im aCS(1,cPt2dr(0,0));
    // in fact this is not necessary , btw maintain just in case and see if the test fail
    MMVII_INTERNAL_ASSERT_bench(&(aCS.F())+1 == &(aCS.PP().x()) ,"Assertion cMapPProj2Im memory model");
    MMVII_INTERNAL_ASSERT_bench(&(aCS.F())+2 == &(aCS.PP().y()) ,"Assertion cMapPProj2Im memory model");

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
/*                 cMapPProj2Im                       */
/*                                                         */
/* ******************************************************* */

cMapPProj2Im::cMapPProj2Im(tREAL8 aFoc,const tPt & aPP) :
    cInvertMappingFromElem<cHomot2D<tREAL8> > (cHomot2D<tREAL8>(aPP,aFoc))
{
}
cMapPProj2Im::cMapPProj2Im(const cMapPProj2Im & aCS) :
    cMapPProj2Im(aCS.F(),aCS.PP())
{
}

cMapIm2PProj cMapPProj2Im::MapInverse() const
{
   return cMapIm2PProj(Map().MapInverse());
}


const double & cMapPProj2Im::F()  const {return Map().Sc();}
const cPt2dr & cMapPProj2Im::PP() const {return Map().Tr();}
double & cMapPProj2Im::F()  {return Map().Sc();}
cPt2dr & cMapPProj2Im::PP() {return Map().Tr();}

/* *********************************** */
/*                                     */
/*           cMapIm2PProj              */
/*                                     */
/* *********************************** */

cMapIm2PProj::cMapIm2PProj(const cHomot2D<tREAL8> & aH) :
    cInvertMappingFromElem<cHomot2D<tREAL8> >(aH)
{
}


}; // MMVII

