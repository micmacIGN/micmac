#include "include/MMVII_all.h"



/**
   \file cConvCalib.cpp

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{

/**  Class for otimizinf a model of camerato  using 3d-2d correspondance and bundle adjustment.  Typically these
 *   corresponance will be synthetic ones coming from another camera. It can be used in, two scenario :
 *
 *    -(1) primary test/bench  on functionnality to do BA
 *    -(2)
 *        (2.a)   conversion between calibration (format/model ...)
 *        (2.b)   comparison of calibrations (to come)
 *
 *    In first case we create artifcially difficult conditions (randomize the initial pose, let free the perspective center).
 *
 *    In the second case,  we use as much information we have : init with identity, and froze the position center
 *
 */

class cCentralPerspConversion
{
    public :
         typedef cIsometry3D<tREAL8>   tPose;
         cCentralPerspConversion
         (
              cPerspCamIntrCalib * ,
              const cSet2D3D &,
              const tPose & aPoseInit = tPose::Identity(),
              bool    HardConstrOnGCP=true , // do we fix GCP,  false make sense in test mode
              bool    CenterFix=true        // do we fix centre of projection,  false make sense in test mode
         );
         ~cCentralPerspConversion();

         void OneIteration();

         const cSensorCamPC  &       CamPC() const;  ///<  Accessor
         const cPerspCamIntrCalib &  Calib() const;  ///<  Accessor

    private :
         tPose                              mPoseInit;
         bool                               mHCG; // HardConstrOnGCP
         bool                               mCFix; // HardConstrOnGCP

         cPerspCamIntrCalib *               mCalib;
         cSensorCamPC                       mCamPC;
         cSet2D3D                           mSetCorresp;
         int                                mSzBuf;
         cCalculator<double> *              mEqColinearity;
         cSetInterUK_MultipeObj<double>     mSetInterv;
         cResolSysNonLinear<double> *       mSys;
};


cCentralPerspConversion::cCentralPerspConversion
(
     cPerspCamIntrCalib *    aCalib,
     const cSet2D3D &        aSetCorresp,
     const tPose &           aPoseInit ,
     bool                    HardConstrOnGCP,
     bool                    CenterFix
) :
    mPoseInit      (aPoseInit),
    mHCG           (HardConstrOnGCP),
    mCFix          (CenterFix),
    mCalib         (aCalib),
    mCamPC         (mPoseInit,mCalib),
    mSetCorresp    (aSetCorresp),
    mSzBuf         (100),
    mEqColinearity (mCalib->EqColinearity(true,mSzBuf))
{
    mSetInterv.AddOneObj(&mCamPC);
    mSetInterv.AddOneObj(mCalib);

    mSys = new cResolSysNonLinear<double>(eModeSSR::eSSR_LsqDense,mSetInterv.GetVUnKnowns());
}


cCentralPerspConversion::~cCentralPerspConversion()
{
    delete mEqColinearity;
    delete mSys;
}


const cSensorCamPC  &       cCentralPerspConversion::CamPC() const {return mCamPC;}
const cPerspCamIntrCalib &  cCentralPerspConversion::Calib() const {return *mCalib;}

void cCentralPerspConversion::OneIteration()
{
     if (mCFix)
     {
        // size_t aNumCx
        // CenterFix
     }
     std::vector<int> aVIndGround{-1,-2,-3};

     // Fill indexe Glob in the same order as in cEqColinearityCamPPC::VNamesUnknowns()
     std::vector<int> aVIndGlob = aVIndGround;
     mCamPC.FillIndexes(aVIndGlob);
     mCalib->FillIndexes(aVIndGlob);

     for (const auto & aCorresp : mSetCorresp.Pairs())
     {
         // structure for points substistion, in mode test
         cSetIORSNL_SameTmp<tREAL8>   aStrSubst
                                      (
                                         aCorresp.mP3.ToStdVector() ,
                                          (mHCG ? aVIndGround : std::vector<int>())
                                      );

         if (! mHCG)
         {
            for (const auto & anInd : aVIndGround)
               aStrSubst.AddFixCurVarTmp(anInd,1.0);
         }

         // "observation" of equation  : PTIm (real obs) + Cur-Rotation to avoid guimbal-lock
         std::vector<double> aVObs = aCorresp.mP2.ToStdVector();
         mCamPC.Pose().Rot().Mat().PushByCol(aVObs);

         mSys->AddEq2Subst(aStrSubst,mEqColinearity,aVIndGlob,aVObs);
         mSys->AddObsWithTmpUK(aStrSubst);
     }

     const auto & aVectSol = mSys->SolveUpdateReset();
     mSetInterv.SetVUnKnowns(aVectSol);
}


void BenchCentralePerspective_ImportV1(cParamExeBench & aParam,const std::string & aName,bool HCG,bool  CenterFix,double aAccuracy)
{
     std::string aFullName = cMMVII_Appli::CurrentAppli().InputDirTestMMVII() + "Ori-MMV1" +  StringDirSeparator() + aName;

     cExportV1StenopeCalInterne  aExp(aFullName,10);

     double aF0 = aExp.mFoc;
     cPt2dr aPP0 = aExp.mPP;
     aExp.mFoc *=  (1.0 + 0.05*RandUnif_C());
     aExp.mPP = MulCByC(aExp.mPP,  cPt2dr(1,1)+cPt2dr::PRandC()*0.05);

     cPerspCamIntrCalib aCalib(aExp.eProj,cPt3di(3,1,1),aExp.mFoc,aExp.mSzCam);

     cIsometry3D<tREAL8> aPose0
                         (
                              cPt3dr::PRandC() * (CenterFix ? 0.0 : 0.1),
                              cRotation3D<tREAL8>::RandomRot(0.05)
                         );

     cCentralPerspConversion aConv(&aCalib,aExp.mCorresp,aPose0,HCG,CenterFix);
     const cSensorCamPC  &       aCamPC =  aConv.CamPC() ;



     double aResidual  = 10;
     for (int aK=0 ; aK<20 ; aK++)
     {
        if (0)
        {
            const cPerspCamIntrCalib &  aCalib =  aConv.Calib();
            StdOut() << "FFF="  <<  aF0  - aCalib.F() <<  " " << aPP0-aCalib.PP() << "\n";
        }
        aConv.OneIteration();
        aResidual  = aCamPC.AvgResidual(aExp.mCorresp);

        if (aResidual<aAccuracy)
        {
            StdOut() << "---------------RR=" <<  aResidual  << "\n";

	    aConv.Calib().ToFile("toto.xml");
	    getchar();
            return;
        }
     }
     StdOut() << "---------------RR=" <<  aResidual  << "\n";
     MMVII_INTERNAL_ASSERT_bench(false ,"No convergence in BenchCentralePerspective_ImportV1");

}

void BenchCentralePerspective_ImportV1(cParamExeBench & aParam)
{
    for (int aK=0 ; aK<3 ; aK++)
    {
        BenchCentralePerspective_ImportV1(aParam,"AutoCal_Foc-60000_Cam-NIKON_D810.xml",false,false,1e-5);
        BenchCentralePerspective_ImportV1(aParam,"AutoCal_Foc-60000_Cam-NIKON_D810.xml",false,true ,1e-5);
        BenchCentralePerspective_ImportV1(aParam,"AutoCal_Foc-60000_Cam-NIKON_D810.xml",true ,false,1e-5);
        BenchCentralePerspective_ImportV1(aParam,"AutoCal_Foc-60000_Cam-NIKON_D810.xml",true ,true ,1e-5);

        BenchCentralePerspective_ImportV1(aParam,"AutoCal_Foc-11500_Cam-imx477imx477-1.xml",false,false,1e-3);
    }

}



}; // MMVII

