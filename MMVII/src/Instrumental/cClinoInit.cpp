#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Clino.h"
#include "MMVII_Tpl_Images.h"


static bool ShowDetails =false;

/**
   \file cClinoInit.cpp

   Compute initial value of clinometers boresight relatively to a camera.

 */

namespace MMVII
{

	/*
enum class eTyClino
           {
              ePendulum,
              eSpring,
              eNbVals    ///< Tag for number of value
           };
	   */

/* ************************************* */
/*           cOneCalibRelClino           */
/* ************************************* */

cOneCalibRelClino::cOneCalibRelClino() :
    mNameRef (""),
    mRot     (tRotR::Identity())
{
}

void AddData(const  cAuxAr2007 & anAux,cOneCalibRelClino & aLnk)
{
    AddData(cAuxAr2007("ClinoRef",anAux),aLnk.mNameRef);
    AddData(cAuxAr2007("Rotation",anAux),aLnk.mRot);
}


/* ************************************* */
/*           cOneCalibClino              */
/* ************************************* */

cOneCalibClino::cOneCalibClino(eTyClino aType,const std::string& aNameClino,const tRotR & aRot, const std::string & aNameCamera) :
   mType       (aType),
   mNameClino  (aNameClino),
   mRot        (aRot),
   mCameraName (aNameCamera)
{
}



cOneCalibClino::cOneCalibClino() :
     mNameClino (""),
     mRot       (tRotR::Identity())
{
}

cOneCalibClino::cOneCalibClino(std::string aNameClino) :
     mNameClino (aNameClino),
     mRot       (tRotR::Identity())
{
}

void cOneCalibClino::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::EnumAddData(anAux,mType,"Type");
    MMVII::AddData(cAuxAr2007("NameClino",anAux),mNameClino);
    MMVII::AddData(cAuxAr2007("Rotation",anAux),mRot);
    MMVII::AddData(cAuxAr2007("NameCamera",anAux),mCameraName);
    MMVII::AddOptData(anAux,"RelCalib",mLinkRel);
}

void AddData(const  cAuxAr2007 & anAux, cOneCalibClino & aCalib)
{
    aCalib.AddData(anAux);
}

const tRotR & cOneCalibClino::Rot()             const {return mRot;};
const std::string& cOneCalibClino::NameClino()  const {return mNameClino;};
const std::string& cOneCalibClino::NameCamera() const {return mCameraName;};
eTyClino    cOneCalibClino::Type() const {return mType;}


void cOneCalibClino::SetRelCalib(const std::string & aNameRef,const tRotR & aRot)
{
     cOneCalibRelClino aRel;
     aRel.mNameRef = aNameRef;
     aRel.mRot     = aRot;
     mLinkRel = aRel;
}


/* ************************************* */
/*           cCalibSetClino              */
/* ************************************* */

cCalibSetClino::cCalibSetClino() :
   mNameCam ("")
{
}




cCalibSetClino::cCalibSetClino(std::string aNameCam, std::vector<cOneCalibClino> aClinosCal) :
   mNameCam (aNameCam),
   mClinosCal (aClinosCal)
{
}


void cCalibSetClino::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("NameCams",anAux),mNameCam);
    MMVII::StdContAddData(cAuxAr2007("ClinoCalibs",anAux),mClinosCal);
}
void AddData(const  cAuxAr2007 & anAux,cCalibSetClino & aSet)
{
    aSet.AddData(anAux);
}

const std::string & cCalibSetClino::NameCam() const {return mNameCam;};
const std::vector<cOneCalibClino> & cCalibSetClino::ClinosCal() const {return mClinosCal;};

void cCalibSetClino::SetClinosCal(const std::vector<cOneCalibClino> &  aClinosCal)
{
   mClinosCal=aClinosCal;
} 

void cCalibSetClino::AddCalib1Clino(const cOneCalibClino& aCalib)
{
    mClinosCal.push_back(aCalib);
}

void cCalibSetClino::SetNameCam(const std::string & aNameCam)
{
    if (mNameCam=="")
    {
       mNameCam = aNameCam;
    }
    else
    {
         if (mNameCam!=aNameCam)
         {
             MMVII_INTERNAL_ERROR("Try to change name of cam in cCalibSetClino : " +mNameCam + " => " + aNameCam);
         }
    }
}






/** Class for storing one measure of clinometer on a camera.
 */

class cClinoCalMes1Cam
{
    public :
        cClinoCalMes1Cam
        (
	      eTyClino,
              cSensorCamPC * aCam,   // camera-> only the pose is usefuul
              const std::vector<tREAL8> & aVAngles, // vector of angles of all clinometer
              const cPt3dr &   aVerticAbs = {0,0,-1}  // position of vertical in current "absolut" system
        );

	/// check that with calibration, we can recover local vertical from angles
	void TestCalib(const cCalibSetClino &,const std::vector<int> & aVKClino ) const;

	/// Simulate the measure we would have if "aR" was the given  calibration
        void SetDirSimul(int aK,const  tRotR &aR) ;

	/// Theoretical Pos of needle, indicating vertical, in plane IJ, for Clino K, given a boresight aCam2Clino
        cPt2dr  PosNeedle(const  tRotR  &aCam2Clino) const;

	///  Vectorial difference, in plane IJ, between measure an theoreticall value for needle
        cPt2dr  EcarVectRot(int aKClino,const  tRotR &aCam2Clino) const;

	/* => Use of WPK is currently deprecated
	    ///  Idem but with WPK
            cPt2dr  EcarVectWPK(int aKClino,const  cPt3dr &aWPK) const;
	    ///  Idem but with WPK
            tREAL8  ScoreWPK(int aKClino,const  cPt3dr &aWPK) const;
	*/

	///  Score (to minimize) between obs and theor for given rot
        tREAL8  ScoreRot(int aKClino,const  tRotR &aR) const;

	///  Gradient / wpk of Needle's  "Ecart vectorial" (in x an y) in plane IJ, computed using finite difference (now used for 
        std::pair<cPt3dr,cPt3dr>  GradEVR(int aKClino,const tRotR & aRot,tREAL8 aEpsilon=1e-3) const;

        /// return the theoreticall position on the spring 
        tREAL8 SpringAngle(const  tRotR &aCam2Clino) const;
    private :
        cSensorCamPC *          mCam;  ///< camera , memorization seems useless
	std::vector<tREAL8>     mVAngles; ///< Copy of angles, for TestCalib
	std::vector<cPt2dr>     mVDir; ///<  measured position of needle, computed from angles
	cPt3dr                  mVertInLoc;  ///<  Vertical in  camera system, this  is the only information usefull of camera orientation
	eTyClino                mTypeClino;
};


cClinoCalMes1Cam::cClinoCalMes1Cam(eTyClino aTypeClino,cSensorCamPC * aCam,const std::vector<tREAL8> & aVAngles,const cPt3dr & aVertAbs) :
    mCam       (aCam),
    mVAngles   (aVAngles),
    mVertInLoc (mCam->Vec_W2L(cPt3dr(0,0,-1))),
    mTypeClino (aTypeClino)
{


    // tranformate angles in vector position of the needle in plane I,J
    for (auto & aTeta : aVAngles)
    {
        mVDir.push_back(FromPolar(1.0,aTeta));
    }
}


void cClinoCalMes1Cam::TestCalib(const cCalibSetClino & aCalib,const std::vector<int> & aVKClino ) const
{
    // copy the selected angles
    std::vector<tREAL8> aVAngleSel;
    for (auto aKClino : aVKClino)
        aVAngleSel.push_back(mVAngles.at(aKClino));

    // extract the local vertical from angles
    cGetVerticalFromClino aGetVert(aCalib,aVAngleSel);
    auto [aScore,aDir1] = aGetVert.OptimGlob(50,1e-7);

    // eventually print 
    if (0)
    {
       StdOut() << "SCORE= " << aGetVert.ScoreDir3D(mVertInLoc) 
	       << " SOP=" << aGetVert.ScoreDir3D(aDir1) 
	       << " Delta=" << Norm2(aDir1-mVertInLoc) << "\n";
    }
}

std::pair<cPt3dr,cPt3dr>  cClinoCalMes1Cam::GradEVR(int aKClino,const tRotR & aR0,tREAL8 aEps) const
{
   cPt3dr aResX;
   cPt3dr aResY;
   //  Parse 3 angles
   for (int aK=0 ; aK<3 ; aK++)
   {
        // compute Epsilon vector in 1 direction
        cPt3dr aPEps = cPt3dr::P1Coord(aK,aEps);

	// small rotation in direction
	tRotR aRPlus  = tRotR::RotFromWPK(aPEps);
	tRotR aRMinus = tRotR::RotFromWPK(-aPEps);

	//  compute difference in needle position
        cPt2dr aDelta = EcarVectRot(aKClino,aR0*aRPlus) -  EcarVectRot(aKClino,aR0*aRMinus);
        aResX[aK] = aDelta.x()/(2*aEps);
        aResY[aK] = aDelta.y()/(2*aEps);
   }
   return std::pair<cPt3dr,cPt3dr>(aResX,aResY);
}


cPt2dr  cClinoCalMes1Cam::PosNeedle(const  tRotR &aCam2Clino) const
{
     // "aVClin"= Direction of Vert in clino repair, mVertInLoc being in camera system
     // we only need to know the boresight "aCam2Clino" to tranfer it in the clino system
     cPt3dr  aVClin =  aCam2Clino.Value(mVertInLoc); 
     // VClin =(x,y,z) ,  the projection of VClin in plane "I,J" is given by Proj (x,y,z) -> (x,y)
     // then we make a unitary vector to maintain the size of "needle" 
     return  VUnit(Proj(aVClin));       // Projection in plane I,J
}


tREAL8 cClinoCalMes1Cam::SpringAngle(const  tRotR &aCam2Clino) const
{
     cPt3dr  aVClin =  aCam2Clino.Value(mVertInLoc); 

     return aVClin.y();
}

cPt2dr  cClinoCalMes1Cam::EcarVectRot(int aKClino,const  tRotR &aCam2Clino) const
{
     //  Theoretical Pos -  Measured Pos
     return PosNeedle(aCam2Clino) - mVDir[aKClino];
}

tREAL8  cClinoCalMes1Cam::ScoreRot(int aKClino,const  tRotR & aCam2Clino) const
{
if (ShowDetails)
{
    StdOut() << " KC=" << aKClino 
	    <<  " SSRR=" << Square(aCam2Clino.Value(mVertInLoc).y() - mVAngles[aKClino] )
	    <<  " " <<  SqN2(EcarVectRot(aKClino,aCam2Clino))
	    << "\n";
}
     switch (mTypeClino)
     {
        case eTyClino::ePendulum:
             return SqN2(EcarVectRot(aKClino,aCam2Clino));

        case eTyClino::eSpring:
             return Square(SpringAngle(aCam2Clino) - std::sin(mVAngles[aKClino]));

        default :
	     ;
     };

     MMVII_INTERNAL_ERROR("No valide type clino");
     return 0.0;
}

void cClinoCalMes1Cam::SetDirSimul(int aKClino,const  tRotR &aR)
{
     mVDir[aKClino] = PosNeedle(aR);
}

/*
tREAL8  cClinoCalMes1Cam::ScoreWPK(int aKClino,const  cPt3dr & aWPK) const
{
    return ScoreRot(aKClino,tRotR::RotFromWPK(aWPK));
}
cPt2dr  cClinoCalMes1Cam::EcarVectWPK(int aKClino,const  cPt3dr &aWPK) const
{
    return EcarVectRot(aKClino,tRotR::RotFromWPK(aWPK));
}
*/




/* ==================================================== */
/*                                                      */
/*          cAppli_CalibratedSpaceResection             */
/*                                                      */
/* ==================================================== */

class cAppli_ClinoInit : public cMMVII_Appli
{
     public :
        cAppli_ClinoInit(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :
	tRotR  OriOfClino(int aKCl,const tRotR & aRot) const {return mOriRelClin.at(aKCl) *aRot;}

	/// for a given clinometern compute the cost of a tested boresight
        tREAL8                     CostRot(const tRotR & aWPK) const;

	/// Vector of score, can be agregated  sum/max
	std::vector<tREAL8>  VectCostRot(const tRotR & aWPK) const;

	///  Make one iteration of research arround a current solution
        cWhichMin<tRotR,tREAL8>   OneIter(const tRotR & aR0,tREAL8 aStep, int aNbStep);

	/// Make a first computation parsing "all" rotation at a given step, using sampling by quaternions
        cWhichMin<tRotR,tREAL8>   ComputeInitialSolution(int aNbStep) const;

	///  Compute the conditionning of least-square matrix
        cPt2dr ComputeCond(const tRotR & aWPK,tREAL8 aEpsilon=1e-4);


        cPhotogrammetricProject        mPhProj;     ///<  Classical structure for photogrammetric project
        eTyClino                       mTypeClino;
	std::vector<bool>              mComputedClino;  
        std::string                    mNameClino;  ///<  Pattern of xml file
        int                            mNbStep0;    ///<  Number of step in initial  first round
        int                            mNbStepIter; ///< Number of step in each iteration
        int                            mNbIter;     ///< Number of iteration
        std::vector<double>            mASim;       ///< Amplitude of simulation parameters


        std::vector<cClinoCalMes1Cam>  mVMeasures;      ///< store vectors of measure after file parsing
	std::vector<int>               mVKClino;        ///< Vecor of indexes of selected clino
	/**  If several Clino, they can be computed globally or independantly, this vector of bool allow to have
	 * the 2 computation, it's mainly for tuning and observing the effect on residual and conditionning
	 */


	std::string                    mNameRel12;  ///< relative orientation of Clino1/Clino2 stored as "i-kj"
	bool                           isOkNoCam;   ///< Is it OK if cam does not exist sometime ?
        std::vector<tRotR>             mOriRelClin; ///< vector of relative orientations Clino[0]/Clino[K]
	bool                           mShowAll;    ///< Do we show all the msg relative to residuals
        cCalibSetClino                 mCalibSetClino; ///< Result of the calibration
        std::string                    mPatFilter;
        tREAL8                         mUnityAng;  ///< Unitiy in radian
        bool                           mDmMGon;    ///< Is Unity Deci milli gon
        std::vector<tREAL8>            mMulAngle;
        bool                           mDebug;
};

cAppli_ClinoInit::cAppli_ClinoInit
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mTypeClino    (eTyClino::eSpring),
     mNbStep0      (65),
     mNbStepIter   (10),
     mNbIter       (50),
     mNameRel12    ("i-kj"),
     isOkNoCam     (false),
     mShowAll      (false),
     mPatFilter    (".*"),
     mUnityAng     (1.0),
     mDmMGon       (false),
     mDebug        (false)
{
}


cCollecSpecArg2007 & cAppli_ClinoInit::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
	      <<  mPhProj.DPMeasuresClino().ArgDirInMand()
              <<  Arg2007(mVKClino,"Index of clinometer",{{eTA2007::ISizeV,"[1,2]"}})
              <<  mPhProj.DPOrient().ArgDirInMand()
              <<  mPhProj.DPClinoMeters().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ClinoInit::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return     anArgOpt
	    << AOpt2007(mNbStep0,"NbStep0","Number of step at  first iteration",{eTA2007::HDV})
	    << AOpt2007(mNbStepIter,"NbStep","Number of step at  current iteration",{eTA2007::HDV})
	    << AOpt2007(mNbIter,"NbIter","Number of iteration",{eTA2007::HDV})
	    << AOpt2007(mASim,"AmplSim","Amplitude of rotation is simul [W,P,K,BS]",{{eTA2007::ISizeV,"[5,5]"}})
	    << AOpt2007(mNameRel12,"Rel12","orientation relative 2 to 1, if several clino",{eTA2007::HDV})
	    << AOpt2007(isOkNoCam,"OkNoCam","is it OK if some cam dont exist",{eTA2007::HDV})
            <<  mPhProj.DPClinoMeters().ArgDirInOpt()  // Just for temporart test we can re-read, to supress later
	    << AOpt2007(mPatFilter,"PatFilter","Pattern for filtering measure on ident",{eTA2007::HDV})
	    << AOpt2007(mDmMGon,"DMGon","Do we print residual in decimilligon",{eTA2007::HDV})
	    << AOpt2007(mTypeClino,"Type","Type of clino",{eTA2007::HDV,AC_ListVal<eTyClino>()})
	    << AOpt2007(mMulAngle,"MulA","Multiplier of Angle to test ...",{eTA2007::Tuning})
	    << AOpt2007(mDebug,"Debug","Debugging flag",{eTA2007::Tuning})
    ;
}

//  The average of cost for all measurements/position
std::vector<tREAL8>   cAppli_ClinoInit::VectCostRot(const tRotR & aRot0) const
{
     std::vector<tREAL8> aRes;  // Average for each Clino

     // Parse all clino currently selected
     for (size_t aIndC=0 ; aIndC<mVKClino.size() ; aIndC++)
     {
         if (mComputedClino.at(aIndC))
         {
             cStdStatRes aStat; // Structure for averaging
	     // Compute Orient, taken into account "aRot0" to test & num of clino for relative orient
             tRotR aRot = OriOfClino(aIndC,aRot0);  
	     // parse all measures
             for (const auto & aMes : mVMeasures)
	     {
                 aStat.Add(aMes.ScoreRot(mVKClino[aIndC],aRot));
	     }
	     //  Push the avg residual of all measure for this clino
	     aRes.push_back(aStat.Avg());
         }
     }

     return aRes;
}

tREAL8   cAppli_ClinoInit::CostRot(const tRotR & aRot0) const
{
     // Compute the vector for each clino
     std::vector<tREAL8> aVCost = VectCostRot(aRot0);
     // extract the average
     return std::accumulate(aVCost.begin(),aVCost.end(),0.0) / aVCost.size();
}



cWhichMin<tRotR,tREAL8>   cAppli_ClinoInit::ComputeInitialSolution(int aNbStep) const
{
    cWhichMin<tRotR,tREAL8>  aWMin(tRotR::Identity(),1e10); // initialize with a "big" residual
    // structure for parsing all rotation using quaternion , with aNbStep division in each direction
    cSampleQuat aSQ(aNbStep,true);

    for (size_t aKQ=0 ; aKQ<aSQ.NbRot() ; aKQ++) // parse all quaternion
    {
        cDenseMatrix<tREAL8> aMat = Quat2MatrRot(aSQ.KthQuat(aKQ)); // quaternion -> rotation
	tRotR aRot(aMat,false); // Rotation from Matrix, w/o optimization

        tREAL8 aCost = CostRot(aRot);  // compute the cost of tested rot
        aWMin.Add(aRot,aCost);  // update lowest cost solution
    }

    if (mShowAll)
       StdOut() << "MIN0=  " << std::sqrt(aWMin.ValExtre())  << std::endl;
    return aWMin;
}

cWhichMin<tRotR,tREAL8>  cAppli_ClinoInit::OneIter(const tRotR & aR0,tREAL8 aStep, int aNbStep)
{
    cWhichMin<tRotR,tREAL8>  aWMin(aR0,1e10); //  initialize with a "big" residual

    // parse the grid in 3 dimension 
    for (int aKw=-aNbStep ; aKw<= aNbStep ; aKw++)  // Parse Omega
    {
        for (int aKp=-aNbStep ; aKp<= aNbStep ; aKp++)  // Parse Phi
        {
            for (int aKk=-aNbStep ; aKk<= aNbStep ; aKk++) // Parse Kapa
            {
                cPt3dr aWPK =  cPt3dr(aKw,aKp,aKk) * aStep; // compute a small Omega-Phi-Kapa
		tRotR aRot = aR0*tRotR::RotFromWPK(aWPK);  // compute a neighbooring rotation

                tREAL8 aCost = CostRot(aRot);  // cost of tested rot
                aWMin.Add(aRot,aCost);  // update lowest cost solution
            }
        }
    }

    if (mShowAll) 
        StdOut() << "MIN " << std::sqrt(aWMin.ValExtre())  << " Step=" << aStep << std::endl;
    return  aWMin;
}


cPt2dr  cAppli_ClinoInit::ComputeCond(const tRotR & aRot0,tREAL8 aEpsilon)
{
     cStrStat2<tREAL8>  aStat(3);  // covariant matrix

     // Add  Gx and Gy of all measures in covariance-matrix
     for (size_t aIndC=0 ; aIndC<mVKClino.size() ; aIndC++)
     {
         if (mComputedClino.at(aIndC))  // If clino is in the current solution
	 {
             tRotR aRot = OriOfClino(aIndC,aRot0);  // Compute rotation, using "aRot0" and relative
             for (const auto & aMes : mVMeasures)  // Parse all measures
             {
                auto [aGx,aGy] = aMes.GradEVR(mVKClino[aIndC],aRot,aEpsilon);  // Compute grad in  x & y
	        aStat.Add(aGx.ToVect());  // Add grad in x for update covariance
	        aStat.Add(aGy.ToVect());  // Add grad in y for update covariance
             }
	 }
     }
     aStat.Normalise(false);  // normalize w/o centering

     // print ratio with highest value
     cDenseVect<tREAL8>  aDV = aStat.DoEigen().EigenValues();

     return cPt2dr (  std::sqrt(aDV(0)/aDV(2)) , std::sqrt(aDV(1)/aDV(2)) );
     //  StdOut() << "R2=" << std::sqrt(aDV(0)/aDV(2))  << " R1=" << std::sqrt(aDV(1)/aDV(2)) << std::endl;
}

int cAppli_ClinoInit::Exe()
{
    mPhProj.FinishInit();

    if (! IsInit(&mMulAngle))
       mMulAngle= std::vector<tREAL8>(mVKClino.size(),0.0);

    if (mDmMGon)
       mUnityAng = (400.0/(2*M_PI)) * 1e3 * 10;
    mComputedClino = std::vector<bool>(mVKClino.size(),true);  // initially all user clino are active


    // --------- Read formated file ----------------
    cSetMeasureClino aSMC = mPhProj.ReadMeasureClino(&mPatFilter) ;
    // const std::vector<cOneMesureClino>&  aVMC =  aSMC.SetMeasures();
    std::vector<cOneMesureClino>  aVMC =  aSMC.SetMeasures();

    // ------------- Compute vector ofrelative position  : usefull only when we have 2 clino
    tRotR aRSim = tRotR::Identity();  // Rotation for simulation
    mOriRelClin.push_back(tRotR::Identity());   // First clinometer : id to itself
    mOriRelClin.push_back(tRotR::RotFromCanonicalAxes(mNameRel12));  // Relative 2 to 1 

    //  if we do simulation, generate 
    size_t aNbMeasures = aVMC.size();
    if (IsInit(&mASim))
    {
       aRSim = tRotR::RandomRot(mASim[3]);
       aNbMeasures = size_t (mASim[4]);
    }
    std::vector<std::string> aVNamesClino = aSMC.NamesClino();

    StdOut() << "ExeExe " << aVNamesClino << "\n";

    // std::string aNameCalibCam;
    cPerspCamIntrCalib * aCalib = nullptr;
    //  put low level in a more structured data

    for (size_t aKMes = 0 ; aKMes<aNbMeasures ; aKMes++)
    {
	    // for now we process the case where clinos are identic on all lines, maybe to change later
        if (IsInit(&mASim))
        {
            auto v1 = RandUnif_C()*mASim[0];
            auto v2 = RandUnif_C()*mASim[1];
            auto v3 = RandUnif_C()*mASim[2];
            cPt3dr aWPK(v1,v2,v3);
            tRotR  aRPose =  tRotR::RotFromWPK(aWPK);

            cSensorCamPC * aCam = new cSensorCamPC("",cIsometry3D<tREAL8>::Identity(),nullptr);
            cMMVII_Appli::AddObj2DelAtEnd(aCam);
            aCam->SetPose(cIsometry3D<tREAL8>(cPt3dr(0,0,0),aRPose));

            // mVMeasures.push_back(cClinoCalMes1Cam(aCam,aVAngles[aKLine]));
            mVMeasures.push_back(cClinoCalMes1Cam(mTypeClino,aCam,std::vector<tREAL8>(10,0.0)));
	    for (size_t aKCl=0 ; aKCl<mVKClino.size() ; aKCl++)
                mVMeasures.back().SetDirSimul(mVKClino[aKCl],OriOfClino(aKCl,aRSim));
        }
	else 
        {
            const cOneMesureClino &  aMes = aVMC.at(aKMes);
            std::string aNameIm =  aSMC.NameOfIm(aMes);

	    cSensorCamPC * aCam = mPhProj.ReadCamPC(aNameIm,true,isOkNoCam);
	    if (aCam != nullptr)
	    {

		 std::vector<double>  aVAngles = aMes.Angles();

                 for (size_t aKC=0 ; aKC<mVKClino.size() ; aKC++)
                 {
                     aVAngles.at(mVKClino.at(aKC)) *= 1.0+ mMulAngle.at(aKC);
                 }

                 mVMeasures.push_back(cClinoCalMes1Cam(mTypeClino,aCam,aVAngles));

	         aCalib = aCam->InternalCalib();

                 mCalibSetClino.SetNameCam(aCalib->Name());
/*
	         aNameCalibCam = aCalib->Name();

	         // We cannnot have multiple camera for now
	         if ((mCalibSetClino.NameCam() !="") &&  (mCalibSetClino.NameCam() != aNameCalibCam))
                    MMVII_UnclasseUsEr("Multiple camera not handled");

                 mCalibSetClino.mNameCam = aNameCalibCam;
*/
	    }
        }
    }

    cWhichMin<tRotR,tREAL8>  aWM0 = ComputeInitialSolution(mNbStep0);

    StdOut() << "RESINIT=" <<  std::sqrt(aWM0.ValExtre()) *  mUnityAng << "\n";
/*
    if (0)
    {
ShowDetails = true;
        CostRot(aWM0.IndexExtre());
ShowDetails = false;
    }
*/

    tREAL8 aStep = 2.0/mNbStep0;
    tREAL8 aDiv = 1.25;
    
    tREAL8 aInitRes = std::sqrt(aWM0.ValExtre());
    for (int aK=0 ; aK<mNbIter ; aK++)
    {
         aWM0 =  OneIter(aWM0.IndexExtre(), (2*aStep)/mNbStepIter,mNbStepIter);
         if (mDebug)
         {
              StdOut() << " RES at iter " << aK << " = " <<  std::sqrt(aWM0.ValExtre()) *mUnityAng << "\n";
         }
         aStep /= aDiv;
    }


    StdOut() <<  "=============== Result of global optimization  =============" << std::endl;
    StdOut() << "Residual=" << std::sqrt(aWM0.ValExtre())  * mUnityAng
            << " Cond=" << ComputeCond(aWM0.IndexExtre())
            << " Resid Init=" << aInitRes * mUnityAng
	    << std::endl;

    for (size_t aKClino=0 ; aKClino<mVKClino.size() ; aKClino++)
    {
       cOneCalibClino aCal
                      (
                          mTypeClino,
                          aVNamesClino.at(mVKClino.at(aKClino)),
                          OriOfClino(aKClino,aWM0.IndexExtre()),
                          mCalibSetClino.NameCam()
                      );
       // aCal.mType =  mTypeClino;
       // aCal.mNameClino = aVNamesClino.at(mVKClino.at(aKClino));
       // aCal.mRot = OriOfClino(aKClino,aWM0.IndexExtre());
       // aCal.SetCameraName(mCalibSetClino.NameCam());
       if (aKClino != 0)
       {
            aCal.SetRelCalib(aVNamesClino.at(mVKClino.at(0)),mOriRelClin.at(aKClino));
/*
            cOneCalibRelClino aCalRel;
            aCalRel.mNameRef = aVNamesClino.at(mVKClino.at(0));
            aCalRel.mRot = mOriRelClin.at(aKClino);
            aCal.mLinkRel = aCalRel;
*/
       }
       mCalibSetClino.AddCalib1Clino(aCal);
    }
    


    // if several clino, after global opt try to make opt independantly
    if (mVKClino.size() > 1)
    {
        StdOut() <<  "=============== Result of individual  optimization  =============" << std::endl;
        std::vector<tRotR>  aREnd;
        for (size_t aKCSel=0 ; aKCSel<mVKClino.size() ; aKCSel++)
	{
            StdOut() << "   ----- OPTIMIZE CLINO " << mVKClino[aKCSel] << "   -------- " << std::endl;
            for (size_t aKC=0 ; aKC<mVKClino.size() ; aKC++)
	    {
                 mComputedClino.at(aKC) = (aKC==aKCSel);
	    }
            cWhichMin<tRotR,tREAL8>  aWMK = aWM0;
	    tREAL8 aStep = 1e-3;
            for (int aK=0 ; aK<30 ; aK++)
            {
                 aWMK =  OneIter(aWMK.IndexExtre(), (2*aStep)/mNbStepIter,mNbStepIter);
                 aStep /= aDiv;
            }
            StdOut() << "Residual=" << std::sqrt(aWMK.ValExtre())  * mUnityAng
                     << " Cond=" << ComputeCond(aWMK.IndexExtre())
		     << std::endl;

           aREnd.push_back(aWMK.IndexExtre());
	}
        /* compute Orthogonality between solution of independant clinos,  Let O be the ortognality matrix, what we optimize
            when using the 2 clino and applying the orthognality is (see  "OriOfClino") :

                 F(R) + F(O R)

            So  R2 R-1 = O is how much clino are Ortho
            
        */
        tRotR  aR1 = aREnd.at(0);
        tRotR  aR2 = aREnd.at(1);
        tRotR  aR12  = aR2 * aR1.MapInverse() ;

        StdOut() << " Orthogonality diff   N="   <<  std::abs(aR12.Angle()) * mUnityAng << "\n";
    }

    // Save the result in standard file
    mPhProj.SaveClino(mCalibSetClino);

    for (const auto & aMes : mVMeasures)
        aMes.TestCalib(mCalibSetClino,mVKClino);

    /*if (mPhProj.DPClinoMeters().DirInIsInit())
    {
       cCalibSetClino* aClinoTest = mPhProj.GetClino(*aCalib, );
       SaveInFile(*aClinoTest,"TestReWriteClino.xml");
       delete aClinoTest;
    }*/


    // Test 2 understand WPK distance / axiator ...
    if (0)
    {
        tREAL8 aEps = 1e-2;

        for (int aKW=-1 ; aKW<=1 ; aKW++)
             for (int aKP=-1 ; aKP<=1 ; aKP++)
                 for (int aKK=-1 ; aKK<=1 ; aKK++)
                     if ((aKW!=0) || (aKP!=0) || (aKK !=0))
                     {
                        cPt3dr aWPK = cPt3dr(aKW,aKP,aKK) * aEps;
                        tRotR aRot = tRotR::RotFromWPK(aWPK);
                        auto aDifId = aRot.Mat() - cDenseMatrix<tREAL8>::Identity(3);
                        tREAL8 aDwpk = Norm2(aWPK);
                        tREAL8 aDId =  aDifId.DIm().L2Norm(false);
                        StdOut() << " R=" << aDwpk / aDId   
                                          << " DW" << aDwpk/aEps   
                                          << " DA=" << aRot.Angle()/aEps
                                          << " DId="  << aDId/aEps << "\n";
                        if ((aKW==1) && (aKP==0) && (aKK==0))
                        {
                              aDifId.Show() ;
                        }
                     }
    }


    /*
    std::string aNameOut = mPhProj.DPClinoMeters().FullDirOut() + "ClinoCalib-" + aNameCalibCam + "."+ GlobTaggedNameDefSerial();
    SaveInFile(mCalibSetClino,aNameOut);
    */
    return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */


tMMVII_UnikPApli Alloc_ClinoInit(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ClinoInit(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ClinoInit
(
     "ClinoInit",
      Alloc_ClinoInit,
      "Initialisation of inclinometer",
      {eApF::Ori},
      {eApDT::Orient},
      {eApDT::Xml},
      __FILE__
);

/*
*/




}; // MMVII

