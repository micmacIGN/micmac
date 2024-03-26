#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"


/**
   \file cClinoInit.cpp

   Compute initial value of clinometers boresight relatively to a camera.

 */

namespace MMVII
{


typedef cRotation3D<tREAL8>  tRot;

/**
 */

class cClinoCalMes1Cam
{
    public :
        cClinoCalMes1Cam
        (
              cSensorCamPC * aCam,   // camera-> only the pose is usefuul
              const std::vector<tREAL8> & aVAngles, // vector of angles of all inclinometer
              const cPt3dr &   aVerticAbs = {0,0,-1}  // position of vertical in current "absolut" system
        );

        void SetDirSimul(int aK,const  cRotation3D<tREAL8> &aR) ;

	/// Theoretical Pos of needle, indicating vertical, in plane IJ, for Clino K, given a boresight aCam2Clino
        cPt2dr  PosNeedle(const  cRotation3D<tREAL8> &aCam2Clino) const;

	///  Vectorial difference, in plane IJ, between measure an theoreticall value for needle
        cPt2dr  EcarVectRot(int aKClino,const  cRotation3D<tREAL8> &aCam2Clino) const;

	/* => Use of WPK is currently deprecated
	    ///  Idem but with WPK
            cPt2dr  EcarVectWPK(int aKClino,const  cPt3dr &aWPK) const;
	    ///  Idem but with WPK
            tREAL8  ScoreWPK(int aKClino,const  cPt3dr &aWPK) const;
	*/

	///  Score (to minimize) between obs and theor for given rot
        tREAL8  ScoreRot(int aKClino,const  cRotation3D<tREAL8> &aR) const;

	///  Gradient / wpk of Needle's  "Ecart vectorial" (in x an y) in plane IJ, computed using finite difference (now used for 
        std::pair<cPt3dr,cPt3dr>  GradEVR(int aKClino,const tRot & aRot,tREAL8 aEpsilon=1e-3) const;

    private :
        cSensorCamPC *          mCam;  ///< camera , memorization seems useless
	std::vector<cPt2dr>     mVDir; ///<  measured position of needle, computed from angles
	cPt3dr                  mVertInLoc;  ///<  Vertical in  camera system, this  is the only information usefull of camera orientation
};


cClinoCalMes1Cam::cClinoCalMes1Cam(cSensorCamPC * aCam,const std::vector<tREAL8> & aVAngles,const cPt3dr & aVertAbs) :
    mCam       (aCam),
    mVertInLoc (mCam->Vec_W2L(cPt3dr(0,0,-1)))
{
    // tranformate angles in vector position of the needle in plane I,J
    for (auto & aTeta : aVAngles)
    {
        mVDir.push_back(FromPolar(1.0,aTeta));
    }
}


std::pair<cPt3dr,cPt3dr>  cClinoCalMes1Cam::GradEVR(int aKClino,const tRot & aR0,tREAL8 aEps) const
{
   cPt3dr aResX;
   cPt3dr aResY;
   //  Parse 3 angles
   for (int aK=0 ; aK<3 ; aK++)
   {
        // compute Epsilon vector in 1 direction
        cPt3dr aPEps = cPt3dr::P1Coord(aK,aEps);

	// small rotation in direction
	tRot aRPlus  = tRot::RotFromWPK(aPEps);
	tRot aRMinus = tRot::RotFromWPK(-aPEps);

	//  compute difference in needle position
        cPt2dr aDelta = EcarVectRot(aKClino,aR0*aRPlus) -  EcarVectRot(aKClino,aR0*aRMinus);
        aResX[aK] = aDelta.x()/(2*aEps);
        aResY[aK] = aDelta.y()/(2*aEps);
   }
   return std::pair<cPt3dr,cPt3dr>(aResX,aResY);
}


cPt2dr  cClinoCalMes1Cam::PosNeedle(const  cRotation3D<tREAL8> &aCam2Clino) const
{
     // "aVClin"= Direction of Vert in clino repair, mVertInLoc being in camera system
     // we only need to know the boresight "aCam2Clino" to tranfer it in the clino system
     cPt3dr  aVClin =  aCam2Clino.Value(mVertInLoc); 
     // VClin =(x,y,z) ,  the projection of VClin in plane "I,J" is given by Proj (x,y,z) -> (x,y)
     // then we make a unitary vector to maintain the size of "needle" 
     return  VUnit(Proj(aVClin));       // Projection in plane I,J
}


cPt2dr  cClinoCalMes1Cam::EcarVectRot(int aKClino,const  cRotation3D<tREAL8> &aCam2Clino) const
{
     //  Theoretical Pos -  Measured Pos
     return PosNeedle(aCam2Clino) - mVDir[aKClino];
}

tREAL8  cClinoCalMes1Cam::ScoreRot(int aKClino,const  cRotation3D<tREAL8>& aCam2Clino) const
{
     // score of a given rotation
     return SqN2(EcarVectRot(aKClino,aCam2Clino));
}

void cClinoCalMes1Cam::SetDirSimul(int aKClino,const  cRotation3D<tREAL8> &aR)
{
     mVDir[aKClino] = PosNeedle(aR);
}

/*
tREAL8  cClinoCalMes1Cam::ScoreWPK(int aKClino,const  cPt3dr & aWPK) const
{
    return ScoreRot(aKClino,cRotation3D<tREAL8>::RotFromWPK(aWPK));
}
cPt2dr  cClinoCalMes1Cam::EcarVectWPK(int aKClino,const  cPt3dr &aWPK) const
{
    return EcarVectRot(aKClino,cRotation3D<tREAL8>::RotFromWPK(aWPK));
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
	tRot  OriOfClino(int aKCl,const tRot & aRot) const {return mOriRelClin.at(aKCl) *aRot;}

	/// for a given clinometern compute the cost of a tested boresight
        tREAL8                     CostRot(const tRot & aWPK) const;

	/// Vector of score, can be agregated  sum/max
	std::vector<tREAL8>  VectCostRot(const tRot & aWPK) const;

	///  Make one iteration of research arround a current solution
        cWhichMin<tRot,tREAL8>   OneIter(const tRot & aR0,tREAL8 aStep, int aNbStep);

	/// Make a first computation parsing "all" rotation at a given step, using sampling by quaternions
        cWhichMin<tRot,tREAL8>   IterInit(int aNbStep) const;

	///  Compute the conditionning of least-square matrix
        cPt2dr ComputeCond(const tRot & aWPK,tREAL8 aEpsilon=1e-3);


        cPhotogrammetricProject        mPhProj;     ///<  Classical structure for photogrammetric project
        std::string                    mNameClino;  ///<  Pattern of xml file
	std::vector<std::string>       mPrePost;    ///<  Pattern of xml file
        int                            mNbStep0;    ///<  Number of step in initial  first round
        int                            mNbStepIter; ///< Number of step in each iteration
        int                            mNbIter;     ///< Number of iteration
        std::vector<double>            mASim;       ///< Amplitude of simulation parameters


        std::vector<cClinoCalMes1Cam>  mVMeasures;      ///< store vectors of measure after file parsing
	std::vector<int>               mVKClino;        ///< Vecor of indexes of selected clino
	/**  If several Clino, they can be computed globally or independantly, this vector of bool allow to have
	 * the 2 computation, it's mainly for tuning and observing the effect on residual and conditionning
	 */
	std::vector<bool>              mComputedClino;  

	std::string                    mNameRel12;  ///< relative orientation of Clino1/Clino2 stored as "i-kj"
        std::vector<tRot>              mOriRelClin; ///< vector of relative orientations Clino[0]/Clino[K]
	bool                           mShowAll;    ///< Do we show all the msg relative to residuals

};

cAppli_ClinoInit::cAppli_ClinoInit
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mNbStep0      (65),
     mNbStepIter   (10),
     mNbIter       (50),
     mNameRel12    ("i-kj"),
     mShowAll      (false)
{
}


cCollecSpecArg2007 & cAppli_ClinoInit::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
              <<  Arg2007(mNameClino,"Name of inclination file") // ,{eTA2007::FileDirProj})
              <<  Arg2007(mPrePost,"[Prefix,PostFix] to compute image name",{{eTA2007::ISizeV,"[2,2]"}})
              <<  Arg2007(mVKClino,"Index of clinometer",{{eTA2007::ISizeV,"[1,2]"}})
              <<  mPhProj.DPOrient().ArgDirInMand()
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
    ;
}

//  just the average of cost for all measurements/position
std::vector<tREAL8>   cAppli_ClinoInit::VectCostRot(const tRot & aRot0) const
{
     std::vector<tREAL8> aRes;

     // Parse all clino currently selected
     for (size_t aIndC=0 ; aIndC<mVKClino.size() ; aIndC++)
     {
         if (mComputedClino.at(aIndC))
         {
             cStdStatRes aStat;
             // tREAL8 aSom = 0.0;
             tRot aRot = OriOfClino(aIndC,aRot0);
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

tREAL8   cAppli_ClinoInit::CostRot(const tRot & aRot0) const
{
     std::vector<tREAL8> aVCost = VectCostRot(aRot0);

     return std::accumulate(aVCost.begin(),aVCost.end(),0.0) / aVCost.size();
}



cWhichMin<tRot,tREAL8>   cAppli_ClinoInit::IterInit(int aNbStep) const
{
    cWhichMin<tRot,tREAL8>  aWMin(tRot::Identity(),1e10); // initialize with a "big" residual
    // structure for parsing all rotation using quaternion , with aNbStep division in each direction
    cSampleQuat aSQ(aNbStep,true);

    for (size_t aKQ=0 ; aKQ<aSQ.NbRot() ; aKQ++) // parse all quaternion
    {
        cDenseMatrix<tREAL8> aMat = Quat2MatrRot(aSQ.KthQuat(aKQ)); // quaternion -> rotation
	tRot aRot(aMat,false);

        tREAL8 aCost = CostRot(aRot);  // compute the cost of tested rot
        aWMin.Add(aRot,aCost);  // update lowest cost solution
    }

    if (mShowAll)
       StdOut() << "MIN0=  " << std::sqrt(aWMin.ValExtre())  << std::endl;
    return aWMin;
}

cWhichMin<tRot,tREAL8>  cAppli_ClinoInit::OneIter(const tRot & aR0,tREAL8 aStep, int aNbStep)
{
    cWhichMin<tRot,tREAL8>  aWMin(aR0,1e10);

    // parse the grid in 3 dimension 
    for (int aKw=-aNbStep ; aKw<= aNbStep ; aKw++)
    {
        for (int aKp=-aNbStep ; aKp<= aNbStep ; aKp++)
        {
            for (int aKk=-aNbStep ; aKk<= aNbStep ; aKk++)
            {
                cPt3dr aWPK =  cPt3dr(aKw,aKp,aKk) * aStep; // compute a small Omega-Phi-Kapa
		tRot aRot = aR0*cRotation3D<tREAL8>::RotFromWPK(aWPK);  // compute a neighbooring rotation

                tREAL8 aCost = CostRot(aRot);  // cost of tested rot
                aWMin.Add(aRot,aCost);  // update lowest cost solution
            }
        }
    }

    if (mShowAll) 
        StdOut() << "MIN " << std::sqrt(aWMin.ValExtre())  << " Step=" << aStep << std::endl;
    return  aWMin;
}


cPt2dr  cAppli_ClinoInit::ComputeCond(const tRot & aRot0,tREAL8 aEpsilon)
{
     cStrStat2<tREAL8>  aStat(3);  // covariant matrix

     // Add  Gx and Gy of all measures in covariance-matrix
     for (size_t aIndC=0 ; aIndC<mVKClino.size() ; aIndC++)
     {
         if (mComputedClino.at(aIndC))
	 {
             tRot aRot = OriOfClino(aIndC,aRot0);
             for (const auto & aMes : mVMeasures)
             {
                auto [aGx,aGy] = aMes.GradEVR(mVKClino[aIndC],aRot,aEpsilon);
	        aStat.Add(aGx.ToVect());
	        aStat.Add(aGy.ToVect());
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

    mComputedClino = std::vector<bool>(mVKClino.size(),true);  // initially all user clino are active

    // std::vector<std::vector<std::string>> aVNames;   // for reading names of camera
    // std::vector<std::vector<double>>      aVAngles;  // for reading angles of  clinometers
    // std::vector<cPt3dr>                   aVFakePts; // not used, required by ReadFilesStruct

    std::string mFormat = "ISFSF";
    cReadFilesStruct aRFS(mNameClino,mFormat,0,-1,'#');
    aRFS.Read();

    tRot aRSim = tRot::Identity();  // Rotation for simulation
    mOriRelClin.push_back(tRot::Identity());
    mOriRelClin.push_back(tRot::RotFromCanonicalAxes(mNameRel12));

    size_t aNbMeasures = aRFS.NbRead();
    //  if we do simulation, generate 
    if (IsInit(&mASim))
    {
       aRSim = tRot::RandomRot(mASim[3]);
       aNbMeasures = size_t (mASim[4]);
    }

    //  put low level in a more structured data
    for (size_t aKLine=0 ; aKLine<aNbMeasures ; aKLine++)
    {
        if (IsInit(&mASim))
        {
            auto v1 = RandUnif_C()*mASim[0];
            auto v2 = RandUnif_C()*mASim[1];
            auto v3 = RandUnif_C()*mASim[2];
            cPt3dr aWPK(v1,v2,v3);
            cRotation3D<tREAL8>  aRPose =  cRotation3D<tREAL8>::RotFromWPK(aWPK);

            cSensorCamPC * aCam = new cSensorCamPC("",cIsometry3D<tREAL8>::Identity(),nullptr);
            cMMVII_Appli::AddObj2DelAtEnd(aCam);
            aCam->SetPose(cIsometry3D<tREAL8>(cPt3dr(0,0,0),aRPose));

            // mVMeasures.push_back(cClinoCalMes1Cam(aCam,aVAngles[aKLine]));
            mVMeasures.push_back(cClinoCalMes1Cam(aCam,std::vector<tREAL8>(10,0.0)));
	    for (size_t aKCl=0 ; aKCl<mVKClino.size() ; aKCl++)
                mVMeasures.back().SetDirSimul(mVKClino[aKCl],OriOfClino(aKCl,aRSim));
        }
	else 
        {
            std::string aNameIm = mPrePost[0] +  aRFS.VNameIm().at(aKLine) + mPrePost[1];
	    cSensorCamPC * aCam = mPhProj.ReadCamPC(aNameIm,true);
            mVMeasures.push_back(cClinoCalMes1Cam(aCam,aRFS.VNums().at(aKLine)));
        }
    }

    cWhichMin<tRot,tREAL8>  aWM0 = IterInit(mNbStep0);

    tREAL8 aStep = 2.0/mNbStep0;
    tREAL8 aDiv = 1.25;
    
    for (int aK=0 ; aK<mNbIter ; aK++)
    {
         aWM0 =  OneIter(aWM0.IndexExtre(), (2*aStep)/mNbStepIter,mNbStepIter);
         aStep /= aDiv;
    }

    StdOut() <<  "=============== Result of global optimization  =============" << std::endl;
    StdOut() << "Residual=" << std::sqrt(aWM0.ValExtre()) 
            << " Cond=" << ComputeCond(aWM0.IndexExtre())
	    << std::endl;
    

    // if several clino, after global opt try to make opt independantly
    if (mVKClino.size() > 1)
    {
        StdOut() <<  "=============== Result of individual  optimization  =============" << std::endl;
        for (size_t aKCSel=0 ; aKCSel<mVKClino.size() ; aKCSel++)
	{
            StdOut() << "   ----- OPTIMIZE CLINO " << mVKClino[aKCSel] << "   -------- " << std::endl;
            for (size_t aKC=0 ; aKC<mVKClino.size() ; aKC++)
	    {
                 mComputedClino.at(aKC) = (aKC==aKCSel);
	    }
            cWhichMin<tRot,tREAL8>  aWMK = aWM0;
	    tREAL8 aStep = 1e-3;
            for (int aK=0 ; aK<30 ; aK++)
            {
                 aWMK =  OneIter(aWMK.IndexExtre(), (2*aStep)/mNbStepIter,mNbStepIter);
                 aStep /= aDiv;
            }
            StdOut() << "Residual=" << std::sqrt(aWMK.ValExtre()) 
                     << " Cond=" << ComputeCond(aWMK.IndexExtre())
		     << std::endl;
	}
    }

#if (0)

    tRot  aRSol =  aWM0.IndexExtre();

    if (IsInit(&mASim))
    {
       StdOut() << "GROUND TRUTH " << CostRot(mKClino,aRSol) << std::endl;
       aRSol.Mat().Show();

       StdOut() << "   ============== GROUND TRUTH ======================" << std::endl;
       StdOut() << "COST " << CostRot(mKClino,aRSim) << std::endl;
       ComputeCond(mKClino, aRSim);
     }
#endif


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

