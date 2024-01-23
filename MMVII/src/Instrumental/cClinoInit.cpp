#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"


/**
   \file GCPQuality.cpp

   MMVII ClinoInit ClinoMeasures.txt [949_,.JPG] MMVII-PhgrProj/Ori/Resec_311_All/ Nbit0=24 AmplSim=[0.1,0.1,1.5,0.1] SeedRand=22


 */

namespace MMVII
{


typedef cRotation3D<tREAL8>  tRot;

class cClinoCalMes1Cam
{
    public :
        cClinoCalMes1Cam(cSensorCamPC * aCam,const std::vector<tREAL8> & aVAngles);

        void SetDirSimul(int aK,const  cRotation3D<tREAL8> &aR) ;

        cSensorCamPC *         mCam;
	std::vector<cPt2dr>     mVDir;
	cPt3dr                 mVertInLoc;  ///  Vertical in camera coordinates

	/// Theoretical Pos of needle, indicating vertical, in plane IJ, for Clino K, given a boresight aCam2Clino
        cPt2dr  PosNeedle(int aKClino,const  cRotation3D<tREAL8> &aCam2Clino) const;

	///  Vectorial difference, in plane IJ, between measure an theoreticall value for needle
        cPt2dr  EcarVectRot(int aKClino,const  cRotation3D<tREAL8> &aCam2Clino) const;

	///  Idem but with WPK
        cPt2dr  EcarVectWPK(int aKClino,const  cPt3dr &aWPK) const;

	///  Score (to minimize) between obs and theor for given rot
        tREAL8  ScoreRot(int aKClino,const  cRotation3D<tREAL8> &aR) const;
	///  Idem but with WPK
        tREAL8  ScoreWPK(int aKClino,const  cPt3dr &aWPK) const;

	///  Gradient / wpk of Needle's in x an y in plane IJ
        std::pair<cPt3dr,cPt3dr>  Grad(int aKClino,const tRot & aRot,tREAL8 aEpsilon=1e-3) const;

};


cClinoCalMes1Cam::cClinoCalMes1Cam(cSensorCamPC * aCam,const std::vector<tREAL8> & aVAngles) :
    mCam       (aCam),
    mVertInLoc (mCam->Vec_W2L(cPt3dr(0,0,-1)))
{
    for (auto & aTeta : aVAngles)
    {
        mVDir.push_back(FromPolar(1.0,aTeta));
    }
}


std::pair<cPt3dr,cPt3dr>  cClinoCalMes1Cam::Grad(int aKClino,const tRot & aR0,tREAL8 aEps) const
{
   cPt3dr aResX;
   cPt3dr aResY;
   for (int aK=0 ; aK<3 ; aK++)
   {
        cPt3dr aPEps(0,0,0);
        aPEps[aK] = aEps;
	tRot aRPlus  = tRot::RotFromWPK(aPEps);
	tRot aRMinus = tRot::RotFromWPK(-aPEps);


        cPt2dr aDelta = EcarVectRot(aKClino,aR0*aRPlus) -  EcarVectRot(aKClino,aR0*aRMinus);
        aResX[aK] = aDelta.x()/(2*aEps);
        aResY[aK] = aDelta.y()/(2*aEps);
   }
   return std::pair<cPt3dr,cPt3dr>(aResX,aResY);
}

cPt2dr  cClinoCalMes1Cam::PosNeedle(int aKClino,const  cRotation3D<tREAL8> &aCam2Clino) const
{
     cPt3dr  aVClin =  aCam2Clino.Value(mVertInLoc); // Direction of Vert in clino repair
     return  VUnit(Proj(aVClin));       // Projection in plane I,J
}


cPt2dr  cClinoCalMes1Cam::EcarVectRot(int aKClino,const  cRotation3D<tREAL8> &aCam2Clino) const
{
     return PosNeedle(aKClino,aCam2Clino) - mVDir[aKClino];
}

tREAL8  cClinoCalMes1Cam::ScoreRot(int aKClino,const  cRotation3D<tREAL8>& aCam2Clino) const
{
     return SqN2(EcarVectRot(aKClino,aCam2Clino));
}

tREAL8  cClinoCalMes1Cam::ScoreWPK(int aKClino,const  cPt3dr & aWPK) const
{
    return ScoreRot(aKClino,cRotation3D<tREAL8>::RotFromWPK(aWPK));
}

cPt2dr  cClinoCalMes1Cam::EcarVectWPK(int aKClino,const  cPt3dr &aWPK) const
{
    return EcarVectRot(aKClino,cRotation3D<tREAL8>::RotFromWPK(aWPK));
}

void cClinoCalMes1Cam::SetDirSimul(int aKClino,const  cRotation3D<tREAL8> &aR)
{
     mVDir[aKClino] = PosNeedle(aKClino,aR);
}



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

	/// Make a first iteration using sampling by quaternions
        cWhichMin<tRot,tREAL8>   IterInit(int aNbStep);

	///  Compute the conditionning of least-square matrix
        void ComputeCond(const tRot & aWPK,tREAL8 aEpsilon=1e-3);


        cPhotogrammetricProject        mPhProj;
        std::string                    mNameClino;   ///  Pattern of xml file
	std::vector<std::string>       mPrePost;     ///  Pattern of xml file
        int                            mNbStep0;
        int                            mNbStep;
        int                            mNbIter;
        std::vector<double>            mASim;


        std::vector<cClinoCalMes1Cam>  mVMeasures;
	std::vector<int>               mVKClino;
	std::vector<bool>              mComputedClino;  ///  Clino can be computed globally or independantly

	std::string                    mNameRel12;
        std::vector<tRot>              mOriRelClin;

};

cAppli_ClinoInit::cAppli_ClinoInit
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mNbStep0      (65),
     mNbStep       (10),
     mNbIter       (50),
     mNameRel12    ("i-kj")
{
}


cCollecSpecArg2007 & cAppli_ClinoInit::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
              <<  Arg2007(mNameClino,"Name of inclination file",{eTA2007::FileDirProj})
              <<  Arg2007(mPrePost,"[Prefix,PostFix] to compute image name",{{eTA2007::ISizeV,"[2,2]"}})
              <<  Arg2007(mVKClino,"Index of clinometer",{{eTA2007::ISizeV,"[1,2]"}})
              <<  mPhProj.DPOrient().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_ClinoInit::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return     anArgOpt
	    << AOpt2007(mNbStep0,"NbStep0","Number of step at  first iteration",{eTA2007::HDV})
	    << AOpt2007(mNbStep,"NbStep","Number of step at  current iteration",{eTA2007::HDV})
	    << AOpt2007(mNbIter,"NbIter","Number of iteration",{eTA2007::HDV})
	    << AOpt2007(mASim,"AmplSim","Amplitude of rotation is simul [W,P,K,BS]",{{eTA2007::ISizeV,"[5,5]"}})
	    << AOpt2007(mNameRel12,"Rel12","orientation relative 2 to 1, if several clino",{eTA2007::HDV})
    ;
}

//  just the average of cost for all measurements/position
std::vector<tREAL8>   cAppli_ClinoInit::VectCostRot(const tRot & aRot0) const
{
     std::vector<tREAL8> aRes;

     for (size_t aIndC=0 ; aIndC<mVKClino.size() ; aIndC++)
     {
         if (mComputedClino.at(aIndC))
         {
             tREAL8 aSom = 0.0;
             tRot aRot = OriOfClino(aIndC,aRot0);
             for (const auto & aMes : mVMeasures)
                  aSom +=  aMes.ScoreRot(mVKClino[aIndC],aRot);

	     aRes.push_back(aSom / mVMeasures.size());
         }
     }

     return aRes;
}

tREAL8   cAppli_ClinoInit::CostRot(const tRot & aRot0) const
{
     std::vector<tREAL8> aVCost = VectCostRot(aRot0);

     return std::accumulate(aVCost.begin(),aVCost.end(),0.0) / aVCost.size();
}



cWhichMin<tRot,tREAL8>   cAppli_ClinoInit::IterInit(int aNbStep)
{
    cWhichMin<tRot,tREAL8>  aWMin(tRot::Identity(),1e10);
    cSampleQuat aSQ(aNbStep,true);

    for (size_t aKQ=0 ; aKQ<aSQ.NbRot() ; aKQ++)
    {
        cDenseMatrix<tREAL8> aMat = Quat2MatrRot(aSQ.KthQuat(aKQ));
	tRot aRot(aMat,false);

        tREAL8 aCost = CostRot(aRot);  // cost of tested rot
        aWMin.Add(aRot,aCost);  // update lowest cost solution
    }

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

    StdOut() << "MIN " << std::sqrt(aWMin.ValExtre())  << " Step=" << aStep << std::endl;
    return  aWMin;
}


void cAppli_ClinoInit::ComputeCond(const tRot & aRot0,tREAL8 aEpsilon)
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
                auto [aGx,aGy] = aMes.Grad(mVKClino[aIndC],aRot,aEpsilon);
	        aStat.Add(aGx.ToVect());
	        aStat.Add(aGy.ToVect());
             }
	 }
     }
     aStat.Normalise(false);  // normalize w/o centering

     // print ratio with highest value
     cDenseVect<tREAL8>  aDV = aStat.DoEigen().EigenValues();
     StdOut() << "R2=" << std::sqrt(aDV(0)/aDV(2))  << " R1=" << std::sqrt(aDV(1)/aDV(2)) << std::endl;
}

int cAppli_ClinoInit::Exe()
{
    mPhProj.FinishInit();

    mComputedClino = std::vector<bool>(mVKClino.size(),true);  // initially all user clino are active

    std::vector<std::vector<std::string>> aVNames;   // for reading names of camera
    std::vector<std::vector<double>>      aVAngles;  // for reading angles of  clinometers
    std::vector<cPt3dr>                   aVFakePts; // not used, required by ReadFilesStruct

    std::string mFormat = "NFFFF";
    //  read angles and camera
    ReadFilesStruct
    (
         mNameClino,
	 mFormat,
	 0,-1,
	 '#',
	 aVNames,
	 aVFakePts,aVFakePts,
	 aVAngles,
	 false
    );

    tRot aRSim = tRot::Identity();  // Rotation for simulation
    mOriRelClin.push_back(tRot::Identity());
    mOriRelClin.push_back(tRot::RotFromCanonicalAxes(mNameRel12));

    size_t aNbMeasures = aVNames.size();
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
            std::string aNameIm = mPrePost[0] +  aVNames[aKLine][0] + mPrePost[1];
	    cSensorCamPC * aCam = mPhProj.ReadCamPC(aNameIm,true);
            mVMeasures.push_back(cClinoCalMes1Cam(aCam,aVAngles[aKLine]));
        }
    }

    cWhichMin<tRot,tREAL8>  aWM0 = IterInit(mNbStep0);

    tREAL8 aStep = 2.0/mNbStep0;
    tREAL8 aDiv = 1.25;
    
    for (int aK=0 ; aK<mNbIter ; aK++)
    {
         aWM0 =  OneIter(aWM0.IndexExtre(), (2*aStep)/mNbStep,mNbStep);
         aStep /= aDiv;
    }

    ComputeCond(aWM0.IndexExtre());
    

    // if several clino, after global opt try to make opt independantly
    if (mVKClino.size() > 1)
    {
        for (size_t aKCSel=0 ; aKCSel<mVKClino.size() ; aKCSel++)
	{
            StdOut() << "OPTIMIZE CLINO " << mVKClino[aKCSel] << std::endl;
            for (size_t aKC=0 ; aKC<mVKClino.size() ; aKC++)
	    {
                 mComputedClino.at(aKC) = (aKC==aKCSel);
	    }
            cWhichMin<tRot,tREAL8>  aWMK = aWM0;
	    tREAL8 aStep = 1e-3;
            for (int aK=0 ; aK<30 ; aK++)
            {
                 aWMK =  OneIter(aWMK.IndexExtre(), (2*aStep)/mNbStep,mNbStep);
                 aStep /= aDiv;
            }
            ComputeCond(aWMK.IndexExtre());
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

