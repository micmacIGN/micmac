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
    mVertInLoc (mCam->V_W2L(cPt3dr(0,0,-1)))
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
     return Norm2(EcarVectRot(aKClino,aCam2Clino));
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
	/// for a given clinometern compute the cost of a tested boresight
        tREAL8                     CostRot(size_t aKClino,const tRot & aWPK);

	///  Make one iteration of research arround a current solution
        cWhichMin<tRot,tREAL8>   OneIter(size_t aKClino,const tRot & aR0,tREAL8 aAmpl, int aNbStep);

	///  Compute the conditionning of least-square matrix
        void ComputeCond(int aKClino,const tRot & aWPK,tREAL8 aEpsilon=1e-3);


        cPhotogrammetricProject        mPhProj;
        std::string                    mNameClino;   ///  Pattern of xml file
	std::vector<std::string>       mPrePost;     ///  Pattern of xml file
        int                            mNbStep0;
        std::vector<double>            mASim;


        std::vector<cClinoCalMes1Cam>  mVMeasures;
	int                            mKClino;

};

cAppli_ClinoInit::cAppli_ClinoInit
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mNbStep0      (10)
{
}


cCollecSpecArg2007 & cAppli_ClinoInit::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
              <<  Arg2007(mNameClino,"Name of inclination file",{eTA2007::FileDirProj})
              <<  Arg2007(mPrePost,"[Prefix,PostFix] to compute image name",{{eTA2007::ISizeV,"[2,2]"}})
              <<  Arg2007(mKClino,"Index of clinometer")
              <<  mPhProj.DPOrient().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_ClinoInit::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return     anArgOpt
	    << AOpt2007(mNbStep0,"NbIt0","Number of step at  first iteration",{eTA2007::HDV})
	    << AOpt2007(mASim,"AmplSim","Amplitude of rotation is simul [W,P,K,BS]",{{eTA2007::ISizeV,"[5,5]"}})
    ;
}

//  just the average of cost for all measurements/position
tREAL8   cAppli_ClinoInit::CostRot(size_t aKClino,const tRot & aRot)
{
    tREAL8 aSom = 0.0;

     for (const auto & aMes : mVMeasures)
         aSom +=  aMes.ScoreRot(aKClino,aRot);

     return aSom / mVMeasures.size();
}

cWhichMin<tRot,tREAL8>  cAppli_ClinoInit::OneIter(size_t aKClino,const tRot & aR0,tREAL8 aStep, int aNbStep)
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

                tREAL8 aCost = CostRot(aKClino,aRot);  // cost of tested rot
                aWMin.Add(aRot,aCost);  // update lowest cost solution
            }
        }
    }

    StdOut() << "MIN " << aWMin.ValExtre()  << "\n";
    return  aWMin;
}

void cAppli_ClinoInit::ComputeCond(int aKClino,const tRot & aRot,tREAL8 aEpsilon)
{
     cStrStat2<tREAL8>  aStat(3);  // covariant matrix

     // Add  Gx and Gy of all measures in covariance-matrix
     for (const auto & aMes : mVMeasures)
     {
        auto [aGx,aGy] = aMes.Grad(aKClino,aRot,aEpsilon);
	aStat.Add(aGx.ToVect());
	aStat.Add(aGy.ToVect());
     }
     aStat.Normalise(false);  // normalize w/o centering

     // print ratio with highest value
     cDenseVect<tREAL8>  aDV = aStat.DoEigen().EigenValues();
     StdOut() << "R2=" << std::sqrt(aDV(0)/aDV(2))  << " R1=" << std::sqrt(aDV(1)/aDV(2)) << "\n";
}

int cAppli_ClinoInit::Exe()
{
    mPhProj.FinishInit();


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

    cPt3dr aWPK_Sim (0,0,0);
    tRot aRSim = cRotation3D<tREAL8>::RotFromWPK(aWPK_Sim);

    size_t aNbMeasures = aVNames.size();
    if (IsInit(&mASim))
    {
       aWPK_Sim =  cPt3dr::PRandC() * mASim[3];
       aRSim = cRotation3D<tREAL8>::RotFromWPK(aWPK_Sim);
       mKClino = 0;
       aNbMeasures = size_t (mASim[4]);
    }

    //  put low level in a more structured data
    for (size_t aKLine=0 ; aKLine<aNbMeasures ; aKLine++)
    {
        if (IsInit(&mASim))
        {
            cPt3dr aWPK(RandUnif_C()*mASim[0],RandUnif_C()*mASim[1],RandUnif_C()*mASim[2]);
            cRotation3D<tREAL8>  aRPose =  cRotation3D<tREAL8>::RotFromWPK(aWPK);

            cSensorCamPC * aCam = new cSensorCamPC("",cIsometry3D<tREAL8>::Identity(),nullptr);
            cMMVII_Appli::AddObj2DelAtEnd(aCam);
            aCam->SetPose(cIsometry3D<tREAL8>(cPt3dr(0,0,0),aRPose));

            // mVMeasures.push_back(cClinoCalMes1Cam(aCam,aVAngles[aKLine]));
            mVMeasures.push_back(cClinoCalMes1Cam(aCam,{0.0}));
            mVMeasures.back().SetDirSimul(mKClino,aRSim);
        }
	else 
        {
            std::string aNameIm = mPrePost[0] +  aVNames[aKLine][0] + mPrePost[1];
	    cSensorCamPC * aCam = mPhProj.ReadCamPC(aNameIm,true);
            mVMeasures.push_back(cClinoCalMes1Cam(aCam,aVAngles[aKLine]));
        }
    }


    /*
    if (IsInit(&mASim))
    {
       for (auto & aMes : mVMeasures)
       {
       }
    }
    */
    

    tREAL8 aStep0 = 1.5/mNbStep0;

    cWhichMin<tRot,tREAL8> aWM0 =  OneIter(mKClino,tRot::Identity(),aStep0,mNbStep0);

    int aNbStep = 15;
    tREAL8 aDiv = 1.1;
    
    for (int aK=0 ; aK<200 ; aK++)
    {
         aStep0 /= aDiv;
         aWM0 =  OneIter(mKClino,aWM0.IndexExtre(), (2*aDiv*aStep0)/aNbStep,aNbStep);
    }

    ComputeCond(mKClino, aWM0.IndexExtre());



    tRot  aRSol =  aWM0.IndexExtre();

    if (IsInit(&mASim))
    {
       StdOut() << "GROUND TRUTH " << CostRot(mKClino,aRSol) << "\n";
       aRSol.Mat().Show();

       StdOut() << "   ============== GROUND TRUTH ======================\n";
       StdOut() << "COST " << CostRot(mKClino,aRSim) << "\n";
       ComputeCond(mKClino, aRSim);
       // cRotation3D<tREAL8>::RotFromWPK(aWPK_Sim).Mat().Show();
     }


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

