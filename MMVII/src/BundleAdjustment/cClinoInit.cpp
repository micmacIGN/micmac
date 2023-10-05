#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"


/**
   \file GCPQuality.cpp


 */

namespace MMVII
{

class cClinoCalMes1Cam
{
    public :
        cClinoCalMes1Cam(cSensorCamPC * aCam,const std::vector<tREAL8> & aVAngles);

        void SetDirSimul(int aK,const  cRotation3D<tREAL8> &aR) ;

        cSensorCamPC *         mCam;
	std::vector<cPt2dr>     mVDir;
	cPt3dr                 mVertInLoc;  ///  Vertical in camera coordinates

        cPt2dr  EcarVectRot(int aKClino,const  cRotation3D<tREAL8> &aR) const;
        cPt2dr  EcarVectWPK(int aKClino,const  cPt3dr &aWPK) const;
        cPt2dr  PosNeedle(int aKClino,const  cRotation3D<tREAL8> &aCam2Clino) const;

        tREAL8  ScoreRot(int aKClino,const  cRotation3D<tREAL8> &aR) const;
        tREAL8  ScoreWPK(int aKClino,const  cPt3dr &aR) const;
        std::pair<cPt3dr,cPt3dr>  Grad(int aKClino,const cPt3dr & aWPK,tREAL8 aEpsilon=1e-3) const;

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


std::pair<cPt3dr,cPt3dr>  cClinoCalMes1Cam::Grad(int aKClino,const cPt3dr & aWPK,tREAL8 aEps) const
{
   cPt3dr aResX;
   cPt3dr aResY;
   for (int aK=0 ; aK<3 ; aK++)
   {
        cPt3dr aPEps(0,0,0);
        aPEps[aK] = aEps;

        cPt2dr aDelta = EcarVectWPK(aKClino,aWPK+aPEps) -  EcarVectWPK(aKClino,aWPK-aPEps);
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
        tREAL8                     CostWPK(size_t aKClino,const cPt3dr & aWPK);
        cWhichMin<cPt3dr,tREAL8>   OneIter(size_t aKClino,const cPt3dr & aV0,tREAL8 aAmpl, int aNbStep);
        void ComputeCond(int aKClino,const cPt3dr & aWPK,tREAL8 aEpsilon=1e-3);


        cPhotogrammetricProject        mPhProj;
        std::string                    mNameClino;   ///  Pattern of xml file
	std::vector<std::string>       mPrePost;     ///  Pattern of xml file
        int                            mNbStep0;
        std::vector<double>            mASim;


        std::vector<cClinoCalMes1Cam>  mVMeasures;

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
              <<  mPhProj.DPOrient().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_ClinoInit::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return     anArgOpt
	    << AOpt2007(mNbStep0,"NbIt0","Number of step at  first iteration",{eTA2007::HDV})
	    << AOpt2007(mASim,"AmplSim","Amplitude of rotation is simul [W,P,K,BS]",{{eTA2007::ISizeV,"[4,4]"}})
    ;
}

tREAL8   cAppli_ClinoInit::CostWPK(size_t aKClino,const cPt3dr & aWPK)
{
    cRotation3D<tREAL8>  aRot = cRotation3D<tREAL8>::RotFromWPK(aWPK);

    tREAL8 aSom = 0.0;

     for (const auto & aMes : mVMeasures)
         aSom +=  aMes.ScoreRot(aKClino,aRot);

     return aSom / mVMeasures.size();
}

 cWhichMin<cPt3dr,tREAL8>  cAppli_ClinoInit::OneIter(size_t aKClino,const cPt3dr & aV0,tREAL8 aStep, int aNbStep)
{
    cWhichMin<cPt3dr,tREAL8>  aWMin(aV0,1e10);

    for (int aKw=-aNbStep ; aKw<= aNbStep ; aKw++)
    {
        for (int aKp=-aNbStep ; aKp<= aNbStep ; aKp++)
        {
            for (int aKk=-aNbStep ; aKk<= aNbStep ; aKk++)
            {
                cPt3dr aV = aV0 + cPt3dr(aKw,aKp,aKk) * aStep;
                tREAL8 aCost = CostWPK(aKClino,aV);

                aWMin.Add(aV,aCost);
            }
        }
    }

    StdOut() << "MIN " << aWMin.ValExtre()  << "\n";
    return  aWMin;
}

void cAppli_ClinoInit::ComputeCond(int aKClino,const cPt3dr & aWPK,tREAL8 aEpsilon)
{
     std::vector<cPt3dr> aVPt;
     for (const auto & aMes : mVMeasures)
     {
        auto [aGx,aGy] = aMes.Grad(aKClino,aWPK,aEpsilon);
        aVPt.push_back(aGx);
        aVPt.push_back(aGy);
     }
     StdOut() << "Pl=" << L2_PlanarityIndex(aVPt)  << " Lin=" << L2_LinearityIndex(aVPt) << "\n";
}

int cAppli_ClinoInit::Exe()
{
    mPhProj.FinishInit();


    std::vector<std::vector<std::string>> aVNames;
    std::vector<std::vector<double>>      aVAngles;
    std::vector<cPt3dr>                   aVFakePts;

    ReadFilesStruct
    (
         mNameClino,
	 "NNFNF",
	 0,-1,
	 -1,
	 aVNames,
	 aVFakePts,aVFakePts,
	 aVAngles,
	 false
    );

    for (size_t aKLine=0 ; aKLine<aVNames.size() ; aKLine++)
    {
        std::string aNameIm = mPrePost[0] +  aVNames[aKLine][0] + mPrePost[1];
	cSensorCamPC * aCam = mPhProj.ReadCamPC(aNameIm,true);

        if (IsInit(&mASim))
        {
            cPt3dr aWPK(RandUnif_C()*mASim[0],RandUnif_C()*mASim[1],RandUnif_C()*mASim[2]);
            cRotation3D<tREAL8>  aRPose =  cRotation3D<tREAL8>::RotFromWPK(aWPK);

// aRPose.Mat().Show() ; getchar();
            aCam->SetPose(cIsometry3D<tREAL8>(cPt3dr(0,0,0),aRPose));
        }
        mVMeasures.push_back(cClinoCalMes1Cam(aCam,aVAngles[aKLine]));
    }

    int aKClino = 1;
    cPt3dr aWPK_Sim;

    if (IsInit(&mASim))
    {
       aWPK_Sim = cPt3dr::PRandC() * mASim[3];
       cRotation3D<tREAL8>  aRBoresight =  cRotation3D<tREAL8>::RotFromWPK(aWPK_Sim);
        
       for (auto & aMes : mVMeasures)
       {
           aMes.SetDirSimul(aKClino,aRBoresight);
       }
    }
    

    tREAL8 aStep0 = 1.5/mNbStep0;

    cWhichMin<cPt3dr,tREAL8> aWM0 =  OneIter(aKClino,cPt3dr(0,0,0),aStep0,mNbStep0);

    int aNbStep = 15;
    
    for (int aK=0 ; aK<20 ; aK++)
    {
         aStep0 /= 4.5;
         aWM0 =  OneIter(aKClino,aWM0.IndexExtre(),aStep0,aNbStep);
    }
    ComputeCond(aKClino, aWM0.IndexExtre());


    StdOut() << "VALUE= " << aWM0.IndexExtre() << "\n";

    cRotation3D<tREAL8>  aR =  cRotation3D<tREAL8>::RotFromWPK(aWM0.IndexExtre());

    if (IsInit(&mASim))
    {
       aR.Mat().Show();
       StdOut() << "GROUND TRUTH " << aWPK_Sim << " C=" << CostWPK(aKClino,aWPK_Sim) << "\n";
       cRotation3D<tREAL8>::RotFromWPK(aWPK_Sim).Mat().Show();
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

