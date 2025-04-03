#include "MMVII_BlocRig.h"

#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_2Include_Serial_Tpl.h"

#include "MMVII_Clino.h"
#include "MMVII_HeuristikOpt.h"


namespace MMVII
{

/* ==================================================== */
/*                                                      */
/*               cGetVerticalFromClino                  */
/*                                                      */
/* ==================================================== */

class cOptimGVFromClino : public  cDataMapping<tREAL8,2,1>
{
    public :
        cOptimGVFromClino(const cGetVerticalFromClino & aGVFC,const cPt3dr & aP0) :
           mGVFC (aGVFC),
	   mP0   (VUnit(aP0))
	{
            tRotR aR = tRotR::CompleteRON(mP0);
	    mP1 = aR.AxeJ();
	    mP2 = aR.AxeK();
	}

	cPt3dr  Delta2Pt(const cPt2dr & aDelta) const {return mP0 + mP1*aDelta.x()+mP2*aDelta.y();}

        cPt1dr Value(const cPt2dr & aDelta) const override
        {
	     return cPt1dr(mGVFC.ScoreDir3D(Delta2Pt(aDelta)));
        }
    private :
         const cGetVerticalFromClino&  mGVFC;
	 cPt3dr                        mP0;
	 cPt3dr                        mP1;
	 cPt3dr                        mP2;
};

cGetVerticalFromClino::cGetVerticalFromClino(const cCalibSetClino & aCalib,const std::vector<tREAL8> & aVAngle) :
	mCalibs (aCalib)
{
    for (const auto & aTeta : aVAngle)
        mDirs.push_back(FromPolar(1.0,aTeta));
}

tREAL8 cGetVerticalFromClino::ScoreDir3D(const cPt3dr & aDirCam) const
{
   tREAL8 aSum=0.0;
   for (size_t aK=0 ; aK<mDirs.size() ; aK++)
   {
       cPt3dr aDirClino = mCalibs.ClinosCal().at(aK).CamToClino(aDirCam);
       cPt2dr aDirNeedle = VUnit(Proj(aDirClino));
       aSum += SqN2(aDirNeedle-mDirs.at(aK));
   }

   return std::sqrt(aSum/mDirs.size());
	    
}

cPt3dr cGetVerticalFromClino::Refine(cPt3dr aP0,tREAL8 StepInit,tREAL8 StepEnd) const
{
    cOptimGVFromClino aMapOpt(*this,aP0);
    cOptimByStep<2>   aOptHeur(aMapOpt,true,10);

    auto [aSc,aDXY] = aOptHeur.Optim(cPt2dr(0,0),StepInit,StepEnd);

    return VUnit(aMapOpt.Delta2Pt(aDXY));
}


cPt3dr cGetVerticalFromClino::OptimInit(int aNbStepInSphere) const
{
    cSampleSphere3D aSS3(aNbStepInSphere);

    cWhichMin<cPt3dr,tREAL8> aWMin;
    for (int aKPt=0 ; aKPt<aSS3.NbSamples() ; aKPt++)
    {
        cPt3dr aPt = aSS3.KthPt(aKPt);
        aWMin.Add(aPt,ScoreDir3D(aPt));
    }

    return aWMin.IndexExtre();
}

cPt3dr cGetVerticalFromClino::OptimGlob(int aNbStep0,tREAL8 aStepEnd) const
{
    cPt3dr aP0 = OptimInit(aNbStep0);
    return Refine(aP0,1.0/aNbStep0,aStepEnd);
}




/* ==================================================== */
/*                                                      */
/*                  cAppli_CernInitRep                  */
/*                                                      */
/* ==================================================== */

class cAppli_CernInitRep : public cMMVII_Appli
{
     public :

        cAppli_CernInitRep(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        void ProcessOneBloc(const std::vector<cSensorCamPC *> &);
	//std::vector<std::string>  Samples() const override;

     private :
        cPhotogrammetricProject  mPhProj;
        std::string              mSpecIm;
        cBlocOfCamera *          mTheBloc;
        cSetMeasureClino         mMesClino;
// ReadMeasureClino(const std::string * aPatSel=nullptr) const;


};

cCollecSpecArg2007 & cAppli_CernInitRep::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
             <<  Arg2007(mSpecIm,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPOrient().ArgDirInMand()
             <<  mPhProj.DPRigBloc().ArgDirInMand()
             <<  mPhProj.DPMeasuresClino().ArgDirInMand()
             <<  mPhProj.DPClinoMeters().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_CernInitRep::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return    anArgOpt
    ;
}

cAppli_CernInitRep::cAppli_CernInitRep
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mTheBloc      (nullptr)
{
}

void cAppli_CernInitRep::ProcessOneBloc(const std::vector<cSensorCamPC *> & aVPC)
{
   MMVII_INTERNAL_ASSERT_tiny(aVPC.size()>=2,"Not enough cam in cAppli_CernInitRep::ProcessOneBloc");

   std::string anId = mTheBloc->IdSync(aVPC.at(0)->NameImage());
   const  cOneMesureClino & aMes = *  mMesClino.MeasureOfId(anId);
   std::string aNameIm = mMesClino.NameOfIm(aMes);
   cPerspCamIntrCalib *  aCalib = mPhProj.InternalCalibFromImage(aNameIm);



   cCalibSetClino aSetC = mPhProj.ReadSetClino(*aCalib,mMesClino.NamesClino());
   cGetVerticalFromClino aGetVert(aSetC,aMes.Angles());

   cPt3dr aVertLoc = aGetVert.OptimGlob(50,1e-8);


   StdOut() << " ID=" << anId << " Ang=" << aMes.Angles()  << " CAM=" << aNameIm << " F=" << aCalib->F() << "\n";
   StdOut() << " NAMES=" <<  mMesClino.NamesClino() << " RESIDUAL VERT=" << aGetVert.ScoreDir3D(aVertLoc) << "\n";
}


int cAppli_CernInitRep::Exe()
{
    mPhProj.FinishInit();  // the final construction of  photogrammetric project manager can only be done now

    mTheBloc = mPhProj.ReadUnikBlocCam();

    mMesClino = mPhProj.ReadMeasureClino();


    std::vector<std::vector<cSensorCamPC *>>  aVVC = mTheBloc->GenerateOrientLoc(mPhProj,VectMainSet(0));
    for (auto & aVC : aVVC)
    {
        ProcessOneBloc(aVC);
        DeleteAllAndClear(aVC);
    }

    delete mTheBloc;

    return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */

tMMVII_UnikPApli Alloc_CernInitRep(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_CernInitRep(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_CernInitRep
(
      "CERN_InitRep",
      Alloc_CernInitRep,
      "Initialize the repere local to wire/sphere/clino",
      {eApF::Ori},
      {eApDT::Orient}, 
      {eApDT::Xml}, 
      __FILE__
);

}; // MMVII

