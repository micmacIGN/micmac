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

class cResultWireD
{
    public :
        std::string  mId;
        tREAL8       mDistV;
        tREAL8       mDistH;
        tREAL8       mDistG;
};
void AddData(const  cAuxAr2007 & anAux,cResultWireD & aRW)
{
    AddData(cAuxAr2007("Id",anAux),aRW.mId);
    AddData(cAuxAr2007("DV",anAux),aRW.mDistV);
    AddData(cAuxAr2007("DH",anAux),aRW.mDistH);
    AddData(cAuxAr2007("DG",anAux),aRW.mDistG);
}

/* ==================================================== */
/*                                                      */
/*               cGetVerticalFromClino                  */
/*                                                      */
/* ==================================================== */

/**  Class for computing the vertical in the repair of of the object linked to the clino.

       Use  "cGetVerticalFromClino" for the score function and interface as a "cDataMapping<tREAL8,2,1>"
    to use  the "cOptimByStep<2>"  . The computation is done in a "tangent space" arround an initial solution.
*/

class cOptimGVFromClino : public  cDataMapping<tREAL8,2,1>
{
    public :
        /** Constructor, take the initial point and the scoring function */
        cOptimGVFromClino(const cGetVerticalFromClino & aGVFC,const cPt3dr & aP0) :
           mGVFC (aGVFC),
	   mP0   (VUnit(aP0))
	{
            tRotR aR = tRotR::CompleteRON(mP0);  // complete an orthogonal bas
	    mP1 = aR.AxeJ();
	    mP2 = aR.AxeK();
	}

        /// Convert a "small" point of the plane to a point in tanget space
	cPt3dr  Delta2Pt(const cPt2dr & aDelta) const {return VUnit(mP0 + mP1*aDelta.x()+mP2*aDelta.y());}

        /// scoring function to be optimized
        cPt1dr Value(const cPt2dr & aDelta) const override
        {
	     return cPt1dr(mGVFC.ScoreDir3D(Delta2Pt(aDelta)));
        }
    private :
         const cGetVerticalFromClino&  mGVFC;
	 cPt3dr                        mP0;   ///< initial solution on the sphere
	 cPt3dr                        mP1;   ///< first direction of the  tangent space
	 cPt3dr                        mP2;   ///< second direction of the  tangent space
};

cGetVerticalFromClino::cGetVerticalFromClino(const cCalibSetClino & aCalib,const std::vector<tREAL8> & aVAngle) :
	mCalibs (aCalib)
{
    mVAngles = aVAngle;
    for (const auto & aTeta : aVAngle)  // convert angle to direction in repair
    {
        mDirs.push_back(FromPolar(1.0,aTeta));
        mVSinAlpha.push_back(std::sin(aTeta));
    }
}



tREAL8 cGetVerticalFromClino::ScoreDir3D(const cPt3dr & aDirCam) const
{
   tREAL8 aSum=0.0;
   for (size_t aK=0 ; aK<mDirs.size() ; aK++)
   {
       const  cOneCalibClino & aCalib =  mCalibs.ClinosCal().at(aK);
       cPt3dr aDirClino =  aCalib.CamToClino(aDirCam);
       if ( aCalib.Type() == eTyClino::ePendulum)
       {
           cPt2dr aDirNeedle = VUnit(Proj(aDirClino));
           aSum += SqN2(aDirNeedle-mDirs.at(aK));
       }
       else if (aCalib.Type() == eTyClino::eSpring)
       {
            aSum += Square(aDirClino.y()-mVSinAlpha.at(aK));

            // The sinus is udefined up to a sign change, so we have to rectify, btw the spring clino cannot ??  be up-down
            if (aDirClino.x()<0)
               aSum +=  -aDirClino.x();
       }
       else
       {
       }
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

std::pair<tREAL8,cPt3dr> cGetVerticalFromClino::OptimGlob(int aNbStep0,tREAL8 aStepEnd) const
{
    cPt3dr aPt = OptimInit(aNbStep0);
    aPt =  Refine(aPt,1.0/aNbStep0,aStepEnd);

    return std::pair<tREAL8,cPt3dr>(ScoreDir3D(aPt),aPt);
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

        void ProcessOneBloc(const std::vector<cSensorCamPC *> &,int aKIter);
	//std::vector<std::string>  Samples() const override;

     private :
        cPhotogrammetricProject  mPhProj;
        std::string              mSpecIm;
        cBlocOfCamera *          mTheBloc;
        cSetMeasureClino         mMesClino;
        bool                     mTestAlreadyV;  ///< If true, repair is already verticalized, just used as test 
        int                      mNbMinTarget;   ///< Required minimal number of Target identified
        std::string              mNameFileSave;
// ReadMeasureClino(const std::string * aPatSel=nullptr) const;

        std::map<std::string,cResultWireD>  mResults;
};

cCollecSpecArg2007 & cAppli_CernInitRep::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
             <<  Arg2007(mSpecIm,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPOrient().ArgDirInMand()
             <<  mPhProj.DPRigBloc().ArgDirInMand()
             <<  mPhProj.DPMeasuresClino().ArgDirInMand()
             <<  mPhProj.DPClinoMeters().ArgDirInMand()
             <<  mPhProj.DPGndPt3D().ArgDirInMand()
             <<  mPhProj.DPGndPt2D().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_CernInitRep::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return      anArgOpt
             << AOpt2007(mTestAlreadyV,"TestAlreadyV","If repair is already verticalized, for test",{{eTA2007::HDV}})
             << AOpt2007(mNbMinTarget,"NbMinTarget","Number minimal of target required",{{eTA2007::HDV}})
             << AOpt2007(mNameFileSave,"NameFileSave","Name file for saving results",{{eTA2007::HDV}})
    ;
}

cAppli_CernInitRep::cAppli_CernInitRep
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mTheBloc      (nullptr),
     mTestAlreadyV (false),
     mNbMinTarget  (5)
{
}

void cAppli_CernInitRep::ProcessOneBloc(const std::vector<cSensorCamPC *> & aVPC,int aKIter)
{
   MMVII_INTERNAL_ASSERT_tiny(aVPC.size()>=2,"Not enough cam in cAppli_CernInitRep::ProcessOneBloc");

   // 0  ================  PREPARATION ==========================================
   std::string anId = mTheBloc->IdSync(aVPC.at(0)->NameImage());    // Extract the Time Identifier
   const  cOneMesureClino & aMes = *  mMesClino.MeasureOfId(anId);  // Extract the set of clino mes of time

   // 
   std::string aNameIm = mMesClino.NameOfIm(aMes); // Extract image  "Master" of clino measure
   cPerspCamIntrCalib *  aCalib = mPhProj.InternalCalibFromImage(aNameIm);  

   cSensorCamPC * aCamClino = nullptr;
   for (auto aCamPtr : aVPC)
   {
       if (aCamPtr->InternalCalib()==aCalib)
       {
            MMVII_INTERNAL_ASSERT_tiny(aCamClino==nullptr,"Multiple cam of calib for clino");
            aCamClino = aCamPtr;
       }
   }
   MMVII_INTERNAL_ASSERT_tiny(aCamClino!=nullptr,"None cam of calib for clino");
  

   cCalibSetClino aSetC = mPhProj.ReadSetClino(*aCalib,mMesClino.NamesClino());
   cGetVerticalFromClino aGetVert(aSetC,aMes.Angles());

   // 1 =========  VERTICAL =====================================================
   auto[aScoreVert, aVertLocCamDown]  = aGetVert.OptimGlob(50,1e-9);  // this vertical in camera coordinates (only use clino calib)
   cPt3dr aVertPannelDow = aCamClino->Vec_L2W(aVertLocCamDown); // verical in "word of pannel" coordinates

   if (mTestAlreadyV)
   {
      StdOut() << "TestV: " << aVertPannelDow << "\n";
      return;
   }

   // 2 =========  FIL ==========================================================

   cSetMesGndPt  aMesPts;
   aMesPts.AddMes3D(mPhProj.LoadGCP3D());

   std::vector<cPlane3D>  aVPlane;
   for (const auto & aCam : aVPC)
   {
       const std::string & aNameIm = aCam->NameImage();
       if (mPhProj.HasFileLines(aNameIm))
       {
           cLinesAntiParal1Im   aSetL  = mPhProj.ReadLines(aNameIm);
           const std::vector<cOneLineAntiParal> & aVL  =     aSetL.mLines;

           // At this step we dont handle multiple lines
           if (aVL.size()==1)
           {
              tSeg2dr aSeg = aVL.at(0).mSeg;
              aVPlane.push_back(aCam->SegImage2Ground(aSeg));
           }
       }
       if (mPhProj.HasMeasureIm(aNameIm))
       {
           aMesPts.AddMes2D(mPhProj.LoadMeasureIm(aNameIm),nullptr,aCam);
       }
   }
   if (aVPlane.size() < 2) return;
   cSegmentCompiled<tREAL8,3> aLocSegWire = cPlane3D::InterPlane(aVPlane);


   // 3 =========  TARGET =======================================================

   std::vector<cPt3dr>  aVecPloc;
   std::vector<cPt3dr>  aVecPSphere;
   for (const auto & aMultImPt : aMesPts.MesImOfPt())
   {
       if (aMultImPt.VMeasures().size()>=2)
       {
           aVecPloc.push_back(aMesPts.BundleInter(aMultImPt));
           aVecPSphere.push_back(aMesPts.MesGCPOfMulIm(aMultImPt).mPt);
       }
   }
   if ((int)aVecPloc.size()<mNbMinTarget) return; // require a bit redundancy

   tPoseR aPosLoc2Sphere = RobustIsometry(aVecPloc,aVecPSphere);
   cWeightAv<tREAL8>  aWAvRes;
   for (size_t aK=0 ; aK<aVecPloc.size() ; aK++)
   {
       aWAvRes.Add(1.0,Norm2(aPosLoc2Sphere.Value(aVecPloc.at(aK))-aVecPSphere.at(aK)));
   }

   cPt3dr  aCenterInSphere = aMesPts.MesGCPOfName("CENTRE").mPt;
   cPt3dr  aPCenterLoc  = aPosLoc2Sphere.Inverse(aCenterInSphere);


//    aVertLoc = - aVertLoc;
//             tRotR aR = tRotR::CompleteRON(mP0);

   //  4 ================  Creat New Reper ========================================
       // 4.1 create the repair IJK such that 
       //    *   K is vertical UP
       //    *   I is ~ the direction of wire and orthog to k
       //    *   J is 3rd axe
   cPt3dr aNewAxeK = - VUnit(aVertPannelDow);                // Axes Vertical , goes Up 
   cPt3dr aNewAxeJ = VUnit(aNewAxeK ^ aLocSegWire.V12());  // Axe  J = K ^ i
   cPt3dr aNewAxeI = aNewAxeJ ^aNewAxeK;                   // I = J ^K , projection of wire on horizontal line

   //   P : Cam-> W1       M : W1->W2       M*NewK=K  ...
   //   NewP =   aM-1  * aP
   //   aTr + M-1 aPCenterLoc = 0,0,0

    tRotR  aNewR(aNewAxeI,aNewAxeJ,aNewAxeK,false);
    if (aKIter==0)
    {
        tPoseR aNewP(-aNewR.Inverse(aPCenterLoc),aNewR.MapInverse());
        // tPoseR aNewP(cPt3dr(0,0,0),aNewR);
        for (auto & aPtrCam :  aVPC)
        {
             aPtrCam->SetPose(aNewP * aPtrCam->Pose());
        }
    }
    else 
    {
       cPt3dr aDirW =  aLocSegWire.V12() ;
       MMVII_INTERNAL_ASSERT_tiny(Norm2(aPCenterLoc)  < 1e-8,"Pb with origin on center");
       MMVII_INTERNAL_ASSERT_tiny(std::abs(aDirW.y()) < 1e-8,"Pb with wire orientation");

       tREAL8 aSteep =  atan2(std::abs(aDirW.z()),std::abs(aDirW.x()));

       cPt3dr aCSphere(0,0,0);
       cPt3dr aVProjSph = aLocSegWire.Proj(aCSphere) - aCSphere;

       cResultWireD aRW;
       aRW.mId   =  anId;
       aRW.mDistH  =   Norm2(Proj(aVProjSph));
       aRW.mDistV  =   std::abs(aVProjSph.z());
       aRW.mDistG =     Norm2(aVProjSph);

       mResults[anId] = aRW;

       StdOut() << "ID=" << anId << " Angles="  << aMes.Angles()  << " ScoreV=" << aScoreVert << "\n";
       StdOut() << " WIRE , Steep : " << aSteep  << " DHor=" << aRW.mDistH << " DVert=" << aRW.mDistV << " DGlob=" << aRW.mDistG << "\n";
    }

   //  NewRep 

/*
   StdOut() << " ===========================================================================\n";
   StdOut() << " WIRE DIR " << aLocSegWire.V12() << "\n";
   StdOut() << " DET=" << aNewR.Mat().Det() << "\n";
   StdOut() << " 3D-NBMES=" << aVecPloc.size() << " Residu=" << aWAvRes.Average() <<  " PCL " << aPCenterLoc << "\n";
   StdOut() << " ID=" << anId << " Ang=" << aMes.Angles()  << " CAM=" << aNameIm << " F=" << aCalib->F() << "\n";
   StdOut() << " NAMES=" <<  mMesClino.NamesClino() << " RESIDUAL VERT=" << aGetVert.ScoreDir3D(aVertLocDown) << "\n";
*/
}


int cAppli_CernInitRep::Exe()
{
    mPhProj.FinishInit();  // the final construction of  photogrammetric project manager can only be done now

    if (IsInit(&mNameFileSave) && ExistFile(mNameFileSave))
    {
          ReadFromFile(mResults,mNameFileSave);
    }

    mTheBloc = mPhProj.ReadUnikBlocCam();

    mMesClino = mPhProj.ReadMeasureClino();


    std::vector<std::vector<cSensorCamPC *>>  aVVC = mTheBloc->GenerateOrientLoc(mPhProj,VectMainSet(0));

    for (auto & aVPannel : aVVC)
    {
        std::vector<cSensorCamPC *> aVecCam = aVPannel;

        if ( mTestAlreadyV)
        {
            aVecCam.clear();
            for (const auto & aPtr : aVPannel)
            {
                aVecCam.push_back(mPhProj.ReadCamPC(aPtr->NameImage(),DelAuto::No));
            }
        }

        int aNbIter = mTestAlreadyV ? 1 : 2;
        for (int aKIter=0 ; aKIter<aNbIter ; aKIter++)
        {
            ProcessOneBloc(aVecCam,aKIter);
        }
        DeleteAllAndClear(aVPannel);
        if (mTestAlreadyV)
           DeleteAllAndClear(aVecCam);
    }

    if (IsInit(&mNameFileSave))
    {
         SaveInFile(mResults,mNameFileSave);
         cStdStatRes aStatH;
         cStdStatRes aStatG;
         for (const auto & [anId,aRW] : mResults)
         {
             aStatH.Add(aRW.mDistH);
             aStatG.Add(aRW.mDistG);
         }
         StdOut() << "HOR  Avg=" << aStatH.Avg() <<  " ECT=" << aStatH.UBDevStd(-1) << "\n";
         StdOut() << "3D   Avg=" << aStatG.Avg() <<  " ECT=" << aStatG.UBDevStd(-1) << "\n";
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

