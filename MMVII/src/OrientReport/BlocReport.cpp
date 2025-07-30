#include "MMVII_BlocRig.h"

#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Clino.h"

/*
 Pt=Center Avg=0.0946877 StdDev=1.62817e-05
 Pt=Center Avg=0.0946488 StdDev=7.36029e-06
 Pt=Center Avg=0.0946661 StdDev=7.67083e-06
 Pt=Center Avg=0.0946694 StdDev=8.27856e-06
 Pt=Center Avg=0.094679  StdDev=1.27643e-05

*/


/**
 
   \file cWire3DInit.cpp

   This file contains a command for computing the 3D position of a wire from multiple
   images. 

 */

namespace MMVII
{


/* ==================================================== */
/*                                                      */
/*          cAppli_CalibratedSpaceResection             */
/*                                                      */
/* ==================================================== */

class  cStatDistPtWire
{
    public :
        cStdStatRes  mStat3d;  // stat on 3D dist
        cStdStatRes  mStatH;  // stat on horiz dist
        cStdStatRes  mStatV;  // stat on vert dist
};

class cAppli_ReportBlock : public cMMVII_Appli
{
     public :

        cAppli_ReportBlock(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

	//  std::vector<std::string>  Samples() const override;

     private :
	void ProcessOneBloc(const std::vector<cSensorCamPC *> &);

        void AddStatDistWirePt(const cPt3dr& aPt,const cPt3dr& aDirLoc,const std::string &);

        void TestWire3D(const std::string& anIdSync,const std::vector<cSensorCamPC *> & aVCam);
        /// For a given Id of Sync and a bloc of cameras, compute the stat on accuracy of intersection
        void TestPoint3D(const std::string& anIdSync,const std::vector<cSensorCamPC *> & aVCam);

        cPhotogrammetricProject  mPhProj;

	std::string                 mSpecImIn;
        cBlocOfCamera *             mTheBloc;

        std::string                  mRepW;
        std::string                  mIdRepPtIndiv;
        std::string                  mIdRepDWirePt;
        std::string                  mIdRepPtGlob;
        std::string                  mPatNameGCP;

        std::string                  mStrM2T;  /// String of measure to test
        std::string                  mAddExReport;
        cWeightAv<tREAL8,tREAL8>     mAvgGlobRes;
        cStdStatRes                  mStatGlobPt;
        std::vector<int>             mPercStat;
        std::map<std::string,cStdStatRes>    mMapStatPair;
        std::map<std::string,cStdStatRes>    mMap1Image;
      
        //  Add a statistic results in csv-file
        void CSV_AddStat(const std::string& anId,const std::string& aMes,const cStdStatRes &) ;
	cSegmentCompiled<tREAL8,3> *  mCurWire ;

        std::map<std::string,cStatDistPtWire>   mStatByPt;
        //  Stuff  for  CERN-Sphere-Center like manip
        bool                                    mCernStat;
        cPt3dr                                  mSphereCenter; // center of the sphere
        bool                                    mSCFreeScale;  // do we have a free scale 
        cSetMesGnd3D                            mGCP3d;
        cSetMeasureClino                        mMesClino;
};


/*
std::vector<std::string>  cAppli_Repo&mAddExReportrtBlock::Samples() const
{
    return {
	     "MMVII BlockCamInit SetFiltered_GCP_OK_Resec.xml   BA_311_B   '(.*)_(.*).JPG' [1,2]  Rig_311_B"
    };
}
*/


cAppli_ReportBlock::cAppli_ReportBlock
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mRepW         ("Wire"),
     mIdRepPtIndiv ("Pt"),
     mIdRepDWirePt ("DistWP"),
     mIdRepPtGlob  ("GlobPt"),
     mPatNameGCP   (".*"),
     mStrM2T       ("TW"),
     mPercStat     {15,25,50,75,85},
     mSphereCenter (0,0,0)
{
}


cCollecSpecArg2007 & cAppli_ReportBlock::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
             <<  Arg2007(mSpecImIn,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPOrient().ArgDirInMand()
             <<  mPhProj.DPGndPt2D().ArgDirInMand()
             <<  mPhProj.DPRigBloc().ArgDirInMand()
           ;
}


cCollecSpecArg2007 & cAppli_ReportBlock::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return      anArgOpt
             << AOpt2007(mPatNameGCP,"PatFiltGCP","Pattern to filter name of GCP",{{eTA2007::HDV}})
             << AOpt2007(mStrM2T,"M2T","Measure to test : T-arget W-ire",{{eTA2007::HDV}})
             << AOpt2007(mAddExReport,"AddExRep","Addditional Extension in Report Name",{{eTA2007::HDV}})
             << AOpt2007(mPercStat,"PercStat","Percentils for stat in global report",{{eTA2007::HDV}})

             << mPhProj.DPGndPt3D().ArgDirInOpt()
             << AOpt2007(mSphereCenter,"ShereC","Additionnal GPC to export")
             << mPhProj.DPMeasuresClino().ArgDirInOpt()
             << mPhProj.DPClinoMeters().ArgDirInOpt()
    ;
}


void cAppli_ReportBlock::AddStatDistWirePt(const cPt3dr& aPt,const cPt3dr& aVertLoc,const std::string & aName)
{
    if (!mCurWire)
       return;

     cPt3dr  aProj = mCurWire->Proj(aPt);
     cPt3dr  anEc = aProj-aPt;
     cPt3dr  aCompV = aVertLoc * Scal(aVertLoc,anEc);
     cPt3dr  aCompH = anEc-aCompV;

     mStatByPt[aName].mStat3d.Add(Norm2(anEc));
     mStatByPt[aName].mStatH.Add(Norm2(aCompH));
     mStatByPt[aName].mStatV.Add(Norm2(aCompV));

     AddOneReportCSV(mIdRepDWirePt,{ToStr(Norm2(anEc)),ToStr(Norm2(aCompH)),ToStr(Norm2(aCompV))});
}

void cAppli_ReportBlock::TestWire3D(const std::string & anIdSync,const std::vector<cSensorCamPC *> & aVCam)
{

     std::vector<cPlane3D>  aVPlane;
     std::vector<cSensorCamPC *>  aVCamOk;
     std::vector<tSeg2dr>         aVSegOk;

     for (const auto & aCam : aVCam)
     {
          const std::string & aNameIm = aCam->NameImage();
          if (mPhProj.HasFileLines(aNameIm))
	  {
              cLinesAntiParal1Im   aSetL  = mPhProj.ReadLines(aNameIm);
	      const std::vector<cOneLineAntiParal> & aVL  = 	aSetL.mLines;

	      // At this step we dont handle multiple lines
	      if (aVL.size()==1)
	      {
                 tSeg2dr aSeg = aVL.at(0).mSeg;

	         aVPlane.push_back(aCam->SegImage2Ground(aSeg));
	         aVCamOk.push_back(aCam);
	         aVSegOk.push_back(aSeg);
	      }
	  }
     }


     int aNbPl = aVPlane.size();

     if (aNbPl>=2)
     {
	mCurWire  = new cSegmentCompiled<tREAL8,3>(cPlane3D::InterPlane(aVPlane));
        if (aNbPl>=3)
        {
	    cWeightAv<tREAL8> aWGr;  // Average of ground distance
	    cWeightAv<tREAL8> aWPix; // Average of pixel distance

            // Parse the camera where a seg was detected
	    for (size_t aKC=0 ; aKC<aVCamOk.size() ; aKC++)
	    {
                cSensorCamPC * aCam = aVCamOk[aKC];
                cPerspCamIntrCalib * aCalib = aCam->InternalCalib();

	        int aNSampleSeg = 5; //  Number of sample on the seg

	        for (int aKS=0 ; aKS<=aNSampleSeg ; aKS++)
	        {
                      // compute a point on curve corresponding to the undistorde line
		      cPt2dr aPIm = aCalib->InterpolOnUDLine(aVSegOk[aKC],aKS/tREAL8(aNSampleSeg));
		      aWGr.Add(1.0,aCam->GroundDistBundleSeg(aPIm,*mCurWire));
		      aWPix.Add(1.0,aCam->PixDistBundleSeg(aPIm,*mCurWire));
	         }
	    }
	    tREAL8 aRatio = aNbPl /(aNbPl-2.0);
            tREAL8 aDist3D =  aWGr.Average() * aRatio;
            tREAL8 aDistPix =  aWPix.Average() * aRatio;

            AddOneReportCSV(mRepW,{anIdSync,ToStr(aNbPl),ToStr(aDist3D),ToStr(aDistPix)});

            StdOut() << " =================== Cam-Wire N2 ======================= \n";
            for (const auto & aCam : aVCam)
            {
               cPt3dr  aProj = mCurWire->Proj(aCam->Center());
               StdOut() << "  * N2=" << Norm2(aProj-aCam->Center()) << " Cam=" << aCam->NameImage() << "\n";
            }

            StdOut() << "  ================== Cam-Cam  N2=========================\n";
            for (size_t aK1=0 ; aK1<aVCam.size() ; aK1++)
            {
                for (size_t aK2=aK1+1 ; aK2<aVCam.size() ; aK2++)
                {
                      StdOut() << " *  N2=" << Norm2(aVCam[aK1]->Center()-aVCam[aK2]->Center()) 
                                                 << " Cam=" << aVCam[aK1]->NameImage() << " " << aVCam[aK2]->NameImage()  << "\n";
                }
            }
        }
    }
}

typedef std::pair<cSensorCamPC *,cMesIm1Pt> tPairCamPt;

void cAppli_ReportBlock::CSV_AddStat(const std::string& anId,const std::string& aMes,const cStdStatRes & aStat) 
{
   AddStdStatCSV(anId,aMes,aStat,mPercStat);
}

void cAppli_ReportBlock::TestPoint3D(const std::string & anIdSync,const std::vector<cSensorCamPC *> & aVCam)
{
     // for a given name of point, store  Mes+Cam , that will allow to compute bundles
     std::map<std::string,std::vector<tPairCamPt>> aMapMatch;
     cStdStatRes  aStatRes;

     // [1]  Parse all the camera to group measur by name of point (A) load points (B) parse them to store image measure + Cam 
     for (const auto & aCam : aVCam)
     {
          // if images measures were  computed
          int aNbMesOK = 0;
          if (mPhProj.HasMeasureIm(aCam->NameImage()))
          {
	     cSetMesPtOf1Im  aSet = mPhProj.LoadMeasureIm(aCam->NameImage()); // (A) Load the points

	     for (const auto & aMes : aSet.Measures()) // (B) parse the points
	     {
                 // Dont select points if NotCodes or not selected by user-regex-filtering
                 if ((!starts_with( aMes.mNamePt,MMVII_NONE)) && MatchRegex(aMes.mNamePt,mPatNameGCP))
	         {
                    aNbMesOK++;
                    aMapMatch[aMes.mNamePt].push_back(tPairCamPt(aCam,aMes));
	         }
	     }
             if (aNbMesOK==0)
             {
                 StdOut() << "NO Measure valide  for " << aCam->NameImage() << "\n";
             }
	  }
          else
             StdOut() << "NO Measure file  for " << aCam->NameImage() << "\n";
     }

     // memrize the 3D points
     std::vector<cPt3dr> mV3dLoc;
     std::vector<std::string> mVNames;
     // [2]  Parse the measure grouped by points
     for (const auto & [aNamePt,aVect] : aMapMatch )
     {
         int aNbPt = aVect.size();
         if (aNbPt  > 2) 
         {
	     // StdOut() << " NAME=" << aStr << " " << aList.size() << "\n";
             std::vector<tSeg3dr> aVSeg;
	     for (const auto & [aCam,aMes] : aVect)
	     {
                 aVSeg.push_back(aCam->Image2Bundle(aMes.mPt));
	     }
	     cPt3dr aPG =   BundleInters(aVSeg);
             mV3dLoc.push_back(aPG);
             mVNames.push_back(aNamePt);

             if (aNamePt== "07183")
             {
                StdOut() << " ============== Cam-"<< aNamePt << "  ======================\n";
                StdOut()  <<  " PGround=" << aPG << "\n";
             }

	     cWeightAv<tREAL8> aWPix;
	     for (const auto & [aCam,aMes] : aVect)
	     {
                 cPt2dr aPProj = aCam->Ground2Image(aPG);
                 aWPix.Add(1.0,Norm2(aMes.mPt-aPProj));
                 if (1) // (aNamePt== "07183")
                 {
                      cSegmentCompiled<tREAL8,3>  aSeg ( aCam->Image2Bundle(aMes.mPt));

                      StdOut() << " REPROJ " << aMes.mPt - aCam->Ground2Image(aSeg.P2()) << "\n";
                      cPt3dr aDirGround = VUnit(aSeg.V12());
                      cPt3dr aDirCam   = VUnit(aCam->InternalCalib()->DirBundle(aMes.mPt));
                       aDirCam = MulCByC(aDirCam,cPt3dr(1.0,-1.0,-1.0));
                      // cPt3dr aProjBundle = aSeg.Proj(aPG);
                      StdOut() /*<< " * N2=" << Norm2(aCam->Center()-aPG) 
                               << " Cam=" << aCam->NameImage() 
                               << " ResPix=" << Norm2(aMes.mPt-aPProj)
                               << " ResMM=" << aSeg.Dist(aPG) * 1e6
                               << " P->Proj=" << (aSeg.Proj(aPG) -aPG) * 1e6 */
                               << " Dir-Ground=" << aDirGround
                               << " Dir-Cam=" << aDirCam
                               << " C=" << aCam->Center()
                               // << " PUD" << aCam->InternalCalib()->Undist(aMes.mPt)
                               << "\n";
                 }
		 // StdOut() << " DDDD = " << Norm2(aMes.mPt-aPProj) << "\n";
	     }
             tREAL8 aDistPix = aWPix.Average() * (aNbPt*2.0) / (aNbPt*2.0 -3.0);
             AddOneReportCSV(mIdRepPtIndiv,{anIdSync,aNamePt,ToStr(aNbPt),ToStr(aDistPix)});
             aStatRes.Add(aDistPix);
             mStatGlobPt.Add(aDistPix);

             //  Now make the computation by pair of camera 
             for (size_t aK1=0 ; aK1<aVect.size() ; aK1++)
             {
	         const auto & [aCam1,aMes1] = aVect.at(aK1);
                 for (size_t aK2=aK1+1 ; aK2<aVect.size() ; aK2++)
                 {
	             const auto & [aCam2,aMes2] = aVect.at(aK2);
                     cHomogCpleIm aCple(aMes1.mPt,aMes2.mPt);
                     tREAL8 aRes12 = aCam1->PixResInterBundle(aCple,*aCam2) * 4.0;  // 4.0 = DOF = 4 / (4-3)
                     std::string anId1 = "Cam:"+mTheBloc->IdBloc(aCam1->NameImage());
                     std::string anId2 = "Cam:"+mTheBloc->IdBloc(aCam2->NameImage());
                     std::string aNamePair = anId1 + "/" + anId2;
                     mMapStatPair[aNamePair].Add(aRes12);
                     mMap1Image[anId1].Add(aRes12);
                     mMap1Image[anId2].Add(aRes12);
                 }
             }

             // eventually compute distance Wire/Pt for stats
             //  AddStatDistWirePt(aPG,aNamePt);

         }
     }

     if (mCernStat  && (mV3dLoc.size()>=3))
     {
         std::vector<cPt3dr> mV3dGround;
         std::vector<cPt3dr> aFilteredV3DLoc;
         for (size_t aKPt=0 ; aKPt<mV3dLoc.size() ; aKPt++)
         {
             const cMes1Gnd3D * aMes3D = mGCP3d.GetAdrMeasureOfNamePt(mVNames.at(aKPt),SVP::Yes);
             if (aMes3D)
             {
                 mV3dGround.push_back(aMes3D->mPt);
                 aFilteredV3DLoc.push_back(mV3dLoc.at(aKPt));
             }
         }
         if (aFilteredV3DLoc.size() >= 3)
         {
            tPoseR aPose = tPoseR::RansacL1Estimate(mV3dGround,aFilteredV3DLoc,10000);
            aPose = aPose.LeastSquareRefine(mV3dGround,aFilteredV3DLoc);
            tREAL8 aResidual;
            aPose = aPose.LeastSquareRefine(mV3dGround,aFilteredV3DLoc,&aResidual);
            StdOut() << "RESIDUAL CHG REP=" << std::sqrt(aResidual/aFilteredV3DLoc.size())  << "\n";

         //  TO REFACTOR !!!!! 
            cSensorCamPC * aCamMaster = nullptr;
            for (const auto & aCam : aVCam)
            {
                 if (mTheBloc->IdBloc(aCam->NameImage()) == mTheBloc->NameMaster()) 
                 {
                    aCamMaster = aCam;
                 }
            }
            MMVII_INTERNAL_ASSERT_tiny(aCamMaster!=nullptr,"Cannot find master");
            cPerspCamIntrCalib * aCalibM = aCamMaster->InternalCalib();
            const  cOneMesureClino & aMes = *  mMesClino.MeasureOfId(anIdSync); 

            cCalibSetClino aSetC = mPhProj.ReadSetClino(*aCalibM,mMesClino.NamesClino());
            cGetVerticalFromClino aGetVert(aSetC,aMes.Angles());
            auto[aScoreVert, aVertLocCamDown]  = aGetVert.OptimGlob(50,1e-9);  
        
         // StdOut() << " IDB=" << anIdSync << " SV=" << aScoreVert << " V=" << aVertLocCamDown << "\n";
         // aVCam
         // mMesClino
         //  cGetVerticalFromClino aGetVert(aSetC,aMes.Angles());


            cPt3dr aPLoc = aPose.Value(mSphereCenter);
            AddStatDistWirePt(aPLoc,aVertLocCamDown,"Center");

            StdOut() << " ============== Cam-Center  ======================\n";
            for (const auto & aCam : aVCam)
            {
               StdOut() << " * N2=" << Norm2(aPLoc-aCam->Center()) << " Cam=" << aCam->NameImage() << "\n";
            }
         }
     }

     // Add the stat for the time synchronization
     CSV_AddStat(mIdRepPtGlob,"AVG "+anIdSync,aStatRes);
}



void cAppli_ReportBlock::ProcessOneBloc(const std::vector<cSensorCamPC *> & aVCam)
{
     mCurWire = nullptr ;

     std::string anIdSync = mTheBloc->IdSync(aVCam.at(0)->NameImage());
     if (contains(mStrM2T,'W'))
        TestWire3D(anIdSync,aVCam);
     if (contains(mStrM2T,'T'))
        TestPoint3D(anIdSync,aVCam);

     delete mCurWire;
}

int cAppli_ReportBlock::Exe()
{
    mPhProj.FinishInit();  // the final construction of  photogrammetric project manager can only be done now

    std::string aDirRep =          mPhProj.DPOrient().DirIn() 
                           + "-" + mPhProj.DPGndPt2D().DirIn()
                           + "-" + mPhProj.DPRigBloc().DirIn() ;
    if (IsInit(&mAddExReport))
       aDirRep =  mAddExReport + "-" + aDirRep;
    SetReportSubDir(aDirRep);


    InitReportCSV(mRepW,"csv",false);
    InitReportCSV(mIdRepPtIndiv,"csv",false);
    AddHeaderReportCSV(mIdRepPtIndiv,{"TimeBloc","Point","Mult","Dist Pix"});

    InitReportCSV(mIdRepDWirePt,"csv",false);
    AddHeaderReportCSV(mIdRepPtIndiv,{"D3","DH","DV"});
    

    InitReportCSV(mIdRepPtGlob,"csv",false);
    AddOneReportCSV(mRepW,{"TimeBloc","NbPlane","Dist Ground","Dist Pix"});
/*
    AddOneReportCSV(mIdRepPtIndiv,{"TimeBloc","NamePt","NbPt","Dist Pix"});
    AddOneReportCSV
    (
           mIdRepPtGlob,
           Append
           (
               std::vector<std::string>{"TimeBloc","NbPMeasure","Avg","Sigma"},
               std::vector<std::string>{"Min","Max"}
           )
    );
*/
    AddStdHeaderStatCSV(mIdRepPtGlob,"NameAggreg",mPercStat);


    // mListBloc = mPhProj.ReadBlocCams();
    // MMVII_INTERNAL_ASSERT_tiny(mListBloc.size()==1,"Number of bloc ="+ ToStr(mListBloc.size()));

    mTheBloc = mPhProj.ReadUnikBlocCam();
    std::vector<std::vector<cSensorCamPC *>>  aVVC = mTheBloc->GenerateOrientLoc(mPhProj,VectMainSet(0));

    mCernStat  = mPhProj.DPGndPt3D().DirInIsInit();
    if (mCernStat)
    {
       mGCP3d = mPhProj.LoadGCP3D();
       mMesClino =  mPhProj.ReadMeasureClino();

       //  for (const auto 
   //cCalibSetClino aSetC = mPhProj.ReadSetClino(*aCalib,mMesClino.NamesClino());

    }
    // StdOut() << "NBILLL " << VectMainSet(0).size() << " NB BL " << aVVC.size() << "\n";

    for (auto & aVC : aVVC)
    {
        //StdOut() << "   * NbInBloc  " << aVC.size() << "\n";
        ProcessOneBloc(aVC);

        DeleteAllAndClear(aVC);
    }

   // Add the stat for all pairs
   for (const auto & [aNameImage,aStatImage] : mMap1Image )
       CSV_AddStat(mIdRepPtGlob,aNameImage,aStatImage);

   // Add the stat for all pairs
   for (const auto & [aNamePair,aStatPair] : mMapStatPair )
       CSV_AddStat(mIdRepPtGlob,aNamePair,aStatPair);

   // Add the stat for all the points
    CSV_AddStat(mIdRepPtGlob,"GlobAVG ",mStatGlobPt);


   //  export stat on dist Wire/pt
   for (const auto & [aName,aStat] : mStatByPt)
   {
       StdOut() <<  " * " << aName  << " : "
               << "[3d Avg=" << aStat.mStat3d.Avg() << " StdDev=" << aStat.mStat3d.UBDevStd(-1)  << " Med=" << aStat.mStat3d.ErrAtProp(0.5)<< "]"
               << "[3d H=" << aStat.mStatH.Avg() << " StdDev=" << aStat.mStatH.UBDevStd(-1)  << "]"
               << "[3d V=" << aStat.mStatV.Avg() << " StdDev=" << aStat.mStatV.UBDevStd(-1)  << "]"
               << "\n";
   }

   delete mTheBloc;
   return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */

tMMVII_UnikPApli Alloc_ReportBlock(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ReportBlock(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_BlocReport
(
     "ReportBlock",
      Alloc_ReportBlock,
     "Report different measures relative to a block of cam",
      {eApF::Ori},
      {eApDT::Orient}, 
      {eApDT::Xml}, 
      __FILE__
);


}; // MMVII

