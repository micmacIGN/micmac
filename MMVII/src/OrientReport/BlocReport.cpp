#include "MMVII_BlocRig.h"

#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_2Include_Serial_Tpl.h"

//  RIGIDBLOC


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

class cAppli_ReportBlock : public cMMVII_Appli
{
     public :

        cAppli_ReportBlock(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

	//  std::vector<std::string>  Samples() const override;

     private :
	void MakeOneBloc(const std::vector<cSensorCamPC *> &);

        void TestWire3D(const std::string& anIdSync,const std::vector<cSensorCamPC *> & aVCam);
        /// For a given Id of Sync and a bloc of cameras, compute the stat on accuracy of intersection
        void TestPoint3D(const std::string& anIdSync,const std::vector<cSensorCamPC *> & aVCam);

        cPhotogrammetricProject  mPhProj;

	std::string                 mSpecImIn;
        cBlocOfCamera *             mTheBloc;

        std::string                  mRepW;
        std::string                  mIdRepPtIndiv;
        std::string                  mIdRepPtGlob;
        std::string                  mPatNameGCP;

        std::string                  mStrM2T;  /// String of measure to test
        std::string                  mAddExReport;
        cWeightAv<tREAL8,tREAL8>     mAvgGlobRes;
        cStdStatRes                  mStatGlobPt;
      
        void CSV_AddStat(const std::string& anId,const std::string& aMes,const cStdStatRes &) ;
    //  AddOneReportCSV(mIdRepPtIndiv,{"AVG "+anIdSync,"",ToStr(aAvgRes.SW()),ToStr(aAvgRes.Average())});
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
     mIdRepPtGlob  ("GlobPt"),
     mPatNameGCP   (".*"),
     mStrM2T       ("TW")
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
    ;
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

     if (aNbPl>=3)
     {
	 cSegmentCompiled<tREAL8,3> aWire = cPlane3D::InterPlane(aVPlane);
	 cWeightAv<tREAL8> aWGr;
	 cWeightAv<tREAL8> aWPix;
	 for (size_t aKC=0 ; aKC<aVCamOk.size() ; aKC++)
	 {
             cSensorCamPC * aCam = aVCamOk[aKC];
             cPerspCamIntrCalib * aCalib = aCam->InternalCalib();
	     int aNbSeg = 5;


	     for (int aKS=0 ; aKS<=aNbSeg ; aKS++)
	     {
		  cPt2dr aPIm = aCalib->InterpolOnUDLine(aVSegOk[aKC],aKS/tREAL8(aNbSeg));
		  aWGr.Add(1.0,aCam->GroundDistBundleSeg(aPIm,aWire));
		  aWPix.Add(1.0,aCam->PixDistBundleSeg(aPIm,aWire));
	     }
	 }
	 tREAL8 aRatio = aNbPl /(aNbPl-2.0);
         tREAL8 aDist3D =  aWGr.Average() * aRatio;
         tREAL8 aDistPix =  aWPix.Average() * aRatio;

         AddOneReportCSV(mRepW,{anIdSync,ToStr(aNbPl),ToStr(aDist3D),ToStr(aDistPix)});

/*
         StdOut()     << " DIST3="  <<  aDist3D
                      << " DDDDd= " <<  aDistPix
		      << " NBPL = " << aNbPl 
		      << " RATIO="  << aRatio
                      << "\n";
*/
     }
}

typedef std::pair<cSensorCamPC *,cMesIm1Pt> tPairCamPt;

void cAppli_ReportBlock::CSV_AddStat(const std::string& anId,const std::string& aMes,const cStdStatRes & aStat) 
{
    int aNbMes = aStat.NbMeasures();
    if (aNbMes)
    {
        
        AddOneReportCSV
        (
            mIdRepPtGlob,
           {aMes,ToStr(aNbMes),ToStr(aStat.Avg()),ToStr(aStat.UBDevStd(-1)),ToStr(aStat.Min()),ToStr(aStat.Max())}
        );

    }
    else
        AddOneReportCSV(anId,{aMes,"0","XXX","XXX","XXX","XXX"});
}

void cAppli_ReportBlock::TestPoint3D(const std::string & anIdSync,const std::vector<cSensorCamPC *> & aVCam)
{
     // for a given name of point, store  Mes+Cam , that will allow to compute bundles
     std::map<std::string,std::list<tPairCamPt>> aMapMatch;
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

     // [2]  Parse the measure grouped by points
     for (const auto & [aNamePt,aList] : aMapMatch )
     {
         int aNbPt = aList.size();
         if (aNbPt  > 2) 
         {
	     // StdOut() << " NAME=" << aStr << " " << aList.size() << "\n";
             std::vector<tSeg3dr> aVSeg;
	     for (const auto & [aCam,aMes] : aList)
	     {
                 aVSeg.push_back(aCam->Image2Bundle(aMes.mPt));
	     }
	     cPt3dr aPG =   BundleInters(aVSeg);
	     cWeightAv<tREAL8> aWPix;
	     for (const auto & [aCam,aMes] : aList)
	     {
                 cPt2dr aPProj = aCam->Ground2Image(aPG);
                 aWPix.Add(1.0,Norm2(aMes.mPt-aPProj));
		 // StdOut() << " DDDD = " << Norm2(aMes.mPt-aPProj) << "\n";
	     }
             tREAL8 aDistPix = aWPix.Average() * (aNbPt*2.0) / (aNbPt*2.0 -3.0);
             AddOneReportCSV(mIdRepPtIndiv,{anIdSync,aNamePt,ToStr(aNbPt),ToStr(aDistPix)});
             aStatRes.Add(aDistPix);
             mStatGlobPt.Add(aDistPix);
         }
     }

     CSV_AddStat(mIdRepPtGlob,"AVG "+anIdSync,aStatRes);
/*
const std::string& anId,const std::string& aMes,const cStdStatRes & aStat) const
     if (aAvgRes.SW())
     {
// mIdRepPtGlob
         AddOneReportCSV(mIdRepPtIndiv,{"AVG "+anIdSync,"",ToStr(aAvgRes.SW()),ToStr(aAvgRes.Average())});
         mAvgGlobRes.Add(aAvgRes.SW(),aAvgRes.Average());
     }
     else
     {
         AddOneReportCSV(mIdRepPtIndiv,{"AVG "+anIdSync,"",ToStr(aAvgRes.SW()),"***"});
     }
*/
}



void cAppli_ReportBlock::MakeOneBloc(const std::vector<cSensorCamPC *> & aVCam)
{
     std::string anIdSync = mTheBloc->IdSync(aVCam.at(0)->NameImage());
     if (contains(mStrM2T,'W'))
        TestWire3D(anIdSync,aVCam);
     if (contains(mStrM2T,'T'))
        TestPoint3D(anIdSync,aVCam);
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
    InitReportCSV(mIdRepPtGlob,"csv",false);
    AddOneReportCSV(mRepW,{"TimeBloc","NbPlane","Dist Ground","Dist Pix"});
    AddOneReportCSV(mIdRepPtIndiv,{"TimeBloc","NamePt","NbPt","Dist Pix"});
    AddOneReportCSV(mIdRepPtGlob,{"TimeBloc","NbPMeasure","Avg","Sigma","Min","Max"});


    // mListBloc = mPhProj.ReadBlocCams();
    // MMVII_INTERNAL_ASSERT_tiny(mListBloc.size()==1,"Number of bloc ="+ ToStr(mListBloc.size()));

    mTheBloc = mPhProj.ReadUnikBlocCam();
    std::vector<std::vector<cSensorCamPC *>>  aVVC = mTheBloc->GenerateOrientLoc(mPhProj,VectMainSet(0));

    // StdOut() << "NBILLL " << VectMainSet(0).size() << " NB BL " << aVVC.size() << "\n";

    for (auto & aVC : aVVC)
    {
        // StdOut() << "   * NbInBloc  " << aVC.size() << "\n";
        MakeOneBloc(aVC);
        DeleteAllAndClear(aVC);
    }
    // StdOut() << " SW " << mAvgGlobRes.SW() << " SVVW" << mAvgGlobRes.SVW() << "\n";
    // AddOneReportCSV(mIdRepPt,{"AVG Glob",ToStr(mAvgGlobRes.SW()),ToStr(mAvgGlobRes.Average())});

    CSV_AddStat(mIdRepPtGlob,"GlobAVG ",mStatGlobPt);
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

