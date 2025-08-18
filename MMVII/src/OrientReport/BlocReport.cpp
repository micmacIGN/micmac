#include "MMVII_BlocRig.h"

#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Clino.h"


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


///  Class to represent points located in rigid frame
class cCern_PtRF
{
    public :
        cPt3dr        mCoordRF;  //  coord local to rigid frame
        std::string   mName;     // name of point
};

class cAppli_ReportBlock : public cMMVII_Appli
{
     public :
        typedef std::pair<cSensorCamPC *,cMesIm1Pt> tPairCamPt;

        cAppli_ReportBlock(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

	//  std::vector<std::string>  Samples() const override;

     private :
	void ProcessOneBloc(const std::vector<cSensorCamPC *> &);
        void AddStatDistWirePt(const cPt3dr& aPt,const cPt3dr& aDirLoc,const std::string & anIdSync,const std::string & aNamePt);

        cPt3dr ExtractVertLoc(const std::string& anIdSync,const std::vector<cSensorCamPC *> & aVCam);

        void TestWire3D(const std::string& anIdSync,const std::vector<cSensorCamPC *> & aVCam);
        /// For a given Id of Sync and a bloc of cameras, compute the stat on accuracy of intersection
        void TestPoint3D(const std::string& anIdSync,const std::vector<cSensorCamPC *> & aVCam);

        ///  Generate the export specific to Cern's manip (dist 3D & 2D Wire/target)
        void GenerateCernExport (const std::string& anIdSync,const std::vector<cSensorCamPC *> & aVCam);
        ///  Generate an export for LGC Format
        void DoLGCExport(const std::vector<cSensorCamPC *> & aVCam);

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
        bool                                    mCernAllPoint;
        cPt3dr                                  mSphereCenter; //< center of the sphere
        bool                                    mSCFreeScale;  //< do we have a free scale 
        cSetMesGnd3D                            mGCP3d;        //< "Absolute" coordinate of the point
        bool                                    mWithClino;
        cSetMeasureClino                        mMesClino;
        std::vector<cCern_PtRF>                 mVCernPR;  //< Coord in frame + names for CERN export

        /// For a given name of point memorize the set of pair "Cam+Im Measure"
        std::map<std::string,std::vector<tPairCamPt>> mMapMatch;
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
     mCernStat     (false),
     mCernAllPoint (false),
     mSphereCenter (0,0,0),
     mWithClino    (false)
{
    FakeUseIt(mSCFreeScale);
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

             << AOpt2007(mCernStat,"DoCernStat","Do statistic specific to Cerns Wire distance",{{eTA2007::HDV}})
             << AOpt2007(mCernAllPoint,"DoCernAllPt","For cern, compute Wire distance for all points",{{eTA2007::HDV}})
             << mPhProj.DPGndPt3D().ArgDirInOpt("","GCP 3D coordinate for computing centre")
             << AOpt2007(mSphereCenter,"ShereC","Additionnal GPC to export")
             << mPhProj.DPMeasuresClino().ArgDirInOpt()
             << mPhProj.DPClinoMeters().ArgDirInOpt()
    ;
}


void cAppli_ReportBlock::AddStatDistWirePt
     (
            const cPt3dr& aPt,
            const cPt3dr& aVertLoc,
            const std::string & anIdSync,
            const std::string & aNamePt
     )
{
    if (!mCurWire)
       return;

     cPt3dr  aProj = mCurWire->Proj(aPt);
     cPt3dr  anEc = aProj-aPt;
     tREAL8 aD3 = Norm2(anEc);
     tREAL8 aDH = -1;
     tREAL8 aDV = -1;
     if (mWithClino)
     {
        cPt3dr  aCompV = aVertLoc * Scal(aVertLoc,anEc);
        cPt3dr  aCompH = anEc-aCompV;
        aDH = Norm2(aCompH);
        aDV = Norm2(aCompV);
        mStatByPt[aNamePt].mStatH.Add(Norm2(aCompH));
        mStatByPt[aNamePt].mStatV.Add(Norm2(aCompV));
     }

     mStatByPt[aNamePt].mStat3d.Add(Norm2(anEc));

     AddOneReportCSV(mIdRepDWirePt,{aNamePt,ToStr(aD3),ToStr(aDH),ToStr(aDV)});
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

        }
    }
}


void cAppli_ReportBlock::CSV_AddStat(const std::string& anId,const std::string& aMes,const cStdStatRes & aStat) 
{
   AddStdStatCSV(anId,aMes,aStat,mPercStat);
}

std::string ToCernStr(const cPt3dr & aPt)
{
   return ToStr(aPt.x()) + " " + ToStr(aPt.y()) + " " + ToStr(aPt.z());
}

void cAppli_ReportBlock::TestPoint3D(const std::string & anIdSync,const std::vector<cSensorCamPC *> & aVCam)
{
     // for a given name of point, store  Mes+Cam , that will allow to compute bundles
     mMapMatch.clear();
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
                    mMapMatch[aMes.mNamePt].push_back(tPairCamPt(aCam,aMes));
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
     mVCernPR.clear();
//cCern_PtRF

     // [2]  Parse the measure grouped by points
     for (const auto & [aNamePt,aVect] : mMapMatch )
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
             mVCernPR.push_back(cCern_PtRF());
             mVCernPR.back().mCoordRF = aPG;
             mVCernPR.back().mName = aNamePt;

	     cWeightAv<tREAL8> aWPix;

	     for (const auto & [aCam,aMes] : aVect)
	     {
                 cPt2dr aPProj = aCam->Ground2Image(aPG);
                 aWPix.Add(1.0,Norm2(aMes.mPt-aPProj));
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


         }
     }
     CSV_AddStat(mIdRepPtGlob,"AVG "+anIdSync,aStatRes);

     GenerateCernExport(anIdSync,aVCam);
}

cPt3dr cAppli_ReportBlock::ExtractVertLoc(const std::string& anIdSync,const std::vector<cSensorCamPC *> & aVCam)
{
    //  TO REFACTOR !!!!! 
    if (! mWithClino)
       return cPt3dr(0,0,0);

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
        
    return aVertLocCamDown;
}

void cAppli_ReportBlock::GenerateCernExport(const std::string& anIdSync,const std::vector<cSensorCamPC *> & aVCam)
{
     if (!mCernStat)
        return;

     cPt3dr aVertLocCamDown =  ExtractVertLoc(anIdSync,aVCam);

     if  (mVCernPR.size()<3)
        return;


     {
         // Compute the pairs "Coord-Spher/Coord Loc" 
         std::vector<cPt3dr> mV3dGround;  // Pairs of coord in
         std::vector<cPt3dr> aFilteredV3DLoc;
         for (size_t aKPt=0 ; aKPt<mVCernPR.size() ; aKPt++)
         {
             const cMes1Gnd3D * aMes3D = mGCP3d.GetAdrMeasureOfNamePt(mVCernPR.at(aKPt).mName,SVP::Yes);
             if (aMes3D)
             {
                 mV3dGround.push_back(aMes3D->mPt);
                 aFilteredV3DLoc.push_back(mVCernPR.at(aKPt).mCoordRF);
             }
         }
         if (aFilteredV3DLoc.size() >= 3)
         {
            tSim3dR aPose = tSim3dR::RansacL1Estimate(mV3dGround,aFilteredV3DLoc,10000);
            // tPoseR aPose = tPoseR::RansacL1Estimate(mV3dGround,aFilteredV3DLoc,10000);
            aPose = aPose.LeastSquareRefine(mV3dGround,aFilteredV3DLoc);

            tREAL8 aResidual;
            aPose = aPose.LeastSquareRefine(mV3dGround,aFilteredV3DLoc,&aResidual);


            cPt3dr aPLoc = aPose.Value(mSphereCenter);
            AddStatDistWirePt(aPLoc,aVertLocCamDown,anIdSync,"Center");
         }
         if (mCernAllPoint)
         {
             for (const auto & aCernPt : mVCernPR)
             {
                AddStatDistWirePt(aCernPt.mCoordRF,aVertLocCamDown,anIdSync,aCernPt.mName);
             }
         }
   }
if (0)
{
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
   DoLGCExport(aVCam);
}

void cAppli_ReportBlock::DoLGCExport (const std::vector<cSensorCamPC *> & aVCam)
{
        cMMVII_Ofs anOffs("LGG.txt",eFileModeOut::CreateText);

         anOffs.Ofs() << "*TITR\n";
         anOffs.Ofs() << "Export from MMVII\n";
         anOffs.Ofs() << "*OLOC\n";
         anOffs.Ofs() << "*PUNC EE\n";
         anOffs.Ofs() << "*PREC 6\n";
         anOffs.Ofs() << "*FAUT\n";
         anOffs.Ofs() << "*JSON\n";
         anOffs.Ofs() << "*INSTR\n";
         anOffs.Ofs() << "*CAMD TS1 TGT1 0.0\n";
         anOffs.Ofs() << "TGT1 0.02 0.02 0.1 0.001\n";


         anOffs.Ofs() << "*POIN\n";

         for (size_t aKPt=0 ; aKPt<mVCernPR.size() ; aKPt++)
         {
             anOffs.Ofs() << mVCernPR.at(aKPt).mName << " " << ToCernStr( mVCernPR.at(aKPt).mCoordRF  ) << "\n";
         }

         for (size_t aKCam=0 ; aKCam<aVCam.size() ;  aKCam++)
         {
             const auto aCamPtr = aVCam.at(aKCam);
             cRotation3D<tREAL8>  aRotAfter = cRotation3D<tREAL8>::RotFromCanonicalAxes("ijk");
             cRotation3D<tREAL8> aRotCern = aRotAfter.MapInverse() * aCamPtr->Pose().Rot() * aRotAfter;

             cPt3dr aC = aRotAfter.Value(aCamPtr->Center());
             tREAL8 aUnitA =  - (400.0/(2*M_PI)) ;
             cPt3dr aWPK = aRotCern.ToWPK() * aUnitA;

             std::string aNameCam = "Cam" + ToStr(aKCam+1);

             anOffs.Ofs() << "*FRAME   " << aNameCam
                      << " "<< ToCernStr(aC)
                      << " " <<  ToCernStr(aWPK) << " 1 \n";

             anOffs.Ofs() << "*CALA\n";
             anOffs.Ofs() << aNameCam << "        0         0        0\n";
             anOffs.Ofs() << "*CAM "<< aNameCam <<" TS1\n";
             anOffs.Ofs() << "*UVEC\n";

             for (const auto & [aNamePt,aVect] : mMapMatch )
             {
	         for (const auto & [aCamLocPtr,aMes] : aVect)
	         {
                      if (aCamLocPtr==aCamPtr)
                      {
                          cPt3dr aDirCam   = VUnit(aCamPtr->InternalCalib()->DirBundle(aMes.mPt));
                          anOffs.Ofs() <<  aNamePt << " " << ToCernStr(aDirCam) << "\n";
                      }
	         }
             }
             anOffs.Ofs() << "*ENDFRAME\n";
             anOffs.Ofs() << "%-----------------------\n";
         }
         anOffs.Ofs() << "*END\n";

/*
         for (size_t aKCam=0 ; aKCam<aVCam.size() ;  aKCam++)
         {
             StdOut() << " =============== Cam : " << aKCam << " ===========================================\n";
             const auto aCalibPtr = aVCam.at(aKCam)->InternalCalib();
             cPt2dr aSzIm = ToR(aCalibPtr->SzPix());
             int aNbSamples = 2;
             for (int aKX=0 ; aKX<=aNbSamples ; aKX++)
                 for (int aKY=0 ; aKY<=aNbSamples ; aKY++)
                 {
                       cPt2dr aPIm = cPt2dr(aKX,aKY) / tREAL8(aNbSamples);
                       aPIm = MulCByC(aSzIm,aPIm);
                       cPt3dr aBundle = VUnit(aCalibPtr->DirBundle(aPIm));
                       StdOut() << "Im=" << aPIm  << " Dir3d=" << aBundle  
                                // << " RP=" << aCalibPtr->Value(aBundle) 
                                << "\n";
                 }

         }
*/
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
    AddOneReportCSV(mRepW,{"TimeBloc","NbPlane","Dist Ground","Dist Pix"});
    
    InitReportCSV(mIdRepPtIndiv,"csv",false);
    AddHeaderReportCSV(mIdRepPtIndiv,{"TimeBloc","Point","Mult","Dist Pix"});

    InitReportCSV(mIdRepDWirePt,"csv",false);
    AddHeaderReportCSV(mIdRepDWirePt,{"TimeStamp","NamePt","D3","DH","DV"});

    InitReportCSV(mIdRepPtGlob,"csv",false);
    AddStdHeaderStatCSV(mIdRepPtGlob,"NameAggreg",mPercStat);


    mTheBloc = mPhProj.ReadUnikBlocCam();
    std::vector<std::vector<cSensorCamPC *>>  aVVC = mTheBloc->GenerateOrientLoc(mPhProj,VectMainSet(0));

    if (mCernStat)
    {
       mGCP3d = mPhProj.LoadGCP3D();
       mWithClino =  mPhProj.DPMeasuresClino().DirInIsInit() && mPhProj.DPClinoMeters().DirInIsInit();
       if (mWithClino)
          mMesClino =  mPhProj.ReadMeasureClino();
    }

    for (auto & aVC : aVVC)
    {
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
               << "[3d Avg=" << aStat.mStat3d.Avg() << " StdDev=" << aStat.mStat3d.UBDevStd(-1)  << " Med=" << aStat.mStat3d.ErrAtProp(0.5)<< "]";
       if (mWithClino)
          StdOut() 
               << "[3d H=" << aStat.mStatH.Avg() << " StdDev=" << aStat.mStatH.UBDevStd(-1)  << "]"
               << "[3d V=" << aStat.mStatV.Avg() << " StdDev=" << aStat.mStatV.UBDevStd(-1)  << "]";

       StdOut() << "\n";
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

