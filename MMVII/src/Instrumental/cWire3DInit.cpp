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

class cAppli_Wire3DInit : public cMMVII_Appli
{
     public :

        cAppli_Wire3DInit(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

	//  std::vector<std::string>  Samples() const override;

     private :
	void MakeOneBloc(const std::vector<cSensorCamPC *> &);

        void TestWire3D(const std::string& anIdSync,const std::vector<cSensorCamPC *> & aVCam);
        void TestPoint3D(const std::string& anIdSync,const std::vector<cSensorCamPC *> & aVCam);

        cPhotogrammetricProject  mPhProj;

	std::string                 mSpecImIn;
        std::list<cBlocOfCamera *>  mListBloc;
        cBlocOfCamera *             mTheBloc;

        std::string                  mRepW;
        std::string                  mRepPt;
        std::string                  mPatNameGCP;
};

/*
std::vector<std::string>  cAppli_Wire3DInit::Samples() const
{
    return {
	     "MMVII BlockCamInit SetFiltered_GCP_OK_Resec.xml   BA_311_B   '(.*)_(.*).JPG' [1,2]  Rig_311_B"
    };
}
*/


cAppli_Wire3DInit::cAppli_Wire3DInit
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mRepW         ("Wire"),
     mRepPt        ("Pt"),
     mPatNameGCP   (".*")
{
}


cCollecSpecArg2007 & cAppli_Wire3DInit::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
             <<  Arg2007(mSpecImIn,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPOrient().ArgDirInMand()
             <<  mPhProj.DPPointsMeasures().ArgDirInMand()
             <<  mPhProj.DPRigBloc().ArgDirInMand()
           ;
}


cCollecSpecArg2007 & cAppli_Wire3DInit::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return      anArgOpt
             << AOpt2007(mPatNameGCP,"PatFiltGCP","Pattern to filter name of GCP",{{eTA2007::HDV}})
    ;
}



void cAppli_Wire3DInit::TestWire3D(const std::string & anIdSync,const std::vector<cSensorCamPC *> & aVCam)
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


void cAppli_Wire3DInit::TestPoint3D(const std::string & anIdSync,const std::vector<cSensorCamPC *> & aVCam)
{
     std::map<std::string,std::list<tPairCamPt>> aMapMatch;

     for (const auto & aCam : aVCam)
     {
          if (mPhProj.HasMeasureIm(aCam->NameImage()))
          {
	     cSetMesPtOf1Im  aSet = mPhProj.LoadMeasureIm(aCam->NameImage());

	     for (const auto & aMes : aSet.Measures())
	     {
                 if ((!starts_with( aMes.mNamePt,MMVII_NONE)) && MatchRegex(aMes.mNamePt,mPatNameGCP))
	         {
		         aMapMatch[aMes.mNamePt].push_back(tPairCamPt(aCam,aMes));
	         }
	     }
	  }
	  // StdOut()  << "IM="  << aSet.NameIm() << " Nb=" << aSet.Measures().size() << "\n";
     }

     for (const auto & [aStr,aList] : aMapMatch )
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
             AddOneReportCSV(mRepPt,{anIdSync,ToStr(aNbPt),ToStr(aDistPix)});
         }
     }
		  
}


void cAppli_Wire3DInit::MakeOneBloc(const std::vector<cSensorCamPC *> & aVCam)
{
     std::string anIdSync = mTheBloc->IdSync(aVCam.at(0)->NameImage());
     TestWire3D(anIdSync,aVCam);
     TestPoint3D(anIdSync,aVCam);
}

int cAppli_Wire3DInit::Exe()
{
    mPhProj.FinishInit();  // the final construction of  photogrammetric project manager can only be done now

    std::string aDirRep =          mPhProj.DPOrient().DirIn() 
                           + "-" + mPhProj.DPPointsMeasures().DirIn() 
                           + "-" + mPhProj.DPRigBloc().DirIn() ;
    SetReportSubDir(aDirRep);


    InitReport(mRepW,"csv",false);
    InitReport(mRepPt,"csv",false);
    AddOneReportCSV(mRepW,{"TimeBloc","NbPlane","Dist Ground","Dist Pix"});
    AddOneReportCSV(mRepPt,{"TimeBloc","NbPt","Dist Pix"});


    mListBloc = mPhProj.ReadBlocCams();
    MMVII_INTERNAL_ASSERT_tiny(mListBloc.size()==1,"Number of bloc ="+ ToStr(mListBloc.size()));

    mTheBloc = *(mListBloc.begin());
    std::vector<std::vector<cSensorCamPC *>>  aVVC = (*(mListBloc.begin()))->GenerateOrientLoc(mPhProj,VectMainSet(0));

    for (auto & aVC : aVVC)
    {
        MakeOneBloc(aVC);
        DeleteAllAndClear(aVC);
    }

    DeleteAllAndClear(mListBloc);
    return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */

tMMVII_UnikPApli Alloc_Wire3DInit(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_Wire3DInit(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_Wire3DInit
(
      "Wire3DInit",
      Alloc_Wire3DInit,
      "Compute 3D position of wire",
      {eApF::Ori},
      {eApDT::Orient}, 
      {eApDT::Xml}, 
      __FILE__
);


}; // MMVII

