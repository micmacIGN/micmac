#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"


/**
   \file GCPQuality.cpp


 */

namespace MMVII
{



/* ==================================================== */
/*                                                      */
/*          cAppli_CalibratedSpaceResection             */
/*                                                      */
/* ==================================================== */

class cAppli_CGPReport : public cMMVII_Appli
{
     public :

        cAppli_CGPReport(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &,bool IsGCP);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :
	/** make the report  by image, for each image a cvs file with all GCP,
	 * optionnaly make a visualisation of the residual fielsd for each image */
        void  MakeOneIm(const std::string & aNameIm);

	/** Make a report with an average for each GCP */
        void  ReportsByGCP();
	/** Make a visualization of residual in sensor plane*/
        void  ReportsByCam();

        std::string              mSpecImIn;   ///  Pattern of xml file
        cPhotogrammetricProject  mPhProj;
	bool                     mIsGCP;  /// GCP vs Tie Point

	std::vector<double>      mGeomFiedlVec;
	std::vector<int>         mPropStat;

	std::string              mPrefixReport;

	std::string              mNameReportDetail;
	std::string              mNameReportIm;
        std::string              mNameReportGCP;
        std::string              mNameReportGCP_Ground;
        std::string              mNameReportGCP_Ground_Glob;
        std::string              mNameReportCam;
        std::string              mNameReportMissed;

	double                   mMarginMiss;  ///  Margin for counting missing targets
        std::string              mSuffixReportSubDir; // additional name for report subdir
};

cAppli_CGPReport::cAppli_CGPReport
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec,
     bool                     isGCP
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mIsGCP        (isGCP),
     mPropStat     ({50,75}),
     mMarginMiss   (50.0),
     mSuffixReportSubDir ("")
{
}



cCollecSpecArg2007 & cAppli_CGPReport::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
              << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              << (mIsGCP ?  mPhProj.DPPointsMeasures().ArgDirInMand()
                         :  mPhProj.DPMulTieP().ArgDirInMand())
              <<  mPhProj.DPOrient().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_CGPReport::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    auto & aRes =  anArgOpt
	        << AOpt2007(mPropStat,"Perc","Percentil for stat exp",{eTA2007::HDV});

    if (mIsGCP)
       return aRes << AOpt2007(mGeomFiedlVec,"GFV","Geom Fiel Vect for visu [Mul,Witdh,Ray,Zoom?=2]",{{eTA2007::ISizeV,"[3,4]"}})
	           << AOpt2007(mMarginMiss,"MargMiss","Margin to border for counting missed target",{eTA2007::HDV})
                   << AOpt2007(mSuffixReportSubDir, "Suffix", "Suffix to report subdirectory name", {eTA2007::HDV})
       ;

    return aRes;
}


//================================================


void cAppli_CGPReport::MakeOneIm(const std::string & aNameIm)
{
    if (! ExistFile(mPhProj.NameMeasureGCPIm(aNameIm,true)) )  
       return ;

    cSetMesImGCP             aSetMes;
    mPhProj.LoadGCP(aSetMes);
    mPhProj.LoadIm(aSetMes,aNameIm);
    const cSetMesPtOf1Im  &  aSetMesIm = aSetMes.MesImInitOfName(aNameIm);

    // cSet2D3D aSet32;
    // mSetMes.ExtractMes1Im(aSet32,aNameIm);
    cSensorImage*  aCam = mPhProj.LoadSensor(aNameIm,false);

    // StdOut() << " aNameImaNameIm " << aNameIm  << " " << aSetMesIm.Measures().size() << " Cam=" << aCam << std::endl;

    cRGBImage aImaFieldRes(cPt2di(1,1));

    if (IsInit(&mGeomFiedlVec))
    {
         tREAL8 aMul    = mGeomFiedlVec.at(0);
         tREAL8 aWitdh  = mGeomFiedlVec.at(1);
         tREAL8 aRay    = mGeomFiedlVec.at(2);
         aImaFieldRes =  cRGBImage::FromFile(aNameIm);

	 //  [Mul,Witdh,Ray]
	 for (const auto & aMes : aSetMesIm.Measures())
	 {
	     cPt2dr aP2 = aMes.mPt;
	     cPt3dr aPGr = aSetMes.MesGCPOfName(aMes.mNamePt).mPt;
	     cPt2dr aProj = aCam->Ground2Image(aPGr);
	     cPt2dr  aVec = (aP2-aProj);

             aImaFieldRes.DrawCircle(cRGBImage::Green,aP2,aRay);
             aImaFieldRes.DrawLine(aP2,aP2+aVec*aMul,cRGBImage::Red,aWitdh);
	 }

    }


    cWeightAv<tREAL8,cPt2dr>  aAvg2d;
    cStdStatRes               aStat;

    for (const auto & aMes : aSetMesIm.Measures())
    {
        cPt2dr aP2 = aMes.mPt;
        cPt3dr aPGr = aSetMes.MesGCPOfName(aMes.mNamePt).mPt;
        cPt2dr aProj = aCam->Ground2Image(aPGr);
        cPt2dr  aVec = (aP2-aProj);

	aAvg2d.Add(1.0,aVec);
	tREAL8 aDist = Norm2(aVec);
	aStat.Add(aDist);
        AddOneReportCSV(mNameReportDetail,{aNameIm,aMes.mNamePt,ToStr(aDist)});
    }


    for (const auto & aGCP : aSetMes.MesGCP())
    {
        if (aCam->IsVisible(aGCP.mPt))
        {
           cPt2dr aPIm = aCam->Ground2Image(aGCP.mPt);
           tREAL8 aDeg = aCam->DegreeVisibilityOnImFrame(aPIm);
           if ((aDeg> mMarginMiss) && (!aSetMesIm.NameHasMeasure(aGCP.mNamePt)))
           {
              AddOneReportCSV(mNameReportMissed,{aNameIm,aGCP.mNamePt,ToStr(aPIm.x()),ToStr(aPIm.y())});
	      if (IsInit(&mGeomFiedlVec))
	      {
		    for (const auto aRay : {2.0,30.0,32.0,34.0})
                        aImaFieldRes.DrawCircle(cRGBImage::Red,aPIm,aRay);
	      }
           }
            if (LevelCall()==0)
               StdOut() << "VISIBLE " << aGCP.mNamePt << std::endl;
        }
	else
	{
            if (LevelCall()==0)
               StdOut() << "### UNVISIBLE " << aGCP.mNamePt << std::endl;
	}
    }

    if (IsInit(&mGeomFiedlVec))
    {
	int aDeZoom    = round_ni(GetDef(mGeomFiedlVec,3,2.0));
        aImaFieldRes.ToJpgFileDeZoom(mPhProj.DirVisu() + "FieldRes-"+aNameIm+".tif",aDeZoom);
    }

    auto aMesX = (aAvg2d.SW()>0.) ? ToStr(aAvg2d.Average().x()) : "XXX";
    auto aMesY = (aAvg2d.SW()>0.) ? ToStr(aAvg2d.Average().y()) : "XXX";
    AddStdStatCSV
    (
       mNameReportIm,aNameIm,aStat,mPropStat,
       {aMesX, aMesY}
    );

}



void cAppli_CGPReport::ReportsByGCP()
{
   cSetMesImGCP             aSetMes;
   mPhProj.LoadGCP(aSetMes);

   for (const auto & aNameIm : VectMainSet(0))
   {
       mPhProj.LoadIm(aSetMes,aNameIm,mPhProj.LoadSensor(aNameIm,false),true);
   }

   const std::vector<cSensorImage*> &  aVSens =  aSetMes.VSens() ;

   InitReport(mNameReportGCP,"csv",false);
   AddStdHeaderStatCSV(mNameReportGCP,"GCP",mPropStat);

   InitReport(mNameReportGCP_Ground,"csv",false);
   AddOneReportCSV(mNameReportGCP_Ground,{"Name","Dx","Dy","Dz"});

   std::vector<cStdStatRes> aVStatXYZ{cStdStatRes(),cStdStatRes(),cStdStatRes()};

   for (const auto &  aMesIm :  aSetMes.MesImOfPt())
   {
        const auto & aGCP  = aSetMes.MesGCPOfMulIm(aMesIm);
	const std::vector<int> &  aVIndI = aMesIm.VImages() ;
        cStdStatRes               aStat;

	for (size_t aKIm = 0 ; aKIm<  aVIndI.size() ; aKIm++)
	{
            aStat.Add(Norm2( aMesIm.VMeasures()[aKIm]  - aVSens[aVIndI[aKIm]]->Ground2Image(aGCP.mPt)));
	}
	AddStdStatCSV(mNameReportGCP,aGCP.mNamePt,aStat,mPropStat);
    if (aVIndI.size()>1)
    {
    	cPt3dr aDelta = aGCP.mPt -  aSetMes.BundleInter(aMesIm);
        AddOneReportCSV(mNameReportGCP_Ground,{aGCP.mNamePt,ToStr(aDelta.x()),ToStr(aDelta.y()),ToStr(aDelta.z())});

        for (int aKC=0 ; aKC<3 ; aKC++)
            aVStatXYZ[aKC].Add(aDelta[aKC]);
    }
    else
       AddOneReportCSV(mNameReportGCP_Ground,{aGCP.mNamePt,"xxx","yyy","zzz"});
   }

   InitReport(mNameReportGCP_Ground_Glob,"csv",false);
   AddStdHeaderStatCSV(mNameReportGCP_Ground_Glob,"Coord",{});
   std::vector<std::string> aVCoord{"x","y","z"};
   for (int aKC=0 ; aKC<3 ; aKC++)
	AddStdStatCSV(mNameReportGCP_Ground_Glob,aVCoord[aKC],aVStatXYZ[aKC],{});
}

void cAppli_CGPReport::ReportsByCam()
{
   std::map<cPerspCamIntrCalib*,std::vector<cSensorCamPC*>>  aMapCam;
   cSetMesImGCP             aSetMes;
   mPhProj.LoadGCP(aSetMes);

   for (const auto & aNameIm : VectMainSet(0))
   {
       cSensorCamPC *  aCam = mPhProj.ReadCamPC(aNameIm,true);
       mPhProj.LoadIm(aSetMes,aNameIm,aCam,true);
       aMapCam[aCam->InternalCalib()].push_back(aCam);
   }

   InitReport(mNameReportCam,"csv",false);
   AddStdHeaderStatCSV(mNameReportCam,"Cam",mPropStat);

   tREAL8 aFactRed = 100.0;
   tREAL8 anExag = 1000;
   for (const auto& aPair : aMapCam)
   {
       cPerspCamIntrCalib * aCalib =  aPair.first;
       cPt2di aSz = Pt_round_up(ToR(aCalib->SzPix())/aFactRed) + cPt2di(1,1);

       cIm2D<tREAL8> aImX(aSz,nullptr,eModeInitImage::eMIA_Null);  // average X residual
       cIm2D<tREAL8> aImY(aSz,nullptr,eModeInitImage::eMIA_Null);  // average Y redidual
       cIm2D<tREAL8> aImW(aSz,nullptr,eModeInitImage::eMIA_Null);  // averagge weight

       cStdStatRes               aStat;

       int aNbPtsTot =0;
       for (const auto & aCam : aPair.second)
       {
           cSet2D3D aSet32;
	   aSetMes.ExtractMes1Im(aSet32,aCam->NameImage(),true);
	   aNbPtsTot += aSet32.NbPair();

           for (const auto & aPair : aSet32.Pairs())
           {
               cPt2dr aP2 = aPair.mP2;
               cPt2dr aRes = (aCam->Ground2Image(aPair.mP3) - aP2) ;
	       aStat.Add(Norm2(aRes));

               aRes = aRes *anExag;
               aImX.DIm().AddVBL(aP2/aFactRed,aRes.x());
               aImY.DIm().AddVBL(aP2/aFactRed,aRes.y());
               aImW.DIm().AddVBL(aP2/aFactRed,1.0);
           }
       }
       tREAL8 aSigma = Norm2(aImX.DIm().Sz()) / std::sqrt(aNbPtsTot);

       aImW.DIm().ToFile(mPhProj.DirVisu() + "W_RawResidual_"+aCalib->Name() +".tif");
       aImX = aImX.GaussFilter(aSigma);
       aImY = aImY.GaussFilter(aSigma);
       aImW = aImW.GaussFilter(aSigma);

       DivImageInPlace(aImX.DIm(),aImX.DIm(),aImW.DIm());
       DivImageInPlace(aImY.DIm(),aImY.DIm(),aImW.DIm());

       aImX.DIm().ToFile(mPhProj.DirVisu() + "X_Residual_"+aCalib->Name() +".tif");
       aImY.DIm().ToFile(mPhProj.DirVisu() + "Y_Residual_"+aCalib->Name() +".tif");
       aImW.DIm().ToFile(mPhProj.DirVisu() + "W_FiltResidual_"+aCalib->Name() +".tif");

       AddStdStatCSV(mNameReportCam,aCalib->Name(),aStat,mPropStat);
   }
}





int cAppli_CGPReport::Exe()
{
   mPhProj.FinishInit();

   auto nameSubDir = mPhProj.DPOrient().DirIn() +  "_Mes-"+  mPhProj.DPPointsMeasures().DirIn();
   if (IsInit(&mSuffixReportSubDir))
       nameSubDir += "_" + mSuffixReportSubDir;
   SetReportSubDir(nameSubDir);

   mNameReportIm   =  "ByImage" ;
   mNameReportDetail   =  "Detail" ;
   mNameReportGCP  =  "ByGCP"   ;
   mNameReportCam   =  "ByCam"   ;

   mNameReportGCP_Ground   =  "ByGCP_3D"   ;
   mNameReportGCP_Ground_Glob   =  "ByGCP_3D_Stat"   ;

   mNameReportMissed   =  "MissedPoint"   ;

   InitReport(mNameReportIm,"csv",true);
   InitReport(mNameReportDetail,"csv",true);
   InitReport(mNameReportMissed,"csv",true);

   if (LevelCall()==0)
   {
       AddStdHeaderStatCSV(mNameReportIm,"Image",mPropStat,{"AvgX","AvgY"});
       AddOneReportCSV(mNameReportDetail,{"Image","GCP","Err"});
       AddOneReportCSV(mNameReportMissed,{"Image","GCP","XTh","YTh"});
   }
   if (RunMultiSet(0,0))  // If a pattern was used, run in // by a recall to itself  0->Param 0->Set
   {
      int aRes = ResultMultiSet();

      if (mIsGCP)
      {
          ReportsByGCP();
          ReportsByCam();
      }

      return aRes;
   }

   if (mIsGCP)
   {
      MakeOneIm(FileOfPath(mSpecImIn,false));
   }

   return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */


tMMVII_UnikPApli Alloc_CGPReport(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_CGPReport(aVArgs,aSpec,true));
}

cSpecMMVII_Appli  TheSpec_CGPReport
(
     "ReportGCP",
      Alloc_CGPReport,
      "Reports on GCP projection",
      {eApF::GCP,eApF::Ori},
      {eApDT::GCP,eApDT::Orient},
      {eApDT::Image,eApDT::Xml},
      __FILE__
);


}; // MMVII

