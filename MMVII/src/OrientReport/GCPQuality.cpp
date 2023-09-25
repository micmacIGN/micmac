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

        cAppli_CGPReport(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :
        void  MakeOneIm(const std::string & aNameIm);
        void  MakeGlobReports();
        void  BeginReport();
        void  ReportsByGCP();
        void  ReportsByCam();

        std::string              mSpecImIn;   ///  Pattern of xml file
        cPhotogrammetricProject  mPhProj;

	std::vector<double>      mGeomFiedlVec;

	std::string              mPostfixReport;
	std::string              mPrefixReport;
	std::string              mNameReportIm;
        std::string              mNameReportGCP;
};

cAppli_CGPReport::cAppli_CGPReport
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this)
{
}



cCollecSpecArg2007 & cAppli_CGPReport::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
              << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              <<  mPhProj.DPPointsMeasures().ArgDirInMand()
              <<  mPhProj.DPOrient().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_CGPReport::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return    anArgOpt
	     << AOpt2007(mGeomFiedlVec,"GFV","Geom Fiel Vect for visu [Mul,Witdh,Ray,Zoom?=2]",{{eTA2007::ISizeV,"[3,4]"}})
    ;
}


//================================================


void cAppli_CGPReport::MakeOneIm(const std::string & aNameIm)
{
    cSetMesImGCP             aSetMes;
    mPhProj.LoadGCP(aSetMes);
    mPhProj.LoadIm(aSetMes,aNameIm);
    const cSetMesPtOf1Im  &  aSetMesIm = aSetMes.MesImInitOfName(aNameIm);

    // cSet2D3D aSet32;
    // mSetMes.ExtractMes1Im(aSet32,aNameIm);
    cSensorImage*  aCam = mPhProj.LoadSensor(aNameIm,false);

    // StdOut() << " aNameImaNameIm " << aNameIm  << " " << aSetMesIm.Measures().size() << " Cam=" << aCam << "\n";

    if (IsInit(&mGeomFiedlVec))
    {
         tREAL8 aMul    = mGeomFiedlVec.at(0);
         tREAL8 aWitdh  = mGeomFiedlVec.at(1);
         tREAL8 aRay    = mGeomFiedlVec.at(2);
	 int aDeZoom    = round_ni(GetDef(mGeomFiedlVec,3,2.0));
         cRGBImage aIma =  cRGBImage::FromFile(aNameIm);

	 //  [Mul,Witdh,Ray]
	 for (const auto & aMes : aSetMesIm.Measures())
	 {
	     cPt2dr aP2 = aMes.mPt;
	     cPt3dr aPGr = aSetMes.MesGCPOfName(aMes.mNamePt).mPt;
	     cPt2dr aProj = aCam->Ground2Image(aPGr);
	     cPt2dr  aVec = (aP2-aProj);

             aIma.DrawCircle(cRGBImage::Green,aP2,aRay);
             aIma.DrawLine(aP2,aP2+aVec*aMul,cRGBImage::Red,aWitdh);
	 }

         aIma.ToJpgFileDeZoom(mPhProj.DirVisu() + "FieldRes-"+aNameIm+".tif",aDeZoom);
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
    }


    std::vector<std::string> aVIm{
                                    aNameIm,
                                    ToStr(aAvg2d.Average().x()),
                                    ToStr(aAvg2d.Average().y()),
				    ToStr(aStat.Avg()),
				    ToStr(aStat.StdDev()),
				    ToStr(aStat.ErrAtProp(0.5)),
				    ToStr(aStat.ErrAtProp(0.75))
                              };
    AddOneReportCSV(mNameReportIm,aVIm);
}

void cAppli_CGPReport::BeginReport()
{
      AddOneReportCSV(mNameReportIm,{"Image","AvgX","AvgY","AvgD","Sigma","V50","V75"});
}


void cAppli_CGPReport::ReportsByGCP()
{
   cSetMesImGCP             aSetMes;
   mPhProj.LoadGCP(aSetMes);

   for (const auto & aNameIm : VectMainSet(0))
   {
       mPhProj.LoadIm(aSetMes,aNameIm,mPhProj.LoadSensor(aNameIm,false));
   }

   const std::vector<cSensorImage*> &  aVSens =  aSetMes.VSens() ;

   InitReport(mNameReportGCP,"csv",false);
   AddOneReportCSV(mNameReportGCP,{"GCP","Avg","StdDev","P50","P75"});

   for (const auto &  aMesIm :  aSetMes.MesImOfPt())
   {
        const auto & aGCP  = aSetMes.MesGCPOfMulIm(aMesIm);
	const std::vector<int> &  aVIndI = aMesIm.VImages() ;
        cStdStatRes               aStat;

	for (size_t aKIm = 0 ; aKIm<  aVIndI.size() ; aKIm++)
	{
            aStat.Add(Norm2( aMesIm.VMeasures()[aKIm]  - aVSens[aVIndI[aKIm]]->Ground2Image(aGCP.mPt)));
	}
	std::vector<std::string>  aVReport
	{
	       aGCP.mNamePt,
	       ToStr(aStat.Avg()),
	       ToStr(aStat.StdDev()),
	       ToStr(aStat.ErrAtProp(0.5)),
	       ToStr(aStat.ErrAtProp(0.75))
	};
        AddOneReportCSV(mNameReportGCP,aVReport);
   }

}

void cAppli_CGPReport::ReportsByCam()
{
   std::map<cPerspCamIntrCalib*,std::vector<cSensorCamPC*>>  aMapCam;
   cSetMesImGCP             aSetMes;
   mPhProj.LoadGCP(aSetMes);

   for (const auto & aNameIm : VectMainSet(0))
   {
       cSensorCamPC *  aCam = mPhProj.ReadCamPC(aNameIm,true);
       mPhProj.LoadIm(aSetMes,aNameIm,aCam);
       aMapCam[aCam->InternalCalib()].push_back(aCam);
   }

   tREAL8 aFactRed = 100.0;
   tREAL8 anExag = 1000;
   for (const auto& aPair : aMapCam)
   {
       cPerspCamIntrCalib * aCalib =  aPair.first;
       cPt2di aSz = Pt_round_up(ToR(aCalib->SzPix())/aFactRed) + cPt2di(1,1);

       cIm2D<tREAL8> aImX(aSz,nullptr,eModeInitImage::eMIA_Null);  // average X residual
       cIm2D<tREAL8> aImY(aSz,nullptr,eModeInitImage::eMIA_Null);  // average Y redidual
       cIm2D<tREAL8> aImW(aSz,nullptr,eModeInitImage::eMIA_Null);  // averagge weight

       for (const auto & aCam : aPair.second)
       {
           cSet2D3D aSet32;
	   aSetMes.ExtractMes1Im(aSet32,aCam->NameImage());
// StdOut() << " aCam->NameImage()aCam->NameImage() " << aCam->NameImage() << "\n";

           for (const auto & aPair : aSet32.Pairs())
           {
               cPt2dr aP2 = aPair.mP2;
               cPt2dr aRes = (aCam->Ground2Image(aPair.mP3) - aP2) *anExag;
               aImX.DIm().AddVBL(aP2/aFactRed,aRes.x());
               aImY.DIm().AddVBL(aP2/aFactRed,aRes.y());
               aImW.DIm().AddVBL(aP2/aFactRed,1.0);
           }
       }
       tREAL8 aSigma = Norm2(aImX.DIm().Sz()) / 60.0;

       aImW.DIm().ToFile(mPhProj.DirVisu() + "W_RawResidual_"+aCalib->Name() +".tif");
       aImX = aImX.GaussFilter(aSigma);
       aImY = aImY.GaussFilter(aSigma);
       aImW = aImW.GaussFilter(aSigma);

       DivImageInPlace(aImX.DIm(),aImX.DIm(),aImW.DIm());
       DivImageInPlace(aImY.DIm(),aImY.DIm(),aImW.DIm());

       aImX.DIm().ToFile(mPhProj.DirVisu() + "X_Residual_"+aCalib->Name() +".tif");
       aImY.DIm().ToFile(mPhProj.DirVisu() + "Y_Residual_"+aCalib->Name() +".tif");
       aImW.DIm().ToFile(mPhProj.DirVisu() + "W_FiltResidual_"+aCalib->Name() +".tif");
   }
}


void cAppli_CGPReport::MakeGlobReports()
{
   ReportsByGCP();
   ReportsByCam();

	/*
   std::map<std::string,std::list<std::string>>  aMapCamLIM;
   for (const auto & aNameIm : VectMainSet(0))
   {
   }

    cSetMesImGCP             aSetMes;
    mPhProj.LoadGCP(aSetMes);
    mPhProj.LoadIm(aSetMes,aNameIm);
    */
}



int cAppli_CGPReport::Exe()
{

   mPrefixReport =  mSpecs.Name() +"-" ;
   mPostfixReport  =  "-"+  mPhProj.DPOrient().DirIn() +  "-"+  mPhProj.DPPointsMeasures().DirIn() + "-" + Prefix_TIM_GMA();
   mNameReportIm = mPrefixReport + "ByImage" + mPostfixReport;
   mNameReportGCP = mPrefixReport + "ByGCP" + mPostfixReport;


   InitReport(mNameReportIm,"csv",true);


   mPhProj.FinishInit();

   if (LevelCall()==0)
   {
       BeginReport();
   }
   if (RunMultiSet(0,0))  // If a pattern was used, run in // by a recall to itself  0->Param 0->Set
   {
      int aRes = ResultMultiSet();

      MakeGlobReports();
      return aRes;
   }

   MakeOneIm(FileOfPath(mSpecImIn));

    return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */


tMMVII_UnikPApli Alloc_CGPReport(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_CGPReport(aVArgs,aSpec));
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

