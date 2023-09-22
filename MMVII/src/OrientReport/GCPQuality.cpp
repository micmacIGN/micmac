#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"


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
        void MakeOneIm(const std::string & aNameIm);
        void  BeginReport();

        std::string              mSpecImIn;   ///  Pattern of xml file
        cPhotogrammetricProject  mPhProj;

	std::vector<double>      mGeomFiedlVec;

	std::string              mPostfixReport;
	std::string              mPrefixReport;
	std::string              mNameReportIm;
	// std::string              mNameReportGCP;
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

         aIma.ToFileDeZoom("ZOOM-"+aNameIm+".tif",aDeZoom);
    }


    std::vector<tREAL8>  aVRes;
    cWeightAv<tREAL8,cPt2dr>  aAvg2d;
    cWeightAv<tREAL8,tREAL8>  aAvgDist;
    cWeightAv<tREAL8,tREAL8>  aAvgDist2;

    for (const auto & aMes : aSetMesIm.Measures())
    {
        cPt2dr aP2 = aMes.mPt;
        cPt3dr aPGr = aSetMes.MesGCPOfName(aMes.mNamePt).mPt;
        cPt2dr aProj = aCam->Ground2Image(aPGr);
        cPt2dr  aVec = (aP2-aProj);

	aAvg2d.Add(1.0,aVec);
	tREAL8 aDist = Norm2(aVec);
	aVRes.push_back(Norm2(aVec));
	aAvgDist.Add(1.0,aDist);
	aAvgDist2.Add(1.0,Square(aDist));
    }

    StdOut() << "Im=" << aNameIm <<  " Avxy : " << aAvg2d.Average()  
	    <<  " AvD=" << aAvgDist.Average() << " AvD2=" << std::sqrt(aAvgDist2.Average())
	    << " P50=" << NC_KthVal(aVRes,0.5) << " P75=" <<  NC_KthVal(aVRes,0.75) 
	    << "\n";


    std::vector<std::string> aVIm{
                                    aNameIm,
                                    ToStr(aAvg2d.Average().x()),
                                    ToStr(aAvg2d.Average().y()),
				    ToStr( aAvgDist.Average()),
				    ToStr(std::sqrt(aAvgDist2.Average())),
				    ToStr(NC_KthVal(aVRes,0.5)),
				    ToStr(NC_KthVal(aVRes,0.75))
                              };
     AddOneReportCSV(mNameReportIm,aVIm);

}

void cAppli_CGPReport::BeginReport()
{
      AddOneReportCSV(mNameReportIm,{"Image","AvgX","AvgY","AvgD","Sigma","V50","V75"});
      // AddOneReportCSV(mNameReportGCP,{"GCP","Avg"});
      // AddOneReportCSV(mNameReportGCP,{"1","2"});
}

int cAppli_CGPReport::Exe()
{

   mPrefixReport =  mSpecs.Name() +"-" ;
   mPostfixReport  =  "-"+  mPhProj.DPOrient().DirIn() +  "-"+  mPhProj.DPPointsMeasures().DirIn() + "-" + Prefix_TIM_GMA();
   mNameReportIm = mPrefixReport + "ByImage" + mPostfixReport;
   // mNameReportGCP = mPrefixReport + "ByGCP" + mPostfixReport;


   InitReport(mNameReportIm,"csv",true);
   // InitReport(mNameReportGCP,"csv",true);


   mPhProj.FinishInit();

   if (LevelCall()==0)
   {
       BeginReport();
   }
   if (RunMultiSet(0,0))  // If a pattern was used, run in // by a recall to itself  0->Param 0->Set
   {
      return ResultMultiSet();
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

