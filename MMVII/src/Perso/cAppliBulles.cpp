//#include "MMVII_PCSens.h"

#include "MMVII_ExtractLines.h"

#include "MMVII_Sensor.h"
//#include "MMVII_ImageInfoExtract.h"
//#include "MMVII_TplGradImFilter.h"

//#include "MMVII_ExtractLines.h"
//#include "MMVII_2Include_CSV_Serial_Tpl.h"




namespace MMVII
{

template <class Type>  class cExtractCircle
{
   public :


   private :
};
/* =============================================== */
/*                                                 */
/*                 cAppliBubbles                   */
/*                                                 */
/* =============================================== */


/**  An application for  computing line extraction. Created for CERN
 *  porject, and parametrization is more or less optimized for this purpose.
 */

//    cIm2D<tIm> anIm = cIm2D<tIm>::FromFile(mNameCurIm);

class cAppliBubbles : public cMMVII_Appli
{
     public :
        cAppliBubbles(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
     private :
        typedef tREAL4            tElIm;
        typedef cIm2D<tElIm >     tIm;
        typedef cDataIm2D<tElIm > tDIm;

        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
         std::vector<std::string>  Samples() const override;
        virtual ~cAppliBubbles();

        void  DoOneImage(const std::string & aNameIm) ;

        void MakeVisu();

     
        cPhotogrammetricProject  mPhProj;

        cExtractCurves<tElIm> * mCurvExtract;

        std::vector<tREAL8>      mParamMatch;
        std::string              mPatImage;
        std::string              mNameCurIm;

        cPt2di                   mCurSz;
        tIm                      mCurIm;
   //     tIm                      mGx;
   //     tIm                      mGy;
   //     tIm                      mGNorm;
        tDIm*                    mDCurIm;
        /*
        tDIm*                    mDGx;
        tDIm*                    mDGy;
        tDIm*                    mDGNorm;
        */

	cTimerSegm               mTimeSeg;

}; 


cAppliBubbles::cAppliBubbles(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli      (aVArgs,aSpec),
    mPhProj           (*this),
    mCurvExtract      (nullptr),
    mParamMatch       {},
    mCurIm            (cPt2di(1,1)),
  //  mGx               (cPt2di(1,1)),
  //  mGy               (cPt2di(1,1)),
  //  mGNorm            (cPt2di(1,1)),
    mTimeSeg          (this)
{
}

cAppliBubbles::~cAppliBubbles()
{
    delete mCurvExtract;
}

cCollecSpecArg2007 & cAppliBubbles::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
      return    anArgObl
            <<  Arg2007(mPatImage,"Name of input Image", {eTA2007::FileDirProj,{eTA2007::MPatFile,"0"}})
      ;
}

cCollecSpecArg2007 & cAppliBubbles::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
	        << AOpt2007(mParamMatch,"MatchParam","[Angl,DMin,DMax]",{eTA2007::HDV,{eTA2007::ISizeV,"[3,3]"}})
            ;
}

std::vector<std::string>  cAppliBubbles::Samples() const
{
   return {
              "MMVII  ... " 
	};
}


void  cAppliBubbles::DoOneImage(const std::string & aNameIm)
{



    mNameCurIm = aNameIm;
    cAutoTimerSegm  anATSInit (mTimeSeg,"Initialisation");

    cAutoTimerSegm  anATSReadIm (mTimeSeg,"ReadImage");
    mCurIm = tIm::FromFile(mNameCurIm);
    mDCurIm = & mCurIm.DIm();
    mCurSz =  mDCurIm->Sz();

    mCurvExtract = new  cExtractCurves<tElIm> (mCurIm);
   // mCurvExtract->SetSobelAndMasq()


  //  void SetSobelAndMasq(eIsWhite,tREAL8 aRayMaxLoc,int aBorder,bool Show=false);

    /*
    mGx = tIm(mCurSz);
    mDGx = &mGx.DIm();
    mGy = tIm(mCurSz);
    mDGy = &mGy.DIm();
    mGNorm = tIm(mCurSz);
    mDGNorm = & mGNorm.DIm();

    ComputeSobel(*mDGx,*mDGy,*mDCurIm);

    for (const auto & aPix : *mDGNorm)
    {
        tREAL4 aGx = mDGx->GetV(aPix);
        tREAL4 aGy = mDGy->GetV(aPix);
        mDGNorm->SetV(aPix,Norm2(cPt2dr(aGx,aGy)));
    }
*/

    /*
    int mNbOct = 5;
    int mNbLevByOct = 6;
    int mNbOverlap = 0;

    cGP_Params aParam();
    */
    //const cPt2di & aSzIm0,int aNbOct,int aNbLevByOct,int aOverlap,const cMMVII_Appli *,bool is4TieP)

    StdOut() << " mNameCurIm " << mCurSz << "\n";
    MakeVisu();
}




void cAppliBubbles::MakeVisu()
{
    /*
    std::string aPref = mPhProj.DirVisuAppli() + LastPrefix(mNameCurIm) ;
    mDGx->ToFile(aPref+"_Gx.tif");
    mDGy->ToFile(aPref+"_Gy.tif");
    mDGNorm->ToFile(aPref+"_GN.tif");
    */
//mPhProj.DirVisuAppli() + 
}




int cAppliBubbles::Exe()
{
    mPhProj.FinishInit();

    //  InitReportCSV(mNameReportByIm,"csv",true,{"NameIm","CodeResult"});

    //  Create a report with header computed from type
    // Tpl_AddHeaderReportCSV<cOneLineAntiParal>(*this,mIdExportCSV,true);
    // Redirect the reports on folder of result
    // SetReportRedir(mIdExportCSV,mPhProj.DPGndPt2D().FullDirOut());

    if (RunMultiSet(0,0))
    {
       return ResultMultiSet();
    }
    DoOneImage(UniqueStr(0));
    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_AppliBubbles(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
      return tMMVII_UnikPApli(new cAppliBubbles(aVArgs,aSpec));
}


cSpecMMVII_Appli  TheSpecAppliBubbles
(
     "ExtractBubbles",
      Alloc_AppliBubbles,
      "Extraction of Bubbles",
      {eApF::ImProc},
      {eApDT::Image},
      {eApDT::Console},
      __FILE__
);

#if (0)
#endif

};
