//#include "MMVII_PCSens.h"

#include "MMVII_ExtractLines.h"

#include "MMVII_Sensor.h"
//#include "MMVII_ImageInfoExtract.h"
//#include "MMVII_TplGradImFilter.h"

//#include "MMVII_ExtractLines.h"
//#include "MMVII_2Include_CSV_Serial_Tpl.h"




namespace MMVII
{

template <class Type>  class cOneRayExtrC
{
     public :
        cOneRayExtrC(const cPt2di &  aSz,tREAL8 aRho,tREAL8 aDownScale);

     private :
        cPt2di      mSzInit;
        cPt2di      mSzIm;
        tREAL8      mRay;
        tREAL8      mDownScale;
        cIm2D<Type> mAccum;

};


template <class Type>
    cOneRayExtrC<Type>::cOneRayExtrC
    (
        const cPt2di & aSz,
        tREAL8 aRay,
        tREAL8 aDownScale
    ) :
    mSzInit    (aSz),
    mSzIm      (Pt_round_up(ToR(aSz)/aDownScale)),
    mRay       (aRay),
    mDownScale (aDownScale),
    mAccum     (mSzIm,nullptr,eModeInitImage::eMIA_Null)
{
}


template <class Type>  class cExtractCircle
{
   public :
       typedef cOneRayExtrC<Type> t1Ray;

       cExtractCircle(const cPt2di aSz,tREAL8 aRhoMin,tREAL8 aRhoMax,tREAL8 aNbByOct);

   private :
      cPt2di mSz;
      tREAL8 mRayMin;
      tREAL8 mRayMax;
      tREAL8 mNbByOct;
      tREAL8 mRatioDRay;
      std::vector<t1Ray>  mVEx1Ray;


};

template <class Type>
    cExtractCircle<Type>::cExtractCircle
    (
       const cPt2di aSz,
       tREAL8 aRayMin,
       tREAL8 aRayMax,
       tREAL8 aNbByOct
     ) :
        mSz        (aSz),
        mRayMin    (aRayMin),
        mRayMax    (aRayMax),
        mNbByOct   (aNbByOct),
        mRatioDRay (std::pow(2.0,1.0/mNbByOct))
{
    for (tREAL8 aRay = mRayMin ; aRay<mRayMax ; aRay *= mRatioDRay )
    {
        tREAL8 aDownScale = std::min(std::sqrt(aRay/aRayMin),5.0);
        mVEx1Ray.push_back(t1Ray(mSz,aRay,aDownScale));
    }

    StdOut() << " NB IMRAY=" << mVEx1Ray.size() << "\n";
}
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

        cExtractCurves<tElIm> *  mCurvExtract;
        cExtractCircle<tREAL4>*  mExtrC;
        tREAL8                   mDerFact;
        tREAL8                   mRayMin;
        tREAL8                   mRayMax;
        tREAL8                   mNbByRay;
        std::vector<tREAL8>      mParamMatch;
        std::string              mPatImage;
        std::string              mNameCurIm;

        cPt2di                   mCurSz;
        tIm                      mCurIm;
        tDIm*                    mDCurIm;
        cTimerSegm               mTimeSeg;

}; 


cAppliBubbles::cAppliBubbles(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli      (aVArgs,aSpec),
    mPhProj           (*this),
    mCurvExtract      (nullptr),
    mExtrC            (nullptr),
    mDerFact          (2.0),
    mRayMin           (5.0),
    mRayMax           (200.0),
    mNbByRay          (5.0),
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
            << AOpt2007(mDerFact,"DerFact","Deriche factor for graident",{eTA2007::HDV})


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
    mCurvExtract->SetDericheAndMasq(mDerFact,3.0,10);     
    mExtrC = new cExtractCircle<tREAL4>(mCurSz,mRayMin,mRayMax,mNbByRay);


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
    std::string aPref = mPhProj.DirVisuAppli() + LastPrefix(mNameCurIm) ;

     mCurvExtract->Grad().NormG().DIm().ToFile(aPref+"_GN.tif");
     mCurvExtract->Grad().mDGx->ToFile(aPref+"_Gx.tif");
     mCurvExtract->Grad().mDGy->ToFile(aPref+"_Gy.tif");

     cRGBImage aRGBIm = mCurvExtract->MakeImageMaxLoc(0.5);
     aRGBIm.ToFile(aPref+"_Cont.tif");
    /*
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
