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

class cAppli_MesImReport : public cMMVII_Appli
{
public :

    cAppli_MesImReport(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &,bool IsGCP);
    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

private :
    /** make the report  by image, for each image a cvs file with all GCP,
     * optionnaly make a visualisation of the residual fielsd for each image */
    void  MakeOneIm(const std::string & aNameIm);

    /** Make a report with an average for each GCP */
    /** Make a visualization of residual in sensor plane*/

    cPhotogrammetricProject  mPhProj;
    std::string              mSpecImIn;   ///  Pattern of xml file
    std::string              mRefFolder;   ///  Pattern of xml file


    tREAL8                   mThresholdMatch;  /// Threshold for validating a match
    std::vector<int>         mPropStat;

    std::string              mPrefixReport;

    std::string              mNameReportGlobResidual;
    std::string              mNameReportDetail;

    std::string              mSuffixReportSubDir; // additional name for report subdir
    // double                   mMarginMiss;  ///  Margin for counting missing targets

    cStdStatRes              mStatGlob;
    std::string              mNameSubDir;
    int                      mNbUndetectedGlob;
    int                      mNbFalseDetGlob;
    int                      mNbRefGlob;
    tREAL8                   mExagRes;
    tREAL8                   mScaleImage;
    std::string              mSuffix;
};

cAppli_MesImReport::cAppli_MesImReport
(
        const std::vector<std::string> &  aVArgs,
        const cSpecMMVII_Appli & aSpec,
        bool                     isGCP
        ) :
    cMMVII_Appli              (aVArgs,aSpec),
    mPhProj                   (*this),
    mThresholdMatch           (1.0),
    mPropStat                 ({50,75,90}),
    mNameReportGlobResidual   ("StatResidual"),
    mNameReportDetail         ("Detail"),
    mSuffixReportSubDir       (""),
    // mMarginMiss               (50.0),
    mNbUndetectedGlob         (0),
    mNbFalseDetGlob           (0),
    mNbRefGlob                (0),
    mScaleImage               (1.0)
{
}



cCollecSpecArg2007 & cAppli_MesImReport::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return     anArgObl
            << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
            << mPhProj.DPGndPt2D().ArgDirInMand("Folder for tested data")
            << mPhProj.DPGndPt2D().ArgDirInMand("Folder for reference data",&mRefFolder)
               ;
}

cCollecSpecArg2007 & cAppli_MesImReport::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return      anArgOpt
            << AOpt2007(mPropStat,"Perc","Percentil for stat exp",{eTA2007::HDV})
            << AOpt2007(mExagRes,"ExagRes","Factor of residual exageration, set make visu",{eTA2007::HDV})
            << AOpt2007(mScaleImage,"ScaleIma","Is any,scale having been applied to image before extract",{eTA2007::HDV})
            << AOpt2007(mSuffix,"Suffix","Suffix for output name",{eTA2007::HDV})
            << AOpt2007(mThresholdMatch,"ThresholdMatch","Threshold for matching points",{eTA2007::HDV})
               ;
}


//================================================


void cAppli_MesImReport::MakeOneIm(const std::string & aNameIm)
{
    StdOut()  << "MakeOneIm : " << aNameIm << "\n";
    // tREAL8 aFactErr= 200.0;
    bool  doImage = IsInit(&mExagRes);
    cRGBImage aIm(cPt2di(1,1));
    if (doImage)
    {
        aIm = cRGBImage::FromFile(aNameIm);
        aIm.ResetGray();
    }


    cSetMesPtOf1Im  aSet2Test = mPhProj.LoadMeasureIm(aNameIm);
    cSetMesPtOf1Im  aSetRef   = mPhProj.LoadMeasureImFromFolder(mRefFolder,aNameIm);
    cStdStatRes     aStatIm;
    int             nbUnDetected=0;

    for (const auto & aRef : aSetRef.Measures())
    {
        std::string aNameRef = aRef.mNamePt;
        cPt2dr aRefPt = aRef.mPt/mScaleImage;
        bool isDetected  = false;
        if (aSet2Test.NameHasMeasure(aNameRef))
        {
            cMesIm1Pt aHom = aSet2Test.MeasuresOfName(aNameRef);
            cPt2dr aVRes  = aRefPt - aHom.mPt;
            tREAL8 aRes = Norm2(aVRes);
            if (aRes < mThresholdMatch)
            {
                mStatGlob.Add(aRes);
                aStatIm.Add(aRes);
                isDetected = true;
                if (doImage)
                {
                    aVRes = aVRes * mExagRes;
                    tREAL8 aMaxN = 50.0;
                    if (Norm2(aVRes) > aMaxN)
                        aVRes = VUnit(aVRes) * aMaxN;

                    aIm.DrawLine(aRefPt,aRefPt+aVRes,cRGBImage::Green);
                }
            }
            // --- StdOut() << aNameRef <<  aRes << "\n";
        }
        if (! isDetected)
        {
            nbUnDetected ++;
            if (doImage)
                aIm.SetRGBrectWithAlpha(ToI(aRefPt),30,cRGBImage::Orange,0.8);
        }
    }

    int aNbFalseDet=0;
    for (const auto & aTest : aSet2Test.Measures())
    {
        std::string aNameTest = aTest.mNamePt;
        bool isDetected  = false;
        if (aSetRef.NameHasMeasure(aNameTest))
        {
            cMesIm1Pt aHom = aSetRef.MeasuresOfName(aNameTest);
            cPt2dr aRefPt = aHom.mPt/mScaleImage;
            if (Norm2(aTest.mPt-aRefPt) < mThresholdMatch)
                isDetected = true;
        }
        if (! isDetected)
        {
            aNbFalseDet ++;
            if (doImage)
                aIm.SetRGBrectWithAlpha(ToI(aTest.mPt),30,cRGBImage::Red,0.8);
        }
    }
    mNbFalseDetGlob = aNbFalseDet;


    int aNbM = aSetRef.Measures().size();
    mNbRefGlob += aNbM;
    mNbUndetectedGlob += nbUnDetected;
    AddStdStatCSV
            (
                mNameReportGlobResidual,aNameIm,aStatIm,mPropStat,
                {ToStr((100.0*nbUnDetected)/aNbM),ToStr((100.0*aNbFalseDet)/aNbM)}
                );


    if (doImage)
        aIm.ToFile(DirReport()+ aNameIm);
}






int cAppli_MesImReport::Exe()
{
    mPhProj.FinishInit();

    mNameSubDir = mPhProj.DPGndPt2D().DirIn() +  "_vs_"+  mRefFolder + mSuffix;
    if (mScaleImage != 1.0)
        mNameSubDir += "_Scale" + ToStr(mScaleImage);
    if (IsInit(&mSuffixReportSubDir))
        mNameSubDir += "_" + mSuffixReportSubDir;
    SetReportSubDir(mNameSubDir);


    InitReportCSV(mNameReportGlobResidual,"csv",false);

    AddStdHeaderStatCSV(mNameReportGlobResidual,"Image",mPropStat,{"% Undetected","% False Detect"});


    for (const std::string & aName : VectMainSet(0))
        MakeOneIm(FileOfPath(aName,false));

    AddStdStatCSV
            (
                mNameReportGlobResidual,"Global",mStatGlob,mPropStat,
                {ToStr((100.0*mNbUndetectedGlob)/mNbRefGlob),ToStr((100.0*mNbFalseDetGlob)/mNbRefGlob)}
                );



    StdOut()  <<  "GLOB RES "
               <<  " AVG=" <<  mStatGlob.Avg()
                <<  " P50=" <<  mStatGlob.ErrAtProp(0.5)
                 <<  " P75=" <<  mStatGlob.ErrAtProp(0.75)
                  <<  " P90=" <<  mStatGlob.ErrAtProp(0.9)
                   <<  " UnDetected=" <<  mNbUndetectedGlob
                    <<  "\n";




    return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */


tMMVII_UnikPApli Alloc_MesImReport(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppli_MesImReport(aVArgs,aSpec,true));
}

cSpecMMVII_Appli  TheSpec_MesImReport
(
        "ReportMesIm",
        Alloc_MesImReport,
        "Reports on Images measures compared to a reference",
        {eApF::GCP,eApF::Ori},
        {eApDT::GndPt2D,eApDT::Orient},
        {eApDT::Image,eApDT::Xml},
        __FILE__
        );


}; // MMVII

