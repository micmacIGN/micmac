#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"

/**
   \file AppliPseudoIntersect.cpp

    Compute 3D coords from 2D coords and ori.
    Can try to filter errors.

 */

namespace MMVII
{


typedef std::pair<cSensorCamPC *,cMesIm1Pt> tPairCamPt;

struct cBundleDebugged
{
    cBundleDebugged(std::list<tPairCamPt> &aListPairCamPt);
    int nbUsed() const;
    bool removeErrors(double aIntersectTolerance); //< returns true if more iterations needed
    void getIntersection(bool isRobust);

    double mDistPix; // image residual
    cPt3dr mPG;      // ground point
    std::vector<char> mUsedBundles; // store 0/1 for bools
    const std::list<tPairCamPt> & mListPairCamPt;
};

cBundleDebugged::cBundleDebugged(std::list<tPairCamPt> &aListPairCamPt):
    mDistPix(NAN), mPG(cPt3dr::Dummy()), mUsedBundles(aListPairCamPt.size(), 1),
    mListPairCamPt(aListPairCamPt)
{
    getIntersection(true); // initialize with robust intersection
}

int cBundleDebugged::nbUsed() const
{
    return std::count(mUsedBundles.begin(),mUsedBundles.end(),1);
}

void cBundleDebugged::getIntersection(bool isRobust)
{
    if (nbUsed()<2)
    {
        mDistPix = NAN; // not good result
        return; // nothing more to do
    }
    std::vector<tSeg3dr> aVSeg;
    int i = 0;
    for (const auto & [aCam,aMes] : mListPairCamPt)
    {
        if (mUsedBundles[i])
            aVSeg.push_back(aCam->Image2Bundle(aMes.mPt));
        ++i;
    }
    if (isRobust)
        mPG = RobustBundleInters(aVSeg);
    else
        mPG = BundleInters(aVSeg);

    //compute residuals
    cWeightAv<tREAL8> aWPix;
    i = 0;
    for (const auto & [aCam,aMes] : mListPairCamPt)
    {
        if (mUsedBundles[i])
        {
            if (aCam->Pt_W2L(mPG).z()>0.) // check that Ground2Image is possible
            {
                cPt2dr aPProj = aCam->Ground2Image(mPG);
                double res = Norm2(aMes.mPt-aPProj);
                aWPix.Add(1.0,res);
            } else {
                mUsedBundles[i] = 0;
            }
        }
        ++i;
    }
    int aNbUsed = nbUsed();
    if (aNbUsed>1)
        mDistPix = aWPix.Average() * (aNbUsed*2.0) / (aNbUsed*2.0 -3.0);
    else
        mDistPix = NAN; // not good result
}

bool cBundleDebugged::removeErrors(double aIntersectTolerance)
{
    if (nbUsed()<2)
    {
        mDistPix = NAN; // not good result
        return false; // nothing more to do
    }

    bool stillToDo = false;
    int i = 0;
    for (const auto & [aCam,aMes] : mListPairCamPt)
    {
        if (mUsedBundles[i])
        {
            cPt2dr aPProj = aCam->Ground2Image(mPG);
            double res = Norm2(aMes.mPt-aPProj);
            if (res>aIntersectTolerance)
            {
                mUsedBundles[i] = 0;
                mDistPix = NAN; // not good result
                stillToDo = true;
            }
        }
        ++i;
    }
    if (stillToDo)
        return true; // still iterations to do
    return false; // iterations finished
}



//-------------------------------------------------------------

class cAppli_PseudoIntersect : public cMMVII_Appli
{
public :

    cAppli_PseudoIntersect(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

    cBundleDebugged debugBundles(std::list<tPairCamPt> aListPairCamPt);

private :
    cPhotogrammetricProject  mPhProj;
    std::string              mSpecImIn;   ///  Pattern of xml file
    std::vector<std::string> mSetNames;
    std::string              mPatNameGCP;
    double                   mIntersectTolerance;
    bool                     mKeepOnlyIntersectableInFiltered;
    void MakeStatByImage();
};

cAppli_PseudoIntersect::cAppli_PseudoIntersect
(
        const std::vector<std::string> &  aVArgs,
        const cSpecMMVII_Appli & aSpec
        ) :
    cMMVII_Appli  (aVArgs,aSpec),
    mPhProj       (*this),
    mPatNameGCP   (".*"),
    mIntersectTolerance (10.),
    mKeepOnlyIntersectableInFiltered   (false)
{
}

cCollecSpecArg2007 & cAppli_PseudoIntersect::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return     anArgObl
            <<  Arg2007(mSpecImIn,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPOrient().ArgDirInMand()
              <<  mPhProj.DPGndPt2D().ArgDirInMand()
              <<  mPhProj.DPGndPt3D().ArgDirOutMand()
                  ;
}


cCollecSpecArg2007 & cAppli_PseudoIntersect::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return   anArgOpt
            << AOpt2007(mPatNameGCP,"PatFiltGCP","Pattern to filter name of GCP",{{eTA2007::HDV}})
            << AOpt2007(mIntersectTolerance,"Tolerance","Maximal pixel error before removing bundle",{{eTA2007::HDV}})
            << mPhProj.DPGndPt2D().ArgDirOutOpt() // filtered 2d mes
            << AOpt2007(mKeepOnlyIntersectableInFiltered,"KeepOnlyIntersectableInFiltered","Keep only intersectable measures in filtered output",{{eTA2007::HDV}})
               ;
}

cBundleDebugged cAppli_PseudoIntersect::debugBundles(std::list<tPairCamPt> aListPairCamPt)
{
    cBundleDebugged aBundleDebugged(aListPairCamPt);
    while (aBundleDebugged.removeErrors(mIntersectTolerance))
    {
        // update ground point
        aBundleDebugged.getIntersection(false); // after 1st cleaning iteration, use non-robust intersection
    }
    // final iteration
    aBundleDebugged.getIntersection(false);

    return aBundleDebugged;
}


int cAppli_PseudoIntersect::Exe()
{
    mPhProj.FinishInit();

    MMVII_INTERNAL_ASSERT_User((!mKeepOnlyIntersectableInFiltered)||mPhProj.DPGndPt2D().DirOutIsInit(),
                               eTyUEr::eBadOptParam, "KeepOnlyIntersectableInFiltered needs output 2D mes dir to be set!");



    mSetNames = VectMainSet(0);

    std::vector<cSensorCamPC *> aVCam;

    for (const auto & aNameIm : mSetNames)
    {
        aVCam.push_back(mPhProj.ReadCamPC(aNameIm,true,false));
    }

    std::map<std::string,std::list<tPairCamPt>> aMapMatch; // for each point name, list of camera and 2d mes for intersection

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
    }

    std::map<std::string, std::set<std::string>> aMapWrongMesPerImage;
    cSetMesGnd3D aMesGCP("PseudoIntersect");
    for (const auto & [aStr,aList] : aMapMatch )
    {
        if (aList.size()>=2)
        {
            auto aBundleDebugged = debugBundles(aList);
            if (!std::isfinite(aBundleDebugged.mDistPix))
            {
                StdOut() << "Point " << aStr << " can't be intersected.\n";
                if (mKeepOnlyIntersectableInFiltered)
                    for (const auto & [aCam,aMes] : aList)
                        aMapWrongMesPerImage[aCam->NameImage()].insert(aStr);
                continue;
            }
            //StdOut() << aStr << " " << aBundleDebugged.mDistPix << " ";

            if (aBundleDebugged.nbUsed() != (long)aBundleDebugged.mUsedBundles.size())
            {
                StdOut() << "Point " << aStr << " has error in:";
                int i = 0;
                for (const auto & [aCam,aMes] : aList)
                {
                    if (aBundleDebugged.mUsedBundles[i]==0)
                    {
                        StdOut() << " " << aCam->NameImage();
                        aMapWrongMesPerImage[aCam->NameImage()].insert(aStr);
                    }
                    ++i;
                }
                StdOut() << "\n";
            }
            if (std::isfinite(aBundleDebugged.mDistPix))
                aMesGCP.AddMeasure3D( cMes1Gnd3D(aBundleDebugged.mPG, aStr, -1., "From pseudo-intersection") );
            //StdOut() << "\n";
        } else {
            if (mKeepOnlyIntersectableInFiltered)
                for (const auto & [aCam,aMes] : aList)
                    aMapWrongMesPerImage[aCam->NameImage()].insert(aStr);
        }
    }

    StdOut() << "Total: " << aMesGCP.Measures().size() << " successfully intersected\n";

    mPhProj.SaveGCP3D(aMesGCP,mPhProj.DPGndPt3D().DirOut());

    tPtrSysCo aSysCo = mPhProj.CurSysCoOri(true);
    if (aSysCo)
        mPhProj.SaveCurSysCoGCP(aSysCo);

    // export a filtered version of 2d mes
    if (mPhProj.DPGndPt2D().DirOutIsInit())
    {
        std::vector<std::string> aListMesFileIN;
        GetFilesFromDir(aListMesFileIN,mPhProj.DPGndPt2D().FullDirIn(),AllocRegex(""));
        for (const auto & aNameFile : aListMesFileIN)
        {
            cSetMesPtOf1Im aSetIn = cSetMesPtOf1Im::FromFile(mPhProj.DPGndPt2D().FullDirIn()+aNameFile);
            cSetMesPtOf1Im aSetOut(aSetIn.NameIm());
            auto aSetOfWrongMes = aMapWrongMesPerImage.find(aSetIn.NameIm());

            for (const auto & aMes : aSetIn.Measures())
            {
                if ( (aSetOfWrongMes==aMapWrongMesPerImage.end())
                    || (aSetOfWrongMes->second.count(aMes.mNamePt) == 0)) // add only if not a reported error
                        aSetOut.AddMeasure(aMes);
            }
            mPhProj.SaveMeasureIm(aSetOut);
        }
    }
    return EXIT_SUCCESS;
}


/* ==================================================== */

tMMVII_UnikPApli Alloc_PseudoIntersect(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppli_PseudoIntersect(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_PseudoIntersect
(
        "PseudoIntersect",
        Alloc_PseudoIntersect,
        "Pseudo Intersect: 2D points to 3D coords",
        {eApF::TieP,eApF::Ori},
        {eApDT::TieP,eApDT::Orient},
        {eApDT::Image,eApDT::Xml},
        __FILE__
        );


}; // MMVII

