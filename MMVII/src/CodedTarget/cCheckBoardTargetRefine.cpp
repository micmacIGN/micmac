//#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"

namespace MMVII
{
const std::vector<std::string> TargetLoc = {"ul","ur","md","ll","lr"};

    class cAppli_CheckBoardTargetRefine : public cMMVII_Appli//heritage de cMMVII_Appli
    {
        public:
            cAppli_CheckBoardTargetRefine(const std::vector<std::string> & aVArgs,
                                          const cSpecMMVII_Appli & aSpec);
            typedef tREAL4 tElem;
            typedef cIm2D<tElem> tIm;
            typedef cDataIm2D<tElem> tDIm;
            typedef cSegment<tREAL8,3> tSeg3dr;

        private:
            int Exe() override;

            //--spec. methods
            int doVisu(const std::string & aImName);
            int doPredict(const std::string & aImName);
            cSetMesPtOf1Im doAffInv(const std::string & aImName);
            std::vector<std::string> MissingTargetsNames(const cSetMesPtOf1Im & aFoundTargets);
            cSetMesPtOf1Im MissingTargetsPredict(const std::string & aImName, const std::vector<std::string> & aVMissingTargetsNames);
            void drawTarget(cSetMesPtOf1Im & aSetOfMes2D, bool isInit);
            void resetInitTargetPts();
            void doBundle(cSensorCamPC * aCam);

            std::string mSpecImIn;
            int mRes;
            bool mShow; //show details
            bool mVisu; //visualisation
            cRGBImage * mCurrVisuIm;
            cSetMesGnd3D mInSet3D;
            std::vector<cPt2dr> mInitTargetPts;
            cSetMesPtOf1Im mExtendedSetOfMes2D;
            cSetMesGnd3D mExtendedSetOfBundles;
            cSetMesGnd3D mExtendedSetOfMes3D;
            std::map<cSensorCamPC *, cSetMesGnd3D> mMImSetOfBundles;


            //----stolen to cCheckBoardTargetExtract
            std::string NameVisu(const std::string & aDestIm, const std::string & aPref,const std::string aPost="");
            //----

            //--mandatory
            cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
            cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
            cPhotogrammetricProject mPhProj;



    };

    cCollecSpecArg2007 & cAppli_CheckBoardTargetRefine::ArgObl(cCollecSpecArg2007 & anArgObl)
    {
        return anArgObl
            << Arg2007(mSpecImIn, "Pattern/file of images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
            << mPhProj.DPOrient().ArgDirInMand()
            << mPhProj.DPGndPt2D().ArgDirInMand()
            << mPhProj.DPGndPt3D().ArgDirInMand()
            << mPhProj.DPGndPt2D().ArgDirOutMand()
            << mPhProj.DPGndPt3D().ArgDirOutMand()

            ;
    }

    cCollecSpecArg2007 & cAppli_CheckBoardTargetRefine::ArgOpt(cCollecSpecArg2007 & anArgOpt)
    {
        return anArgOpt
               << AOpt2007(mRes,"Res","specifies target resolution", {eTA2007::HDV})
               << AOpt2007(mVisu,"Visu","offers visualisation of refined measurements", {eTA2007::HDV})
               << AOpt2007(mShow,"Show","show some useful details", {eTA2007::HDV})//hdv = has default value
            ;
    }

    cAppli_CheckBoardTargetRefine::cAppli_CheckBoardTargetRefine(const std::vector<std::string> & aVArgs,
                                                                const cSpecMMVII_Appli & aSpec):
        cMMVII_Appli(aVArgs, aSpec),
        mRes (600),
        mCurrVisuIm (nullptr),
        mPhProj (*this)

    {
        //···> constructor does nothing
    }


    int cAppli_CheckBoardTargetRefine::Exe()
    {
        //FINAL OUTPUT = temporary set of "fake" 2d mes
        // TO DO : add HasGCPMesIm facility to mPhProj or defined here

        mPhProj.FinishInit();

        //·0·> iterates on images
            //gets the first set
        std::vector<std::string> aVIm = VectMainSet(0);

        //·1·>loads global input
            //creates & fills 3d mes. set
        mInSet3D=mPhProj.LoadGCP3DFromFolder(mPhProj.DPGndPt3D().DirIn());
        std::vector<cSetMesGnd3D> aVExtendedSetOfBundles;

        StdOut() << "1·>>Prédiction des cibles non détectées" << std::endl;
        for (const auto & aImName:aVIm)
        {
            StdOut() << aImName << std::endl;
            cSensorCamPC * aCam = mPhProj.ReadCamPC(aImName, true);

            //·2->prediction of missing targets
            doPredict(aImName);

            //·4->inverse affinity projection for each target in each image
            //what about void methods and clearing sets at each iteration?
            mExtendedSetOfMes2D=doAffInv(aImName);
            doBundle(aCam);
            //mPhProj.DPOrient().DirOut();
            //·3->res. visualisation
            if (mVisu) doVisu(aImName);
        }

        StdOut() << "2·>>Intersection de points motifs sur les cibles détectées" << std::endl;

        for (std::string const & aCode:mInSet3D.ListOfNames())
        {
            std::vector<std::map<cSensorCamPC *,std::vector<cPt3dr>>> aVCodeMImBundles;//to store Bundle for each im

            for (const auto & aCam:mMImSetOfBundles)
            {
                std::map<cSensorCamPC *,std::vector<cPt3dr>> aMCodeImBundles;
                bool isFirst=true;
                for (cMes1Gnd3D const & aBundle:aCam.second.Measures())
                {
                    if (aBundle.mNamePt==aCode && isFirst)
                    {
                        isFirst=false;
                        aMCodeImBundles[aCam.first] = {aBundle.mPt};
                    } else if (aBundle.mNamePt==aCode) aMCodeImBundles[aCam.first].push_back(aBundle.mPt);
                }
                if (!isFirst) aVCodeMImBundles.push_back(aMCodeImBundles);
            }

            for (decltype(mInitTargetPts.size()) ix=0; ix<mInitTargetPts.size(); ++ix)
            {
                std::vector<tSeg3dr> aVPtBundles;
                for (auto const & aMImBundles:aVCodeMImBundles)
                {
                    for (auto const & aCam:aMImBundles)
                    {
                        aVPtBundles.push_back(tSeg3dr(aCam.first->Center(),
                                                      aCam.first->Center()+aCam.second[ix]));
                    }
                }
                if (aVPtBundles.size() >= 2){
                    cPt3dr a3DCorresp = BundleInters(aVPtBundles);
                    //StdOut() << "3d point" << aCode << ix << a3DCorresp << std::endl;
                    mExtendedSetOfMes3D.AddMeasure3D(cMes1Gnd3D(a3DCorresp, aCode+TargetLoc[ix]));
                }else{
                    StdOut() << aCode << "not enough bundles:"<< aVPtBundles.size() << std::endl;
                }
            }
        }
        mPhProj.SaveGCP3D(mExtendedSetOfMes3D, mPhProj.DPGndPt3D().DirOut());
        return EXIT_SUCCESS;
    }

    void cAppli_CheckBoardTargetRefine::doBundle(cSensorCamPC * aCam)
    {
        cSetMesGnd3D aExtendedSetOfBundles;

        for (auto const & aMes:mExtendedSetOfMes2D.Measures())
        {
            tSeg3dr aBundle = aCam->Image2Bundle(aMes.mPt);
            aExtendedSetOfBundles.AddMeasure3D(cMes1Gnd3D(aBundle.V12(), aMes.mNamePt));
        }
        mMImSetOfBundles[aCam] = aExtendedSetOfBundles;
    }

    cSetMesPtOf1Im cAppli_CheckBoardTargetRefine::doAffInv(const std::string & aImName)
    {
        //load 2d Mes im attributes
        cSetMesPtOf1Im aSetOfMes2D = mPhProj.LoadMeasureIm(aImName);
        cSetMesPtOf1Im aExtendedSetOfMes2D;
        //std::vector<cPt2dr> tmpRes;
        std::vector<cSaveExtrEllipe>aVSavedEllipses;
        ReadFromFile(aVSavedEllipses, cSaveExtrEllipe::NameFile(mPhProj, aSetOfMes2D, true));

        for (const cSaveExtrEllipe & aSavedEllipse:aVSavedEllipses)
        {
            resetInitTargetPts();
            for (const auto & aPt:mInitTargetPts)
            {
                aExtendedSetOfMes2D.AddMeasure(cMesIm1Pt(aSavedEllipse.mAffIm2Ref.Inverse(aPt),
                                                         aSavedEllipse.mNameCode + "", 0.1));
            }
        }
        return aExtendedSetOfMes2D;
    };

    void cAppli_CheckBoardTargetRefine::resetInitTargetPts()
    {
        mInitTargetPts={cPt2dr(0,0), cPt2dr(mRes-1,0),
                          cPt2dr(((mRes-1)/2), (mRes-1)/2),
                          cPt2dr(0,mRes-1),cPt2dr(mRes-1,mRes-1)};
    }

    int cAppli_CheckBoardTargetRefine::doPredict(const std::string & aImName)
    {
    //·1->loads curr. img. input/output
        auto aFoundTargets = mPhProj.LoadMeasureImFromFolder(mPhProj.DPGndPt2D().DirIn(), aImName);
    //·2->filter mes2D that have not been detected
        auto aVMissingTargetsNames = MissingTargetsNames(aFoundTargets);
    //·3->loads filtered 3D points & compute 2D projection (=prediction)
        if (aVMissingTargetsNames.empty())
        {
            StdOut() << "No missing targets" << std::endl;
            return 0;
        }
        cSetMesPtOf1Im aSetOfPredictions = MissingTargetsPredict(aImName, aVMissingTargetsNames);
        mPhProj.SaveMeasureIm(aSetOfPredictions);
    //·4->for each target find affinity parameters & compute correspondances

        return 0;
    }

    std::vector<std::string> cAppli_CheckBoardTargetRefine::MissingTargetsNames(const cSetMesPtOf1Im & aFoundTargets)
    {//diff. btw. detected points and GCPs set
        std::vector<std::string> aVMissingTargetsNames;//creates wanted set
        auto aExpectedTargetsNames = mInSet3D.ListOfNames();

        for (const auto & aTargetName:aExpectedTargetsNames)//iterates on mes3d checks if pt. p exists in known mes2d
        {
            if (!aFoundTargets.NameHasMeasure(aTargetName))
            {
                aVMissingTargetsNames.push_back(aTargetName);
            }
        }
        return aVMissingTargetsNames;
    }

    cSetMesPtOf1Im cAppli_CheckBoardTargetRefine::MissingTargetsPredict(const std::string & aImName, const std::vector<std::string> & aVMissingTargetsNames)
    {//3d->2d prediction of a set of 3d gcps
        cSetMesPtOf1Im aSetOfPredictions(aImName);//creates 2d set to fill
        cSensorCamPC * aCam = mPhProj.ReadCamPC(aImName,true);//loads im ori
        auto & camSz = aCam->InternalCalib()->PixelDomain().Sz();//avoid to load data im.

        for (const auto & aTargetName:aVMissingTargetsNames)//iterates on aWantedNames
        {
            auto aPrediction = aCam->Ground2Image(mInSet3D.GetMeasureOfNamePt(aTargetName).mPt);
            auto & xMax = camSz[0];
            auto & yMax = camSz[1];
            if (0 < aPrediction[0] && aPrediction[0] < xMax//add 2d mes. only if it is in image
                && 0 < aPrediction[1] && aPrediction[1] < yMax)
            {
                if (mShow)
                {
                    StdOut() << " --> Adding point " << aTargetName << " : "
                             << aPrediction[0] << ";" << aPrediction[1] << std::endl;
                }
                aSetOfPredictions.AddMeasure(cMesIm1Pt(aPrediction, aTargetName, 1));
            }
        }
        return aSetOfPredictions;
    }

    /* Visualisation methods */

    int cAppli_CheckBoardTargetRefine::doVisu(const std::string & aImName)
    {
        cRGBImage aIm = cRGBImage::FromFile(aImName);
        mCurrVisuIm = & aIm;
        if (mPhProj.HasMeasureImFolder(mPhProj.DPGndPt2D().DirOut(), aImName))
        {
            cSetMesPtOf1Im aSetOfNewMes2D = mPhProj.LoadMeasureImFromFolder(mPhProj.DPGndPt2D().DirOut(), aImName);
            drawTarget(aSetOfNewMes2D, false);//draw initial targets
        }
        if (mPhProj.HasMeasureImFolder(mPhProj.DPGndPt2D().DirIn(), aImName))
        {
            cSetMesPtOf1Im aSetOfInitMes2D = mPhProj.LoadMeasureImFromFolder(mPhProj.DPGndPt2D().DirIn(), aImName);
            drawTarget(aSetOfInitMes2D, true);//draw new targets
        }
        for (auto aMes : mExtendedSetOfMes2D.Measures())
        {
            //StdOut() << "drawing mes:" << aMes.mPt << "--" << aMes.mNamePt<<std::endl;
            mCurrVisuIm->DrawCircle(cRGBImage::Cyan, aMes.mPt, 1);
        }
        mCurrVisuIm->ToJpgFileDeZoom(NameVisu(aImName, "Ref"), 1, {"QUALITY=90"});

        return 0;
    }

    void cAppli_CheckBoardTargetRefine::drawTarget(cSetMesPtOf1Im & aSetOfMes2D, bool isInit)
    {//draws targets from a set of mes2d & adapts drawing color whether targets are from
     //initial measurements (nothing else than the b/w circle)
     //or not (Orange border represents a window size)
        for (auto aPt2D : aSetOfMes2D.Measures())
        {
            if (!isInit)
            {
                mCurrVisuIm->SetRGBBorderRectWithAlpha(ToI(aPt2D.mPt),100,10,cRGBImage::Orange,0.1);//0.1 means final opacity = 1-0.1
            }
            mCurrVisuIm->DrawCircle(cRGBImage::Black, aPt2D.mPt, 20);
            mCurrVisuIm->DrawCircle(cRGBImage::White, aPt2D.mPt, 22);
            mCurrVisuIm->DrawCircle(cRGBImage::White, aPt2D.mPt, 1);
            mCurrVisuIm->SetRGBPix(ToI(aPt2D.mPt), cRGBImage::Black);
            mCurrVisuIm->DrawString(aPt2D.mNamePt,cRGBImage::White,aPt2D.mPt,cPt2dr(0.5,0.05));
        }
    }

    //----stolen to cCheckBoardTargetExtract
    std::string cAppli_CheckBoardTargetRefine::NameVisu(const std::string & aDestIm, const std::string & aPref, const std::string aPost)
    {
        std::string aRes = mPhProj.DirVisuAppli() +  aPref +"-" + LastPrefix(FileOfPath(aDestIm));
        if (aPost!="") aRes = aRes + "-"+aPost;
        return    aRes + ".tif";
    }
    //----

    //pour faire des trucs avec la memoire - obligatoire
    tMMVII_UnikPApli Alloc_CheckBoardTargetRefine(const std::vector<std::string> & aVArgs,
                                                  const cSpecMMVII_Appli & aSpec)
    {
        return tMMVII_UnikPApli(new cAppli_CheckBoardTargetRefine(aVArgs, aSpec));
    }

    cSpecMMVII_Appli TheSpec_CheckBoardTargetRefine
        (
            "CheckBoardTargetRefine",
            Alloc_CheckBoardTargetRefine,
            "Refines target detection after CheckBoardTargetExtract",
            //metadonnees
            {eApF::Ori,eApF::GCP},//features
            {eApDT::ObjCoordWorld, eApDT::ObjMesInstr},//inputs
            {eApDT::Console},//output
            __FILE__
        );
}
