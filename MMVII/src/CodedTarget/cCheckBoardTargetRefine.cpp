//#include "MMVII_Sensor.h"
#include "CodedTarget.h"
#include "MMVII_Interpolators.h"
#include "MMVII_PCSens.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Matrix.h"

namespace MMVII
{
const std::vector<std::string> TargetLoc = {"ul","ur","ll","lr"};

    class cAppli_CheckBoardTargetRefine : public cMMVII_Appli//heritage de cMMVII_Appli
    {
        public:
            cAppli_CheckBoardTargetRefine(const std::vector<std::string> & aVArgs,
                                          const cSpecMMVII_Appli & aSpec);
            typedef tU_INT1 tElem;
            typedef cIm2D<tElem> tIm;
            typedef cDataIm2D<tElem> tDIm;
            typedef cSegment<tREAL8,3> tSeg3dr;
            typedef cTplBox<double,2>  cBox2dr;


        private:
            int Exe() override;

            //--spec. methods
            int doVisu(const cSensorCamPC * aCam);
            int doPredict(const std::string & aImName);
            void doAffInv(const cSensorCamPC * aCam);
            void doReproj(const std::string & aCode);
            cBox2dr doTargetRefine(bool& isOk);
            cBox2dr doCurrImageTgtExtent(bool& isOk);
            std::vector<std::string> MissingTargetsNames(const cSetMesPtOf1Im & aFoundTargets);
            cSetMesPtOf1Im MissingTargetsPredict(const std::string & aImName, const std::vector<std::string> & aVMissingTargetsNames);
            void drawTarget(cSetMesPtOf1Im & aSetOfMes2D, bool isInit);
            void resetInitTargetPts();
            void doBundle(const cSensorCamPC * aCam);
            std::vector<cPt3dr> getVMotifMes3D(std::list<std::string> lNamesPts);
            cSimilitud3D<cPt3dr> doSimil3D(std::vector<cPt3dr> &aVPtsIn, std::vector<cPt3dr> &aVPtsOut);
            //std::map<std::string, cSimilitud3D<tREAL8>> computeTargets2WorldMappings();
            cSetTargetMap computeTargets2WorldMappings();
            std::map<std::string, std::vector<cPt3dr>> computeTargetsWorldBasePoints();
            cPt2dr world2Target(const std::string& aTargetCode, const cPt3dr& aWorldPt);
            std::vector<cPt3dr> target2World(const std::string& aTargetCode, std::vector<cPt2dr>& aVWorldPt);
            cPt3dr target2World(const std::string& aTargetCode, const cPt2dr& aTargetPt);
            std::vector<cPt2dr> get2DBasePoints();
            std::vector<cPt3dr> get3DBasePoints();
            std::vector<cPt3dr> get3DTargetCorners(std::string& aCode);
            void doPseudoBench();
            tIm doTargetSample(cBox2dr& aExtent);
            cAffin2D<tREAL8> getLocalAff2D(cPt2dr aPix, tREAL8 delta);
            //--mandatory
            cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
            cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
            cPhotogrammetricProject mPhProj;
            /*
             *
             */
            std::vector<cSensorCamPC*> mVCams;
            std::string mSpecImIn;
            int mRes;
            bool mShow; //show details
            bool mVisu; //visualisation
            cSetTargetMap mTargets2WorldMappings;
            std::set <std::string> mIntersectedCodes;
            cRGBImage * mCurrVisuIm;
            cSensorCamPC* mCurrCam;
            std::string mCurrTgtCode;
            cPlane3D* mCurrTgtPlane;
            cSetMesGnd3D mMeasuredTargets;
            std::vector<cPt2dr> mInitTargetPts;
            cSetMesPtOf1Im mExtendedSetOfMes2D;
            cSetMesGnd3D mExtendedSetOfBundles;
            cSetMesGnd3D mExtendedSetOfMes3D;
            std::map<const cSensorCamPC *,cSetMesPtOf1Im> mMImExtendedSetOfMes2D;
            //cSetTargetSim3D mSetTargetSim3D;
            std::map<const cSensorCamPC *, cSetMesGnd3D> mMImSetOfBundles;
            std::unique_ptr<cFullSpecifTarget> mFullSpec;


            //----stolen to cCheckBoardTargetExtract
            std::string NameVisu(const std::string & aDestIm, const std::string & aPref,const std::string aPost="");
            //----





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
        mPhProj (*this),
        mRes (600),
        mTargets2WorldMappings (""),
        mCurrVisuIm (nullptr),
        mFullSpec (nullptr)


    {
        //···> constructor does nothing
    }


    int cAppli_CheckBoardTargetRefine::Exe()
    {
        /*
         * first task is to iterate on images and product useful informations:
         *  ->if a target has not been detected compute projection in the image
         *    based on image's pose -> results goes to a "a priori" 2d measures
         *    file
         *    -->doPredict
         *  ->for each detected target compute bundle to extended target points
         *    (=corners points and middle point projected to image from affinity
         *    mapping from pattern to image plan
         *    -->doAffInv & doBundle
         *
         * second task is to iterate on targets and compute mapping from pattern
         * to space based on useful informations we computed earlier:
         *  -> first we gather 2d/3d correspondances from all images in which we
         *     were able to compute them
         *  -> then we adjust a 3d similarity giving us a the searched mapping
         *     and we store it
         *
         * third one would be to create the image of the target as it would have
         * been viewed in an image an look for the best correlation around a
         * point that we would have predict
         *  -> first we want to get the extend of the target in the view so we
         *     compute the projection in the view of the upper-left and lower
         *     right corner in order to get a 2d rectangle that we will
         *     consider as being the extent
         */

        // PROPOSALS :
            // HasGCPMesIm facility to mPhProj
            // GetMeasures(aCode) to cSetMesGnd3D
       // return EXIT_SUCCESS;  OK

        mPhProj.FinishInit();

        mFullSpec.reset(cFullSpecifTarget::CreateFromFile("./IGNIndoor_Nbb14_Freq2_Hamm4_Run2_3_FullSpecif.xml"));
        std::vector<std::string> aVIm = VectMainSet(0);//gets the first set

        for (const auto& aImName:aVIm)
        {
            cSensorCamPC* aCam = mPhProj.ReadCamPC(aImName, true);
            mVCams.push_back(aCam);
        }

        mMeasuredTargets = mPhProj.LoadGCP3DFromFolder(mPhProj.DPGndPt3D().DirIn());

        mTargets2WorldMappings = computeTargets2WorldMappings();
        SaveInFile(mTargets2WorldMappings, cSetTargetMap::NameFile(mPhProj, mTargets2WorldMappings.Name(), false));

        for (const auto& aCam:mVCams)
        {
            mCurrCam = aCam;
            auto aCodes = mTargets2WorldMappings.ListOfCodes();

            //cRGBImage aIm = cRGBImage::FromFile(mCurrCam->NameImage());

            for (const auto& aTgt:aCodes)
            {
                mCurrTgtCode = aTgt;
                bool isOk;
                //auto aBox = doTargetRefine(isOk);
                //if (mCurrCam->NameImage() == "K127_202409211622-00-cam-22348125-81-66384840717079-66.tiff" && mCurrTgtCode != "35")
                //if (mCurrCam->NameImage() == "cam125_az1_0000.tiff")
                {
                    StdOut() << aTgt<<std::endl;
                    doTargetRefine(isOk);
                }
                /*if(isOk)
                {
                    aIm.SetRGBBorderRectWithAlpha(ToI(aBox.Middle()),Norm2(aBox.Sz())/2,10,cRGBImage::Orange,0.1);//0.1 means final opacity = 1-0.1
                    aIm.DrawCircle(cRGBImage::Green, aBox.P0(), 4);
                    aIm.DrawCircle(cRGBImage::Red, aBox.P1(), 4);
                }*/

            }
            StdOut() << aCam->NameImage() << std::endl;

            //aIm.ToJpgFileDeZoom(NameVisu(mCurrCam->NameImage(), "Ref"), 1, {"QUALITY=90"});


        }

        //doPseudoBench();

        return EXIT_SUCCESS;
    }

    void cAppli_CheckBoardTargetRefine::doPseudoBench()
    {
        for (auto aPt:mMeasuredTargets.Measures())
        {
            cPt2dr aTgtPt = world2Target(aPt.mNamePt, aPt.mPt);
            StdOut() << aPt.mNamePt << " target coords. : " << aTgtPt << std::endl;
        }
    }

    /*
     * From here clean refacted methods
     */

    cSetTargetMap cAppli_CheckBoardTargetRefine::computeTargets2WorldMappings()
    {
        std::map<std::string, std::vector<cPt3dr>> aMTargetsWorldBasePoints = computeTargetsWorldBasePoints();
        //then adjust
        cSetTargetMap aTarget2WorldMappings(mPhProj.DPGndPt3D().DirOut());

        for (const auto& aTgtWrldBsePts:aMTargetsWorldBasePoints)
        {
            const std::string& aTgtCode = aTgtWrldBsePts.first;
            tREAL8 aRes;
            cSimilitud3D<tREAL8> aTgt2WrldMap;
            aTgt2WrldMap = aTgt2WrldMap.StdGlobEstimate(get3DBasePoints(), aTgtWrldBsePts.second,
                                                        &aRes, nullptr, cParamCtrlOpt::Default());
            aTarget2WorldMappings.AddTargetMap(cTargetMap(aTgtCode, aTgt2WrldMap));
            StdOut() << "Input pts: " << get2DBasePoints() << " Output pts: " << aTgtWrldBsePts.second << std::endl;
            StdOut() << "3D Simil. Adjust. Res. " << aTgtCode << " --> " << aRes << std::endl;
        }

        return aTarget2WorldMappings;
    }

    std::map<std::string, std::vector<cPt3dr>> cAppli_CheckBoardTargetRefine::computeTargetsWorldBasePoints()
    {
        std::map<std::string, std::vector<std::vector<tSeg3dr>>> aMTargetsBaseBundles;

        for (const auto& aCam:mVCams)
        {
            cSetMesPtOf1Im aImMeasures = mPhProj.LoadMeasureIm(aCam->NameImage());
            std::vector<cSaveExtrEllipe> aVEllipsesExtrinsics;
            ReadFromFile(aVEllipsesExtrinsics, cSaveExtrEllipe::NameFile(mPhProj, aImMeasures, true));

            for (const auto& aEllipseExtrinsics:aVEllipsesExtrinsics)
            {
                const std::string& aTargetCode = aEllipseExtrinsics.mNameCode;

                std::vector<tSeg3dr> aVBaseBundles = {};
                for (const auto& aBasePt:get2DBasePoints())
                {
                    cPt2dr aTransformedBasePt = aEllipseExtrinsics.mAffIm2Ref.Inverse(aBasePt);
                    tSeg3dr aBundle = aCam->Image2Bundle(aTransformedBasePt);
                    aVBaseBundles.push_back(aBundle);

                    if (!aMTargetsBaseBundles.count(aTargetCode)) aMTargetsBaseBundles[aTargetCode] = {{},{},{},{}};
                }

                for (decltype(get2DBasePoints().size()) ix = 0; ix < get2DBasePoints().size(); ++ix)
                {
                    aMTargetsBaseBundles[aTargetCode][ix].push_back(aVBaseBundles[ix]);
                }
            }
        }

        std::map<std::string, std::vector<cPt3dr>> aMTargetsWorldBasePoints;

        for (const auto& aTargetCode:mMeasuredTargets.ListOfNames())
        {
            if (!aMTargetsBaseBundles.count(aTargetCode)) continue;
            bool isFirst = true;
            for (const auto& aVBaseBundles:aMTargetsBaseBundles[aTargetCode])
            {
                if (aVBaseBundles.size()<=2) continue;

                cPt3dr aInterBasePt;
                aInterBasePt = BundleInters(aVBaseBundles);

                if (isFirst)
                {
                    aMTargetsWorldBasePoints[aTargetCode] = {aInterBasePt};
                    isFirst = false;
                    continue;
                }
                aMTargetsWorldBasePoints[aTargetCode].push_back(aInterBasePt);
            }
        }

        return aMTargetsWorldBasePoints;
    }

    cPt3dr cAppli_CheckBoardTargetRefine::target2World(const std::string& aTargetCode, const cPt2dr& aTargetPt)
    {
        cPt3dr aTgt0Pt(aTargetPt.x(), aTargetPt.y(), 0);
        cPt3dr aWrldPt;
        auto aTgt2WrldMap = mTargets2WorldMappings.GetMapOfCode(aTargetCode);
        return aTgt2WrldMap->mMap.DiffInOut(aTgt0Pt, aWrldPt);
    }

    std::vector<cPt3dr> cAppli_CheckBoardTargetRefine::target2World(const std::string& aTargetCode, std::vector<cPt2dr>& aVWorldPt)
    {
        std::vector<cPt3dr> aVRes;
        for (auto aPt:aVWorldPt)
        {
            aVRes.push_back(target2World(aTargetCode, aPt));
        }
        return aVRes;
    }

    cPt2dr cAppli_CheckBoardTargetRefine::world2Target(const std::string& aTargetCode, const cPt3dr& aWorldPt)
    {
        auto aTgt2WrldMap = mTargets2WorldMappings.GetMapOfCode(aTargetCode);
        cPt3dr aTgt0Pt = aTgt2WrldMap->mMap.Inverse(aWorldPt);
        return cPt2dr(aTgt0Pt.x(), aTgt0Pt.y());
    }

    std::vector<cPt2dr> cAppli_CheckBoardTargetRefine::get2DBasePoints()
    {
        return {cPt2dr(0,0), cPt2dr(mRes-1,0),
                cPt2dr(0,mRes-1),cPt2dr(mRes-1,mRes-1)};
    }

    std::vector<cPt3dr> cAppli_CheckBoardTargetRefine::get3DBasePoints()
    {
        return {cPt3dr(0,0,0), cPt3dr(mRes-1,0,0),
                cPt3dr(0,mRes-1,0),cPt3dr(mRes-1,mRes-1,0)};
    }

    /*
     * From here ~ temp. methods to refact/delete (should begin by doSmthg.)
     */

    std::vector<cPt3dr> cAppli_CheckBoardTargetRefine::get3DTargetCorners(std::string& aCode)
    {
        std::vector<cPt3dr> aRes;
        auto aVBasePts = get2DBasePoints();
        for (const auto& aTgtPt:aVBasePts)
        {
            auto aWrlPt = target2World(aCode, aTgtPt);
            aRes.push_back(aWrlPt);
        }
        return aRes;
    }

    cAffin2D<tREAL8> cAppli_CheckBoardTargetRefine::getLocalAff2D(cPt2dr aRPix, tREAL8 aDelta)
    {
        cPt2dr x1 = aRPix + cPt2dr(aDelta, 0);
        cPt2dr y1 = aRPix + cPt2dr(0, aDelta);

        //base point
        cPt3dr aWrldPix = mCurrCam->Image2PlaneInter(*mCurrTgtPlane, aRPix);
        cPt2dr aTgtPix = world2Target(mCurrTgtCode, aWrldPix);

        //base point + delta
        cPt3dr aWrldX1 = mCurrCam->Image2PlaneInter(*mCurrTgtPlane, x1);
        cPt2dr aTgtX1 = world2Target(mCurrTgtCode, aWrldX1);
        cPt3dr aWrldY1 = mCurrCam->Image2PlaneInter(*mCurrTgtPlane, y1);
        cPt2dr aTgtY1 = world2Target(mCurrTgtCode, aWrldY1);

        cPt2dr aVx = 1/aDelta * (aTgtX1 - aTgtPix);
        cPt2dr aVy = 1/aDelta * (aTgtY1 - aTgtPix);
        cPt2dr aTr = aTgtPix - aRPix.x() * aVx - aRPix.y() * aVy;

        return cAff2D_r(aTr, aVx, aVy);
    }

    /*
     * From here ~ temp. methods to refact/delete (should begin by doSmthg.)
     */

    cBox2dr cAppli_CheckBoardTargetRefine::doTargetRefine(bool& isOk)
    {
        cBox2dr aExtent = doCurrImageTgtExtent(isOk);
        //

        if (isOk)
        {
            doTargetSample(aExtent);

        }
        return cBox2dr::Empty();
    }

    cBox2dr cAppli_CheckBoardTargetRefine::doCurrImageTgtExtent(bool& isOk)
    {
        auto aVBasePts = get2DBasePoints();
        auto aVWrldBasePts = target2World(mCurrTgtCode, aVBasePts);
        std::vector<tREAL8> aVXImBasePts;
        std::vector<tREAL8> aVYImBasePts;

        for (const auto& aWrldPt:aVWrldBasePts) {
            auto aImPt = mCurrCam->Ground2Image(aWrldPt);
            aVXImBasePts.push_back(aImPt[0]);
            aVYImBasePts.push_back(aImPt[1]);
        }

        auto minMaxVX = minmax_element(aVXImBasePts.begin(), aVXImBasePts.end());
        auto minMaxVY = minmax_element(aVYImBasePts.begin(), aVYImBasePts.end());

        cPt2dr aImUL = cPt2dr(*minMaxVX.first, *minMaxVY.first);
        cPt2dr aImLR = cPt2dr(*minMaxVX.second, *minMaxVY.second);

        if (mCurrCam->IsVisibleOnImFrame(aImUL) && mCurrCam->IsVisibleOnImFrame(aImLR))
        {
            isOk = true;
            return cBox2dr(aImUL,aImLR);
        }
        isOk = false;
        return cBox2dr::Empty();
   }

    cIm2D<tU_INT1> cAppli_CheckBoardTargetRefine::doTargetSample(cBox2dr& aExtent)
   {
        cPt2dr& aImOffSet = aExtent.P0ByRef();
        tIm aIm(ToI(aExtent.Sz()));
        auto& aDim = aIm.DIm();
        auto aVTgt3DPts = get3DTargetCorners(mCurrTgtCode);
        cPlane3D aTgtPlane = cPlane3D::From3Point(aVTgt3DPts[0], aVTgt3DPts[1], aVTgt3DPts[2]);
        mCurrTgtPlane = &aTgtPlane;
        auto aCode = mFullSpec->EncodingFromName(mCurrTgtCode);
        tIm aImTgt = mFullSpec->OneImTarget(*aCode);
        tDIm& aDImTgt = aImTgt.DIm();
        auto aRect2 = cRect2(cPt2di(0,0), ToI(aExtent.Sz()));

        for (const auto & aPix : aRect2)
        {
            cPt3dr aWrldPix = mCurrCam->Image2PlaneInter(aTgtPlane, ToR(aPix)+aImOffSet);
            cPt2dr aTgtPix = world2Target(mCurrTgtCode, aWrldPix);

            if (aDImTgt.Inside(ToI(aTgtPix)))
            {
                auto aLocalIm2Tgt = getLocalAff2D(ToR(aPix)+aImOffSet, 0.1);

                //stolen from cSimulTarget.cpp
                //StdOut() << "Begin gaussbicub" << std::endl;
                cRessampleWeigth aRW = cRessampleWeigth::GaussBiCub(ToR(aPix)+aImOffSet,aLocalIm2Tgt,1.0);
                //StdOut() << "Ok gaussbicub" << std::endl;
                const std::vector<cPt2di>  & aVPts = aRW.mVPts;
                if (!aVPts.empty())
                {
                    double aVal=0;
                    for (int aK=0; aK<int(aVPts.size()) ; aK++)
                    {
                        if (aDImTgt.Inside(aVPts[aK]))
                        {
                            double aW = aRW.mVWeight[aK];
                            aVal += aW * aDImTgt.GetV(aVPts[aK]);
                        }
                    }
                    aDim.SetV(aPix,aVal);
                }
            }
            else
            {
                aDim.SetV(aPix,0);
            }
        }
        aDim.ToFile(NameVisu(mCurrCam->NameImage(), "Tgt" + mCurrTgtCode));
        //SaveInFile(aIm, );
        return aIm;
    }

    void cAppli_CheckBoardTargetRefine::doReproj(const std::string & aCode)
    {
        for (auto aCam:mMImExtendedSetOfMes2D)
        {
            int ix=0;
            cPt2dr aRes;
            for (const auto & aMes:aCam.second.Measures())
            {
                if (aMes.mNamePt==aCode)
                {
                    aRes+=(aMes.mPt-aCam.first->Ground2Image(mExtendedSetOfMes3D.GetMeasureOfNamePt(aCode+TargetLoc[ix]).mPt));
                    ++ix;
                }
            }
            if (ix!=0) StdOut()<<aCode<< " : "<<Norm2(aRes)/ix<<" px res."<<std::endl;
        }
    }

    std::vector<cPt3dr> cAppli_CheckBoardTargetRefine::getVMotifMes3D(std::list<std::string> lNamesPts)
    {
        std::vector<cPt3dr> aVMes3D;
        for (const auto & aNamePt:lNamesPts)
        {
            std::string loc = aNamePt.substr(aNamePt.size()-2,aNamePt.size()-1);
            int ix = find(TargetLoc.begin(),TargetLoc.end(),loc) - TargetLoc.begin();
            aVMes3D.push_back(cPt3dr(mInitTargetPts[ix].x(),
                                     mInitTargetPts[ix].y(),
                                     0));
        }
        return aVMes3D;
    }

    void cAppli_CheckBoardTargetRefine::doBundle(const cSensorCamPC * aCam)
    {
        cSetMesGnd3D aExtendedSetOfBundles;

        for (auto const & aMes:mMImExtendedSetOfMes2D[aCam].Measures())
        {
            tSeg3dr aBundle = aCam->Image2Bundle(aMes.mPt);
            aExtendedSetOfBundles.AddMeasure3D(cMes1Gnd3D(aBundle.V12(), aMes.mNamePt));
        }
        mMImSetOfBundles[aCam] = aExtendedSetOfBundles;
    }

    void cAppli_CheckBoardTargetRefine::doAffInv(const cSensorCamPC * aCam)
    {
        //load 2d Mes im attributes
        cSetMesPtOf1Im aSetOfMes2D = mPhProj.LoadMeasureIm(aCam->NameImage());
        cSetMesPtOf1Im aExtendedSetOfMes2D;
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

        mMImExtendedSetOfMes2D[aCam] = aExtendedSetOfMes2D;
    };

    void cAppli_CheckBoardTargetRefine::resetInitTargetPts()
    {
        mInitTargetPts={cPt2dr(0,0), cPt2dr(mRes-1,0),
                          //cPt2dr(((mRes-1)/2), (mRes-1)/2),
                          cPt2dr(0,mRes-1),cPt2dr(mRes-1,mRes-1)};
    }


/*
    int cAppli_CheckBoardTargetRefine::doPredict(const std::string & aImName)
    {
        StdOut()<<"doPredict :";
    //·1->loads curr. img. input/output
        auto aFoundTargets = mPhProj.LoadMeasureImFromFolder(mPhProj.DPGndPt2D().DirIn(), aImName);
    //·2->filter mes2D that have not been detected
        auto aVMissingTargetsNames = MissingTargetsNames(aFoundTargets);
    //·3->loads filtered 3D points & compute 2D projection (=prediction)
        if (aVMissingTargetsNames.empty())
        {
            StdOut() << "no missing targets" << std::endl;
            return 0;
        }
        StdOut()<<aVMissingTargetsNames.size()<<" targets predicted"<<std::endl;
        cSetMesPtOf1Im aSetOfPredictions = MissingTargetsPredict(aImName, aVMissingTargetsNames);
        mPhProj.SaveMeasureIm(aSetOfPredictions);
    //·4->for each target find affinity parameters & compute correspondances

        return 0;
    }
*/
    /*
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
    */
    /*
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
    */
    /* Visualisation methods */

    int cAppli_CheckBoardTargetRefine::doVisu(const cSensorCamPC * aCam)
    {
        cRGBImage aIm = cRGBImage::FromFile(aCam->NameImage());
        mCurrVisuIm = & aIm;
        if (mPhProj.HasMeasureImFolder(mPhProj.DPGndPt2D().DirOut(), aCam->NameImage()))
        {
            cSetMesPtOf1Im aSetOfNewMes2D = mPhProj.LoadMeasureImFromFolder(mPhProj.DPGndPt2D().DirOut(), aCam->NameImage());
            drawTarget(aSetOfNewMes2D, false);//draw initial targets
        }
        if (mPhProj.HasMeasureImFolder(mPhProj.DPGndPt2D().DirIn(), aCam->NameImage()))
        {
            cSetMesPtOf1Im aSetOfInitMes2D = mPhProj.LoadMeasureImFromFolder(mPhProj.DPGndPt2D().DirIn(), aCam->NameImage());
            drawTarget(aSetOfInitMes2D, true);//draw new targets
        }
        for (const auto & aMes:mMImExtendedSetOfMes2D[aCam].Measures())
        {
            //StdOut() << "drawing mes:" << aMes.mPt << "--" << aMes.mNamePt<<std::endl;
            mCurrVisuIm->DrawCircle(cRGBImage::Cyan, aMes.mPt, 1);
        }
        mCurrVisuIm->ToJpgFileDeZoom(NameVisu(aCam->NameImage(), "Ref"), 1, {"QUALITY=90"});

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
