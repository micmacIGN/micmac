//#include "MMVII_Sensor.h"
#include "CodedTarget.h"
#include "MMVII_ImageMorphoMath.h"
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
            /*
             * naming conv. : void return methode : JeanMichel
             *                methode actually returning smthg : jeanMichel
             *                */
            //--spec. methods
            int doVisu(const cSensorCamPC * aCam);
            int doPredict(const std::string & aImName);
            void doAffInv(const cSensorCamPC * aCam);
            void doReproj(const std::string & aCode);
            void TargetRefine(bool& isOk);
            void CurrTgtExtent(bool& isOk);
            std::set<std::string> getMissingTgtNames(cSensorCamPC& aCam);
            cSetMesPtOf1Im MissingTargetsPredict(const std::string & aImName, const std::vector<std::string> & aVMissingTargetsNames);
            void DrawTarget(cSetMesPtOf1Im & aSetOfMes2D, bool isInit);
            void ResetInitTargetPts();
            void doBundle(const cSensorCamPC * aCam);
            std::vector<cPt3dr> getVMotifMes3D(std::list<std::string> lNamesPts);
            cSimilitud3D<cPt3dr> doSimil3D(std::vector<cPt3dr> &aVPtsIn, std::vector<cPt3dr> &aVPtsOut);
            //std::map<std::string, cSimilitud3D<tREAL8>> computeTargets2WorldMappings();
            cSetTargetMap computeTargets2WorldMappings();
            std::map<std::string, std::vector<cPt3dr>> computeTargetsWorldBasePoints();
            cPt2dr world2Target(const std::string& aTargetCode, const cPt3dr& aWorldPt);
            std::vector<cPt3dr> target2World(const std::string& aTargetCode, const std::vector<cPt2dr>& aVWorldPt);
            cPt3dr target2World(const std::string& aTargetCode, const cPt2dr& aTargetPt);
            std::vector<cPt2dr> get2DBasePoints();
            std::vector<cPt3dr> get3DBasePoints();
            std::vector<cPt3dr> get3DTargetCorners(std::string& aCode);
            void doPseudoBench();
            void TargetSample(bool& isOk);
            void LSCorresp(bool& isOk);
            cAffin2D<tREAL8> getLocalAff2D(cPt2dr aPix, tREAL8 delta);
            //--mandatory
            cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
            cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
            std::vector<cPt2di> getRansacPts();
            void doComputeCoOccIm(tDIm* aDImIn);
            void ComputeBasePtsRes();
            void RansacCorresp(bool& isTarget);
            void RansacTransFunc(cPt2dr& theSol, std::vector<cPt2di> aVBitCenters);
            cStdStatRes wL1Score(cPt2dr& aSol, cDataIm2D<tREAL8>* aDWStdIm);
            cPhotogrammetricProject mPhProj;
            std::string mNameSpecif;
            std::vector<cSensorCamPC*> mVCams;
            std::string mSpecImIn;
            tREAL8 mWL1Limit;
            int mRes;
            bool mShow; //show details
            bool mVisu; //visualisation
            cSetTargetMap mTargets2WorldMappings;
            std::set <std::string> mIntersectedCodes;
            cRGBImage * mCurrVisuIm;
            cSensorCamPC* mCurrCam;
            std::string mCurrTgtCode;
            cPlane3D* mCurrTgtPlane;
            tIm mTgtMasq;
            tDIm* mDTgtMasq;
            cSetMesGnd3D mMeasuredTargets;
            std::vector<cPt2dr> mInitTargetPts;
            cSetMesPtOf1Im mExtendedSetOfMes2D;
            cSetMesGnd3D mExtendedSetOfBundles;
            cSetMesGnd3D mExtendedSetOfMes3D;
            std::map<const cSensorCamPC *,cSetMesPtOf1Im> mMImExtendedSetOfMes2D;
            //cSetTargetSim3D mSetTargetSim3D;
            std::map<const cSensorCamPC *, cSetMesGnd3D> mMImSetOfBundles;
            std::unique_ptr<cFullSpecifTarget> mFullSpec;
            cBox2dr mCurrTgtExtent;
            tElem mMasqVal;
            tIm mCurrRef;
            tDIm* mDCurrRef;
            tIm mCurrPred;
            tDIm* mDCurrPred;
            tDIm* mDCurrRefLabel;
            tIm mCurrTrue;
            tDIm* mDCurrTrue;
            tIm mCoOccMat;
            tDIm* mDCoOccMat;
            tREAL8 mCurrWL1Med;
            cIm2D<tREAL8> mWStdDeltaIm;
            cDataIm2D<tREAL8>* mDWStdDeltaIm;
            std::map<std::string, std::vector<cPt3dr>> mMTargetsWorldBasePoints;
            std::map<cSensorCamPC*,std::map<std::string,std::vector<cPt2dr>>> mMCamTgtBasePts;


            //----stolen to cCheckBoardTargetExtract
            std::string NameVisu(const std::string & aDestIm, const std::string & aPref,const std::string aPost="");
            //----





    };

    cCollecSpecArg2007 & cAppli_CheckBoardTargetRefine::ArgObl(cCollecSpecArg2007 & anArgObl)
    {
        return anArgObl
            << Arg2007(mSpecImIn, "Pattern/file of images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
            << Arg2007(mNameSpecif,"Xml/Json name for bit encoding struct",{{eTA2007::XmlOfTopTag,cFullSpecifTarget::TheMainTag}})
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
        mWL1Limit (30.0),
        mRes (600),
        mTargets2WorldMappings (""),
        mCurrVisuIm (nullptr),
        mTgtMasq (cPt2di(1,1)),
        mFullSpec (nullptr),
        mCurrTgtExtent (cBox2dr::Empty()),
        mMasqVal (255),
        mCurrRef (cPt2di(1,1)),
        mCurrPred (cPt2di(1,1)),
        mCurrTrue (cPt2di(1,1)),
        mCoOccMat (cPt2di(1,1)),
        mWStdDeltaIm (cPt2di(1,1))

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

        mPhProj.FinishInit();
        mFullSpec.reset(cFullSpecifTarget::CreateFromFile(mNameSpecif));
        std::vector<std::string> aVIm = VectMainSet(0);//gets the first set

        //**** target metadata enrichment

        for (const auto& aImName:aVIm)
        {
            cSensorCamPC* aCam = mPhProj.ReadCamPC(aImName, true);
            mVCams.push_back(aCam);
        }

        mMeasuredTargets = mPhProj.LoadGCP3DFromFolder(mPhProj.DPGndPt3D().DirIn());

        //compute target to ground mappings as 3D similarities
        mTargets2WorldMappings = computeTargets2WorldMappings();

        //save computed mappings as target attributes
        SaveInFile(mTargets2WorldMappings, cSetTargetMap::NameFile(mPhProj, mTargets2WorldMappings.Name(), false));

        //****

        //*** real refinement starts from here

        for (const auto& aCam:mVCams)
        {
            mCurrCam = aCam;
            auto aCodes = mTargets2WorldMappings.ListOfCodes();

            StdOut() << aCam->NameImage() << std::endl;
            auto aSMissingNames = getMissingTgtNames(*aCam);
            //StdOut() << "missing targets : " << aSMissingNames<<"\n";

            std::vector<std::string> aVRecovNames = {};

            for (const auto& aTgt:aCodes)
            {
                mCurrTgtCode = aTgt;
                bool isOk;
                TargetRefine(isOk);
                if (isOk && (aSMissingNames.find(aTgt)!=aSMissingNames.end()))
                {
                    aVRecovNames.push_back(aTgt);
                }
            }
            StdOut() << "recoverable targets:" << aVRecovNames<<std::endl;
            /*if(isOk)
                {
                    aIm.SetRGBBorderRectWithAlpha(ToI(aBox.Middle()),Norm2(aBox.Sz())/2,10,cRGBImage::Orange,0.1);//0.1 means final opacity = 1-0.1
                    aIm.DrawCircle(cRGBImage::Green, aBox.P0(), 4);
                    aIm.DrawCircle(cRGBImage::Red, aBox.P1(), 4);
                }*/
            //aIm.ToJpgFileDeZoom(NameVisu(mCurrCam->NameImage(), "Ref"), 1, {"QUALITY=90"});


        }

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
        //for each target compute ground coordinates of arbitrary target base points (typically corners)
        mMTargetsWorldBasePoints = computeTargetsWorldBasePoints();

        //here comes the big check
        if (mShow)
        {
            ComputeBasePtsRes();
        }

        cSetTargetMap aTarget2WorldMappings(mPhProj.DPGndPt3D().DirOut());//set of target 2 ground mappings (useful for serialisation)

        for (const auto& aTgtWrldBsePts:mMTargetsWorldBasePoints)
        {
            const std::string& aTgtCode = aTgtWrldBsePts.first;

            tREAL8 aRes;
            cSimilitud3D<tREAL8> aTgt2WrldMap;
            aTgt2WrldMap = aTgt2WrldMap.StdGlobEstimate(get3DBasePoints(), aTgtWrldBsePts.second,
                                                        &aRes, nullptr, cParamCtrlOpt::Default());
            aTarget2WorldMappings.AddTargetMap(cTargetMap(aTgtCode, aTgt2WrldMap));

            if (mShow) {StdOut() << "nb. base bundle for tg. " << aTgtCode << " --> " << aTgtWrldBsePts.second.size()
                         <<std::endl;
            StdOut() << "3D Simil adj.: " << get2DBasePoints() << " --> " << aTgtWrldBsePts.second << std::endl;
            StdOut() << " : " << aRes << std::endl;}
        }

        return aTarget2WorldMappings;
    }

    std::map<std::string, std::vector<cPt3dr>> cAppli_CheckBoardTargetRefine::computeTargetsWorldBasePoints()
    {
        //this will associate a target code to a vector of vector of 3d "base" bundles
        std::map<std::string, std::vector<std::vector<tSeg3dr>>> aMTargetsBaseBundles;

        for (const auto& aCam:mVCams)
        {
            mMCamTgtBasePts[aCam] = {};//init. cam. map.
            //load im. measurements/2d affinity obtained from previous extraction
            cSetMesPtOf1Im aImMeasures = mPhProj.LoadMeasureIm(aCam->NameImage());
            std::vector<cSaveExtrEllipe> aVEllipsesExtrinsics;
            ReadFromFile(aVEllipsesExtrinsics, cSaveExtrEllipe::NameFile(mPhProj, aImMeasures, true));

            for (const auto& aEllipseExtrinsics:aVEllipsesExtrinsics)//= for each extracted target
            {
                const std::string& aTargetCode = aEllipseExtrinsics.mNameCode;
                std::vector<tSeg3dr> aVBaseBundles = {};
                std::vector<cPt2dr> aVCamBasePts = {};//to be use for residual computation

                for (const auto& aBasePt:get2DBasePoints())//get base points (usually corners)
                {
                    //based on affinity computed at the extraction step, predict base pt. coords. in current view
                    cPt2dr aTransformedBasePt = aEllipseExtrinsics.mAffIm2Ref.Inverse(aBasePt);
                    aVCamBasePts.push_back(aTransformedBasePt);
                    tSeg3dr aBundle = aCam->Image2Bundle(aTransformedBasePt);//compute base bundle
                    aVBaseBundles.push_back(aBundle);//add the bundle to the vector of base bundles

                    //initialize the key,value pair if it's the first vector for current target
                    if (!aMTargetsBaseBundles.count(aTargetCode)) aMTargetsBaseBundles[aTargetCode] = {{},{},{},{}};
                }

                mMCamTgtBasePts[aCam][aTargetCode] = aVCamBasePts;

                for (decltype(get2DBasePoints().size()) ix = 0; ix < get2DBasePoints().size(); ++ix)//there is one base bundles vector
                {                                                                                   //for each target base point
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

    void cAppli_CheckBoardTargetRefine::ComputeBasePtsRes()
    {
        for (const auto& aCam : mMCamTgtBasePts)
        {
            StdOut() << "Cam. residuals: " << aCam.first->NameImage() << "\n";
            for (const auto& aTgtCode : aCam.second)
            {
                tREAL8 aSumRes = 0;
                int ix = 0;
                auto search = mMTargetsWorldBasePoints.find(aTgtCode.first);
                if (search != mMTargetsWorldBasePoints.end())
                {
                    for (const auto& aWorldBsePt : mMTargetsWorldBasePoints.at(aTgtCode.first))
                    {
                        aSumRes += Norm2(aCam.first->Ground2Image(aWorldBsePt) - aTgtCode.second[ix]);
                        ++ix;
                    }
                    (void) aTgtCode;
                    StdOut() << "AvgRes. [px] on TGT n°" << aTgtCode.first << ": " << aSumRes/ix << std::endl;
                }
            }
        }
    }

    void cAppli_CheckBoardTargetRefine::TargetRefine(bool& isOk)
    {
        CurrTgtExtent(isOk);//computes current target extent
        if (isOk && mCurrCam->NameImage() == "K127_202409211622-00-cam-22348125-38-66255657607881-23.tiff")
        {
            TargetSample(isOk);
            if(isOk){RansacCorresp(isOk);}//first mapping
            if(isOk){LSCorresp(isOk);}//refine mapping
        }
    }

    void cAppli_CheckBoardTargetRefine::CurrTgtExtent(bool& isOk)
    {
        //StdOut() << "BEGIN TARGET EXTENT" << std::endl;

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
            mCurrTgtExtent = cBox2dr(aImUL,aImLR);
        }
        else
        {
            isOk = false;
            mCurrTgtExtent = cBox2dr::Empty();
        }
   }

    void cAppli_CheckBoardTargetRefine::TargetSample(bool& isOk)
   {
        //StdOut() << "BEGIN TARGET SAMPLE" << std::endl;
        cPt2dr& aImOffSet = mCurrTgtExtent.P0ByRef();//save the offset for camera back projection
        mCurrPred = tIm(ToI(mCurrTgtExtent.Sz()));
        mDCurrPred = &mCurrPred.DIm();

        auto aVTgt3DPts = get3DTargetCorners(mCurrTgtCode);//tgt. base pts. w.r.t. ground frame
        cPlane3D aTgtPlane = cPlane3D::From3Point(aVTgt3DPts[0], aVTgt3DPts[1], aVTgt3DPts[2]);//gnd. tgt. plane
        mCurrTgtPlane = &aTgtPlane;

        auto aCode = mFullSpec->EncodingFromName(mCurrTgtCode);
        mCurrRef = mFullSpec->OneImTarget(*aCode);//from current target encoding
        mDCurrRef = &mCurrRef.DIm();
        auto aRect2 = cRect2(cPt2di(0,0), ToI(mCurrTgtExtent.Sz()));//rect. to fill from tgt.

        mTgtMasq = tIm(ToI(mCurrTgtExtent.Sz()));
        mDTgtMasq = &mTgtMasq.DIm();

        for (const auto & aPix : aRect2)
        {
            cPt3dr aWrldPix = mCurrCam->Image2PlaneInter(aTgtPlane, ToR(aPix)+aImOffSet);//inter. of gnd. tgt. plane & camera pix. (knowing cam. pose)
            cPt2dr aTgtPix = world2Target(mCurrTgtCode, aWrldPix);// pix. coordinates in wrt canonic tgt. frame

            if (mDCurrRef->Inside(ToI(aTgtPix)))
            {
                auto aLocalIm2Tgt = getLocalAff2D(ToR(aPix)+aImOffSet, 0.1);

                //stolen from cSimulTarget.cpp: anti-aliasing algorithm
                cRessampleWeigth aRW = cRessampleWeigth::GaussBiCub(ToR(aPix)+aImOffSet,aLocalIm2Tgt,2);
                const std::vector<cPt2di>  & aVPts = aRW.mVPts;
                if (!aVPts.empty())
                {
                    double aVal=0;
                    for (int aK=0; aK<int(aVPts.size()) ; aK++)
                    {
                        if (mDCurrRef->Inside(aVPts[aK]))
                        {
                            double aW = aRW.mVWeight[aK];
                            aVal += aW * mDCurrRef->GetV(aVPts[aK]);
                        }
                    }
                    mDCurrPred->SetV(aPix,aVal);
                }
            }
            else
            {
                mDCurrPred->SetV(aPix,0);
                mDTgtMasq->SetV(aPix,mMasqVal);
            }
        }


            mCurrTrue = tIm(ToI(mCurrTgtExtent.Sz()));//prepare true target extract
            mDCurrTrue = &mCurrTrue.DIm();
            auto aCurrIm = tIm::FromFile(mCurrCam->NameImage());//loads curr. img.
            auto* aDCurrIm = &aCurrIm.DIm();
            mDCurrTrue->CropIn(ToI(aImOffSet), *aDCurrIm);//fills true tgt.

            if (mVisu)
        {
            mDCurrTrue->ToFile(NameVisu(mCurrCam->NameImage(),"True" + mCurrTgtCode));
            mDCurrPred->ToFile(NameVisu(mCurrCam->NameImage(), "Tgt" + mCurrTgtCode));
            mDTgtMasq->ToFile(NameVisu(mCurrCam->NameImage(), "Masq" + mCurrTgtCode));
        }
    }

    void cAppli_CheckBoardTargetRefine::RansacCorresp(bool& isTarget)
    {
        //1-look if it exists a "good" transfunc on bits centers if not leave.
        std::vector<cPt3dr> aVWorldBitCenters = target2World(mCurrTgtCode, mFullSpec->BitsCenters());
        std::vector<cPt2di> aVTrueBitCenters ={};
        for (const auto& aWrldBit:aVWorldBitCenters)
        {
            aVTrueBitCenters.push_back(ToI(mCurrCam->Ground2Image(aWrldBit)-mCurrTgtExtent.P0ByRef()));
        }//get currbits centers
        cPt2dr theSol{0,0};//assuming affine mapping I'(x',y')=aI(x,y)+b
        RansacTransFunc(theSol, aVTrueBitCenters);
        if (theSol == cPt2dr{0,0}){isTarget=false; return;}

        cIm2D<tREAL8> aWStdIm(mDCurrPred->Sz());
        cDataIm2D<tREAL8>* aDWStdIm = &aWStdIm.DIm();
        cRect2 aRectInt = mDCurrPred->Dilate(-1);
        aDWStdIm->InitCste(0);

        for (const auto & aPix : aRectInt)
        {
            if (mDTgtMasq->GetV(aPix)==mMasqVal) continue;

            cComputeStdDev<tREAL8>  aCDev;
            for (int aK=0 ; aK<8 ; aK++)
            {
                aCDev.Add(mDCurrPred->GetV(aPix+FreemanV8[aK]));
            }
            aDWStdIm->SetVTrunc(aPix,aCDev.StdDev());
        }

        cStdStatRes aScore = wL1Score(theSol, aDWStdIm);
        if (aScore.Avg() > mWL1Limit){isTarget=false; return;}

        isTarget=true;

        //set residual median value
        std::vector<tREAL8> aVSortedRes = aScore.VRes();
        std::sort(aVSortedRes.begin(), aVSortedRes.end());
        mCurrWL1Med = aVSortedRes[aVSortedRes.size()/2];
    }

    void cAppli_CheckBoardTargetRefine::LSCorresp(bool& isOk)
    {
        //1-create LS struct
        cLeasSqtAA<tREAL8> aLSSystem(10);
        cDiffInterpolator1D * aInterpol = cDiffInterpolator1D::AllocFromNames({"Linear"});
        //2-iterate on "good" pixels and add corresponding observations to the struct
        int ix=0;
        for (const auto& aPix : *mDCurrPred)
        {
            if (mDTgtMasq->GetV(aPix)!=mMasqVal && mDWStdDeltaIm->GetV(aPix)<=mCurrWL1Med)
            {
                ++ix;
                (void) ix;
                auto [aValue,aGrad] = mDCurrPred->GetValueAndGradInterpol(*aInterpol,ToR(aPix));
                tREAL8 aI=(tREAL8)mDCurrPred->GetV(aPix);
                tREAL8 ai=(tREAL8)mDCurrTrue->GetV(aPix);
                auto& aPartix=aGrad.x();//interpol
                auto& aPartiy=aGrad.y();//interpol
                cDenseVect<tREAL8> aVEqObs({aI*aPix.x(),
                                            aI*aPix.y(),
                                            aI,
                                            aI,
                                            -aPartix*aPix.x(),
                                            -aPartix*aPix.y(),
                                            -aPartix,
                                            -aPartiy*aPix.x(),
                                            -aPartiy*aPix.y(),
                                            -aPartiy});
                aLSSystem.PublicAddObservation(1, aVEqObs, ai);
            }
        }
        //3-solve the system
        StdOut() << "BEGIN LSQUARE SOLVING" << std::endl;
        auto aVObs = aLSSystem.V_tAA();
        auto aSol = aLSSystem.PublicSolve();
        StdOut() << "LS Solution : " << aSol;
        isOk = true;
    }

    /* From here ~ utils methods which called above */

    cStdStatRes cAppli_CheckBoardTargetRefine::wL1Score(cPt2dr& aSol, cDataIm2D<tREAL8>* aDWStdIm)
    {
        mWStdDeltaIm = aDWStdIm->Sz();
        mDWStdDeltaIm = &mWStdDeltaIm.DIm();
        mDWStdDeltaIm->InitCste(0);

        tREAL8 aSig0 = 1;
        cStdStatRes aStat;

        for (auto aPix:*mDCurrTrue)
        {
            if(mDTgtMasq->GetV(aPix) != mMasqVal)
            {
                tREAL8 aDelta = abs(mDCurrPred->GetV(aPix) - (mDCurrTrue->GetV(aPix)*aSol[0] + aSol[1]));
                tREAL8 aWVal = aDelta * (aSig0/(aDWStdIm->GetV(aPix)+aSig0));
                mDWStdDeltaIm->SetVTrunc(aPix,aDelta);
                aStat.Add(aWVal);
            }
        }

        if (mVisu)
        {
            mDWStdDeltaIm->ToFile(NameVisu(mCurrCam->NameImage(),"WStdDeltaIm"+mCurrTgtCode));
        }

        return aStat;
    }

    cPt3dr cAppli_CheckBoardTargetRefine::target2World(const std::string& aTargetCode, const cPt2dr& aTargetPt)
    {
        cPt3dr aTgt0Pt(aTargetPt.x(), aTargetPt.y(), 0);
        cPt3dr aWrldPt;
        auto aTgt2WrldMap = mTargets2WorldMappings.GetMapOfCode(aTargetCode);
        return aTgt2WrldMap->mMap.DiffInOut(aTgt0Pt, aWrldPt);
    }

    std::vector<cPt3dr> cAppli_CheckBoardTargetRefine::target2World(const std::string& aTargetCode, const std::vector<cPt2dr>& aVWorldPt)
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

        cPt2dr aVx = (aTgtX1 - aTgtPix)/aDelta;
        cPt2dr aVy = (aTgtY1 - aTgtPix)/aDelta;
        cPt2dr aTr = aTgtPix - aRPix.x() * aVx - aRPix.y() * aVy;

        return cAff2D_r(aTr, aVx, aVy);
    }

    void cAppli_CheckBoardTargetRefine::RansacTransFunc(cPt2dr& theSol, std::vector<cPt2di> aVBitCenters)
    {
        //StdOut() << "BEGIN RANSAC->"<<std::endl;
        int it = 200;//nb. iterations
        std::vector<cPt2di> aSet = {};
        std::vector<cPt2di> theSet = {};
        tREAL8 a1, a2;
        tREAL8 theScore = 100000000;
        std::vector<tREAL8> theSolPts ={};

        for (int ix=0;ix<it;++ix)
        {
            int aL1Score = 0;
            tU_INT1 bit1 = RandUnif_N(aVBitCenters.size()), bit2 = RandUnif_N(aVBitCenters.size());

            if (mDCurrRef->GetV(ToI(mFullSpec->BitsCenters()[bit1])) == mDCurrRef->GetV(ToI(mFullSpec->BitsCenters()[bit2])))
            {
                continue;
            }

            tPt2di p1 = aVBitCenters[bit1];
            tPt2di p2 = aVBitCenters[bit2];

            tREAL8 G1 = mDCurrTrue->GetV(p1), G2 = mDCurrTrue->GetV(p2);
            tREAL8 G3 = mDCurrPred->GetV(p1), G4 = mDCurrPred->GetV(p2);

            if(G1==G2) {continue;}

            a1 = (G3-G4)/(G1-G2);
            a2 = G3-a1*G1;

            for (const auto& aPix : * mDCurrPred)
            {
                if (mDTgtMasq->GetV(aPix)==mMasqVal) continue;
                tREAL8 val = a1*mDCurrTrue->GetV(aPix) + a2;
                tREAL8 delta = val - mDCurrPred->GetV(aPix);
                aL1Score += abs(delta);
            }

            if (theScore > aL1Score)
            {
                theScore = aL1Score;
                theSol = cPt2dr(a1,a2);
            }
        }
        //StdOut() << "OK RANSAC"<<std::endl;
    }

    std::set<std::string> cAppli_CheckBoardTargetRefine::getMissingTgtNames(cSensorCamPC& aCam)
    {
        std::set<std::string> aSRes;
        auto aVCTargets = mPhProj.LoadGCP3D();

        for (const auto & aTgt : aVCTargets.ListOfNames())//iterates on mes3d checks if pt. p exists in known mes2d
        {
            if (!mPhProj.LoadMeasureIm(aCam.NameImage()).NameHasMeasure(aTgt) && aCam.IsVisible(aVCTargets.GetMeasureOfNamePt(aTgt).mPt))
            {
                aSRes.insert(aTgt);
            }
        }
        return aSRes;
    }

    /*---
     * from here kind of garbage collector*/

/*
    void cAppli_CheckBoardTargetRefine::doComputeCoOccIm(tDIm* aDImIn)
    {
        auto aTMax = std::numeric_limits<tElem>::max()+1;
        mCoOccMat = tIm(cPt2di(aTMax,aTMax));
        mDCoOccMat = &mCoOccMat.DIm();
        mDCoOccMat->InitCste(0);

        std::vector<cPt2di> dir_offset = {cPt2di(0,1), cPt2di(1,0),cPt2di(-1,-1), cPt2di(1,1)};

        for (auto const& aPix : *aDImIn)
        {
            if(mDTgtMasq->GetV(aPix)== mMasqVal) continue;
            for (auto const& aOffSet : dir_offset)
            {
                if (aDImIn->Inside(aPix+aOffSet))
                {
                    tElem left=aDImIn->GetV(aPix), right=aDImIn->GetV(aPix+aOffSet);
                    tElem value = mDCoOccMat->GetV(cPt2di(left,right)) + 1;
                    mDCoOccMat->SetVTrunc(cPt2di(left,right), value);
                }
            }
        }
    }
*/
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
            ResetInitTargetPts();
            for (const auto & aPt:mInitTargetPts)
            {
                aExtendedSetOfMes2D.AddMeasure(cMesIm1Pt(aSavedEllipse.mAffIm2Ref.Inverse(aPt),
                                                         aSavedEllipse.mNameCode + "", 0.1));
            }
        }

        mMImExtendedSetOfMes2D[aCam] = aExtendedSetOfMes2D;
    };

    void cAppli_CheckBoardTargetRefine::ResetInitTargetPts()
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
            DrawTarget(aSetOfNewMes2D, false);//draw initial targets
        }
        if (mPhProj.HasMeasureImFolder(mPhProj.DPGndPt2D().DirIn(), aCam->NameImage()))
        {
            cSetMesPtOf1Im aSetOfInitMes2D = mPhProj.LoadMeasureImFromFolder(mPhProj.DPGndPt2D().DirIn(), aCam->NameImage());
            DrawTarget(aSetOfInitMes2D, true);//draw new targets
        }
        for (const auto & aMes:mMImExtendedSetOfMes2D[aCam].Measures())
        {
            //StdOut() << "drawing mes:" << aMes.mPt << "--" << aMes.mNamePt<<std::endl;
            mCurrVisuIm->DrawCircle(cRGBImage::Cyan, aMes.mPt, 1);
        }
        mCurrVisuIm->ToJpgFileDeZoom(NameVisu(aCam->NameImage(), "Ref"), 1, {"QUALITY=90"});

        return 0;
    }

    void cAppli_CheckBoardTargetRefine::DrawTarget(cSetMesPtOf1Im & aSetOfMes2D, bool isInit)
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
