//#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "cMMVII_Appli.h"

namespace MMVII
{

//static constexpr tU_INT1 eNone = 0 ;
//static constexpr tU_INT1 eTopo0  = 1 ;
//static constexpr tU_INT1 eTopoTmpCC  = 2 ;
//static constexpr tU_INT1 eTopoMaxOfCC  = 3 ;
//static constexpr tU_INT1 eTopoMaxLoc  = 4 ;
//static constexpr tU_INT1 eFilterSym  = 5 ;
//static constexpr tU_INT1 eFilterRadiom  = 6 ;
//static constexpr tU_INT1 eFilterEllipse  = 7 ;
//static constexpr tU_INT1 eFilterCodedTarget  = 8 ;

    class cAppli_CheckBoardTargetRefine : public cMMVII_Appli//heritage de cMMVII_Appli
    {
        public:
            cAppli_CheckBoardTargetRefine(const std::vector<std::string> & aVArgs,
                                          const cSpecMMVII_Appli & aSpec);
            typedef tREAL4            tElem;
            typedef cIm2D<tElem>      tIm;
            typedef cDataIm2D<tElem>  tDIm;

        private:
            int Exe() override;

            //--spec. methods
            int doVisu(const std::string & aNameIm);
            int doOneImage(const std::string & aNameIm);
            std::vector<std::string> MissingTargetsNames(const cSetMesPtOf1Im & aFoundTargets);
            cSetMesPtOf1Im TargetPredict(const std::string & aNameIm, const std::vector<std::string> & aVMissingTargetsNames);
            void drawTarget(cSetMesPtOf1Im & aSetOfMes2D, bool isInit);

            std::string mSpecImIn;
            bool mShow; //show details
            bool mVisu; //visualisation
            cRGBImage * mCurrVisuIm;
            cSetMesGnd3D mInSet3D;

            //----stolen to cCheckBoardTargetExtract
            std::string NameVisu(const std::string & aDestIm, const std::string & aPref,const std::string aPost="");
            void ReadImageAndBlurr();
            cDataIm2D<tU_INT1> *  mDImLabel;    ///< Data Image of label
            cIm2D<tU_INT1>        mImLabel;     ///< Image storing labels of centers
            tIm                   mImInCur;     ///< Input current image
            cPt2di                mSzImCur;     ///< Size of current image
            cIm2D<tU_INT1>        mImTmp;       ///< Temporary image for connected components
            cDataIm2D<tU_INT1> *  mDImTmp;      ///< Data Image of "mImTmp"
            tIm                   mImBlur;      ///< Blurred image, used in pre-detetction
            tDIm *                mDImBlur;     ///< Data input image
            int                   mNbBlur1;     ///< = 4,  Number of initial blurring
            //std::vector<cCdSadle> mVCdtSad;     ///< Candidate  that are selected as local max of saddle criteria
            //std::vector<cCdSym>   mVCdtSym;     ///< Candidate that are selected on the symetry criteria



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
            ;
    }

    cCollecSpecArg2007 & cAppli_CheckBoardTargetRefine::ArgOpt(cCollecSpecArg2007 & anArgOpt)
    {
        return anArgOpt
               << AOpt2007(mVisu,"Visu","offers visualisation of refined measurements", {eTA2007::HDV})
               << AOpt2007(mShow,"Show","show some useful details", {eTA2007::HDV})//hdv = has default value
            ;
    }

    cAppli_CheckBoardTargetRefine::cAppli_CheckBoardTargetRefine(const std::vector<std::string> & aVArgs,
                                                                const cSpecMMVII_Appli & aSpec):
        cMMVII_Appli(aVArgs, aSpec),
        mCurrVisuIm (nullptr),
        //----stolen to cCheckBoardTargetExtract
        mDImLabel         (nullptr),
        mImLabel          (cPt2di(1,1)),
        mImInCur          (cPt2di(1,1)),
        mImTmp            (cPt2di(1,1)),
        mDImTmp           (nullptr),
        mImBlur           (cPt2di(1,1)),
        mDImBlur          (nullptr),
        mNbBlur1          (1),
        //----
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
        std::vector<std::string> aVImg = VectMainSet(0);

        //·1·>loads global input
            //creates & fills 3d mes. set
        mInSet3D= mPhProj.LoadGCP3DFromFolder(mPhProj.DPGndPt3D().DirIn());

        //·2->exe. algo for each im.
        for (const auto & aNameIm:aVImg)
        {
            doOneImage(aNameIm);

            if (mVisu)
            {
                doVisu(aNameIm);
            }
        }

        // fin de la tranquillite
        return EXIT_SUCCESS;
    }

    /*tDIm cAppli_CheckBoardTargetRefine::getWindow(cBox2di & aBox)
    {//will have to extract image from original one from 2d box
        ...
    }*/

    int cAppli_CheckBoardTargetRefine::doOneImage(const std::string & aNameIm)
    {
    //·1->loads curr. img. input/output
        auto aFoundTargets = mPhProj.LoadMeasureImFromFolder(mPhProj.DPGndPt2D().DirIn(), aNameIm);
    //·2->filter mes2D that have not been detected
        auto aVMissingTargetsNames = MissingTargetsNames(aFoundTargets);
    //·3->loads filtered 3D points & compute 2D projection (=prediction)
        if (aVMissingTargetsNames.empty())
        {
            return 1;
        }//else:
        cSetMesPtOf1Im aPredictions = TargetPredict(aNameIm, aVMissingTargetsNames);
        mPhProj.SaveMeasureIm(aPredictions);
    //·4->classical target detection calls -> to be continued ??

    /*    mVCdtSad.clear();
        mVCdtSym.clear();
        mCurScale = aScale;
        ReadImageAndBlurr();
    */
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

    cSetMesPtOf1Im cAppli_CheckBoardTargetRefine::TargetPredict(const std::string & aNameIm, const std::vector<std::string> & aVMissingTargetsNames)
    {//3d->2d prediction of a set of 3d gcps
        cSetMesPtOf1Im aPredictions(aNameIm);//creates 2d set to fill
        cSensorCamPC * aCam = mPhProj.ReadCamPC(aNameIm,true);//loads im ori
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
                aPredictions.AddMeasure(cMesIm1Pt(aPrediction, aTargetName, 1));
            }
        }
        return aPredictions;
    }

    int cAppli_CheckBoardTargetRefine::doVisu(const std::string & aNameIm)
    {
        cRGBImage aIm = cRGBImage::FromFile(aNameIm);
        mCurrVisuIm = & aIm;

        cSetMesPtOf1Im aSetOfRefMes2D = mPhProj.LoadMeasureImFromFolder(mPhProj.DPGndPt2D().DirOut(), aNameIm);
        cSetMesPtOf1Im aSetOfInitMes2D = mPhProj.LoadMeasureImFromFolder(mPhProj.DPGndPt2D().DirIn(), aNameIm);

        drawTarget(aSetOfInitMes2D, true);//draw Init targets
        drawTarget(aSetOfRefMes2D, false);//draw Refined targets

        mCurrVisuIm->ToJpgFileDeZoom(NameVisu(aNameIm, "Ref"), 1, {"QUALITY=90"});

        return 0;
    }

    void cAppli_CheckBoardTargetRefine::drawTarget(cSetMesPtOf1Im & aSetOfMes2D, bool isInit)
    {//draws targets from a set of mes2d & adapts drawing color whether targets are from
     //initial measurements (only Cyan circle) or not (Orange border represents window size)
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
