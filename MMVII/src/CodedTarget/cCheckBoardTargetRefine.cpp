//#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "cMMVII_Appli.h"

namespace MMVII
{

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
            int doVisu(const std::string & aNameIm);
            //----stolen to cCheckBoardTargetExtract
            std::string NameVisu(const std::string & aDestIm, const std::string & aPref,const std::string aPost="");
            //----
            void drawTarget(cSetMesPtOf1Im & aSetOfMes2D, bool isInit);
            cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
            cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
            cPhotogrammetricProject mPhProj;
            std::string mSpecImIn;
            bool mShow;
            //--refine
            int mWinSz;
            //--visu
            bool mVisu;
            cRGBImage * mCurrVisuIm;


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
               << AOpt2007(mWinSz,"WinSz","sets size of the window centered on target prediction in which we will look for true target measurement")
               << AOpt2007(mVisu,"Visu","offers visualisation of refined measurements", {eTA2007::HDV})
               << AOpt2007(mShow,"Show","show some useful details", {eTA2007::HDV})//hdv = has default value
            ;
    }

    cAppli_CheckBoardTargetRefine::cAppli_CheckBoardTargetRefine(const std::vector<std::string> & aVArgs,
                                                                const cSpecMMVII_Appli & aSpec):
        cMMVII_Appli(aVArgs, aSpec),
        mPhProj (*this),
        mWinSz (100),
        mCurrVisuIm (nullptr)
    {
        //···> constructor does nothing
    }


    int cAppli_CheckBoardTargetRefine::Exe()
    {
        //FINAL OUTPUT = temporary set of "fake" 2d mes
        // TO DO : add HasGCPMesIm facility to mPhProj or defined here

        mPhProj.FinishInit();

        //·0·> iterates on images
            //gets the firts set
        std::vector<std::string> aVImg = VectMainSet(0);

        //·1·>loads global input
            //creates & fills 3d mes. set
        cSetMesGnd3D aInSet3D = mPhProj.LoadGCP3DFromFolder(mPhProj.DPGndPt3D().DirIn());

        for (const auto & aNameIm:aVImg)
        {
        //·1-im·>loads curr. img. input/output
            //creates & fills 2d mes. of curr. img.
            StdOut() << "Loading " << aNameIm << " 2D mes. from "
                     << mPhProj.DPGndPt2D().DirIn() << std::endl;
            auto aInSet2D = mPhProj.LoadMeasureImFromFolder(mPhProj.DPGndPt2D().DirIn(), aNameIm);

        //·2-im·> diff. btw. detected points and GCPs set

            //creates wanted set
            std::vector<std::string> aWantedNames;
            //iterates on mes3d and checks if pt. p exists in mes2d if not add to
            //the wanted set
            auto aKnownNames = aInSet3D.ListOfNames();
            //StdOut() << a3DNames.size() << " elements names loaded." << std::endl;
            for (const auto & aNameOfGCP:aKnownNames)
            {
                if (!aInSet2D.NameHasMeasure(aNameOfGCP))
                {
                    //StdOut() << aNameOfGCP << " not found in " << aNameIm << std::endl;
                    aWantedNames.push_back(aNameOfGCP);
                }
            }

        //·3-im·> 3d->2d projection from input ori
            //creates 2d set to fill
            cSetMesPtOf1Im aSetWantedMes2D(aNameIm);
            //loads im ori
            cSensorCamPC * aCam = mPhProj.ReadCamPC(aNameIm,true);
            auto & camSz = aCam->InternalCalib()->PixelDomain().Sz();
            //iterates on aWantedNames
            for (const auto & aWantedName:aWantedNames)
            {
                auto aWanted2DPt = aCam->Ground2Image(aInSet3D.GetMeasureOfNamePt(aWantedName).mPt);
                //·4-im·> add 2d mes. only if it is in image
                auto & xMax = camSz[0];
                auto & yMax = camSz[1];
                if (0 < aWanted2DPt[0] && aWanted2DPt[0] < xMax
                    && 0 < aWanted2DPt[1] && aWanted2DPt[1] < yMax)
                {
                    StdOut() << " --> Adding point " << aWantedName << " : "
                             << aWanted2DPt[0] << ";" << aWanted2DPt[1] << std::endl;
                    aSetWantedMes2D.AddMeasure(cMesIm1Pt(aWanted2DPt, aWantedName, 1));
                }
            }

            mPhProj.SaveMeasureIm(aSetWantedMes2D);

            if (mVisu)
            {
                doVisu(aNameIm);
            }

            cIm2D<tREAL4> aIm = cIm2D<tREAL4>::FromFile(aNameIm);
            //extract windows around points
            auto & aVecMes2D = aSetWantedMes2D.Measures();
            for (const auto & aMes2D:aVecMes2D){
                //auto & aDataIm = aIm.DIm();
                int x = ToI(aMes2D.mPt).x();
                StdOut() << x;
                cPt2di lCorn = {ToI(aMes2D.mPt) - cPt2di(100,100)};
                cPt2di rCorn = {ToI(aMes2D.mPt) + cPt2di(100,100)};
                //cIm2D<tREAL4> aCroppedIm(cPt2di(200,200));
                if (aIm.DIm().Inside(lCorn) && aIm.DIm().Inside(rCorn))
                {
                    aIm.DIm().ClipToFile(NameVisu(aNameIm, "Tmp1"), cBox2di(cPt2di(0,0),cPt2di(1000,1000)));
                    aIm.DIm().ClipToFile(NameVisu(aNameIm, "Tmp"), cBox2di(lCorn,rCorn));
                    /*auto imPtr = aDataIm.ExtractRawData2D();
                    for (int & i=lCorn.x();i<rCorn.x();++i)
                    {
                        for (int & j=lCorn.y();j<rCorn.y();++j)
                        {

                            auto pixVal = ;
                            StdOut() << pixVal << std::endl;
                            break;
                        }
                    }*/
                    //
                    //aDataIm.CropIn(uLCorn, aCroppedIm.DIm());
                    //aCroppedIm.DIm().ToFile(NameVisu(aNameIm, "Tmp"));
                    //StdOut() << "Saved to " << NameVisu(aNameIm, "Tmp");
                }
            }
        }

        // fin de la tranquillite
        return EXIT_SUCCESS;
    }

    /*tDIm cAppli_CheckBoardTargetRefine::getWindow(cBox2di & aBox)
    {//will have to extract image from original one from 2d box
        ...
    }*/

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
                mCurrVisuIm->SetRGBBorderRectWithAlpha(ToI(aPt2D.mPt),mWinSz,10,cRGBImage::Orange,0.1);//0.1 means final opacity = 1-0.1
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
