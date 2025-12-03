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

        private:
            int Exe() override;
            cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
            cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
            cPhotogrammetricProject mPhProj;
            std::string mSpecImIn;
            bool mShow;

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
               << AOpt2007(mShow,"show","show some useful details", {eTA2007::HDV})//hdv = has default value
            ;
    }

    cAppli_CheckBoardTargetRefine::cAppli_CheckBoardTargetRefine(const std::vector<std::string> & aVArgs,
                                                                const cSpecMMVII_Appli & aSpec):
        cMMVII_Appli(aVArgs, aSpec),
        mPhProj (*this)
    {
        //···> constructor does nothing
    }


    int cAppli_CheckBoardTargetRefine::Exe()
    {
        //FINAL OUTPUT = un set temporaire de mes2d virtuel

        mPhProj.FinishInit();

        // a partir d'ici on peut coder tranquille

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
        }

        // fin de la tranquillite
        return EXIT_SUCCESS;
    }


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
