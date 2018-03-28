#include "cAppuis2homol.h"

cAppuis2Homol::cAppuis2Homol(int argc, char** argv):
    mDebug(0),
    mExpTxt(0),
    mSH("-Appui")

{

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(mIm1, "Image 1 name",eSAM_IsExistFile)
                << EAMC(mIm2, "Image 2 name",eSAM_IsExistFile)
                << EAMC(m2DMesFileName, "  2D measures of GCPs, as results of SaisieAppuiInit ",eSAM_IsExistFile),
                LArgMain()
                << EAM(mDebug,"Debug",true,"Print Messages to help debugging process")
                << EAM(mSH, "SH", true, "Set of Homol postfix, def '-Appui' will write homol to Homol-Appui/ directory")
                << EAM(mExpTxt,"ExpTxt",true,"Save as text? default false, mean binary format")
                );

    mICNM=cInterfChantierNameManipulateur::BasicAlloc("./");
    std::string aExt("dat");
    if (mExpTxt) aExt="txt";

    // initialiser le pack de points homologues
    ElPackHomologue  aPackHom;
    if (mDebug) std::cout << "open 2D mesures\n";
    cSetOfMesureAppuisFlottants aSetOfMesureAppuisFlottants=StdGetFromPCP(m2DMesFileName,SetOfMesureAppuisFlottants);
    if (mDebug) std::cout << "Done\n";
       int count(0);
    for (auto &aMesAppuisIm1 : aSetOfMesureAppuisFlottants.MesureAppuiFlottant1Im())
    {
        if(aMesAppuisIm1.NameIm() == mIm1)
        {
            if (mDebug) std::cout << "Found 2D mesures for Image 1\n";
            for (auto &aMesAppuisIm2 : aSetOfMesureAppuisFlottants.MesureAppuiFlottant1Im())
            {
                if(aMesAppuisIm2.NameIm() == mIm2)
                {
                     if (mDebug) std::cout << "Found 2D mesures for Image 2\n";

                    for (auto & OneAppuiIm1 : aMesAppuisIm1.OneMesureAF1I())
                    {

                        for (auto & OneAppuiIm2 : aMesAppuisIm2.OneMesureAF1I())
                        {
                            if (OneAppuiIm1.NamePt()==OneAppuiIm2.NamePt())
                            {
                                ElCplePtsHomologues Homol(OneAppuiIm1.PtIm(),OneAppuiIm2.PtIm());

                                aPackHom.Cple_Add(Homol);

                                count++;
                            }
                        }
                    }
                 break;
                }
            }
        break;
        }
    }

    // save result
    if (mDebug) std::cout << "Save Results\n";

    if (count!=0)
    {
    std::string aKeyAsocHom = "NKS-Assoc-CplIm2Hom@"+ mSH +"@" + aExt;
    if (mDebug) std::cout << "NKS " << aKeyAsocHom << "\n";
    std::string aHomolFile= mICNM->Assoc1To2(aKeyAsocHom, mIm1, mIm2,true);
    if (mDebug) std::cout << "generate " << aHomolFile << "\n";

    aPackHom.StdPutInFile(aHomolFile);
    aPackHom.SelfSwap();
    aHomolFile=  mICNM->Assoc1To2(aKeyAsocHom, mIm2, mIm1,true);
    if (mDebug) std::cout << "generate " << aHomolFile << "\n";

    aPackHom.StdPutInFile(aHomolFile);
    std::cout << "Finished, " << count << " manual seasing of GCP have been converted in homol format\n";
    std::string aKH("NB");
    if (mExpTxt) aKH="NT";

    std::cout << "Launch SEL for visualisation: SEL ./ " << mIm1 << " " << mIm2 << " KH=" << aKH << " SH=" << mSH << "\n";
    } else { std::cout << "I haven't found couple of 2D measure for these images pairs, sorry \n";}
}

int GCP2Hom_main(int argc,char ** argv)
{
   cAppuis2Homol(argc,argv);
   return EXIT_SUCCESS;
}
