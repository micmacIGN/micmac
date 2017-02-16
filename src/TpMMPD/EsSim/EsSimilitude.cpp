#include "EsSimilitude.h"


int EsSim_main(int argc,char ** argv)
{
    string aImgX;
    string aImgY;
    string aDir;

    Pt3di aSzDisp;

    Pt2dr aPtCtr;
    int aSzW;

    int nInt = 0;

    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()
                << EAMC(aDir, "Dir", eSAM_None)
                << EAMC(aImgX, "Img de deplacement X", eSAM_IsExistFile)
                << EAMC(aImgY, "Img de deplacement Y", eSAM_IsExistFile),

                //optional arguments
                LArgMain()
                << EAM(aSzDisp, "SzDisp", true, "Size Win")
                << EAM(aPtCtr, "Pt", true, "Pt Correl central")
                << EAM(aSzW, "aSzW", true, "Sz win Correl (demi)")
             );

    if (MMVisualMode)     return EXIT_SUCCESS;
    if (EAMIsInit(&aSzDisp))
    {
        nInt = 1;
    }
    cParamEsSim * aParam = new cParamEsSim(aDir, aImgX, aImgY, aPtCtr, aSzW, aSzDisp, nInt);
    cAppliEsSim * aAppli = new cAppliEsSim(aParam);



 return EXIT_SUCCESS;
}
