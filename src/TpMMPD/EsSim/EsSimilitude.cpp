#include "EsSimilitude.h"


int EsSim_main(int argc,char ** argv)
{
    string aImgX;
    string aImgY;
    string aDir;

    Pt3di aSzDisp(50,50,5);

    Pt2dr aPtCtr(0,0);
    int aSzW = 5;

    int nInt = 0;

    Pt2di aNbGrill(1,1);

    double aSclDepl = 50.0;

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
                << EAM(aSzDisp, "SzDisp", true, "Size Display Win")
                << EAM(aPtCtr, "Pt", true, "Pt Correl central")
                << EAM(aSzW, "aSzW", true, "Sz win Correl (demi)")
                << EAM(aNbGrill, "aNbGrill", true, "Nb de Grill")
                << EAM(aSclDepl, "Scl", true, "=50 def (scale Vec Depl)")
             );

    if (MMVisualMode)     return EXIT_SUCCESS;
    if (EAMIsInit(&aSzDisp))
    {
        nInt = 1;
    }
    cParamEsSim * aParam = new cParamEsSim(aDir, aImgX, aImgY, aPtCtr, aSzW, aSzDisp, nInt, aNbGrill, aSclDepl);
    cAppliEsSim * aAppli = new cAppliEsSim(aParam);

    Pt2dr aRotCosSin;
    Pt2dr aTransXY;

    if ( !EAMIsInit(&aNbGrill) && nInt == 0)
    {
        ElPackHomologue aPack;
        aAppli->getHomolInVgt(aPack, aPtCtr, aSzW);
        aAppli->EsSimFromHomolPack(aPack, aRotCosSin, aTransXY);
    }
    if (EAMIsInit(&aNbGrill))
    {
        aAppli->EsSimEnGrill(aAppli->VaP0Grill(), aSzW, aRotCosSin, aTransXY);
    }
    if ( !EAMIsInit(&aNbGrill) && nInt == 1)
    {
        aAppli->EsSimAndDisp(aPtCtr, aSzW, aRotCosSin, aTransXY);
    }

 return EXIT_SUCCESS;
}
