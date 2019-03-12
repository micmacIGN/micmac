#include "StdAfx.h"
#include <algorithm>


int CreateMasq_main (int argc,char **argv)
{
    std::string aNameOut;
    Box2di       aBoxOut;

    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(aNameOut,"File for resulta"),
         LArgMain()  << EAM(aBoxOut,"Box",true,"Box of result, def computed according to files definition")
    );

    Pt2di aSzOut = aBoxOut.sz();

    Tiff_Im aTifOut
            (
                aNameOut.c_str(),
                aSzOut,
                GenIm::bits1_msbf,
                Tiff_Im::Group_4FAX_Compr,
                Tiff_Im::BlackIsZero
                );
    Output  anOutGlob = aTifOut.out();


    ELISE_COPY
    (
        rectangle(Pt2di(0,0),aSzOut),
        true,
        anOutGlob
    );

    return EXIT_SUCCESS;
}

