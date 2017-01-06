#include "ZBufferRaster.h"

string aPatFIm, aMesh, aOri;
int ZBufferRaster_main(int argc,char ** argv)
{
    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()
                << EAMC(aMesh, "Mesh",     eSAM_IsExistFile)
                << EAMC(aPatFIm, "Image",  eSAM_IsPatFile)
                << EAMC(aOri, "Ori",       eSAM_IsExistDirOri),
                //optional arguments
                LArgMain()
                );

    if (MMVisualMode) return EXIT_SUCCESS;

    string aDir, aPatIm;
    SplitDirAndFile(aDir, aPatIm, aPatFIm);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    vector<string>  vImg = *(aICNM->Get(aPatIm));

    StdCorrecNameOrient(aOri,aDir,true);

    InitOutil * aInitMesh = new InitOutil(aMesh);

    vector<triangle*> aVTriMesh = aInitMesh->getmPtrListTri();

    vector<cTri3D> aVTri;

    for (int aKTri=0; aKTri<aVTriMesh.size(); aKTri++)
    {
        triangle * aTriMesh = aVTriMesh[aKTri];
        cTri3D aTri (   aTriMesh->getSommet(0),
                        aTriMesh->getSommet(1),
                        aTriMesh->getSommet(2)
                    );
        aVTri.push_back(aTri);
    }
    //delete(aInitMesh);



    cAppliZBufferRaster * aAppli = new cAppliZBufferRaster(aICNM, aDir, aOri, aVTri, vImg);

    aAppli->DoAllIm();

    return EXIT_SUCCESS;
}
