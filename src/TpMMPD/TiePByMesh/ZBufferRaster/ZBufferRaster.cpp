#include "ZBufferRaster.h"

string aPatFIm, aMesh, aOri;
int nInt = 0;
Pt2di aSzW;
int rech=1;
double distMax = DBL_MAX;
bool withLbl = true;
bool aNoTif = false;


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
                << EAM(nInt, "nInt", true, "niveau Interaction")
                << EAM(aSzW,  "aSzw",true,"if visu [x,y]")
                << EAM(rech,  "rech",true,"cal ZBuff in img Resample - default =1.0 - 2 => 2 times <")
                << EAM(distMax,  "distMax",true,"limit distant cover Maximum from camera - default = NO LIMIT")
                << EAM(withLbl,  "withLbl",true,"Do image label (image label of triangle in surface)")
                );

    if (MMVisualMode) return EXIT_SUCCESS;

    string aDir, aPatIm;
    SplitDirAndFile(aDir, aPatIm, aPatFIm);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    vector<string>  vImg = *(aICNM->Get(aPatIm));

    //===========Modifier ou chercher l'image si l'image ne sont pas tif============//
       std::size_t found = aPatFIm.find_last_of(".");
       std::cout << " extension: " << aPatFIm.substr(found+1) << '\n';
       string ext = aPatFIm.substr(found+1);
       if (ext != "tif" || ext!= "TIF")
       {
           aNoTif = true;
           cout<<" No Tif"<<endl;
       }

    //===============================================================================//

    StdCorrecNameOrient(aOri,aDir,true);

    vector<cTri3D> aVTri;

    cout<<"Lire mesh...";
    ElTimer aChrono;
    cMesh myMesh(aMesh, true);
    const int nFaces = myMesh.getFacesNumber();
    for (double aKTri=0; aKTri<nFaces; aKTri++)
    {
        cTriangle* aTri = myMesh.getTriangle(aKTri);
        vector<Pt3dr> aSm;
        aTri->getVertexes(aSm);
        cTri3D aTri3D (   aSm[0],
                          aSm[1],
                          aSm[2],
                          aKTri
                      );
        aVTri.push_back(aTri3D);
    }
    cout<<"Finish - time "<<aChrono.uval()<<" - NbTri : "<<aVTri.size()<<endl;

    cAppliZBufferRaster * aAppli = new cAppliZBufferRaster(aICNM, aDir, aOri, aVTri, vImg, aNoTif);

    aAppli->NInt() = nInt;
    if (EAMIsInit(&aSzW))
    {
        aAppli->SzW() = aSzW;
    }
    if (EAMIsInit(&distMax))
    {
        aAppli->DistMax() = distMax;
    }
    aAppli->WithImgLabel() = withLbl;
    aAppli->Reech() = 1.0/double(rech);
    aAppli->SetNameMesh(aMesh);
    aAppli->DoAllIm();


    return EXIT_SUCCESS;
}
