#include "ZBufferRaster.h"
#include "../DrawOnMesh.h"


cParamZbufferRaster::cParamZbufferRaster():
    mFarScene (false),
    mInt (0),
    mrech (1),
    mDistMax (DBL_MAX),
    mWithLbl (true),
    mNoTif   (false),
    mMethod  (3),
    MD_SEUIL_SURF_TRIANGLE (TT_SEUIL_SURF),
    mPercentVisible (80.0),
    mSafe   (true),
    mInverseOrder (false)
{
}

int ZBufferRaster_main(int argc,char ** argv)
{
    cParamZbufferRaster aParam;
    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()
                << EAMC(aParam.mMesh, "Mesh",     eSAM_IsExistFile)
                << EAMC(aParam.mPatFIm, "Pattern Image",  eSAM_IsPatFile)
                << EAMC(aParam.mOri, "Ori",       eSAM_IsExistDirOri),
                //optional arguments
                LArgMain()
                << EAM(aParam.mInt, "nInt", true, "niveau Interaction")
                << EAM(aParam.mSzW,  "aSzw",true,"if visu [x,y]")
                << EAM(aParam.mrech,  "rech",true,"cal ZBuff in img Resample - default =1.0 - 2 => 2 times <")
                << EAM(aParam.mDistMax,  "distMax",true,"limit distant cover Maximum from camera - default = NO LIMIT")
                << EAM(aParam.mWithLbl,  "withLbl",true,"Do image label (image label of triangle in surface)")
                << EAM(aParam.mMethod,  "method",true,"method of grab pixel in triangle (1=very good (low), 3=fast (not so good - def))")
                << EAM(aParam.MD_SEUIL_SURF_TRIANGLE, "surfTri", true, "Threshold of surface to filter triangle too small (def=100)")
                << EAM(aParam.mFarScene, "farScene", true, "Detect far scene part")
                << EAM(aParam.mPercentVisible, "pVisible", true, "condition to decide far scene part : triangle visible in % nb of image (def=80%)")
                << EAM(aParam.mSafe, "Safe", true, "check if pt 3D raster visible in img before calcul (safe but slow) - def=true")
                << EAM(aParam.mInverseOrder, "InvOrder", true, "Inverse order of triangle's vertices (as we don't know how 's the mesh generated - def=false")
             );

    if (MMVisualMode) return EXIT_SUCCESS;
    string aDir, aPatIm;
    SplitDirAndFile(aDir, aPatIm, aParam.mPatFIm);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    vector<string>  vImg = *(aICNM->Get(aPatIm));

    //===========Modifier ou chercher l'image si l'image ne sont pas tif============//
    std::size_t found = aParam.mPatFIm.find_last_of(".");
    string ext = aParam.mPatFIm.substr(found+1);
    cout<<"Ext : "<<ext<<endl;
    if ( ext.compare("tif") )   //ext equal tif
    {
        aParam.mNoTif = true;
        cout<<" No Tif"<<endl;
    }
    if (aParam.mNoTif)
    {
        list<string> cmd;
        for (uint aK=0; aK<vImg.size(); aK++)
        {
             string aCmd = MM3DStr +  " PastDevlop "+  vImg[aK] + " Sz1=-1 Sz2=-1 Coul8B=0";
             cmd.push_back(aCmd);
        }
        cEl_GPAO::DoComInParal(cmd);
    }
    //===============================================================================//

    StdCorrecNameOrient(aParam.mOri,aDir,true);

    vector<cTri3D> aVTri;

    cout<<"Lire mesh...";
    ElTimer aChrono;
    cMesh myMesh(aParam.mMesh, true);
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

    cAppliZBufferRaster * aAppli = new cAppliZBufferRaster(aICNM, aDir, aParam.mOri, aVTri, vImg, aParam.mNoTif, aParam);

    aAppli->NInt() = aParam.mInt;
    if (EAMIsInit(&aParam.mSzW))
    {
        aAppli->SzW() = aParam.mSzW;
    }
    if (EAMIsInit(&aParam.mDistMax))
    {
        aAppli->DistMax() = aParam.mDistMax;
    }
    if (EAMIsInit(& aParam.MD_SEUIL_SURF_TRIANGLE))
        aAppli->SEUIL_SURF_TRIANGLE()=aParam.MD_SEUIL_SURF_TRIANGLE;
    aAppli->Method() = aParam.mMethod;
    aAppli->WithImgLabel() = aParam.mWithLbl;
    aAppli->Reech() = 1.0/double(aParam.mrech);
    aAppli->SetNameMesh(aParam.mMesh);
    aAppli->DoAllIm();

    cout<<"Cal ZBuf: time "<<aChrono.uval()<<" - NbTri : "<<aVTri.size()<<endl;


    // statistic far scene part
    if (aParam.mFarScene)
    {
        set <int> aTriToWrite;
        vector<cXml_TriAngulationImMaster> aFarTask;
        vector<int> vNumImSec;
        //DrawOnMesh aDraw;
        //vector<vector<Pt3dr> > aTriToWrite;

        cXml_TriAngulationImMaster aXMLTri;
        aXMLTri.NameMaster() = vImg[0]; // on s'en fou le master
        for (uint aK=0; aK<vImg.size(); aK++)
        {
            if(aAppli->vImgVisibleFarScene()[aK])
            {
                aXMLTri.NameSec().push_back(vImg[aK]); // les secondaire est list d'image du far scene
                vNumImSec.push_back(aK);
            }
        }


        string farSceneMesh = aParam.mMesh.substr(0,aParam.mMesh.length()-4) + "_Far.ply";
        //sortDescendPt2diY(aAppli->AccNbImgVisible());

        int aCount{0};
        for (int aKK=0; aKK<int(aAppli->AccNbImgVisible().size()); aKK++)
        {
            if (aAppli->AccNbImgVisible()[aKK].y >= (vImg.size() * aParam.mPercentVisible/100.0))
            {
                aCount++;
                /*
                vector<Pt3dr> aOneTri;
                aOneTri.push_back(aVTri[aKK].P1());
                aOneTri.push_back(aVTri[aKK].P2());
                aOneTri.push_back(aVTri[aKK].P3());
                aTriToWrite.push_back(aOneTri);
                */
                aTriToWrite.insert(aAppli->AccNbImgVisible()[aKK].x);
                cXml_Triangle3DForTieP aTri3D;
                aTri3D.P1() = aVTri[aKK].P1();
                aTri3D.P2() = aVTri[aKK].P2();
                aTri3D.P3() = aVTri[aKK].P3();
                aTri3D.NumImSec() = vNumImSec;
                aXMLTri.Tri().push_back(aTri3D);
            }
        }
        cout<<"Write mesh & export XML.. "<<endl;
        //aDraw.drawListTriangle(aTriToWrite, farSceneMesh, Pt3dr(255,0,0));
        myMesh.Export(farSceneMesh, aTriToWrite, true);
        MakeFileXML(aXMLTri, "FarScene.xml");
        cout<<"Nb Tri View by "<<aParam.mPercentVisible<<"% of Img : "<<aCount<<" / "<< aVTri.size()<<endl;
    }

    return EXIT_SUCCESS;
}
