/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"

/*#if(ELISE_QT_VERSION >= 4)
    #ifdef Int
        #undef Int
    #endif

    #include <QMessageBox>
    #include <QApplication>
#endif*/

// Test if distance in depth image is close to distance between pt3d and center (threshold = aDst)
bool DistanceTest(double aDst, Pt3dr const &pt3d, Pt3dr const &center, double downScale, CamStenope* cam, Im2D_REAL4* ImgProf, Im2D_INT4* Masq)
{
    Pt2dr Pt = cam->R3toF2(pt3d);
    if (cam->IsInZoneUtile(Pt))
    {
        Pt2di Pti(round_down(Pt.x/downScale), round_down(Pt.y/downScale));
        /*if (Masq->GetI(Pti) == 1)
        {
            cout << "dist  : " << ImgProf->GetR(Pti) << endl;
        }*/

        if ((Masq->GetI(Pti)) && (abs(abs(ImgProf->GetR(Pti)) - euclid(pt3d - center)) < aDst)) return true;
    }
    return false;
}

int TiPunch_main(int argc,char ** argv)
{
    bool verbose = true;

    std::string aDir, aPat, aFullName, aOri, aPly, aOut, aMode, aCom;
    bool aBin = true;
    bool aRmPoissonMesh = false;
    int aDepth = 8;
    double aDst = 0.5f;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullName,"Full Name (Dir+Pat)",eSAM_IsPatFile)
                            << EAMC(aOri,"Orientation path",eSAM_IsExistDirOri)
                            << EAMC(aPly,"Ply file", eSAM_IsExistFile)
                            << EAMC(aMode,"C3DC mode", eSAM_None,ListOfVal(eNbTypeMMByP)),
                LArgMain()  << EAM(aOut,"Out",true,"Mesh name (def=plyName+ _mesh.ply)")
                            << EAM(aBin,"Bin",true,"Write binary ply (def=true)")
                            << EAM(aDepth,"Depth",true,"Maximum reconstruction depth for PoissonRecon (def=8)")
                            << EAM(aRmPoissonMesh,"Rm",true,"Remove intermediary Poisson mesh (def=false)")
                            << EAM(aDst,"Dist",true,"Threshold on distance between mesh and point cloud (def=1)")
            );

    if (MMVisualMode) return EXIT_SUCCESS;

    SplitDirAndFile(aDir,aPat,aFullName);

    if (!EAMIsInit(&aOut)) aOut = StdPrefix(aPly) + "_mesh.ply";

    stringstream ss;
    ss << aDepth;
    std::string poissonMesh = StdPrefix(aPly) + "_poisson_depth" + ss.str() +".ply";

    bool computeMesh = true;

    if (ELISE_fp::exist_file(poissonMesh))
    {
        /*if (MMVisualMode)
        {
            #if(ELISE_QT_VERSION >= 4)
              std::string question = "File " + poissonMesh + " already exists. Do you want to replace it? (y/n)";
              QMessageBox::StandardButton reply = QMessageBox::question(NULL, "Warning", question.c_str(),
                                            QMessageBox::Yes|QMessageBox::No);
              if (reply == QMessageBox::Yes) {
                computeMesh = true;
                QApplication::quit();
              } else {
                computeMesh = false;
              }
            #endif
        }
        else*/
        {
            std::string yn;
            cout << "File " << poissonMesh << " already exists. Do you want to replace it? (y/n)" << endl;
            cin >> yn;
            while ((yn != "y") && (yn != "n"))
            {
                cout << "Invalid value, try again." << endl;
                cin >> yn;
            }
            if (yn == "y")
                computeMesh = true;
            else if (yn == "n")
                computeMesh = false;
        }
    }

    if (computeMesh)
    {
        //#if USE_OPEN_MP
        int nbProc = NbProcSys();
        stringstream sst;
        sst << nbProc;
        //#endif

        aCom = g_externalToolHandler.get( "PoissonRecon" ).callName()
                + std::string(" --in ") + aPly.c_str()
                + std::string(" --out ") + poissonMesh.c_str()
                + " --depth " + ss.str()
        //#if USE_OPEN_MP //TODO: � activer quand WITH_OPEN_MP=1
                + " --threads " + sst.str()
        //#endif
        ;

        if (verbose) cout << "Com= " << aCom << endl;

        cout << "\n**********************Running Poisson reconstruction**********************" << endl;

        system_call(aCom.c_str());

        cout << "\nMesh built and saved in " << poissonMesh << endl;
    }

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    std::list<std::string>  aLS = aICNM->StdGetListOfFile(aPat);

    StdCorrecNameOrient(aOri,aDir);

    std::vector<CamStenope*> vCam;
    std::vector<Pt3dr> vCamCenter;

    bool help;
    eTypeMMByP  type;
    StdReadEnum(help,type,aMode,eNbTypeMMByP);

    cMMByImNM *PIMsFilter = cMMByImNM::FromExistingDirOrMatch(aDir + "PIMs-" + aMode + ELISE_CAR_DIR,false);

    vector <Im2D_REAL4 *> vImgProf;
    vector <Im2D_INT4 *>  vImgMasq;

    cout << endl;
    for (std::list<std::string>::const_iterator itS=aLS.begin(); itS!=aLS.end() ; itS++)
    {
        std::string NOri=aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aOri,*itS,true);

        vCam.push_back(CamOrientGenFromFile(NOri,aICNM));
        vCamCenter.push_back(vCam.back()->VraiOpticalCenter()); //pour eviter le test systematique dans VraiOpticalCenter() lors du parcours du maillage

        std::string aNameProf = PIMsFilter->NameFileProf(eTMIN_Depth,*itS);

        if (ELISE_fp::exist_file(aNameProf))
        {
            Tiff_Im tif = Tiff_Im::StdConvGen(aNameProf, 1, true);

            Im2D_REAL4* pImg = new Im2D_REAL4(tif.sz().x, tif.sz().y);

            ELISE_COPY
            (
                pImg->all_pts(),
                tif.in(),
                pImg->out()
            );

            vImgProf.push_back(pImg);
            //cout << "img sz" << vImgProf.back()->sz() << endl;
        }
        else
            cout << "File does not exist : " << aNameProf << endl;

        std::string aNameMasq = PIMsFilter->NameFileMasq(eTMIN_Depth,*itS);

        if (ELISE_fp::exist_file(aNameMasq))
        {
            Tiff_Im tif = Tiff_Im::StdConvGen(aNameMasq, 1, false);

            Im2D_INT4* pImg = new Im2D_INT4(tif.sz().x, tif.sz().y);

            ELISE_COPY
            (
                pImg->all_pts(),
                tif.in(),
                pImg->out()
            );

            vImgMasq.push_back(pImg);
            //cout << "img sz" << vImgMasq.back()->sz() << endl;
        }
        else
            cout << "File does not exist : " << aNameMasq << endl;

        cout << "Image "<<*itS<<", with ori : " << NOri << endl;
        cout << "Depth : "<< aNameProf << endl;
    }

    double downScaleFactor = (double) vCam[0]->Sz().x / vImgProf[0]->sz().x;

    //cout << "downScaleFactor = " << downScaleFactor << endl;

    cMesh myMesh(poissonMesh);

    cout << endl;
    cout <<"**********************Filtering faces*************************"<<endl;
    cout << endl;

    std::vector < int > toRemove;

    const int nCam = vCam.size();
    for(int aK=0; aK <myMesh.getFacesNumber(); aK++)
    {
        if (aK%1000 == 0) cout << aK << " / " << myMesh.getFacesNumber() << endl;

        cTriangle * Triangle = myMesh.getTriangle(aK);

        vector <Pt3dr> Vertex;
        Triangle->getVertexes(Vertex);

        bool found = false;
        for(int bK=0 ; bK<nCam; bK++)
        {
            found = DistanceTest(aDst, (Vertex[0]+Vertex[1]+Vertex[2])/3.f, vCamCenter[bK], downScaleFactor, vCam[bK], vImgProf[bK], vImgMasq[bK]);

            if (found) break;
        }

        if (!found)
        {
            toRemove.push_back(Triangle->getIdx());
            //myMesh.removeTriangle(*Triangle, false);
            //aK--;
        }
    }
    cout << myMesh.getFacesNumber() << " / " << myMesh.getFacesNumber() << endl;

    cout << "Removing " << toRemove.size() << endl;

    std::sort(toRemove.begin(),toRemove.end(),std::greater<int>());
    for (unsigned int var = 0; var < toRemove.size(); ++var) {
         myMesh.removeTriangle(*(myMesh.getTriangle(toRemove[var])), false);
    }

    cout << endl;
    cout <<"**************************Writing ply file***************************"<<endl;
    cout <<endl;

    myMesh.write(aOut, aBin);

    cout<<"********************************Done*********************************"<<endl;
    cout<<endl;

    if (aRmPoissonMesh)
    {
        aCom = std::string(SYS_RM) + " " + poissonMesh;
        system_call(aCom.c_str());
    }

    return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/




