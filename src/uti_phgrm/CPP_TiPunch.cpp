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

/*#if ELISE_QT
    #ifdef Int
    #undef Int
    #endif

    #include <QMessageBox>
    #include <QApplication>
#endif ELISE_QT*/

static const REAL Eps = 1e-7;

REAL SqrDistSum(vector <Pt3dr> const & Sommets, cElNuage3DMaille* nuage)
{
    REAL Res = 0.f;

    if (Sommets.size() == 3)
    {
        cBasicGeomCap3D* aCam = nuage->Cam();

        Pt2dr A = aCam->Ter2Capteur(Sommets[0]);
        Pt2dr B = aCam->Ter2Capteur(Sommets[1]);
        Pt2dr C = aCam->Ter2Capteur(Sommets[2]);

        Pt2dr AB = B-A;
        Pt2dr AC = C-A;
        REAL aDet = AB^AC;

        if (aDet!=0)
        {
            Pt2di aP0 = round_down(Inf(A,Inf(B,C)));
            aP0 = Sup(aP0,Pt2di(0,0));
            Pt2di aP1 = round_up(Sup(A,Sup(B,C)));
            aP1 = Inf(aP1,nuage->SzUnique()-Pt2di(1,1));

            for (INT x=aP0.x ; x<= aP1.x ; x++)
            for (INT y=aP0.y ; y<= aP1.y ; y++)
            {
                Pt2dr Pt(x,y);
                Pt2dr AP = Pt-A;

                // Coordonnees barycentriques de P(x,y)
                REAL aPdsB = (AP^AC) / aDet;
                REAL aPdsC = (AB^AP) / aDet;
                REAL aPdsA = 1 - aPdsB - aPdsC;
                if ((aPdsA>-Eps) && (aPdsB>-Eps) && (aPdsC>-Eps) &&
                    (nuage->ImMask().GetI(Pt2di(x,y)) > 0))
                {
                Pt3dr Pt1 = Sommets[0]*aPdsA + Sommets[1]*aPdsB + Sommets[2]*aPdsC;
                Pt3dr Pt2 = nuage->PreciseCapteur2Terrain(Pt);

                Res += square_euclid(Pt1, Pt2);
                }
            }
        }
    }
    return Res;
}

int TiPunch_main(int argc,char ** argv)
{
	if ( !g_externalToolHandler.get( "PoissonRecon" ).isCallable()) ELISE_ERROR_RETURN("cannot find PoissonRecon tool, did build micmac with option BUILD_POISSON=1 ?");

    bool verbose = true;

    string aDir, aPat, aFullName, aPly, aOut, aMode, aCom;
    bool aBin = true;
    bool aRmPoissonMesh = false;
    int aDepth = 8;
    bool aFilter = true;
    aMode = "Statue";
    int aZBuffSSEch = 1;
    float defValZBuf = 1e9;
    bool aFilterFromBorder = true;

    ElInitArgMain
        (
        argc,argv,
        LArgMain()  << EAMC(aPly,"Ply file", eSAM_IsExistFile),
        LArgMain()  << EAM(aFullName,"Pattern",false,"Full Name (Dir+Pat)",eSAM_IsPatFile)
                    << EAM(aOut,"Out",false,"Mesh name (def=plyName+ _mesh.ply)")
                    << EAM(aBin,"Bin",true,"Write binary ply (def=true)")
                    << EAM(aDepth,"Depth",true,"Maximum reconstruction depth for PoissonRecon (def=8)")
                    << EAM(aRmPoissonMesh,"Rm",true,"Remove intermediary Poisson mesh (def=false)")
                    << EAM(aFilter,"Filter",true,"Filter mesh (def=true)")
                    << EAM(aMode,"Mode",true,"C3DC mode (def=Statue)", eSAM_None,ListOfVal(eNbTypeMMByP))
                    << EAM(aZBuffSSEch,"Scale", true, "Z-buffer downscale factor (def=2)",eSAM_InternalUse)
                    << EAM(aFilterFromBorder,"FFB",true,"Filter from border (def=true)")
        );


    if (MMVisualMode) return EXIT_SUCCESS;

    SplitDirAndFile(aDir,aPat,aFullName);

    if (!EAMIsInit(&aOut)) aOut = StdPrefix(aPly) + "_mesh.ply";

    stringstream ss;
    ss << aDepth;
    string poissonMesh = StdPrefix(aPly) + "_poisson_depth" + ss.str() +".ply";

    bool computeMesh = true;

    if (ELISE_fp::exist_file(poissonMesh))
    {
    /*if (MMVisualMode)
    {
       #if ELISE_QT
          std::string question = "File " + poissonMesh + " already exists. Do you want to replace it? (y/n)";
          QMessageBox::StandardButton reply = QMessageBox::question(NULL, "Warning", question.c_str(),
                        QMessageBox::Yes|QMessageBox::No);
          if (reply == QMessageBox::Yes) {
        computeMesh = true;
        QApplication::quit();
          } else {
        computeMesh = false;
          }
        #endif // ELISE_QT
    }
    else*/
    {
        string yn;
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
        #if USE_OPEN_MP
            int nbProc = NbProcSys();
            stringstream sst;
            sst << nbProc;
        #endif

        aCom = g_externalToolHandler.get( "PoissonRecon" ).callName()
            + string(" --in ") + aPly.c_str()
            + string(" --out ") + poissonMesh.c_str()
            + " --depth " + ss.str()
        #if USE_OPEN_MP
            + " --threads " + sst.str()
        #endif
        ;

        if (verbose) cout << "Com= " << aCom << endl;

        cout << "\n**********************Running Poisson reconstruction**********************" << endl;

        system_call(aCom.c_str());

        cout << "\nMesh built and saved in " << poissonMesh << endl;
    }

    cMesh myMesh(poissonMesh, aFilterFromBorder);

    if (aFilter)
    {
        ELISE_ASSERT(EAMIsInit(&aFullName),"Filter=true and image pattern is missing");

        cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
        list<string>  aLS = aICNM->StdGetListOfFile(aPat);



        bool help;
        eTypeMMByP  type;
        StdReadEnum(help,type,aMode,eNbTypeMMByP);

        // cMMByImNM *PIMsFilter = cMMByImNM::FromExistingDirOrMatch(aDir + "PIMs-" + aMode + ELISE_CAR_DIR,false);
        cMMByImNM *PIMsFilter = cMMByImNM::FromExistingDirOrMatch(aDir + "PIMs-" + aMode + ELISE_CAR_DIR,false);

        vector <cElNuage3DMaille *> vNuages;
        vector <Im2D_BIN> vMasqImg;
        vector <cZBuf> vZBuffers;

        cout << endl;
        for (list<string>::const_iterator itS=aLS.begin(); itS!=aLS.end() ; itS++)
        {
            string aNameXml  = PIMsFilter->NameFileXml(eTMIN_Merge,*itS);
            string aNameMasqDepth = PIMsFilter->NameFileMasq(eTMIN_Depth,*itS);

            if (ELISE_fp::exist_file(aNameXml) && ELISE_fp::exist_file(aNameMasqDepth))
            {
                vNuages.push_back(cElNuage3DMaille::FromFileIm(aNameXml,"XML_ParamNuage3DMaille"));
                CamStenope * aCS = vNuages.back()->Cam()->DownCastCS();

                if (aCS)
                   aCS->SetIdentCam(aNameXml); //debug

                cBasicGeomCap3D * Cam = vNuages.back()->Cam();
                cZBuf aZBuffer(Cam->SzBasicCapt3D(), defValZBuf, aZBuffSSEch);

                aZBuffer.BasculerUnMaillage(myMesh, *(dynamic_cast <CamStenope*> (Cam)));

                vZBuffers.push_back(aZBuffer);

                cout << "Image " << *itS << " with masq " << aNameMasqDepth << endl;

                Tiff_Im aImg(aNameMasqDepth.c_str());

                Pt2di sz = aImg.Sz2();

                Im2D_BIN aImBin(sz.x, sz.y, 0);

                ELISE_COPY
                (
                   aImg.all_pts(),
                   aImg.in(),
                   aImBin.out()
                );

                vMasqImg.push_back(aImBin);
            }
            else cout << aNameXml << " or " << aNameMasqDepth << " does not exist for " << *itS << endl;
        }

        ELISE_ASSERT(vNuages.size() == vMasqImg.size(), "Missing masq image");

        cout << endl;
        cout <<"**********************Filtering faces*************************"<<endl;
        cout << endl;

        Pt2dr A2, B2, C2, AB, AC;
        Pt2di A2i, B2i, C2i, aP0, aP1;

        const int nNuages = (int)vNuages.size();
        for(int bK=0 ; bK<nNuages; bK++)
        {
            set <int> vTri;
            vZBuffers[bK].getVisibleTrianglesIndexes(vTri);

            set <int>::const_iterator it = vTri.begin();
            for(;it!=vTri.end();++it)
            {
                cTriangle * Triangle = myMesh.getTriangle(*it);

                if (!Triangle->isTextured())
                {
                    vector <Pt3dr> Vertex;
                    Triangle->getVertexes(Vertex);

                    cBasicGeomCap3D * Cam = vNuages[bK]->Cam();

                    A2 = Cam->Ter2Capteur(Vertex[0]);
                    B2 = Cam->Ter2Capteur(Vertex[1]);
                    C2 = Cam->Ter2Capteur(Vertex[2]);

                    TIm2DBits<1> im (vMasqImg[bK]);

                    // Tiff_Im::CreateFromIm(vMasqImg[bK],"./toto" + ToString(bK) + ".tif");

                    A2i = round_ni(A2);
                    B2i = round_ni(B2);
                    C2i = round_ni(C2);

                    if (im.inside(A2i) && im.inside(B2i) && im.inside(C2i))
                    {
                        AB = B2-A2;
                        AC = C2-A2;
                        REAL aDet = AB^AC;

                        if (aDet!=0)
                        {
                            aP0 = round_down(Inf(A2,Inf(B2,C2)));
                            aP0 = Sup(aP0,Pt2di(0,0));
                            aP1 = round_up(Sup(A2,Sup(B2,C2)));
                            aP1 = Inf(aP1,im.sz()-Pt2di(1,1));

                            bool doBreak = false;
                            for (INT x=aP0.x ; x<= aP1.x ; x++)
                            {

                                for (INT y=aP0.y ; y<= aP1.y ; y++)
                                {
                                    Pt2dr AP = Pt2dr(x,y)-A2;

                                    // Coordonnees barycentriques de P(x,y)
                                    REAL aPdsB = (AP^AC) / aDet;
                                    REAL aPdsC = (AB^AP) / aDet;
                                    REAL aPdsA = 1 - aPdsB - aPdsC;
                                    if ((aPdsA>-Eps) && (aPdsB>-Eps) && (aPdsC>-Eps))
                                    {
                                        if (im.get(Pt2di(x,y)))
                                        {
                                            /*if (Triangle->Idx() == 46614)
                                            {
                                                cout << "ok " << bK << " " << x << " " << y  << endl;
                                                cout << "A2  = " << A2 << endl;
                                                cout << "B2  = " << B2 << endl;
                                                cout << "C2  = " << C2 << endl;
                                            }*/

                                            Triangle->setBestImgIndex(1); //trick to check if triangle is viewed
                                            doBreak = true;
                                            break;
                                        }
                                    }
                                }
                                if (doBreak) break;
                            }
                        }
                    }
                }
            }
        }

        /*
        //TODO devrait suffire mais reste un bug (rotation des masques ?)

        set < int > stri;
        const int nbTriangles = myMesh.getFacesNumber();
        for (int aK=0; aK < nbTriangles;++aK)
        {
            cTriangle * triangle = myMesh.getTriangle(aK);
            if (triangle->isTextured()) stri.insert(aK);
        }
        myMesh.Export(aOut,stri);*/

        set < int, greater<int> > toRemove;


        if (aFilterFromBorder)
        {
// MPD : bloque ds myMesh.clean ?? std::cout << "TTtttttt\n";
            myMesh.clean();

//  MPD :  std::cout << "  aaaaaaa \n";
            //after clean, some isolated triangles can remain, we remove them by keeping only the biggest region
            vector<cTextureBox2d> vTexBox = myMesh.getRegions();

            //looking for biggest region
            unsigned int id = 0;
            size_t nbTri = vTexBox[0].triangles.size();
            for (unsigned int aK = 1; aK < vTexBox.size();++aK)
            {
                if (vTexBox[aK].triangles.size() > nbTri) id = aK;
            }

            //remove other
            for (unsigned int aK = 0; aK < vTexBox.size();++aK)
            {
                if (aK != id)
                {
                    vector<int> *vtri = &(vTexBox[aK].triangles);
                    vector<int>::const_iterator it = vtri->begin();
                    for (;it!=vtri->end();++it) toRemove.insert(*it);
                }
            }

            cout << "Removing " << toRemove.size() << " / " << myMesh.getFacesNumber() << " faces" << endl;

            set < int, greater<int> >::const_iterator itr = toRemove.begin();
            int aCpt = (int)toRemove.size();
            for (; itr != toRemove.end(); ++itr)
            {
                  myMesh.removeTriangle(*itr);
                  aCpt--;
                  // if (aCpt%100==0) std::cout << "Still " << aCpt << " to do \n";
            }
        }
        else
        {
            const int nbTriangles = myMesh.getFacesNumber();
            for (int aK=0; aK < nbTriangles;++aK)
            {
                cTriangle * triangle = myMesh.getTriangle(aK);
                if (!triangle->isTextured()) toRemove.insert(aK);
            }

            cout << "Removing " << toRemove.size() << " / " << nbTriangles << " faces" << endl;

            set<int, greater<int> >::const_iterator itr = toRemove.begin();
            for(; itr!=toRemove.end();++itr) myMesh.removeTriangle(*itr, false);
        }
    }

    cout << endl;
    cout <<"**************************Writing ply file***************************"<<endl;
    cout <<endl;

    myMesh.write(aOut, aBin);

    cout <<"********************************Done*********************************"<<endl;
    cout <<endl;

    if (aRmPoissonMesh)
    {
        aCom = string(SYS_RM) + " " + poissonMesh;
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




