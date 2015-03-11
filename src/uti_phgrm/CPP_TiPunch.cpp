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

int TiPunch_main(int argc,char ** argv)
{
    bool verbose = true;

    std::string aDir, aPat, aFullName, aOri, aPly, aOut;
    bool aBin = true;
    bool aRmPoissonMesh = false;
    int aDepth = 8;
    double aDst = 1.f;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullName,"Full Name (Dir+Pat)",eSAM_IsPatFile)
                            << EAMC(aOri,"Orientation path",eSAM_IsExistDirOri)
                            << EAMC(aPly,"Ply file", eSAM_IsExistFile),
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

    int nbProc = NbProcSys();
    stringstream sst;
    sst << nbProc;

    std::string poissonMesh = StdPrefix(aPly) + "_poisson_depth" + ss.str() +".ply";

    std::string aCom = g_externalToolHandler.get( "PoissonRecon" ).callName()
            + std::string(" --in ") + aPly.c_str()
            + std::string(" --out ") + poissonMesh.c_str()
            + " --depth " + ss.str()
    //#if USE_OPEN_MP //TODO: � activer quand WITH_OPEN_MP=1
            + " --threads " + sst.str()
    //#endif
    ;

    if (verbose) cout << "Com= " << aCom << endl;

    cout << "\nRunning Poisson reconstruction" << endl;

    system_call(aCom.c_str());

    cout << "\nMesh built and saved in " << poissonMesh << endl;

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    std::list<std::string>  aLS = aICNM->StdGetListOfFile(aPat);

    StdCorrecNameOrient(aOri,aDir);

    std::vector<CamStenope*> ListCam;

    cout << endl;
    for (std::list<std::string>::const_iterator itS=aLS.begin(); itS!=aLS.end() ; itS++)
    {
        std::string NOri=aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aOri,*itS,true);

        ListCam.push_back(CamOrientGenFromFile(NOri,aICNM));

        cout <<"Image "<<*itS<<", with ori : "<< NOri <<endl;
    }

    //TODO: charger Z_Num / Depth-img

    Im2D_REAL4 * pImg = new Im2D_REAL4(1000,1000,0.f);

    cMesh myMesh(poissonMesh);

    const int nbFaces = myMesh.getFacesNumber();
    for(int aK=0; aK <nbFaces; aK++)
    {
        cTriangle * Triangle = myMesh.getTriangle(aK);

        vector <Pt3dr> Vertex;
        Triangle->getVertexes(Vertex);

        const int nCam = ListCam.size();
        for(int bK=0 ; bK<nCam; bK++)
        {
            CamStenope* Cam = ListCam[bK];

            Pt2dr Pt1 = Cam->R3toF2(Vertex[0]);
            Pt2dr Pt2 = Cam->R3toF2(Vertex[1]);
            Pt2dr Pt3 = Cam->R3toF2(Vertex[2]);

            if (Cam->IsInZoneUtile(Pt1) && Cam->IsInZoneUtile(Pt2) && Cam->IsInZoneUtile(Pt3))
            {
                Pt2dr _min(DBL_MAX, DBL_MAX);
                Pt2dr _max;

                _min = Inf(Pt1, Inf(Pt2, Inf(Pt3, _min)));
                _max = Sup(Pt1, Sup(Pt2, Sup(Pt3, _max)));

                bool found = false;
                //Parcours de l'image de profondeur
                for(int cK= _min.x; cK < _max.x; cK++)
                {
                    for(int dK=_min.y; dK < _max.y; dK++)
                    {
                        if (pImg->GetR(Pt2di(cK,dK)) < aDst) found = true;
                    }
                }

                if (!found) myMesh.removeTriangle(*Triangle);
            }
        }
    }

    myMesh.write(aOut, aBin);

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




