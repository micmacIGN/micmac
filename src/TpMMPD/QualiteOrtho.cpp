#include "StdAfx.h"
#include <vector>
#include <string>

int QualiteOrtho_main(int argc, char** argv)
{

    bool verbose = false;
    bool stop1 = false;

    std::string aNameFileMNT;// un fichier xml de type FileOriMnt pour le MNT
    std::string aFullMNEPattern;// pattern des nuages en entree
    std::string aFullOrientationPattern;// pattern des images correspondant aux nuages
    std::string aNameResult ;// un fichier resultat
    double resolutionPlani=1.;
    Box2dr aBoxTerrain;
    double OffsetXMNS, OffsetYMNS, ResolMNS;


    ElInitArgMain
    (
        argc, argv,
        LArgMain()  << EAMC(aNameFileMNT,"xml file (FileOriMnt) for the DTM")
                    << EAMC(aFullMNEPattern, "input DSM (image tif + offset ) ou (nuage xml)")
                    << EAMC(aFullOrientationPattern, "input Orientation")
                    << EAMC(aNameResult," output ortho filename"),
        LArgMain()  << EAM(resolutionPlani,"Resol",true,"output ortho resolution")
                    << EAM(aBoxTerrain,"BoxTerrain",true,"([Xmin,Ymin,Xmax,Ymax])")
                    << EAM(ResolMNS,"ResolMNS",true,"pix size MNS")
                    << EAM(OffsetXMNS,"OffsetX",true,"Xmin MNS")
                    << EAM(OffsetYMNS,"OffsetY",true,"Ymin MNS")
     );

    std::cout   <<  "Input MNT : "     <<  aNameFileMNT    <<std::endl;
    std::cout   <<  "Input MNE : "     <<  aFullMNEPattern <<std::endl;
    std::cout   <<  "Output Ortho : "  <<  aNameResult     <<std::endl;
    std::cout   <<  "Resol Ortho : "   <<  resolutionPlani <<std::endl;

    std::string aDirMNE,aPatMNE;
    SplitDirAndFile(aDirMNE,aPatMNE,aFullMNEPattern);
    std::cout<<"Nuage dir: "<<aDirMNE<<std::endl;
    std::cout<<"Nuage pattern: "<<aPatMNE<<std::endl;

    std::string aDirOrientations,aPatOrientations;
    SplitDirAndFile(aDirOrientations,aPatOrientations,aFullOrientationPattern);
    std::cout << "Orientations dir : "<<aDirOrientations<<std::endl;
    std::cout << "Orientation pattern : "<<aPatOrientations<<std::endl;

    // Chargement des orientations
    cInterfChantierNameManipulateur * aICNM2=cInterfChantierNameManipulateur::BasicAlloc(aDirOrientations);
    std::vector<std::string> aSetOrientations = *(aICNM2->Get(aPatOrientations));
    std::vector<CamStenope *> aVOrientations;
    for(size_t i=0;i<aSetOrientations.size();++i)
    {
        std::string nom = aDirOrientations+aSetOrientations[i];
        std::cout << "Image "<<i<<" : "<<nom<<std::endl;
        if (nom.substr(nom.size()-3,nom.size()) == "XML")
            std::cout << "Attention, l'extension des fichiers xml doit etre en minuscule..."<<std::endl;

        std::cout << "chargement ..."<<std::endl;
        //cResulMSO aRMso =  aICNM->MakeStdOrient(nom,false);
        //aVNuages.push_back(aRMso.Nuage());
        aVOrientations.push_back(CamOrientGenFromFile(nom,aICNM2));
    }

    // Chargement des MNS
    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirMNE);
    std::vector<std::string> aSetMNE = *(aICNM->Get(aPatMNE));

    std::vector<const TIm2D<REAL4,REAL8> *> aVMNE_tif;
    std::vector<const cElNuage3DMaille *> aVMNE_xml;

    if ((aSetMNE[0].substr(aSetMNE[0].size()-3,aSetMNE[0].size()) == "xml") || (aSetMNE[0].substr(aSetMNE[0].size()-3,aSetMNE[0].size()) == "XML"))
    {   // Si on a un MNE en nuage
        for(size_t i=0;i<aSetMNE.size();++i)
        {
            std::string nom = aDirMNE+aSetMNE[i];
            std::cout << "MNE "<<i<<" : "<<nom<<std::endl;
            std::cout << "chargement ..."<<std::endl;
            aVMNE_xml.push_back(cElNuage3DMaille::FromFileIm(nom)); // on lit le MNE comme un nuage
        }
    }
    else // Si on a un MNS en tif
    {
        for(size_t i=0;i<aSetMNE.size();++i)
        {
            std::string nom = aDirMNE+aSetMNE[i];
            std::cout << "MNE "<<i<<" : "<<nom<<std::endl;
            std::cout << "chargement ..."<<std::endl;
            aVMNE_tif.push_back(new TIm2D<REAL4,REAL8>(Im2D<REAL4,REAL8>::FromFileStd(nom)));
        }
    }

    // Chargement du MNT
    cFileOriMnt aMntOri=  StdGetFromPCP(aNameFileMNT,FileOriMnt);
    std::cout << "Taille du MNT : "<<aMntOri.NombrePixels().x<<" "<<aMntOri.NombrePixels().y<<std::endl;
    TIm2D<REAL4,REAL8> aMntImg(Im2D<REAL4,REAL8>::FromFileStd(aMntOri.NameFileMnt()));

    double NoData = -9999.;
    int NC = (aBoxTerrain._p1.x - aBoxTerrain._p0.x)/resolutionPlani;
    int NL = (aBoxTerrain._p1.y - aBoxTerrain._p0.y)/resolutionPlani;

    std::cout << "Taille de l'image resultat : "<<NC<<" x "<<NL<<std::endl;
    Pt2di SzOrtho(NC,NL);
    TIm2D<REAL4,REAL8>* ptrQualite = new TIm2D<REAL4,REAL8>(SzOrtho);
    std::cout << "image pre creee"<<std::endl;

/*    Verification de l'origine du MNE
    Pt2di ptOO;
    ptOO.x = 0;
    ptOO.y = 0;
    Pt3dr PtOrigMNE;
    PtOrigMNE = aVMNE[0]->PtOfIndex(ptOO);
    if (verbose)
        std::cout << "PtOrigMNE : " << PtOrigMNE.x << " " << PtOrigMNE.y << std::endl;
*/

    std::cout << "debut balayage"<<std::endl;
    for(int l=0;l<NL;++l)
    {
        for(int c=0;c<NC;++c)
        {
            Pt2di ptI;
            ptI.x = c;
            ptI.y = l;
            if (verbose)
                std::cout << "Point image de qualite : " << c << " " << l << std::endl;

            Pt2dr ptProj;
            ptProj.x = aBoxTerrain._p0.x + c*resolutionPlani;
            ptProj.y = aBoxTerrain._p1.y - l*resolutionPlani;
            if (verbose)
                std::cout << "Point projete : " << ptProj.x << " " << ptProj.y << std::endl;

            //Recuperation du point 3d sur le MNT
            Pt3dr P3d_MNT;
            P3d_MNT.x = ptProj.x;
            P3d_MNT.y = ptProj.y;
            Pt2dr ptMnt;
            ptMnt.x = (P3d_MNT.x - aMntOri.OriginePlani().x)/aMntOri.ResolutionPlani().x;
            ptMnt.y = (P3d_MNT.y - aMntOri.OriginePlani().y)/aMntOri.ResolutionPlani().y;
            P3d_MNT.z = aMntImg.getr(ptMnt,NoData)*aMntOri.ResolutionAlti() + aMntOri.OrigineAlti();
            if (verbose)
            {
                std::cout << "Index MNT    : " << ptMnt.x << " " << ptMnt.y << " " << std::endl;
                std::cout << "Point 3d MNT : " << P3d_MNT.x << " " << P3d_MNT.y << " " << P3d_MNT.z << std::endl;
                std::cout << std::endl;
            }

            /// On sait qu'on a bascule donc le Z n'a pas d'importance
            ///
            Pt3dr P3d_MNE;
            P3d_MNE.x = ptProj.x;
            P3d_MNE.y = ptProj.y;
            P3d_MNE.z = 0;

            Pt2dr ptIndex_mne;
            double z_mne;

            if ((aSetMNE[0].substr(aSetMNE[0].size()-3,aSetMNE[0].size()) == "xml") || (aSetMNE[0].substr(aSetMNE[0].size()-3,aSetMNE[0].size()) == "XML"))
            {   //Nuage
                ptIndex_mne = aVMNE_xml[0]->Terrain2Index(P3d_MNE);
                if (verbose)
                    std::cout << "point Index MNE : " << ptIndex_mne.x << " " << ptIndex_mne.y << std::endl;

                z_mne = aVMNE_xml[0]->ProfOfIndexInterpol(ptIndex_mne);
                if (verbose)
                    std::cout << "Z MNE : " << z_mne <<std::endl;
            }
            else //Image
            {
                ptIndex_mne.x = (P3d_MNE.x - OffsetXMNS)/ResolMNS;
                ptIndex_mne.y = -(P3d_MNE.y - OffsetYMNS)/ResolMNS;
                if (verbose)
                    std::cout << "point Index MNE : " << ptIndex_mne.x << " " << ptIndex_mne.y << std::endl;

                z_mne = aVMNE_tif[0]->getr(ptIndex_mne);
                if (verbose)
                    std::cout << "Z MNE : " << z_mne <<std::endl;
            }

            P3d_MNE.z = z_mne;

            //Retour dans les images
            //Point du MNT
            Pt2dr CoordImgZmnt = aVOrientations[0]->Ter2Capteur(P3d_MNT);
            //Point du MNE
            Pt2dr CoordImgZmne = aVOrientations[0]->Ter2Capteur(P3d_MNE);
            double dist = sqrt(pow((CoordImgZmnt.x-CoordImgZmne.x),2)+pow((CoordImgZmnt.y-CoordImgZmne.y),2));
            if (verbose)
            {
                std::cout << "point image Z MNT : " << CoordImgZmnt.x << " " << CoordImgZmnt.y << std::endl;
                std::cout << "point image Z MNE : " << CoordImgZmne.x << " " << CoordImgZmne.y << std::endl;
                std::cout << std::endl;
                std::cout << "distance : " << dist << std::endl;
            }

            ptrQualite->oset(ptI,dist);

            /* Image de difference de Z
            double Zdiff = abs(P3d_MNE.z-P3d_MNT.z);
            ptrZdiff->oset(ptI,Zdiff);
            */

            if (stop1)
                return 0;
        }
    }

    Tiff_Im out(aNameResult.c_str(), ptrQualite->sz(),GenIm::real4,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
    ELISE_COPY(ptrQualite->_the_im.all_pts(),ptrQualite->_the_im.in(),out.out());
    /* Image de difference de Z
    Tiff_Im out2(aNameResultZ.c_str(), ptrZdiff->sz(),GenIm::real4,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
    ELISE_COPY(ptrZdiff->_the_im.all_pts(),ptrZdiff->_the_im.in(),out2.out());
    */

    return EXIT_SUCCESS;
}
