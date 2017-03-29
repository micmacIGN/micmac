#include "StdAfx.h"
#include <vector>
#include <string>

int QualiteOrtho_main(int argc, char** argv)
{

    bool verbose = false;

    std::string aNameFileMNT;// un fichier xml de type FileOriMnt pour le MNT
    std::string aFullMNEPattern;// pattern des nuages en entree
    std::string aFullImagePattern;// pattern des images correspondant aux nuages
    std::string aNameResult;// un fichier resultat
    double resolutionPlani=1.;
    Box2dr aBoxTerrain;


    ElInitArgMain
    (
        argc, argv,
        LArgMain()  << EAMC(aNameFileMNT,"xml file (FileOriMnt) for the DTM")
                    << EAMC(aFullMNEPattern, "Pattern of input MNS",  eSAM_IsPatFile)
                    << EAMC(aFullImagePattern, "Pattern of corresponding input Images", eSAM_IsPatFile)
                    << EAM(aBoxTerrain,"BoxTerrain",true,"([Xmin,Ymin,Xmax,Ymax])")
                    << EAMC(aNameResult," output ortho filename"),
        LArgMain()  << EAM(resolutionPlani,"Resol",true,"output ortho resolution")
     );

    std::cout << "Input MNT : "<<aNameFileMNT<<std::endl;
    std::cout << "Input MNE : "<<aFullMNEPattern<<std::endl;
    std::cout << "Output Ortho : "<<aNameResult<<std::endl;
    std::cout << "Resol Ortho : "<<resolutionPlani<<std::endl;


    std::string aDirMNE,aPatMNE;
    SplitDirAndFile(aDirMNE,aPatMNE,aFullMNEPattern);
    std::cout<<"Nuage dir: "<<aDirMNE<<std::endl;
    std::cout<<"Nuage pattern: "<<aPatMNE<<std::endl;

    std::string aDirImages,aPatImages;
    SplitDirAndFile(aDirImages,aPatImages,aFullImagePattern);
    std::cout << "Image dir : "<<aDirImages<<std::endl;
    std::cout << "Image pattern : "<<aPatImages<<std::endl;

    // Chargement des MNS
    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirMNE);
    std::vector<std::string> aSetMNE = *(aICNM->Get(aPatMNE));
//    std::vector<const TIm2D<REAL4,REAL8> *> aVMNE;
    std::vector<const cElNuage3DMaille *> aVMNE;
    for(size_t i=0;i<aSetMNE.size();++i)
    {
        std::string nom = aDirMNE+aSetMNE[i];
        std::cout << "MNE "<<i<<" : "<<nom<<std::endl;
        std::cout << "chargement ..."<<std::endl;
//        aVMNE.push_back(new TIm2D<REAL4,REAL8>(Im2D<REAL4,REAL8>::FromFileStd(nom))); // Si on veut lire le MNE comme une image
        aVMNE.push_back(cElNuage3DMaille::FromFileIm(nom)); // on lit le MNE comme un nuage
        std::cout << "ok"<<std::endl;
    }

    // Chargement des images
    cInterfChantierNameManipulateur * aICNM2=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    std::vector<std::string> aSetImages = *(aICNM2->Get(aPatImages));
    std::vector<const TIm2D<U_INT1,INT4> *> aVImages;
    for(size_t i=0;i<aSetImages.size();++i)
    {
        std::string nom = aDirImages+aSetImages[i];
        std::cout << "Image "<<i<<" : "<<nom<<std::endl;
        std::cout << "chargement ..."<<std::endl;
        //cResulMSO aRMso =  aICNM->MakeStdOrient(nom,false);
        //aVNuages.push_back(aRMso.Nuage());
        aVImages.push_back(new TIm2D<U_INT1,INT4>(Im2D<U_INT1,INT4>::FromFileStd(nom)));
        std::cout << "ok"<<std::endl;
    }


    // Chargement du MNT
    cFileOriMnt aMntOri=  StdGetFromPCP(aNameFileMNT,FileOriMnt);
    std::cout << "Taille du MNT : "<<aMntOri.NombrePixels().x<<" "<<aMntOri.NombrePixels().y<<std::endl;

    TIm2D<REAL4,REAL8> aMntImg(Im2D<REAL4,REAL8>::FromFileStd(aMntOri.NameFileMnt()));

    double NoData = -9999.;
//    int NC = aVMNE[0]->_the_im.tx();
//    int NL = aVMNE[0]->_the_im.ty();
    int NC = (aBoxTerrain._p1.x - aBoxTerrain._p0.x)/resolutionPlani;
    int NL = (aBoxTerrain._p1.y - aBoxTerrain._p0.y)/resolutionPlani;

    std::cout << "Taille de l'image resultat : "<<NC<<" x "<<NL<<std::endl;
    Pt2di SzOrtho(NC,NL);
    TIm2D<REAL4,REAL8>* ptrQualite = new TIm2D<REAL4,REAL8>(SzOrtho);
    std::cout << "image pré créée"<<std::endl;

/*
//    TIm2D<REAL4,REAL8> mne = (*aVMNE[0]);
//    Pt2di origMNE = mne.p0();

//    cElNuage3DMaille mne = (*aVMNE[0]);
//    std::cout << "origineProf " << aVMNE[0]->OrigineProf() << std::endl;
//    std::cout << "ResolSolGlob " << aVMNE[0]->ResolSolGlob() << std::endl;
//    return 0;

    Pt2di origMNE ;//= mne.p0();
    origMNE.x = 954179.724999999976717;
    origMNE.y = 6229553.175000000745058;
//    std::cout << "origine plani MNE " << origMNE.x << ";" << origMNE.y << std::endl;
*/

/*
    if (verbose == true)
    {
        NL = 2;
        NC = 2;
    }
*/
    Pt2di ptOO;
    ptOO.x = 0;
    ptOO.y = 0;
    Pt3dr PtOrigMNE;
    PtOrigMNE = aVMNE[0]->PtOfIndex(ptOO);
    if (verbose)
        std::cout << "PtOrigMNE : " << PtOrigMNE.x << " " << PtOrigMNE.y << std::endl;

    std::cout << "début balayage"<<std::endl;
    for(int l=0;l<NL;++l)
    {
        for(int c=0;c<NC;++c)
        {
            Pt2di ptI;
            ptI.x = c;
            ptI.y = l;
            if (verbose)
                std::cout << "Point image : " << c << " " << l << std::endl;

            Pt2dr ptProj;
            ptProj.x = aBoxTerrain._p0.x + c*resolutionPlani;
            ptProj.y = aBoxTerrain._p1.y - l*resolutionPlani;
            if (verbose)
                std::cout << "Point projete : " << ptProj.x << " " << ptProj.y << std::endl;

            //Récupération du point 3d sur le MNT
            Pt3dr P3d_MNT;
            P3d_MNT.x = ptProj.x;
            P3d_MNT.y = ptProj.y;
            Pt2dr ptMnt;
            ptMnt.x = (P3d_MNT.x - aMntOri.OriginePlani().x)/aMntOri.ResolutionPlani().x;
            ptMnt.y = (P3d_MNT.y - aMntOri.OriginePlani().y)/aMntOri.ResolutionPlani().y;
            P3d_MNT.z = aMntImg.getr(ptMnt,NoData)*aMntOri.ResolutionAlti() + aMntOri.OrigineAlti();
            if (verbose)
                std::cout << "Point 3d MNT : " << P3d_MNT.x << " " << P3d_MNT.y << " " << P3d_MNT.z << std::endl;

/*
            //Récupération du point 3d sur le MNE
            Pt3dr P3d_MNE;
            P3d_MNE.x = ptProj.x;
            P3d_MNE.y = ptProj.y;
            P3d_MNE.z = aVMNE[0]->ProjTerrain(P3d_MNE);
            if (verbose)
                std::cout << "Point 3d MNE : " << P3d_MNE.x << " " << P3d_MNE.y << " " << P3d_MNE.z << std::endl;
*/

            Pt2dr P2d_indexMNEviaMNT;
            P2d_indexMNEviaMNT = aVMNE[0]->Terrain2Index(P3d_MNT);
            if (verbose)
                std::cout << "Index MNE via MNT: " << P2d_indexMNEviaMNT.x << " " << P2d_indexMNEviaMNT.y << std::endl;

            Pt3dr P3d_MNEviaMNT;
//            P3d_MNEviaMNT.x = ptProj.x;
//            P3d_MNEviaMNT.y = ptProj.y;
            Pt2di P2di_indexMNEviaMNT;
            P2di_indexMNEviaMNT.x = P2d_indexMNEviaMNT.x;
            P2di_indexMNEviaMNT.y = P2d_indexMNEviaMNT.y;
            P3d_MNEviaMNT = aVMNE[0]->PtOfIndex(P2di_indexMNEviaMNT);
            if (verbose)
                std::cout << "point 3d MNE via MNT: " << P3d_MNEviaMNT.x << " " << P3d_MNEviaMNT.y << " " << P3d_MNEviaMNT.z << std::endl;

            Pt2di ptMne;
            ptMne.x = (ptProj.x - PtOrigMNE.x);
            ptMne.y = (ptProj.y - PtOrigMNE.y);
//            ptMne.x = (ptProj.x - origMNE.x)/0.1;
//            ptMne.y = (ptProj.y - origMNE.y)/0.1;

            //Pt3dr AltiMNE = aVMNE[0]->PtOfIndex(ptMne);
            if (verbose)
                std::cout << "Index MNE : " << ptMne.x << " " << ptMne.y /*<< " " << AltiMNE.z */<< std::endl;
            if (verbose)
                std::cout << std::endl;

            double radio = P3d_MNT.z-P3d_MNEviaMNT.z;
            ptrQualite->oset(ptI,radio);


/*
            Pt3dr P3d_MNEviaMNT;

            if (verbose)
                std::cout << "Point index MNE via MNT: " << P2d_indexMNEviaMNT.x << " " << P2d_indexMNEviaMNT.y << std::endl;
*/
/*
            Pt2dr ptMne;
            ptMne.x = (P3d_MNE.x - origMNE.x)/0.1;
            ptMne.y = (P3d_MNE.y - origMNE.y)/0.1;

//            P3d_MNE.z = aVMNE[0]->getr(ptMne,NoData);
            Pt2di ptMne2;
            ptMne2.x = ptMne.x;
            ptMne2.y = ptMne.y;

//            std::cout << "Calcul du Znuage " << std::endl;
            std::cout << "MNE     " << ptMne.x << " " << ptMne.y << std::endl;
*/
            /*if (aVMNE[0]->CaptHasData(ptMne))
            {
                std::cout << "HasData " << std::endl;
                P3d_MNE.z = aVMNE[0]->ProfEnPixel(ptMne2);
                std::cout << "Lambert " << P3d_MNT.x << " " << P3d_MNT.y << std::endl;
                std::cout << "MNE     " << ptMne.x << " " << ptMne.y << std::endl;
                std::cout << "MNT     " << ptMnt.x << " " << ptMnt.y << std::endl;
                std::cout << "Zmnt " << P3d_MNT.z << " Zmne " << P3d_MNE.z << std::endl;
            }*/

            /*if (P3d_MNT.x == 954470 && P3d_MNT.y == 6229021)
            {
                std::cout << "Lambert " << P3d_MNT.x << " " << P3d_MNT.y << std::endl;
                std::cout << "MNE     " << ptMne.x << " " << ptMne.y << std::endl;
                std::cout << "MNT     " << ptMnt.x << " " << ptMnt.y << std::endl;
                std::cout << "Zmnt " << P3d_MNT.z << " Zmne " << P3d_MNE.z << std::endl;
            }*/
        }
    }


    Tiff_Im out(aNameResult.c_str(), ptrQualite->sz(),GenIm::real4,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
    ELISE_COPY(ptrQualite->_the_im.all_pts(),ptrQualite->_the_im.in(),out.out());

/*
    std::cout << "-----------"<<std::endl;

    Pt2dr ptMne;
    ptMne.x = 5740;
    ptMne.y = 3460;
    bool data = aVMNE[0]->CaptHasData(ptMne);
    std::cout << data << std::endl;

    ptMne.x = 7000;
    ptMne.y = 8000;
    data = aVMNE[0]->CaptHasData(ptMne);
    std::cout << data << std::endl;

*/
    return EXIT_SUCCESS;
}
