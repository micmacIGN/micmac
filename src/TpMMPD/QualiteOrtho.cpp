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
    std::vector<const TIm2D<REAL4,REAL8> *> aVMNE;
    for(size_t i=0;i<aSetMNE.size();++i)
    {
    std::string nom = aDirMNE+aSetMNE[i];
    std::cout << "MNE "<<i<<" : "<<nom<<std::endl;
    std::cout << "chargement ..."<<std::endl;
        //cResulMSO aRMso =  aICNM->MakeStdOrient(nom,false);
        //aVNuages.push_back(aRMso.Nuage());
    aVMNE.push_back(new TIm2D<REAL4,REAL8>(Im2D<REAL4,REAL8>::FromFileStd(nom)));
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

    std::cout << "Taille de l'ortho : "<<NC<<" x "<<NL<<std::endl;
    Pt2di SzOrtho(NC,NL);
    TIm2D<REAL4,REAL8>* ptrQualite = new TIm2D<REAL4,REAL8>(SzOrtho);


    TIm2D<REAL4,REAL8> mne = (*aVMNE[0]);
    Pt2di origMNE = mne.p0();
    origMNE.x = 954179.724999999976717;
    origMNE.y = 6229553.175000000745058;
    std::cout << "origine plani MNE " << origMNE.x << ";" << origMNE.y << std::endl;

    for(int l=0;l<NL;++l)
    {
        for(int c=0;c<NC;++c)
        {
            Pt2di ptI;
            ptI.x = c;
            ptI.y = l;
            if (verbose)
                std::cout << "Point image (mnt) : " << c << " " << l << std::endl;

            //Récupération du point 3d sur le MNT
            Pt3dr P3d_MNT;
            P3d_MNT.x = aBoxTerrain._p0.x + c;
            P3d_MNT.y = aBoxTerrain._p1.y - l;
            Pt2dr ptMnt;
            ptMnt.x = (P3d_MNT.x - aMntOri.OriginePlani().x)/aMntOri.ResolutionPlani().x;
            ptMnt.y = (P3d_MNT.y - aMntOri.OriginePlani().y)/aMntOri.ResolutionPlani().y;

            P3d_MNT.z = aMntImg.getr(ptMnt,NoData)*aMntOri.ResolutionAlti() + aMntOri.OrigineAlti();

            //Récupération du point 3d sur le MNE
            Pt3dr P3d_MNE;
            P3d_MNE.x = P3d_MNT.x;
            P3d_MNE.y = P3d_MNT.y;
//            P3d_MNE.z = aVMNE[0]->getr(ptMne,NoData)*aMntOri.ResolutionAlti() + aMntOri.OrigineAlti();



        }
    }

    return EXIT_SUCCESS;
}
