#include <vector>
#include <string>

int PseudoOrtho_main(int argc, char **argv) {

    std::string aNameFileMNT;// un fichier xml de type FileOriMnt pour le MNT
    std::string aFullInputPattern;// pattern des nuages en entree
	std::string aFullImagePattern;// pattern des images correspondant aux nuages
    std::string aNameResult;// un fichier resultat
    
    double resolutionPlani=1.;
    Box2dr aBoxTerrain;
    

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aNameFileMNT,"xml file (FileOriMnt) for the DTM")
                    << EAMC(aFullInputPattern, "Pattern of input Nuages",  eSAM_IsPatFile)
		    << EAMC(aFullImagePattern, "Pattern of corresponding input Images", eSAM_IsPatFile)
                    << EAM(aBoxTerrain,"BoxTerrain",true,"([Xmin,Ymin,Xmax,Ymax])")
                    << EAMC (aNameResult," output ortho filename"),
        LArgMain() << EAM(resolutionPlani,"Resol",true,"output ortho resolution")
     );

    std::cout << "Input MNT : "<<aNameFileMNT<<std::endl;
    std::cout << "Input Nuages : "<<aFullInputPattern<<std::endl;
    std::cout << "Output Ortho : "<<aNameResult<<std::endl;
    std::cout << "Resol Ortho : "<<resolutionPlani<<std::endl;


    std::string aDir,aPatNuages;
    SplitDirAndFile(aDir,aPatNuages,aFullInputPattern);
    std::cout<<"Nuage dir: "<<aDir<<std::endl;
    std::cout<<"Nuage pattern: "<<aPatNuages<<std::endl;

	std::string aDirImages,aPatImages;
	SplitDirAndFile(aDirImages,aPatImages,aFullImagePattern);
	std::cout << "Image dir : "<<aDirImages<<std::endl;
	std::cout << "Image pattern : "<<aPatImages<<std::endl;

	// Chargement des nuages
    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDir);
    std::vector<std::string> aSetNuages = *(aICNM->Get(aPatNuages));
    std::vector<const cElNuage3DMaille *> aVNuages;
    for(size_t i=0;i<aSetNuages.size();++i)
    {
	std::string nom = aDir+aSetNuages[i];
	std::cout << "Nuage "<<i<<" : "<<nom<<std::endl;
	std::cout << "chargement ..."<<std::endl;
        //cResulMSO aRMso =  aICNM->MakeStdOrient(nom,false);
        //aVNuages.push_back(aRMso.Nuage());
	aVNuages.push_back(cElNuage3DMaille::FromFileIm(nom));
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
    int NC = (aBoxTerrain._p1.x - aBoxTerrain._p0.x)/resolutionPlani;
    int NL = (aBoxTerrain._p1.y - aBoxTerrain._p0.y)/resolutionPlani;
    std::cout << "Taille de l'ortho : "<<NC<<" x "<<NL<<std::endl;
    

	// Creation de l'ortho
	Pt2di SzOrtho(NC,NL);
        TIm2D<U_INT2,INT4>* ptrOrtho = new TIm2D<U_INT2,INT4>(SzOrtho);

	// Pour l'instant on cree une ortho superposable au MNT en entree
    for(int l=0;l<NL;++l)
    {
	for(int c=0;c<NC;++c)
	{
		Pt2di ptI;
		ptI.x = c;
		ptI.y = l;

        Pt3dr P3d;
        P3d.x = aBoxTerrain._p0.x + c*resolutionPlani;
        P3d.y = aBoxTerrain._p1.y - l*resolutionPlani;
        
		Pt2dr ptMnt;
		ptMnt.x = (P3d.x - aMntOri.OriginePlani().x)/aMntOri.ResolutionPlani().x;
        	ptMnt.y = (P3d.y - aMntOri.OriginePlani().y)/aMntOri.ResolutionPlani().y;
		std::cout << "Point image (mnt) : "<<c<<" "<<l<<std::endl;
		P3d.z = aMntImg.getr(ptMnt,NoData)*aMntOri.ResolutionAlti() + aMntOri.OrigineAlti();
		//P3d.x = aMntOri.OriginePlani().x + ptMnt.x * aMntOri.ResolutionPlani().x;
        	//P3d.y = aMntOri.OriginePlani().y - ptMnt.y * aMntOri.ResolutionPlani().y;
		std::cout << "Point terrain : "<<P3d.x<<" "<<P3d.y<<" "<<P3d.z<<std::endl;
		int best_image = -1;
		double dmin = 0.;
		Pt2dr best_Pt;
		for(size_t i=0;i<aVNuages.size();++i)
		{
			Pt2dr aPIm, aPCapteur;
            		aPIm = aVNuages[i]->Terrain2Index(P3d);
			std::cout << "Point image dans le nuage "<<i<<" : "<<aPIm.x<<" "<<aPIm.y<<std::endl;
			aPCapteur = aVNuages[i]->ImRef2Capteur (aPIm);
			std::cout << "Point Capteur dans le nuage "<<i<<" : "<<aPCapteur.x<<" "<<aPCapteur.y<<std::endl;
                        if (aVNuages[i]->CaptHasData(aPCapteur))
                        {
                                Pt3dr aP  = aVNuages[i]->PreciseCapteur2Terrain(aPCapteur);
				std::cout <<"Point 3D dans le nuage "<<i<<" : "<<aP.x<<" "<<aP.y<<" "<<aP.z<<std::endl;
                        	double d = aP.z - P3d.z;
				//std::cout << "diff pour le nuage "<<i<<" : "<<d<<std::endl;
				if ((best_image == -1)||(dmin>d))
				{
					best_image = i;
					dmin = d;
					best_Pt = aPIm;
				}
			}
			//else
				//std::cout << "Pas de Z pour ce point dans le nuage"<<std::endl;
		}
		if (best_image != -1)
		{
			std::cout << "Point terrain : "<<P3d.x<<" "<<P3d.y<<" "<<P3d.z<<std::endl;
			std::cout << "Choix du nuage "<<best_image<<" : "<<best_Pt.x<<" "<<best_Pt.y<<" avec une diff de "<<dmin<<std::endl;
			double radio = aVImages[best_image]->getr(best_Pt,0);
			std::cout << "radio : "<<radio<<" "<<(int)radio<<std::endl;
			ptrOrtho->oset(ptI,(int)radio);
		}
		else
			ptrOrtho->oset(ptI,(int)0);
	}
    }

	Tiff_Im out(aNameResult.c_str(), ptrOrtho->sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
        ELISE_COPY(ptrOrtho->_the_im.all_pts(),ptrOrtho->_the_im.in(),out.out());

    return 0;
}
