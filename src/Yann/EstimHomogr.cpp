


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
#include "hassan/reechantillonnage.h"

// --------------------------------------------------------------------------------------
// Classes permettant d'estimer, d'appliquer et d'inverser les points de liaison d'un 
// homographie (calculée à partir d'un jeu de points terrain <-> image).
// --------------------------------------------------------------------------------------
class cAppli_YannEstimHomog{
	
    public :
       cAppli_YannEstimHomog(int argc, char ** argv);

       std::string mName3D;                        // Fichier des mesures 3D
       std::string mName2D;                        // fichier des mesures 2D
       std::string mNameImRef;                     // Image référence (optionnelle)
       std::string mDirOri;                        // Direction d'orientation
	   std::string mResolution;                    // Resolution images de sortie
	   std::string mMargeRel;                      // Marge relative des bords
	   std::string aNameCam;					   // Chemin calibration interne
	   bool mCamRot;                               // Camera frame rotation
       cElemAppliSetFile  mEASF;                   // Pour gerer un ensemble d'images
       cInterfChantierNameManipulateur * mICNM;    // Name manipulateur
       std::string                       mDir;     // Directory of data
       cDicoAppuisFlottant               mDAF3D;   // All 3D
	   cSetOfMesureAppuisFlottants       mDAF2D;   // All 2D
};
// --------------------------------------------------------------------------------------
class cAppli_YannApplyHomog{
	
    public :
       cAppli_YannApplyHomog(int argc, char ** argv);

       std::string mNameIm;                        // Images à convertir
       std::string mNameHomog;                     // Fichier d'homographie
	   std::string mDir;                           // Répertoire d'images
	   std::string mDirOut;                        // Répertoire de sortie
       cElemAppliSetFile  mEASF;                   // Pour gerer un ensemble d'images
       cInterfChantierNameManipulateur * mICNM;    // Name manipulateur
};
// --------------------------------------------------------------------------------------
class cAppli_YannInvHomolHomog{
	
    public :
       cAppli_YannInvHomolHomog(int argc, char ** argv);

	   std::string ImPattern;                      // Images redressées
	   std::string ImPatternInit;				   // Images initiales
       std::string mFolderIn;                      // Dossier input
       std::string mFolderOut;                     // Dossier output
	   std::string mHomogFile;                     // Fichier d'homographie
	   std::string mExt;                           // Extension d'image cible
	   std::string exportTxtIn;                    // Entrée en binaire ou txt
	   std::string exportTxtOut;                   // Sortie en binaire ou txt
       std::string mDir;                           // Directory of data
	   std::string mDirOri;                        // Répertoire d'orientation
	   std::string mRawFolder;					   // Dossier des images bruts
	   cElemAppliSetFile mEASF;                    // Pour gerer un ensemble d'images
	   cInterfChantierNameManipulateur * mICNM;    // Name manipulateur

};


// --------------------------------------------------------------------------------------
// Fonction de lecture du fichier d'homographie
// --------------------------------------------------------------------------------------
// Inputs :
//   - nom du fichier à lire
//   - matrice des paramètres d'homographie
//   - emprise xmin, ymin, xmax, ymax
//   - nombre de pixels en x de l'image
// --------------------------------------------------------------------------------------
void readHomogFile(std::string mHomogFile, ElMatrix<REAL> &H, double &xmin, double &ymin, double &xmax, 
				   double &ymax, int &Nx_org, int &Ny_org, int &Nx, int &Ny, std::string &dirOri){

	std::string line;
	std::ifstream infile(mHomogFile);
	
	// Test existence du fichier
	std::string message = "ERROR: can't find homography file [" + std::string(mHomogFile) + "]";
	ELISE_ASSERT(infile.good(), message.c_str());
	
	std::getline(infile, line);
	
	for (int i=0; i<3; i++){
		
		std::getline(infile, line);
		
		for (int j=0; j<3; j++){
			std::getline(infile, line);
			line = line.substr(10,30);
			line = line.substr(0, line.size()-5);
    		H(0,3*i+j) = std::stof(line);
		}
	
		std::getline(infile, line);
	}
	
	std::getline(infile, line);
	std::getline(infile, line);
	
	std::getline(infile, line);
	line = line.substr(6,line.size()-10);
	xmin = std::stof(line);
	
	std::getline(infile, line);
	line = line.substr(6,line.size()-10);
	ymin = std::stof(line);
	
	std::getline(infile, line);
	std::getline(infile, line);
	
	std::getline(infile, line);
	line = line.substr(6,line.size()-10);
	xmax = std::stof(line);
	
	std::getline(infile, line);
	line = line.substr(6,line.size()-10);
	ymax = std::stof(line);
	
	std::getline(infile, line);
	std::getline(infile, line);
	
	std::getline(infile, line);
	Nx_org = std::stof(line.substr(6,line.size()-10));
	std::getline(infile, line);
	Ny_org = std::stof(line.substr(6,line.size()-10));
	
	std::getline(infile, line);
	std::getline(infile, line);
	
	std::getline(infile, line);
	Nx = std::stof(line.substr(6,line.size()-10));
	std::getline(infile, line);
	Ny = std::stof(line.substr(6,line.size()-10));
	
	std::getline(infile, line);
	std::getline(infile, line);
	dirOri = line.substr(7,line.size()-15);
	
	infile.close();
	
}

// --------------------------------------------------------------------------------------
// Fonction d'inversion de l'homographie
// --------------------------------------------------------------------------------------
// Inputs :
//   - pattern des images homographiées
//   - pattern des images initiales
//   - fichier des paramètres d'homographie
//   - dossier d'orientation des images initiales
//   - dossier des points homologues transformés (defaut : nom_input + "InvHomog")
//   - arguments de types de points (binaire ou txt) et d'images produites
// --------------------------------------------------------------------------------------
// Outputs :
//   - points homologues dans l'espace image de depart
// --------------------------------------------------------------------------------------
cAppli_YannInvHomolHomog::cAppli_YannInvHomolHomog(int argc, char ** argv){
	
	ElInitArgMain(argc,argv,
        LArgMain()  <<  EAMC(ImPattern,"Rectified images pattern")  
				    <<  EAMC(mRawFolder,"Raw images folder")
				    <<  EAMC(mHomogFile,"Homography parameters file"),
        LArgMain()  <<  EAM(mFolderOut,"Out", "NONE", "Homol output folder")
				    <<  EAM(exportTxtIn, "ExpTxtIn", "0", "Input in txt")
				    <<  EAM(exportTxtOut, "ExpTxtOut", "0", "Output in txt")
		        	<<  EAM(mExt, "Ext", "NONE", "Target image extension"));

	
	std::string sep =  "-----------------------------------------------------------------------";
	
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "                  HOMOGRAPHIE INVERSE TRANSFORMATION                   " << std::endl;
	std::cout << "-----------------------------------------------------------------------" << std::endl;

	
	// ---------------------------------------------------------------
	// Lecture des paramètres d'homographie
	// ---------------------------------------------------------------
	ElMatrix<REAL> H(1,9,0.0);
	
	int Nx; int Ny;
	int Nx_org; int Ny_org;
	double xmin; double ymin;
	double xmax; double ymax;
	std::string aNameCam;
	
	readHomogFile(mHomogFile, H, xmin, ymin, xmax, ymax, Nx_org, Ny_org, Nx, Ny, aNameCam);
	
	double resolution = (xmax-xmin)/(Nx-1);
	
	// ---------------------------------------------------------------
	// Correction éventuelle de la distorsion
	// ---------------------------------------------------------------
	
	CamStenope * aCam = 0;
	
	if (aNameCam.size() > 0){
    	cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
    	aCam = CamOrientGenFromFile(mRawFolder+"/"+aNameCam,anICNM);
	}
	
	// ---------------------------------------------------------------
	// Gestion du répertoire de sortie
	// ---------------------------------------------------------------
	std::string output_folder = "InvHomog";
	if (EAMIsInit(&mFolderOut)){
		output_folder = mFolderOut;
	}

	
	// ---------------------------------------------------------------
	// Impression console pour confirmation des paramètres
	// ---------------------------------------------------------------
	printf ("H = %10.3f %10.3f %10.3f    BBOX = %7.2f %7.2f \n", H(0,0), H(0,1), H(0,2), xmin, xmax);
	printf ("    %10.3f %10.3f %10.3f           %7.2f %7.2f \n", H(0,3), H(0,4), H(0,5), ymin, ymax);
	printf ("    %10.3f %10.3f %10.3f      \n", H(0,6), H(0,7), 1.0);
	printf ("GENERATED IMAGE SIZE: [%i x %i]   RAW IMAGE SIZE:  [%i x %i] \n", Nx, Ny, Nx_org, Ny_org);
	
	std::cout << sep << std::endl;
	
	cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
	
	bool ExpTxtIn = false;
	bool ExpTxtOut = false;
	
	if (EAMIsInit(&exportTxtIn)){
		ExpTxtIn  = ((exportTxtIn  == "1") || (exportTxtIn  == "true")  || (exportTxtIn  == "T"));
	}
	if (EAMIsInit(&exportTxtOut)){
		ExpTxtOut = ((exportTxtOut == "1") || (exportTxtOut == "true") || (exportTxtOut == "T"));
	}
	
	std::string aPostIn= "";

	std::string anExtIn = ExpTxtIn ? "txt" : "dat";
    std::string anExtOut = ExpTxtOut ? "txt" : "dat";
	
	std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(aPostIn)
                       +  std::string("@")
                       +  std::string(anExtIn);
    std::string aKHOut =  std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(output_folder)
                       +  std::string("@")
                       +  std::string(anExtOut);
	
	double x1, y1, x1h, y1h, d1; 
	double x2, y2, x2h, y2h, d2;
	
	int count = 0;
	int count_warning_domain_error = 0;

	const std::vector<std::string> *  aVN = anICNM->Get(ImPattern);
	for (int aKN1=0; aKN1<int(aVN->size()); aKN1++){
		
		std::string aNameIm1 = (*aVN)[aKN1];
		std::cout << aNameIm1 << ":" << std::endl;
       
		for (int aKN2=0; aKN2<int(aVN->size()); aKN2++){ 
			
            std::string aNameIm2 = (*aVN)[aKN2];
			
			// Récupération des fichiers homologues
			std::string aNameIn = "./" + anICNM->Assoc1To2(aKHIn,aNameIm1,aNameIm2,true);
			std::string aNameOut = "./" + anICNM->Assoc1To2(aKHOut,aNameIm1,aNameIm2,true);
			
			if (EAMIsInit(&mExt)){
				aNameOut = "./" + anICNM->Assoc1To2(aKHOut,StdPrefix(aNameIm1)+"."+mExt,StdPrefix(aNameIm2)+"."+mExt,true);
			}

            bool ExistFileIn =  ELISE_fp::exist_file(aNameIn);
			
			if (ExistFileIn){
			
				ElPackHomologue aPackIn =  ElPackHomologue::FromFile(aNameIn);
                ElPackHomologue aPackOut;
				
				for (ElPackHomologue::const_iterator itP=aPackIn.begin(); itP!=aPackIn.end() ; itP++){
					
					Pt2dr aP1 = itP->P1();
                	Pt2dr aP2 = itP->P2();
					
					x1 = xmin + aP1.x*resolution; y1 = ymax - aP1.y*resolution;
					x2 = xmin + aP2.x*resolution; y2 = ymax - aP2.y*resolution;
					
					d1 = H(0,6)*x1 + H(0,7)*y1 + 1.0;
					x1h = (H(0,0)*x1 + H(0,1)*y1 + H(0,2))/d1;
					y1h = (H(0,3)*x1 + H(0,4)*y1 + H(0,5))/d1;
					
					d2 = H(0,6)*x2 + H(0,7)*y2 + 1.0;
					x2h = (H(0,0)*x2 + H(0,1)*y2 + H(0,2))/d2;
					y2h = (H(0,3)*x2 + H(0,4)*y2 + H(0,5))/d2;
					
					if ((x1h < 0) || (y1h < 0) || (x2h < 0) || (y1h < 0)){
						count_warning_domain_error ++;
						continue;
					}
					
					if ((x1h > Nx_org) || (y1h > Ny_org) || (x2h > Nx_org) || (y1h > Ny_org)){
						count_warning_domain_error ++;
						continue;
					}
					
					Pt2dr pt1 = Pt2dr(x1h,y1h);
					Pt2dr pt2 = Pt2dr(x2h,y2h);
					
					// Distorsion
					if (aNameCam.size() > 0){
						pt1 = aCam->DistDirecte(pt1);
						pt2 = aCam->DistDirecte(pt2);
					}
					
					ElCplePtsHomologues aCple(pt1, pt2, itP->Pds());
					aPackOut.Cple_Add(aCple);
					
				}
				
				count += aPackOut.size();
				aPackOut.StdPutInFile(aNameOut);
				std::cout << "   " << aNameIm2 << " (" << aPackOut.size() << " pts)" << std::endl;
					
			}
	
		}
		
	}
	
	std::cout << sep << std::endl;
	std::cout << count << " tie points transformed into " << "[Homol" << output_folder << "]" << std::endl;
	if (count_warning_domain_error > 0){
		int count_rel = (100*count_warning_domain_error)/(count + count_warning_domain_error);
		std::cout << "Warning: " << count_warning_domain_error;
		std::cout << " (" << count_rel <<" %) points out of image frame removed" << std::endl;
	}
	std::cout << sep << std::endl;
	
}


// --------------------------------------------------------------------------------------
// Fonction d'application de l'homographie
// --------------------------------------------------------------------------------------
// Inputs :
//   - string: Pattern d'images sur lesquelles appliquer la transformation
//   - string: Fichier des paramètres d'homographie
//   - string: Répertoire de sortie (optionnel) pour la création des images
// --------------------------------------------------------------------------------------
// Outputs :
//   - Les images transformées par l'homographie
// --------------------------------------------------------------------------------------
cAppli_YannApplyHomog::cAppli_YannApplyHomog(int argc, char ** argv){

	 ElInitArgMain(argc,argv,
        LArgMain()  <<  EAMC(mNameIm,"Images pattern")
				    <<  EAMC(mNameHomog,"Homography parameters file"),
        LArgMain()  <<  EAM(mDirOut,"Out", "./", "Output folder"));

	
	std::string sep =  "-----------------------------------------------------------------------";
	
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "                      HOMOGRAPHIE TRANSFORMATION                       " << std::endl;
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	
	// ---------------------------------------------------------------
	// Lecture des images
	// ---------------------------------------------------------------
	if (EAMIsInit(&mNameIm)){
		mEASF.Init(mNameIm);
        mICNM = mEASF.mICNM;
        mDir = mEASF.mDir;
	}
	
	unsigned N = mEASF.SetIm()->size();
	
	// ---------------------------------------------------------------
	// Gestion du répertoire de sortie
	// ---------------------------------------------------------------
	std::string output_folder = "./";
	if (EAMIsInit(&mDirOut)){
		output_folder = mDirOut;
		ELISE_fp::MkDir(output_folder);
	}

	// ---------------------------------------------------------------
	// Lecture des paramètres d'homographie
	// ---------------------------------------------------------------
	ElMatrix<REAL> H(1,9,0.0);
	
	int Nx; int Ny;
	int Nx_org; int Ny_org;
	double xmin; double ymin;
	double xmax; double ymax;
	std::string aNameCam;
	
	readHomogFile(mNameHomog, H, xmin, ymin, xmax, ymax, Nx_org, Ny_org, Nx, Ny, aNameCam);

	// ---------------------------------------------------------------
	// Impression console pour confirmation des paramètres
	// ---------------------------------------------------------------
	printf ("H = %10.3f %10.3f %10.3f    BBOX = %7.2f %7.2f \n", H(0,0), H(0,1), H(0,2), xmin, xmax);
	printf ("    %10.3f %10.3f %10.3f           %7.2f %7.2f \n", H(0,3), H(0,4), H(0,5), ymin, ymax);
	printf ("    %10.3f %10.3f %10.3f      \n", H(0,6), H(0,7), 1.0);
	printf ("GENERATED IMAGE SIZE: [%i x %i]   RAW IMAGE SIZE:  [%i x %i] \n", Nx, Ny, Nx_org, Ny_org);
	std::cout << sep << std::endl;
	
	// ---------------------------------------------------------------
	// Liste des images a transformer
	// ---------------------------------------------------------------
	std::string aNameIn;
	std::string aNameOut;
	std::string aNameFileOut;
	std::cout << "Transformation of images (" << N << ")" << std::endl;
	std::cout << sep << std::endl;
	
	CamStenope * aCam = 0;
	
	// Correction éventuelle de la distorsion
	if (aNameCam.size() > 0){
    	cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
    	aCam = CamOrientGenFromFile(aNameCam,anICNM);
	}
	
	ElMatrix<REAL> PATTERN_X(Nx,Ny,0.0);
	ElMatrix<REAL> PATTERN_Y(Nx,Ny,0.0);
	
	double D;
	double X; 
	double Y;
	
	for (int iy=0; iy<Ny; iy++) {
		Y = ymax - iy*(ymax-ymin)/Ny;
        for (int ix=0; ix<Nx; ix++) {
			X = xmin + ix*(xmax-xmin)/Nx;
			
			// Homographie
            D = H(0,6)*X + H(0,7)*Y + 1.0;
			PATTERN_X(ix,iy) = (H(0,0)*X + H(0,1)*Y + H(0,2))/D;
			PATTERN_Y(ix,iy) = (H(0,3)*X + H(0,4)*Y + H(0,5))/D;
			
			// Distorsion éventuelle
			if (aNameCam.size() > 0){
				Pt2dr aCenterOut = aCam->DistDirecte(Pt2dr(PATTERN_X(ix,iy), PATTERN_Y(ix,iy)));
				PATTERN_X(ix,iy) = aCenterOut.x;
				PATTERN_Y(ix,iy) = aCenterOut.y;	
			}	
		}
	}
	
	for (unsigned i=0; i<N; i++){
		
		aNameIn = mEASF.SetIm()[0][i];
		aNameFileOut = aNameIn.substr(0, aNameIn.size()-4);
		aNameFileOut = aNameFileOut + ".tif";
		aNameOut = output_folder + "/" + aNameFileOut;
		Tiff_Im aTF = Tiff_Im::StdConvGen(aNameIn,3,false);
		Pt2di im_size = aTF.sz();
		
		Im2D_U_INT1  aImR(im_size.x,im_size.y);
   	 	Im2D_U_INT1  aImG(im_size.x,im_size.y);
    	Im2D_U_INT1  aImB(im_size.x,im_size.y);
    	Im2D_U_INT1  aImROut(Nx,Ny);
    	Im2D_U_INT1  aImGOut(Nx,Ny);
    	Im2D_U_INT1  aImBOut(Nx,Ny);
		
		 ELISE_COPY(
       		aTF.all_pts(),
       		aTF.in(),
       		Virgule(aImR.out(),aImG.out(),aImB.out())
    	);

    	U_INT1 ** aDataR = aImR.data();
    	U_INT1 ** aDataG = aImG.data();
    	U_INT1 ** aDataB = aImB.data();
    	U_INT1 ** aDataROut = aImROut.data();
    	U_INT1 ** aDataGOut = aImGOut.data();
    	U_INT1 ** aDataBOut = aImBOut.data();
	
		
		Pt2dr ptIn; 
				
   	 	for (int iy=0; iy<Ny; iy++) {
        	for (int ix=0; ix<Nx; ix++) {
				
				ptIn.x = PATTERN_X(ix,iy);
				ptIn.y = PATTERN_Y(ix,iy);
				
				// Interpolation bilinéaire
            	aDataROut[iy][ix] = Reechantillonnage::biline(aDataR, im_size.y, im_size.x, ptIn);
            	aDataGOut[iy][ix] = Reechantillonnage::biline(aDataG, im_size.y, im_size.x, ptIn);
            	aDataBOut[iy][ix] = Reechantillonnage::biline(aDataB, im_size.y, im_size.x, ptIn);
				
       	 	}
    	}
		
		// Impression image
		Pt2di aSzOut = Pt2di(Nx, Ny);
		Tiff_Im aTOut(aNameOut.c_str(), aSzOut, GenIm::u_int1, Tiff_Im::No_Compr, Tiff_Im::RGB);
		ELISE_COPY(
			aTOut.all_pts(), 
			Virgule(aImROut.in(), aImGOut.in(), aImBOut.in()), 
			aTOut.out());
		
		std::cout << "Image " << aNameFileOut << " generated  [" << (i+1) << "/" << N << "]" << std::endl;
		
	}
	
};


// --------------------------------------------------------------------------------------
// Fonction principale de calcul de l'homographie
// --------------------------------------------------------------------------------------
// Inputs :
//   - string: Fichier de type "mesure-S3D.xml" de points dans un repère terrain
//   - string: Fichier de type "mesure-S2D.xml" de points mesurés dans plusieurs images
//   - string: Répertoire d'orientation (optionnel) pour rotation dans un repère caméra
//   - string: Nom de l'image centrale (optionnel) pour forcer le choix de la référence
//   - int: Résolution en x des images (défaut: 2000)
//   - double: marge (entre 0 et 1) à appliquer sur les bords (défaut: 0.3))
// --------------------------------------------------------------------------------------
// Outputs :
//   - Un fichier xml contenant les 9 paramètres de l'homographie calculée + emprise
//   - Affichage à l'écran des résidus (d'homographie et de recalage sur la plan)
// --------------------------------------------------------------------------------------
// Note : la fonction nécessite qu'au moins une image de mesure-S2D.xml contienne un 
// minimum de 4 points mesurés. Lorsque l'image centrale n'est pas définie, le programme
// récupère automatiquement l'image contenant le plus de points dans mesure-S2D.xml.
// --------------------------------------------------------------------------------------

cAppli_YannEstimHomog::cAppli_YannEstimHomog(int argc, char ** argv){
	
    ElInitArgMain(argc,argv,
        LArgMain()  <<  EAMC(mName3D,"Ground 3D points coordinates file")
                    <<  EAMC(mName2D,"Image 2D points measurement file"),
        LArgMain()  <<  EAM(mDirOri,"Ori", "NONE", "Orientation folder")
				    <<  EAM(mNameImRef,"ImRef", "NONE", "Name of reference image")
				  	<<  EAM(mResolution,"ImRes", "2000", "Output resolution of images")
				    <<  EAM(mMargeRel,"Margin", "0.3", "Ground 3D points margin")
				    <<  EAM(mCamRot,"CamRot", "True", "Camera frame rotation"));


	
	std::string sep =  "-----------------------------------------------------------------------";
	
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "                        HOMOGRAPHIE ESTIMATION                         " << std::endl;
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	
	// ---------------------------------------------------------------
	// Si l'image de référence est définie
	// ---------------------------------------------------------------
	
	if (EAMIsInit(&mNameImRef)){
		
		mEASF.Init(mNameImRef);
        mICNM = mEASF.mICNM;
        mDir = mEASF.mDir ;
		
		// Test image centrale ne contient qu'une image (si définie)
		ELISE_ASSERT((mEASF.SetIm()->size() == 1), "ImRef must refer to exactly 1 image");
		
	}
	
	// ---------------------------------------------------------------
	// Cohérence dossier d'orientation
	// ---------------------------------------------------------------
	if (EAMIsInit(&mCamRot)){
		ELISE_ASSERT(EAMIsInit(&mDirOri), "No orientation folder for camera frame rotation")
	}
	
		
	// ---------------------------------------------------------------
	// Gestion de la marge
	// ---------------------------------------------------------------
	double marge = 0.3;
	if (EAMIsInit(&mMargeRel)){
		marge = std::stof(mMargeRel);
	}
	
	// ---------------------------------------------------------------
	// Gestion de la résolution de sortie
	// ---------------------------------------------------------------
	int resolution = 2000;
	if (EAMIsInit(&mResolution)){
		resolution = std::stoi(mResolution);
	}
	
	
	// ---------------------------------------------------------------
	// Récupération des points dans le repère image (2D)
	// ---------------------------------------------------------------
	mDAF2D = StdGetFromPCP(mName2D, SetOfMesureAppuisFlottants);
	std::cout << "Number of images in measurement file [" << mName2D;
	std::cout << "]: " << mDAF2D.MesureAppuiFlottant1Im().size() << std::endl;


	// Conversion liste -> vecteur
	std::vector<cMesureAppuiFlottant1Im> MESURE_IMAGES;
	for (std::list<cMesureAppuiFlottant1Im>::const_iterator itM= mDAF2D.MesureAppuiFlottant1Im().begin();
         itM != mDAF2D.MesureAppuiFlottant1Im().end();
         itM++){
	 	MESURE_IMAGES.push_back(*itM);
	}
	

	// ---------------------------------------------------------------
	// Recherche de l'image ayant le plus de points
	// ---------------------------------------------------------------
	
	unsigned max = 0;
	unsigned val = 0;
	unsigned argmax = 0;
	int selected = -1;

	
	for (unsigned i=0; i<MESURE_IMAGES.size(); i++){
		
		val = MESURE_IMAGES[i].OneMesureAF1I().size();
		if (val > max){
			max = val;
			argmax = i;
		}
		
		if (EAMIsInit(&mNameImRef)){
			if (MESURE_IMAGES[i].NameIm() == mNameImRef){
				selected = i;
			}
		}
		
		if (MESURE_IMAGES[i].OneMesureAF1I().size() >= 4){
			std::cout << MESURE_IMAGES[i].NameIm() << ": ";
			std::cout << MESURE_IMAGES[i].OneMesureAF1I().size();
			std::cout << " pts" << std::endl;
		}
		
	}
	
	if (EAMIsInit(&mNameImRef)){
		ELISE_ASSERT(selected != -1, "ImRef is not referenced in measurement file");
	}
	
	
	if (!EAMIsInit(&mNameImRef)){
		mNameImRef = MESURE_IMAGES[argmax].NameIm();
		selected = argmax;
	}
	
	std::cout << "SELECTED IMAGE: " << mNameImRef << "  ";
	std::cout << "(" << MESURE_IMAGES[selected].OneMesureAF1I().size();
	std::cout << " pts)"<< std::endl;
	
	// Lecture taille de l'image
	Tiff_Im aTF = Tiff_Im::StdConvGen(mNameImRef,3,false);
	Pt2di im_size_origin = aTF.sz();
	int Nx_org = im_size_origin.x;
	int Ny_org = im_size_origin.y;

	// Test nombre de points suffisant
	const int N = MESURE_IMAGES[selected].OneMesureAF1I().size();
	std::string tmp = "ERROR: not enough points in image measurement file (" + std::to_string(N) + ")";

	ELISE_ASSERT(N >= 4, tmp.c_str());
	
	
    // Récupération des points 2D
	std::vector<double> X2D;
	std::vector<double> Y2D;
	std::vector<string> NAME2D;
	for (std::list<cOneMesureAF1I>::iterator i=MESURE_IMAGES[selected].OneMesureAF1I().begin(); 
		i != MESURE_IMAGES[selected].OneMesureAF1I().end(); i++){
		X2D.push_back(i->PtIm().x);
		Y2D.push_back(i->PtIm().y);
		NAME2D.push_back(i->NamePt());
	}
	
	std::cout << sep << std::endl;
	
	// ---------------------------------------------------------------
	// Récupération des points dans le repère terrain (3D)
	// ---------------------------------------------------------------
	mDAF3D = StdGetFromPCP(mName3D, DicoAppuisFlottant);
	std::cout << "Number of 3D ground points in [" << mName3D;
	std::cout << "]: " << mDAF3D.OneAppuisDAF().size() << std::endl;
	
	// Récupération des points 3D
	std::vector<double> X3D;
	std::vector<double> Y3D;
	std::vector<double> Z3D;
	std::vector<string> NAME3D;
	for (std::list<cOneAppuisDAF>::const_iterator itT= mDAF3D.OneAppuisDAF().begin();
         itT != mDAF3D.OneAppuisDAF().end();
         itT++){
		X3D.push_back(itT->Pt().x);
		Y3D.push_back(itT->Pt().y);
		Z3D.push_back(itT->Pt().z);
		NAME3D.push_back(itT->NamePt());
	 }


	// ---------------------------------------------------------------
	// Appariemment des points IMAGE <-> TERRAIN
	// ---------------------------------------------------------------
	std::vector<double> P2D[2];
	std::vector<double> P3D[3];
	
	for (unsigned i=0; i<X3D.size(); i++){
		for (unsigned j=0; j<X2D.size(); j++){
			if (NAME3D[i] == NAME2D[j]){
				P2D[0].push_back(X2D[j]);
				P2D[1].push_back(Y2D[j]);
				P3D[0].push_back(X3D[i]);
				P3D[1].push_back(Y3D[i]);
				P3D[2].push_back(Z3D[i]);
				printf ("IMAGE: %9.3f %9.3f     ", X2D[j], Y2D[j]);
				printf ("TERRAIN: %5.3f %5.3f %5.3f\n", X3D[i], Y3D[i], Z3D[i]);
			}
		}
	}
	
	std::cout << sep << std::endl;
	
	// ---------------------------------------------------------------
	// Rotation de la prise de vue + distorsions + coplanarisation
	// ---------------------------------------------------------------
	if (EAMIsInit(&mDirOri)){
		
		StdCorrecNameOrient(mDirOri,mDir);
		
		// Correction de la distorsion
		aNameCam="Ori-"+mDirOri+"/Orientation-"+mNameImRef+".xml";
    	cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
    	CamStenope * aCam = CamOrientGenFromFile(aNameCam,anICNM);
		std::cout << "Distorsion correction and rotation in camera frame from file: ";
		std::cout << aNameCam << std::endl;
		
		for (unsigned i=0; i<P2D[0].size(); i++){
			Pt2dr aCenterOut = aCam->DistInverse(Pt2dr(P2D[0][i], P2D[1][i]));
			P2D[0][i] = aCenterOut.x;
			P2D[1][i] = aCenterOut.y;
		}
		
		
		// Rotation + coplanarisation
		ElMatrix<REAL> ROT(3,3,0.0);
		if ((!EAMIsInit(&mCamRot)) || (mCamRot)){
			cOrientationConique aOriConique=StdGetFromPCP(aNameCam,OrientationConique); 

			ROT = MatFromCol(
				aOriConique.Externe().ParamRotation().CodageMatr().Val().L1(),
				aOriConique.Externe().ParamRotation().CodageMatr().Val().L2(),
				aOriConique.Externe().ParamRotation().CodageMatr().Val().L3()
			).transpose();
			
			// Rotation des points du repère terrain
			for (unsigned i=0; i<P3D[0].size(); i++){
				ElMatrix<REAL> P(1,3,0.0);
				ElMatrix<REAL> P_ROT(1,3,0.0);
				P(0,0) = P3D[0][i]; P(0,1) = P3D[1][i];P(0,2) = P3D[2][i];
				P_ROT = ROT*P;
				P3D[0][i] = P_ROT(0,0); P3D[1][i] = P_ROT(0,1); P3D[2][i] = P_ROT(0,2);
			}
		
			// ---------------------------------------------------------------
			// "Co-planarisation" des points par moindres carrés
			// ---------------------------------------------------------------

			ElMatrix<REAL> Aplan(3,N,0.0);
			ElMatrix<REAL> Bplan(1,N,1.0);
			for (int i=0; i<N; i++){
				Aplan(0,i) = P3D[0][i];
				Aplan(1,i) = P3D[1][i];
				Aplan(2,i) = P3D[2][i];
			}

			// Régression du plan
			ElMatrix<REAL> Xplan = gaussj(Aplan.transpose()*Aplan)*(Aplan.transpose()*Bplan);

			// Calcul de la matrice de rotation
			ElMatrix<REAL> normale = Xplan*(-1.0/sqrt(Xplan.L2()));
			Pt3dr rot_axis(-normale(0,2), 0.0, normale(0,0));
			ElMatrix<REAL> VR = MatProVect(rot_axis).transpose();
			ElMatrix<REAL> Id(3,3,0); Id(0,0) = Id(1,1) = Id(2,2) = 1;
			ElMatrix<REAL> R = Id + VR + VR*VR*(1.0/(1.0+normale(0,1)));

			// Rotation
			for (unsigned i=0; i<P3D[0].size(); i++){
				for (unsigned j=0; j<3; j++){
					P3D[j][i] = R(j,0)*P3D[0][i] + R(j,1)*P3D[1][i] + R(j,2)*P3D[2][i];
				}
			}

			// Résidus et rmse
			ElMatrix<REAL> ERR_PLAN = Aplan*Xplan - Bplan;
			printf ("Point planarization: RMSE = %4.2f GROUND UNITS\n", sqrt(ERR_PLAN.L2()/N));
			std::cout << sep << std::endl;
			
		}
	
	}
	
	
	// ---------------------------------------------------------------	
	// Emprise de la zone
	// ---------------------------------------------------------------
	auto xmin = min_element(std::begin(P3D[0]), std::end(P3D[0]));
	auto xmax = max_element(std::begin(P3D[0]), std::end(P3D[0]));
	auto ymin = min_element(std::begin(P3D[2]), std::end(P3D[2]));
	auto ymax = max_element(std::begin(P3D[2]), std::end(P3D[2]));

	std::cout << "Bounding box:" << std::endl;
	printf ("xmin: %6.2f  xmax: %6.2f\n", *xmin, *xmax);
	printf ("ymin: %6.2f  ymax: %6.2f\n", *ymin, *ymax);
	
	std::cout << sep << std::endl;
	
	// ---------------------------------------------------------------	
	// Calcul de l'homographie
	// ---------------------------------------------------------------
	
	L2SysSurResol  system(8);               // Solveur moindres carrés

	for (int i=0; i<N; i++){
		
		double eq1_coeff[8];
		double eq2_coeff[8];
		double x = P3D[0][i];
		double y = P3D[2][i];
		double xp = P2D[0][i];
		double yp = P2D[1][i];

    	eq1_coeff[0] = x;  eq2_coeff[0] = 0;
    	eq1_coeff[1] = y;  eq2_coeff[1] = 0;
    	eq1_coeff[2] = 1;  eq2_coeff[2] = 0;

    	eq1_coeff[3] = 0;  eq2_coeff[3] = x;
    	eq1_coeff[4] = 0;  eq2_coeff[4] = y;
    	eq1_coeff[5] = 0;  eq2_coeff[5] = 1;

    	eq1_coeff[6] = -x*xp;  eq2_coeff[6] = -x*yp;
    	eq1_coeff[7] = -y*xp;  eq2_coeff[7] = -y*yp;

    	system.AddEquation(1.0, eq1_coeff, xp);
		system.AddEquation(1.0, eq2_coeff, yp);
		
	}
	
	
	bool Ok;
	Im1D_REAL8  aSol = system.Solve(&Ok);
	ELISE_ASSERT(Ok, "Error: least squares resolution")
	
	double HTAB[9];  memcpy(HTAB,aSol.data(),8*sizeof(double));
	HTAB[8] = 1;

	ElMatrix<REAL> H(1,8,0.0);
	for (int i=0; i<8; i++){
		H(0,i) = HTAB[i];
	}
	
	// Résidu
	double residu = sqrt(system.ResiduAfterSol()/(2.0*N));
		
	// Sortie console
	printf ("H =  %10.3f %10.3f %10.3f\n", H(0,0), H(0,1), H(0,2));
	printf ("     %10.3f %10.3f %10.3f\n", H(0,3), H(0,4), H(0,5));
	printf ("     %10.3f %10.3f %10.3f", H(0,6), H(0,7), 1.0);
	printf ("     RMSE = %4.2f PX\n", residu);
	
	std::cout << sep << std::endl;
	
	// ---------------------------------------------------------------	
	// Sauvegarde de l'homographie
	// ---------------------------------------------------------------
	
	cElComposHomographie aHX(H(0,0),H(0,1),H(0,2));
    cElComposHomographie aHY(H(0,3),H(0,4),H(0,5));
    cElComposHomographie aHZ(H(0,6),H(0,7),     1);

    cElHomographie homography = cElHomographie(aHX,aHY,aHZ);
	
	cElXMLFileIn aFileXML("homog.xml");
	aFileXML.PutElHomographie(homography,"Homographie");
	
	
		
	// ---------------------------------------------------------------
	// Prise en compte de la marge
	// ---------------------------------------------------------------
	double margex = abs(*xmax-*xmin)*marge;
	double margey = abs(*ymax-*ymin)*marge;
	*xmin = *xmin - margex; *xmax = *xmax + margex;
	*ymin = *ymin - margey; *ymax = *ymax + margey;
	
	Pt2dr pt_min (*xmin, *ymin);
	Pt2dr pt_max (*xmax, *ymax);
	aFileXML.PutPt2dr(pt_min,"MinCoords");
	aFileXML.PutPt2dr(pt_max,"MaxCoords");
	
	int Nx = resolution;
	int Ny = (*ymax-*ymin)/(*xmax-*xmin)*Nx;
	aFileXML.PutPt2dr(Pt2dr(Nx_org, Ny_org), "Input_resolution");
	aFileXML.PutPt2dr(Pt2dr(Nx, Ny), "Output_resolution");
	aFileXML.PutString(aNameCam, "Calib");
	
	std::cout << "Homography parameters written in [homog.xml] file" << std::endl;
	std::cout << sep << std::endl;	
	
}



// ----------------------------------------------------------------------------
//  A chaque commande MicMac correspond une fonction ayant la signature du main
//  Le lien se fait dans src/CBinaires/mm3d.cpp
// ----------------------------------------------------------------------------

// (1) Estimation de l'homographie
int CPP_YannEstimHomog(int argc,char ** argv){
   cAppli_YannEstimHomog(argc,argv);
   return EXIT_SUCCESS;
}

// (2) Application de l'homographie
int CPP_YannApplyHomog(int argc,char ** argv){
   cAppli_YannApplyHomog(argc,argv);
   return EXIT_SUCCESS;
}

// (3) Inversion des TP par l'homographie
int CPP_YannInvHomolHomog(int argc,char ** argv){
   cAppli_YannInvHomolHomog(argc,argv);
   return EXIT_SUCCESS;
}
  
/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a la mise en
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
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
