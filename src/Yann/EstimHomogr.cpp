


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
	   std::string mResolution;                    // Resolution images de sortie
	   std::string mCalib;						   // Calibration interne xml
	   std::string mInterpType;					   // Interpolation sub-px
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
	   std::string mCalib;						   // Calibration interne xml
	   std::string mDirOri;                        // Répertoire d'orientation
	   std::string mRawFolder;					   // Dossier des images bruts
	   cElemAppliSetFile mEASF;                    // Pour gerer un ensemble d'images
	   cElemAppliSetFile mEASF2;                   // Pour gerer un ensemble d'images
	   cInterfChantierNameManipulateur * mICNM;    // Name manipulateur
	   cInterfChantierNameManipulateur * mICNM2;   // Name manipulateur
};

// --------------------------------------------------------------------------------------
class cAppli_YannViewIntersect{
	
    public :
       cAppli_YannViewIntersect(int argc, char ** argv);

	   std::string ImPattern;                      // Images à traiter
	   std::string mDirOri;                        // Dossier d'orientations
       std::string mDepth;                         // Profondeur maximale d'intersection
       std::string mCpleFile;                      // Fichier output
       std::string mDir;                           // Directory of data
	   std::string mCalib;						   // Calibration interne xml
	   std::string aNameIn;					       // Nom d'image temporaire
	   std::string mRes;                           // Pas de discrétisation des cones
	   std::string mDist;                          // Distance maximale entre caméras
	   std::string mBuffer;                        // Buffer (pos ou neg) sur le champ
	   std::string mAngle;                         // Angle de visée maximal
	   std::string mDebug;                         // Pour générer des ply des cones
	   cElemAppliSetFile mEASF;                    // Pour gerer un ensemble d'images
	   cInterfChantierNameManipulateur * mICNM;    // Name manipulateur
};



// ======================================================================================
// Méthodes de calcul d'intersections entre champs de vision
// ======================================================================================


// --------------------------------------------------------------------------------------
// Fonction de calcul de rayon
// --------------------------------------------------------------------------------------
Pt3dr rayon(CamStenope * aCam, Pt2dr pt, double factor){
	Pt3dr p0 = aCam->Capteur2RayTer(pt).P0();
	Pt3dr p1 = aCam->Capteur2RayTer(pt).P1();
	double x = p0.x + factor*(p1.x-p0.x);
	double y = p0.y + factor*(p1.y-p0.y);
	double z = p0.z + factor*(p1.z-p0.z);
	return Pt3dr(x, y, z);
}

// --------------------------------------------------------------------------------------
// Fonction de calcul du "cone de visée"
// --------------------------------------------------------------------------------------
// Inputs  :
//   - cam : un pointeur vers une caméra
//   - N   : un nombre de points dans la discrétisation
//   - fz  : facteur de profondeur
//   - buf ; buffer (pos. ou neg.) en pixels
// Outputs : 
//	 - un vecteur de points Pt3dr (N+4+1 points avec le sommet de prise de vue)
// --------------------------------------------------------------------------------------
std::vector<Pt3dr> discretizedFieldOfView(CamStenope * aCam, unsigned N, int fz, double buf){
	
	std::vector<Pt3dr> FIELD;
	
	int nx = aCam->Sz().x + 2*buf;
	int ny = aCam->Sz().y + 2*buf;
	
	FIELD.push_back(aCam->PseudoOpticalCenter());
	
	FIELD.push_back(rayon(aCam, Pt2dr(0.0-buf, 0.0-buf), fz));

	for (unsigned n=1; n<N+1; n++){
		double xn = nx*(0.0+n)/(N+1)-buf;
		double yn = 0.0-buf;
		FIELD.push_back(rayon(aCam, Pt2dr(xn, yn),fz));
	}	
		
	FIELD.push_back(rayon(aCam, Pt2dr(nx-buf, 0.0-buf), fz));	
		
	for (unsigned n=1; n<N+1; n++){
		double xn = nx-buf;
		double yn = ny*(0.0+n)/(N+1)-buf;
		FIELD.push_back(rayon(aCam, Pt2dr(xn, yn), fz));
	}	
		
	FIELD.push_back(rayon(aCam, Pt2dr(nx-buf, ny-buf), fz));	
		
	for (unsigned n=1; n<N+1; n++){
		double xn = nx*(N+1-n)/(N+1)-buf;
		double yn = ny-buf;
		FIELD.push_back(rayon(aCam, Pt2dr(xn, yn), fz));
	}	
	
	FIELD.push_back(rayon(aCam, Pt2dr(0.0-buf, ny-buf), fz));
		
	for (unsigned n=1; n<N+1; n++){	
		double xn = 0.0-buf;
		double yn = ny*(N+1-n)/(N+1)-buf;
		FIELD.push_back(rayon(aCam, Pt2dr(xn, yn), fz));
	}	
	
	return FIELD;

} 

// --------------------------------------------------------------------------------------
// Fonction de transformation d'un "cone de visée" en nuage de points ply
// --------------------------------------------------------------------------------------
// Inputs  :
//   - cone : une liste de points définissant le cone de visée
//   - path : le chemin d'un fichier de sortie
// Outputs : 
//	 - Impression dans un fichier
// --------------------------------------------------------------------------------------
void fieldOfView2Ply(std::vector<Pt3dr>& cone, std::string path){
	
	unsigned resA = 500;
	unsigned resB = 4*resA/cone.size();
		
	ofstream myfile;
	myfile.open(path);

	myfile << "ply\n";
	myfile << "format ascii 1.0\n";
	myfile << "comment VCGLIB generated\n";
	myfile << "element vertex " << resB*(cone.size()-1) + resA*(cone.size()-1) << "\n";
	myfile << "property float x\n";
	myfile << "property float y\n";
	myfile << "property float z\n";
	myfile << "property uchar red\n";
	myfile << "property uchar green\n";
	myfile << "property uchar blue\n";
	myfile << "property uchar alpha\n";
	myfile << "element face 0\n";
	myfile << "property list uchar int vertex_indices\n";
	myfile << "end_header\n";
		
	for (unsigned i=1; i<cone.size()-1; i++){
		for (unsigned k=0; k<resB; k++){
			double t = ((float)k)/((float)resB);
			double x = cone.at(i).x*t + cone.at(i+1).x*(1-t);
			double y = cone.at(i).y*t + cone.at(i+1).y*(1-t);
			double z = cone.at(i).z*t + cone.at(i+1).z*(1-t); 
			myfile << x << " " << y << " " << z << " 0 0 255 255\n";	
		}
	}
	
	for (unsigned k=0; k<resB; k++){
		double t = ((float)k)/((float)resB);
		double x = cone.at(cone.size()-1).x*t + cone.at(1).x*(1-t);
		double y = cone.at(cone.size()-1).y*t + cone.at(1).y*(1-t);
		double z = cone.at(cone.size()-1).z*t + cone.at(1).z*(1-t); 
		myfile << x << " " << y << " " << z << " 0 0 255 255\n";	
	}
	
	for (unsigned j=1; j<cone.size(); j++){
		for (unsigned k=0; k<resA; k++){
			double t = ((float)k)/((float)resA);
			double x = cone.at(0).x*t + cone.at(j).x*(1-t);
			double y = cone.at(0).y*t + cone.at(j).y*(1-t);
			double z = cone.at(0).z*t + cone.at(j).z*(1-t);
			myfile << x << " " << y << " " << z << " " << "250 164 1 255\n";
		}
	}
	
	myfile.close();	
	
}


// --------------------------------------------------------------------------------------
// Fonction de calcul du vecteur unitaire de visée de la caméra
// --------------------------------------------------------------------------------------
Pt3dr sightDirectionVector(CamStenope * aCam){
	ElSeg3D s = aCam->Capteur2RayTer(Pt2dr(aCam->Sz().x/2.0, aCam->Sz().y/2.0));
	double dx = s.P1().x-s.P0().x;
	double dy = s.P1().y-s.P0().y;
	double dz = s.P1().z-s.P0().z;
	double norm = sqrt(dx*dx + dy*dy + dz*dz);
	return Pt3dr(dx/norm, dy/norm, dz/norm);
}

// --------------------------------------------------------------------------------------
// Fonction d'intersection d'un triangle et d'un segment
// --------------------------------------------------------------------------------------
bool segmentInTriangle(Pt3dr pt1, Pt3dr pt2, Pt3dr pt3, Pt3dr ps1, Pt3dr ps2){
	
	double x1  = pt1.x; double y1  = pt1.y; double z1  = pt1.z;
	double x2  = pt2.x; double y2  = pt2.y; double z2  = pt2.z;
	double x3  = pt3.x; double y3  = pt3.y; double z3  = pt3.z;
	double p0x = ps1.x; double p0y = ps1.y; double p0z = ps1.z;
	double p1x = ps2.x; double p1y = ps2.y; double p1z = ps2.z;
	
	double vx = p1x-p0x; double vy = p1y-p0y; double vz = p1z-p0z;

	double det = x1*y2*z3 + x2*y3*z1 + x3*y1*z2 - x3*y2*z1 - y3*z2*x1 - z3*x2*y1;

	double a = (-(y2*z3-y3*z2) + (y1*z3-y3*z1) - (y1*z2-y2*z1))/det;
	double b = (+(x2*z3-x3*z2) - (x1*z3-x3*z1) + (x1*z2-x2*z1))/det;
	double c = (-(x2*y3-x3*y2) + (x1*y3-x3*y1) - (x1*y2-x2*y1))/det;

	// Teste qu'on a bien un point du segment de chaque côté du plan du triangle
	double each_side = (a*p0x + b*p0y + c*p0z + 1)*(a*p1x + b*p1y + c*p1z + 1);
	if (each_side > 0){
		return false;
	}

	// Test parallelisme plan / droite
	double parallel = a*vx + b*vy + c*vz;
	double inclusion = a*p0x + b*p0y + c*p0z + 1;
	if (parallel == 0){
			return (inclusion < 1e-10);
	}
	
	// Sinon recherche unique intersection
	double t = -inclusion/parallel;

	// Intersection
	double xi = p0x + vx*t;
	double yi = p0y + vy*t;
	double zi = p0z + vz*t;

	// Test appartenance triangle par les produits vectoriels
	double pv1x = (y1-yi)*(z2-zi)-(z1-zi)*(y2-yi); double pv2x = (y2-yi)*(z3-zi)-(z2-zi)*(y3-yi); double pv3x = (y3-yi)*(z1-zi)-(z3-zi)*(y1-yi);
	double pv1y = (z1-zi)*(x2-xi)-(x1-xi)*(z2-zi); double pv2y = (z2-zi)*(x3-xi)-(x2-xi)*(z3-zi); double pv3y = (z3-zi)*(x1-xi)-(x3-xi)*(z1-zi);
	double pv1z = (x1-xi)*(y2-yi)-(y1-yi)*(x2-xi); double pv2z = (x2-xi)*(y3-yi)-(y2-yi)*(x3-xi); double pv3z = (x3-xi)*(y1-yi)-(y3-yi)*(x1-xi);

	double dot1 = pv1x*pv2x + pv1y*pv2y + pv1z*pv2z;
	double dot2 = pv2x*pv3x + pv2y*pv3y + pv2z*pv3z;
	
	return ((dot1 >= 0) && (dot2 >= 0));
	
}

// --------------------------------------------------------------------------------------
// Fonction de test d'intersection des surfaces deux cones
// --------------------------------------------------------------------------------------
bool sightIntersect(std::vector<Pt3dr>& cone1, std::vector<Pt3dr>& cone2){
	unsigned N1 = cone1.size();
	unsigned N2 = cone2.size();
	Pt3dr ps1 = cone1.at(0);
	Pt3dr pt1 = cone2.at(0);
	for (unsigned i1=1; i1<N1; i1++){
		Pt3dr ps2 = cone1.at(i1);
		for (unsigned i2=1; i2<N2-1; i2++){
			 Pt3dr pt2 = cone2.at(i2);
			 Pt3dr pt3 = cone2.at(i2+1);
			 if (segmentInTriangle(pt1, pt2, pt3, ps1, ps2)){
				return true;
			}
		}
	} 
	return false;
}


// --------------------------------------------------------------------------------------
// Fonction principale de calcul des paires de champs de vision
// --------------------------------------------------------------------------------------
// Inputs :
//   - string: liste des images à prendre en compte
//   - string: dossier d'orientation des images
//   - string: profondeur maximal de calcul (optionnelle, défaut -1 = Inf)
//   - string: nom du fichier de sortie (optionnel, défaut cple.xml)
// --------------------------------------------------------------------------------------
// Outputs :
//   - Un fichier xml contenant les paires de champs de vision
// --------------------------------------------------------------------------------------
cAppli_YannViewIntersect::cAppli_YannViewIntersect(int argc, char ** argv){
	
	ElInitArgMain(argc,argv,
        LArgMain()  <<  EAMC(ImPattern,"Images pattern")  
				    <<  EAMC(mDirOri,"Orientation directory"),
	   LArgMain()	<<  EAM(mDepth,"Depth", "NONE", "Maximal Depth (default -1 = Inf)")
					<<  EAM(mDist,"Dist", "None", "Max. distance (ground units) between camera optical centers")
					<<  EAM(mAngle,"Angle", "None", "Max. angle difference(degrees) between camera sights")
					<<  EAM(mRes,"Pts", "10", "Number of discretized points on fields of view (default 10)")
					<<  EAM(mBuffer,"Buffer", "None", "Signed buffer around camera field of view (default 0.0 PX)")
					<<  EAM(mDebug,"Ply", "0", "Set Ply=1 to generate ply files of fields of view")
          			<<  EAM(mCpleFile,"Out", "None", "Name of output file (default CpleFrom[OriName].xml)"));
	
	
	
	// ---------------------------------------------------------------
	// Lecture des images
	// ---------------------------------------------------------------
	if (EAMIsInit(&ImPattern)){
		mEASF.Init(ImPattern);
        mICNM = mEASF.mICNM;
        mDir = mEASF.mDir;
	}
	
	size_t N = mEASF.SetIm()->size();
	
	std::string message = "ERROR: can't find pairs with only 1 image";
	ELISE_ASSERT(N > 1, message.c_str());

	// ---------------------------------------------------------------
	// Gestion des paramètres d'input
	// ---------------------------------------------------------------
	int pts = 10;
	if (EAMIsInit(&mRes)){
		pts = std::stoi(mRes);
	}	
	
	bool debug = false;
	if (EAMIsInit(&mDebug)){
		debug = (std::stoi(mDebug) == 1);
	}	
	
	double max_dist = 1e300;
	if (EAMIsInit(&mDist)){
		max_dist = std::stod(mDist);
	}	
	
	double max_angle = 180;
	if (EAMIsInit(&mAngle)){
		max_angle = std::stod(mAngle);
	}	
	
	double max_depth = 1e300;
	if (EAMIsInit(&mDepth)){
		max_depth = std::stod(mDepth);
	}	
	
	double buffer = 0.0;
	if (EAMIsInit(&mBuffer)){
		buffer = std::stod(mBuffer);
	}
	
	std::string name_cple = +"CpleFrom"+mDirOri+".xml";
	if (EAMIsInit(&mCpleFile)){
		name_cple = mCpleFile;
	}	
	
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "                       FIELD OF VIEW COMPUTATION                       " << std::endl;
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "NUMBER OF IMAGES                       " << N                            << std::endl;
	std::cout << "NUMBER OF PAIRS                        " << N*(N-1)/2.0                  << std::endl;
	std::cout << "SIGHT CONE RESOLUTION                  " << pts                          << std::endl;
	std::cout << "BUFFER AROUND IMAGE                    " << buffer    << " PX"           << std::endl;
	std::cout << "MAX. ALLOWED ANGLE                     " << max_angle << " DEGREES"      << std::endl;
	std::cout << "MAX. ALLOWED DEPTH                     " << max_depth << " GROUND UNITS" << std::endl;
	std::cout << "MAX. ALLOWED DISTANCE                  " << max_dist  << " GROUND UNITS" << std::endl;
	std::cout << "-----------------------------------------------------------------------" << std::endl;

	// ---------------------------------------------------------------
	// Calcul des paires
	// ---------------------------------------------------------------
	
	ofstream myfile;
	myfile.open(name_cple);
	
	myfile << "<?xml version=\"1.0\" ?>\n";
	myfile << "<SauvegardeNamedRel>\n";
	
	unsigned nb_pair_found = 0;
	unsigned nb_pair_found_for_im = 0;
	
	double min_dot_product = cos(max_angle*3.14159/180.0);
	
	int factor = 200;   //   Attention : à régler !

	// ---------------------------------------------------------------
	// Loop on image 1
	// ---------------------------------------------------------------
	for (unsigned i=0; i<N; i++){
		
		aNameIn = mEASF.SetIm()[0][i];
		std::cout << "[" << i << "/" << N << "]  " <<  aNameIn << "      ";
		
		cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
		StdCorrecNameOrient(mDirOri, anICNM->Dir());
		CamStenope * aCam = CamOrientGenFromFile("Ori-"+mDirOri+"/Orientation-"+aNameIn+".xml", anICNM);
		
		Pt3dr p1 = aCam->PseudoOpticalCenter();
		Pt3dr v1 = sightDirectionVector(aCam);
		
		std::vector<Pt3dr> F1 = discretizedFieldOfView(aCam, pts, factor, buffer);
		
		if (debug){
			fieldOfView2Ply(F1, "AperiCone_"+aNameIn+".ply");
		}
		
		// ---------------------------------------------------------------
		// Loop on image 2
		// ---------------------------------------------------------------
		nb_pair_found_for_im = 0;
		for (unsigned j=i; j<N; j++){
			if (i == j) continue;
			
			std::string anOtherNameIn = mEASF.SetIm()[0][j];
			CamStenope * anOtherCam = CamOrientGenFromFile("Ori-"+mDirOri+"/Orientation-"+anOtherNameIn+".xml", anICNM);
			
			// ---------------------------------------------------------------
			// Distance test
			// ---------------------------------------------------------------
			Pt3dr p2 = anOtherCam->PseudoOpticalCenter();
			if (sqrt(pow(p1.x-p2.x,2) + pow(p1.y-p2.y,2) + pow(p1.z-p2.z,2)) > max_dist){
					continue;
			}
			
			// ---------------------------------------------------------------
			// Orientation test
			// ---------------------------------------------------------------
			Pt3dr v2 = sightDirectionVector(anOtherCam);
			if (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z < min_dot_product){
					continue;
			}
			
			// ---------------------------------------------------------------
			// Inclusion test
			// ---------------------------------------------------------------
			// TO DO !
		
			
			// ---------------------------------------------------------------
			// Sight intersection test
			// ---------------------------------------------------------------
			std::vector<Pt3dr> F2 = discretizedFieldOfView(anOtherCam, pts, factor, buffer);
			if (!sightIntersect(F1, F2)){
				continue;
			}
			
			myfile << "<Cple>" << aNameIn << " " << anOtherNameIn << "</Cple>\n"; nb_pair_found_for_im ++;
			
		}	
		
		nb_pair_found += nb_pair_found_for_im;
		double frac = ((int)(1000*nb_pair_found/(N*(N-1)/2.0)))/10.0;
		std::cout << nb_pair_found_for_im << " homolog pair(s)    [" << frac << " %]"  << std::endl;
		
	}
	
	myfile << "</SauvegardeNamedRel>\n";
	myfile.close();	
	
	double frac = ((int)(1000*nb_pair_found/(N*(N-1)/2.0)))/10.0;
	std::cout << "-----------------------------------------------------------------------"               << std::endl;
	std::cout << "NUMBER OF PAIRS FOUND       " << nb_pair_found << "                [" << frac << " %]" << std::endl;
	std::cout << "Output file [" << name_cple << "] generated"                                             << std::endl;
	std::cout << "-----------------------------------------------------------------------"               << std::endl;

}


// --------------------------------------------------------------------------------------
// Fonction de calcul des résolutions Nx et Ny en fonction d'un paramètre de maximal
// --------------------------------------------------------------------------------------
// Inputs :
//   - Nx et Ny (par référence)
//   - Résolution maximale en x et en y
//   - Extension de la zone d'étude
// --------------------------------------------------------------------------------------
void setResolution(int& Nx, int& Ny, int resol_max, double xmin, double xmax, double ymin, double ymax){
	Nx = resol_max;
	Ny = (ymax-ymin)/(xmax-xmin)*Nx;
	if (Ny > Nx){
		Ny = resol_max;
		Nx = (xmax-xmin)/(ymax-ymin)*Ny;
	}
} 

// --------------------------------------------------------------------------------------
// Fonction de lecture du fichier d'homographie
// --------------------------------------------------------------------------------------
// Inputs :
//   - nom du fichier à lire
//   - matrice des paramètres d'homographie
//   - emprise xmin, ymin, xmax, ymax
//   - nombre de pixels en x de l'image
//   - résolution souhaitée en x
// --------------------------------------------------------------------------------------
void readHomogFile(std::string mHomogFile, ElMatrix<REAL> &H, double &xmin, double &ymin, double &xmax, 
				   double &ymax, int &Nx_org, int &Ny_org, std::string &dirOri){

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
	   LArgMain()	<<  EAM(mCalib,"Calib", "NONE", "Calibration xml file")
          			<<  EAM(mFolderOut,"Out", "InvHomog", "Homol output folder")
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
	
	readHomogFile(mHomogFile, H, xmin, ymin, xmax, ymax, Nx_org, Ny_org, aNameCam);
	
	// ---------------------------------------------------------------
	// Récupération de la taille (en x) des images redressées
	// ---------------------------------------------------------------
	mEASF2.Init(ImPattern);
    mICNM2 = mEASF2.mICNM;
    std::string mDir = mEASF2.mDir;
	Tiff_Im aTF = Tiff_Im::StdConvGen(mEASF2.SetIm()[0][0],3,false);
	Pt2di im_size = aTF.sz();
	
	Nx = im_size.x;
	Ny = im_size.y;
	
	double resolution = (xmax-xmin)/(Nx-1);
	
	setResolution(Nx, Ny, resolution, xmin, xmax, ymin, ymax);

	// ---------------------------------------------------------------
	// Correction éventuelle de la distorsion
	// ---------------------------------------------------------------

	
	CamStenope * aCam = 0;
	

	if ((aNameCam.size() > 0) && (EAMIsInit(&mCalib))){
    	//cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
    	//aCam = CamOrientGenFromFile(mRawFolder+"/"+aNameCam, anICNM);
		aCam = Std_Cal_From_File(mCalib);
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
	
	// ---------------------------------------------------------------
	// Corps du module
	// ---------------------------------------------------------------
	
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
					if (aCam != 0){
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
//   - string: Résolution (maximale) des images générées
// --------------------------------------------------------------------------------------
// Outputs :
//   - Les images transformées par l'homographie
// --------------------------------------------------------------------------------------
cAppli_YannApplyHomog::cAppli_YannApplyHomog(int argc, char ** argv){

	 ElInitArgMain(argc,argv,
        LArgMain()  <<  EAMC(mNameIm,"Images pattern")
				    <<  EAMC(mNameHomog,"Homography parameters file"),
        LArgMain()  <<  EAM(mDirOut,"Out", "Homog", "Output folder") 
				    <<  EAM(mCalib,"Calib", "NONE", "Calibration xml file") 
				   	<<  EAM(mResolution,"ImRes", "2000", "Output resolution of images")
				    <<  EAM(mInterpType, "Interp", "bilin", "Interpolation method (ppv/bilin)"));

	
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
	
	size_t N = mEASF.SetIm()->size();
	
	bool hasCalib = false;
	
	// ---------------------------------------------------------------
	// Gestion du répertoire de sortie
	// ---------------------------------------------------------------
	std::string output_folder = "Homog";
	if (EAMIsInit(&mDirOut)){
		output_folder = mDirOut;
	}
	if (output_folder != "./"){
		ELISE_fp::MkDir(output_folder);
	}
	
	// ---------------------------------------------------------------
	// Gestion de la résolution de sortie
	// ---------------------------------------------------------------
	int resolution = 2000;
	if (EAMIsInit(&mResolution)){
		resolution = std::stoi(mResolution);
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
	
	readHomogFile(mNameHomog, H, xmin, ymin, xmax, ymax, Nx_org, Ny_org, aNameCam);
	setResolution(Nx, Ny, resolution, xmin, xmax, ymin, ymax);

	// ---------------------------------------------------------------
	// Impression console pour confirmation des paramètres
	// ---------------------------------------------------------------
	printf ("H = %10.3f %10.3f %10.3f    BBOX = %7.2f %7.2f \n", H(0,0), H(0,1), H(0,2), xmin, xmax);
	printf ("    %10.3f %10.3f %10.3f           %7.2f %7.2f \n", H(0,3), H(0,4), H(0,5), ymin, ymax);
	printf ("    %10.3f %10.3f %10.3f      \n", H(0,6), H(0,7), 1.0);
	printf ("GENERATED IMAGE SIZE: [%i x %i]   RAW IMAGE SIZE:  [%i x %i] \n", Nx, Ny, Nx_org, Ny_org);
	
	std::cout << "INTERPOLATION METHOD: ";
	if (EAMIsInit(&mInterpType) && mInterpType == "ppv"){
		std::cout << "Nearest Neighbor Interpolation" << std::endl;
	}else{
		std::cout << "Bilinear Interpolation" << std::endl;
	}
	
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
	if (mCalib.size() > 0){
		aCam = Std_Cal_From_File(mCalib);
		hasCalib = true;
	}else{
		if (aNameCam.size() > 0){
    		cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
    		aCam = CamOrientGenFromFile(aNameCam,anICNM);
			hasCalib = true;
		}
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
			if (hasCalib){
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
				
				int px = (int) ptIn.x;
				int py = (int) ptIn.y;
				
				if (px < 0){continue;}
				if (py < 0){continue;}
				if (px >= im_size.x-1){continue;}
				if (py >= im_size.y-1){continue;}
				
				if (EAMIsInit(&mInterpType) && mInterpType == "ppv"){
				
					// ----------------------------------------------
					// Interpolation plus proche voisin
					// ----------------------------------------------
					aDataROut[iy][ix] = aDataR[py][px];
					aDataGOut[iy][ix] = aDataG[py][px];
					aDataBOut[iy][ix] = aDataB[py][px];
				
				}else{
				
					// ----------------------------------------------
					// Interpolation bilinéaire
					// ----------------------------------------------
            		aDataROut[iy][ix] = Reechantillonnage::biline(aDataR, im_size.y, im_size.x, ptIn);
            		aDataGOut[iy][ix] = Reechantillonnage::biline(aDataG, im_size.y, im_size.x, ptIn);
            		aDataBOut[iy][ix] = Reechantillonnage::biline(aDataB, im_size.y, im_size.x, ptIn);
				
				}
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
		ELISE_ASSERT(EAMIsInit(&mDirOri), "No orientation folder provided for camera frame rotation")
	}
	
			
	// ---------------------------------------------------------------
	// Gestion de la marge
	// ---------------------------------------------------------------
	double marge = 0.3;
	if (EAMIsInit(&mMargeRel)){
		marge = std::stof(mMargeRel);
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
	
	size_t val = 0;
	size_t max = 0;
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
	const size_t N = MESURE_IMAGES[selected].OneMesureAF1I().size();
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
	std::vector<string> NAME_PT_SELECTED;
	for (unsigned i=0; i<X3D.size(); i++){
		for (unsigned j=0; j<X2D.size(); j++){
			if (NAME3D[i] == NAME2D[j]){
				P2D[0].push_back(X2D[j]);
				P2D[1].push_back(Y2D[j]);
				P3D[0].push_back(X3D[i]);
				P3D[1].push_back(Y3D[i]);
				P3D[2].push_back(Z3D[i]);
				NAME_PT_SELECTED.push_back(NAME3D[i]);
				printf ("IMAGE: %9.3f %9.3f     ", X2D[j], Y2D[j]);
				printf ("TERRAIN: %7.3f %7.3f %7.3f\n", X3D[i], Y3D[i], Z3D[i]);
			}
		}
	}
	
	std::cout << sep << std::endl;
	
	// ---------------------------------------------------------------
    // Correction des distorsions
    // ---------------------------------------------------------------
    if (EAMIsInit(&mDirOri)){
       
        StdCorrecNameOrient(mDirOri,mDir);
       
        // Correction de la distorsion
        aNameCam="Ori-"+mDirOri+"/Orientation-"+mNameImRef+".xml";
        cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
        CamStenope * aCam = CamOrientGenFromFile(aNameCam,anICNM);
        std::cout << "Distorsion correction ";
       
        for (unsigned i=0; i<P2D[0].size(); i++){
            Pt2dr aCenterOut = aCam->DistInverse(Pt2dr(P2D[0][i], P2D[1][i]));
            P2D[0][i] = aCenterOut.x;
            P2D[1][i] = aCenterOut.y;
        }
		
		std::cout << "from orientation: [";
        std::cout << mDirOri << "]" << std::endl;
		std::cout << sep << std::endl;
   
    }
	
	// ---------------------------------------------------------------
	// Rotation des points du repère terrain dans l'espace caméra
	// ---------------------------------------------------------------
    ElMatrix<REAL> ROT(3,3,0.0);
    if ((EAMIsInit(&mCamRot)) && (mCamRot)){
			
		std::cout << "Rotation in camera frame: " << mNameImRef << std::endl;
		std::cout << sep << std::endl;
			
        cOrientationConique aOriConique=StdGetFromPCP(aNameCam,OrientationConique);

        ROT = MatFromCol(
           aOriConique.Externe().ParamRotation().CodageMatr().Val().L1(),
           aOriConique.Externe().ParamRotation().CodageMatr().Val().L2(),
           aOriConique.Externe().ParamRotation().CodageMatr().Val().L3()
        ).transpose();
           
		ElMatrix<REAL> C(1,3,0.0);
		C(0,0) = aOriConique.Externe().Centre().x;
		C(0,1) = aOriConique.Externe().Centre().y;
		C(0,2) = aOriConique.Externe().Centre().z;
		
		ElMatrix<REAL> P(1,3,0.0);
        for (unsigned i=0; i<P3D[0].size(); i++){ 
            P(0,0) = P3D[0][i]; P(0,1) = P3D[1][i];P(0,2) = P3D[2][i];
            ElMatrix<REAL> P_ROT = ROT.transpose()*(P-C);
            P3D[0][i] = -P_ROT(0,2); P3D[1][i] = P_ROT(0,0); P3D[2][i] = P_ROT(0,1);
        }
    }
	
	// ---------------------------------------------------------------
	// "Co-planarisation" des points par moindres carrés
	// ---------------------------------------------------------------
	
	ElMatrix<REAL> Aplan(3,N,0.0);
	ElMatrix<REAL> Bplan(1,N,1.0);
	for (unsigned i=0; i<N; i++){
		Aplan(0,i) = P3D[0][i];
		Aplan(1,i) = P3D[1][i];
		Aplan(2,i) = P3D[2][i];
	}

	// Régression du plan
	ElMatrix<REAL> Xplan = gaussj(Aplan.transpose()*Aplan)*(Aplan.transpose()*Bplan);
	
	double a = Xplan(0,0);
	double b = Xplan(0,1);
	double c = Xplan(0,2);
	double d = -1;
	
	// Résidus et rmse
	double rmse_plan = 0;
	std::cout << "Point planarization";
	double abc_norm = sqrt(Xplan.L2());
	for (unsigned i=0; i<N; i++){
		std::cout << std::endl;
		double val = abs(Xplan(0,0)*P3D[0][i] + Xplan(0,1)*P3D[1][i] + Xplan(0,2)*P3D[2][i] - 1)/abc_norm;
		rmse_plan += val*val;
		printf ("RESIDUAL POINT %s %7.3f", NAME_PT_SELECTED[i].c_str(), val);
	}
	rmse_plan = sqrt(rmse_plan/N);
	printf("     RMSE = %4.3f GROUND UNITS\n", rmse_plan);

		
	a /= abc_norm;
	b /= abc_norm;
	c /= abc_norm;
	d /= abc_norm;
	
	
	// Rotation
	double xp = 0; double yp = 0; double zp = -d/c;
	double xq = 1; double yq = 0; double zq = -(a + d)/c;
	double pq_norm = sqrt((xq-xp)*(xq-xp) +  (yq-yp)*(yq-yp) + (zq-zp)*(zq-zp));
	
	double vpqx = (xq-xp)/pq_norm; 
	double vpqy = (yq-yp)/pq_norm; 
	double vpqz = (zq-zp)/pq_norm;
	
	double wx = b*vpqz - c*vpqy;
	double wy = c*vpqx - a*vpqz;
	double wz = a*vpqx - b*vpqy;
	
	double w_norm = sqrt(wx*wx +  wy*wy + wz*wz);
	wx /= w_norm;
	wy /= w_norm;
	wz /= w_norm;
	
	ElMatrix<REAL> RP(3,3,0.0);
	
	RP(0,0) = vpqx;  RP(1,0) = vpqy;  RP(2,0) = vpqz;
	RP(0,1) = wx;    RP(1,1) = wy;    RP(2,1) = wz;
	RP(0,2) = a;     RP(1,2) = b;     RP(2,2) = c;

	
	ElMatrix<REAL> P(1,3,0.0);
	for (unsigned i=0; i<P3D[0].size(); i++){
		P(0,0) = P3D[0][i]; 
		P(0,1) = P3D[1][i];
		P(0,2) = P3D[2][i] - zp;
		ElMatrix<REAL> P_ROT = RP*P;
		P3D[0][i] = P_ROT(0,0); 
		P3D[1][i] = P_ROT(0,1); 
		P3D[2][i] = P_ROT(0,2);
	}
	

	std::cout << sep << std::endl;
		
	// ---------------------------------------------------------------	
	// Creation du jeu de points 3D final
	// ---------------------------------------------------------------
	std::vector<double> X3DF;
	std::vector<double> Y3DF;
	std::cout << "Corrected ground points: " << std::endl;
	for (unsigned i=0; i<P3D[0].size(); i++){
		X3DF.push_back(P3D[1][i]);
		Y3DF.push_back(-P3D[0][i]);
		printf ("X = %7.3f   Y = %7.3f   Z = %7.3f\n", P3D[0][i], P3D[1][i], P3D[2][i]);
	}
	std::cout << sep << std::endl;
	
	// ---------------------------------------------------------------	
	// Emprise de la zone
	// ---------------------------------------------------------------
	auto xmin = min_element(std::begin(X3DF), std::end(X3DF));
	auto xmax = max_element(std::begin(X3DF), std::end(X3DF));
	auto ymin = min_element(std::begin(Y3DF), std::end(Y3DF));
	auto ymax = max_element(std::begin(Y3DF), std::end(Y3DF));

	std::cout << "Bounding box:" << std::endl;
	printf ("xmin: %6.2f  xmax: %6.2f\n", *xmin, *xmax);
	printf ("ymin: %6.2f  ymax: %6.2f\n", *ymin, *ymax);
	
	std::cout << sep << std::endl;
	
	// ---------------------------------------------------------------	
	// Calcul de l'homographie
	// ---------------------------------------------------------------
	
	L2SysSurResol  system(8);               // Solveur moindres carrés

	for (unsigned i=0; i<N; i++){
		
		double eq1_coeff[8];
		double eq2_coeff[8];
		double x = X3DF[i];
		double y = Y3DF[i];
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
	std::cout << "Solving least squares";
	double residu = 0;
	double Z, xp_predict, yp_predict;
	double er2, residu_x, residu_y;
	for (unsigned i=0; i<N; i++){
		std::cout << std::endl;
		Z = (H(0,6)*X3DF[i] + H(0,7)*Y3DF[i] + 1);
		xp_predict = (H(0,0)*X3DF[i] + H(0,1)*Y3DF[i] + H(0,2))/Z;
		yp_predict = (H(0,3)*X3DF[i] + H(0,4)*Y3DF[i] + H(0,5))/Z;
		residu_x = xp_predict-P2D[0][i];
		residu_y = yp_predict-P2D[1][i];
		er2 = sqrt(residu_x*residu_x + residu_y*residu_y);
		residu += er2;
		printf ("RESIDUAL POINT %s %7.3f PX", NAME_PT_SELECTED[i].c_str(), sqrt(er2));
	}
	residu = sqrt(residu/N);
	printf ("     RMSE = %4.2f PX\n", residu);
	
	std::cout << sep << std::endl;
		
	// Sortie console
	printf ("H =  %10.3f %10.3f %10.3f\n", H(0,0), H(0,1), H(0,2));
	printf ("     %10.3f %10.3f %10.3f\n", H(0,3), H(0,4), H(0,5));
	printf ("     %10.3f %10.3f %10.3f\n", H(0,6), H(0,7), 1.0);
	
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
	
	aFileXML.PutPt2dr(Pt2dr(Nx_org, Ny_org), "Input_resolution");
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
  
// (4) Calcul de champ de vision
int CPP_YannViewIntersect(int argc,char ** argv){
   cAppli_YannViewIntersect(argc,argv);
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
