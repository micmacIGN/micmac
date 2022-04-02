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

#include "gnss/Gnss.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#define APPROX_HEIGHT_SAT_TARGETS            1000

// --------------------------------------------------------------------------------------
// Classes permettant de compléter les images avec le champ de datation absolue
// --------------------------------------------------------------------------------------
class cAppli_YannSetTimestamps{

	public :
	
		cAppli_YannSetTimestamps(int argc, char ** argv);
	
		std::string ImPattern;                      // Images caméra
		std::string mTimestamps;					// Fichier des timestamps
		std::string mTimestampsFmt;					// Format du fichier des timestamps
		std::string mDateIni;                       // Date initiale (si conversion)
	
		cElemAppliSetFile mEASF;                    // Pour gerer un ensemble d'images
		std::string mDir;                           // Répertoire des données
	 	cInterfChantierNameManipulateur * mICNM;    // Name manipulateur

};

// --------------------------------------------------------------------------------------
// Classes permettant d'exclure les satellites d'un fichier d'observations rinex à 
// partir d'un ensemble de masque de ciel, d'une orientation externe (absolue) et d'un 
// fichier d'éphémérides (rinex .nav + éventuellement éphémérides précises sp3).
// --------------------------------------------------------------------------------------
class cAppli_YannExcludeSats{

	public :
	
		cAppli_YannExcludeSats(int argc, char ** argv);
	
		std::string ImPattern;                      // Images caméra
		std::string mDirOri;                        // Fichier d'orientation
		std::string mRinexObs;						// Fichier rinex d'observations GNSS
		std::string mNavFile;						// Fichiers d'éphémérides rinex/sp3
		std::string mSysCode;						// Code de l'orientation absolue
		std::string mMasqKey;						// Clé des images de masque dans MMLCD
		std::string mGloNavFile;					// Glonass navigation file rinex/sp3
		std::string mGalNavFile;					// Galileo navigation file rinex/sp3
		std::string mTolerance;						// GNSS <-> image capture time tolerance
		std::string mPlotFolder;					// Dossier pour tracer les satellites
		std::string mOffset;						// Offset sur les coordonnées
		std::string mAddSat;						// Ajout de satellites à traiter
		std::string mRemSat;						// Retrait de satellites à traiter
		std::string mSlice;						    // Sous-échantillonnage du fichier rinex
		std::string mAux;							// Fichier auxiliaire (optionnel)
		std::string mOutRinex;						// Output rinex observation file
	
		cElemAppliSetFile mEASF;                    // Pour gerer un ensemble d'images
		std::string mDir;                           // Répertoire des données
	 	cInterfChantierNameManipulateur * mICNM;    // Name manipulateur

};

// --------------------------------------------------------------------------------------
// Classes permettant de compléter les images avec le champ de datation absolue
// --------------------------------------------------------------------------------------
class cAppli_YannSkyMask{

	public :
	
		cAppli_YannSkyMask(int argc, char ** argv);
	
		std::string ImPattern;                      // Images caméra
		std::string outDir;                         // Répertoire de sortie
		std::string mProba;							// Probabilistic mode
		std::string mInv;							// Inverse mode
		std::string mThresh;						// Decision threshold
		std::string mFilter;						// Filtering artifacts
	    std::string mInstall;						// Installation lib python
		std::string mUninstall;						// Desinstallation lib python
		
		cElemAppliSetFile mEASF;                    // Pour gerer un ensemble d'images
		std::string mDir;                           // Répertoire des données
	 	cInterfChantierNameManipulateur * mICNM;    // Name manipulateur

};


// --------------------------------------------------------------------------------------
// Classes de test peremettant de stocker le code de type script
// --------------------------------------------------------------------------------------
class cAppli_YannScript{
	
	public :
	
	cAppli_YannScript(int argc, char ** argv);
	
	std::string ImPattern;
	std::string aPostIn;
	std::string mOut;

};

// --------------------------------------------------------------------------------------
// Fonction de conversion d'un chaîne de caractères en GPSTime
// --------------------------------------------------------------------------------------
GPSTime str2GPSTime(std::string time) {
	
	int day = std::stoi(time.substr(0,2));
	int mon = std::stoi(time.substr(3,2));
	int yea = std::stoi(time.substr(6,4));
	int hou = std::stoi(time.substr(11,2));
	int min = std::stoi(time.substr(14,2));
	int sec = std::stoi(time.substr(17,2));
	int mls = std::stoi(time.substr(20,3));
				
	GPSTime tps(yea, mon, day, hou, min, sec, mls);
    
    return tps;
}


// --------------------------------------------------------------------------------------
// Fonction d'appel system pour récupérer la sortie d'une commande
// --------------------------------------------------------------------------------------
std::string execCmdOutput(const char* cmd) {

	// Version Windows
	#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
		char buffer[128];
    	std::string result = "";
    	std::shared_ptr<FILE> pipe(_popen(cmd, "r"), _pclose);
    	if (!pipe) throw std::runtime_error("popen() failed!");

    	while (!feof(pipe.get())) {
        	if (fgets(buffer, 128, pipe.get()) != NULL)
            	result += buffer;
    	}
    	return result;
    	
	#else   // Version Linux ou Mac
		
		std::array<char, 128> buffer;
   		std::string result;
    	std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
	    if (!pipe) {
        	throw std::runtime_error("popen() failed!");
    	}
    	while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        	result += buffer.data();
    	}
    	return result;
	
	#endif

}


// --------------------------------------------------------------------------------------
// Fonction d'ajout des timestamps dans les images (champ exif Date/Time Original)
// --------------------------------------------------------------------------------------
// Inputs :
//   - string: Pattern des images
//   - string: Fichier contenant les timestamps
//   - string: Format de fichier des timestamps (doit contenir N et T)
//   - string: Date initiale pour la conversion secondes -> date absolue
// --------------------------------------------------------------------------------------
cAppli_YannSetTimestamps::cAppli_YannSetTimestamps(int argc, char ** argv){
	
	 ElInitArgMain(argc,argv,
        LArgMain()  <<  EAMC(ImPattern,"Pattern of images") 
				  	<<  EAMC(mTimestamps, "File containing timestamps")
				    <<  EAMC(mTimestampsFmt, "Timestamp file format specification"),
        LArgMain()  <<  EAM(mDateIni,"DateIni", "NONE", "Start date [dd/mm/yyyy hh:mm:ss.msc]"));
		
	// ---------------------------------------------------------------
	// Analyse du format
	// ---------------------------------------------------------------

	std::cout << "Format:        " << mTimestampsFmt << std::endl;
	
	size_t posix = mTimestampsFmt.find("Ix"); if (posix < mTimestampsFmt.size()) mTimestampsFmt.replace(posix,2,"I");
	size_t posiy = mTimestampsFmt.find("Iy"); if (posiy < mTimestampsFmt.size()) mTimestampsFmt.replace(posiy,2,"J");
	size_t posiz = mTimestampsFmt.find("Iz"); if (posiz < mTimestampsFmt.size()) mTimestampsFmt.replace(posiz,2,"K");
	
	std::string delimiter = mTimestampsFmt.substr(4,1);
	size_t time_position = (mTimestampsFmt.find("T")-3)/2;
	size_t name_position = (mTimestampsFmt.find("N")-3)/2;
	
	std::cout << "Delimiter:     " << delimiter << std::endl;
	std::cout << "Time position: " << time_position << std::endl;
	std::cout << "Name position: " << name_position << std::endl;
	
	if (delimiter == "_") delimiter = " ";
	
	// ---------------------------------------------------------------
	// Lecture des images
	// ---------------------------------------------------------------
		
	std::vector<std::string> IMAGES;
	
	if (EAMIsInit(&ImPattern)){
		mEASF.Init(ImPattern);
        mICNM = mEASF.mICNM;
        mDir = mEASF.mDir;
	}
	
	size_t N = mEASF.SetIm()->size();
	
	for (unsigned i=0; i<N; i++){
		IMAGES.push_back(mEASF.SetIm()[0][i]);
	}

	
	// ---------------------------------------------------------------
	// Lecture du fichier de timestamps
	// ---------------------------------------------------------------
	std::ifstream infile(mTimestamps);
	std::string line = "";
	std::string time = "";
	std::string name = "";
	
	while (std::getline(infile, line)){
		
		
		if (line.substr(line.size()-1, 1) == "\r") line = line.substr(0, line.size()-1);
		std::vector<std::string> fields = Utils::tokenize(line, " ");
		
		time = fields.at(time_position);
		name = fields.at(name_position);
		
		// Recherche dans les images
		bool found = false;
		
		//"TestVincennes_1909121306_04_00829_0.tif"
		for (unsigned i=0; i<IMAGES.size(); i++){
			if (IMAGES.at(i) == name){
				found = true; break;
			}
		}
		
		// -----------------------------------------------------------------------
		// Impression de la date dans l'exif
		// -----------------------------------------------------------------------
		if (found) {
			
			// Modification de l'exif
			std::string dateToPrint = "";
			
			if (!EAMIsInit(&mDateIni)){ // Dates sous forme dd/mm/yyyy hh:mm:ss:msc
				
				dateToPrint = time;
				
			}else{    // Date sous forme de secondes écoulées depuis mDateIni 
				
				GPSTime start = str2GPSTime(mDateIni);
				
				double tps_float = std::stod(time.c_str());
				long absTime = (long)(start.convertToAbsTime() + tps_float);
				GPSTime timestamp(absTime);
				timestamp.ms = (tps_float - std::floor(tps_float))*1000;
				dateToPrint = timestamp.to_complete_string();
				
			}
			
			std::cout << dateToPrint << " ";
			
			cElemAppliSetFile anEASF(name);
			std::string aNameFile =  Dir2Write(anEASF.mDir) + "ExivBatchFile.txt";
			FILE * aFP = FopenNN(aNameFile,"w","CPP_SetExif");
			fprintf(aFP, "set Exif.Photo.DateTimeOriginal  Ascii  \"%s\"\n", dateToPrint.c_str());
			fclose(aFP);
			
			std::list<std::string> aLCom;
    		const cInterfChantierNameManipulateur::tSet * aVIm = anEASF.SetIm();

   			for (int aKIm=0 ; aKIm<int(aVIm->size()) ; aKIm++){
        		std::string aCom ="exiv2 -m " + aNameFile + " " + (*aVIm)[aKIm];
       			aLCom.push_back(aCom);
        		System(aCom);
    		}
			
    		cEl_GPAO::DoComInParal(aLCom);
			
		}
		
	}
	
	infile.close();
  	
}


// --------------------------------------------------------------------------------------
// Fonction principale d'exclusion des satellites à partir de masques de ciel
// --------------------------------------------------------------------------------------
// Inputs :
//   - string: Pattern des images orientées par photogrammétrie
//   - string: Répertoire d'orientation (orientation externe absolue)
//   - string: Fichier rinex des observations GNSS (datées dans les exif des images)
//   - string: Fichier(s) d'éphémérides GNSS rinex/sp3 (min 1 fichier)
//   - string: Clé des images de masque de ciel dans MMLCD
//   - string: Répertoire pour les images avec tracé des satellites (si besoin)
//   - string: Fichier d'observation rinex de sortie (défaut excluded_sats.o)
// --------------------------------------------------------------------------------------
// Outputs :
//   - Un fichier rinex d'observations épurés des satellites non-visibles
//   - Optionnel : les images d'entrée avec affichage des positions des satellites
// --------------------------------------------------------------------------------------
// Notes : la fonction nécessite que les images contiennent un exif Date/Time Original
// éventuellement fixé avec la commande setTimetamps. Les champs de timestamps doivent 
// être dans le même système de temps que ceux des fichiers rinex d'obs et de nav.
// --------------------------------------------------------------------------------------

cAppli_YannExcludeSats::cAppli_YannExcludeSats(int argc, char ** argv){
	
    ElInitArgMain(argc,argv,
        LArgMain()  <<  EAMC(ImPattern,"Pattern of images") 
                    <<  EAMC(mDirOri, "Orientation folder")
				  	<<  EAMC(mRinexObs, "Raw GNSS observation rinex file")
				    <<  EAMC(mNavFile, "GPS ephemeride files (rinex .nav or sp3)")
				  	<<  EAMC(mSysCode, "Absolute orientation system code"),
        LArgMain()  <<  EAM(mMasqKey,"MasqKey", "NONE", "Key to find sky masq images in MMLCD")
				  	<<  EAM(mGloNavFile,"GloNavFile", "NONE", "Glonass ephemeride file (rinex .nav or sp3)")
				  	<<  EAM(mGalNavFile,"GalNavFile", "NONE", "Galileo ephemeride file (rinex .nav or sp3)")
				    <<  EAM(mPlotFolder,"PlotDir", "NONE", "Directory to print out satellite positions")
				    <<  EAM(mTolerance,"Tol", "0.5", "Time tolerance (s) btw GNSS and image capture")
				 	<<  EAM(mOffset,"Offset", "[0.0,0.0]", "Offset on coordinates X and Y")
				  	<<  EAM(mAddSat,"AddSat", "NONE", "Add satellite(s) in process")
				  	<<  EAM(mRemSat,"RemSat", "NONE", "Remove satellite(s) in process")
				  	<<  EAM(mSlice,"Slice", "1", "Rinex observation file down-sampling factor")
				 	<<  EAM(mAux,"Aux", "0", "Auxiliary output file [default: no]")
				  	<<  EAM(mOutRinex,"Out", "excluded_sats.o", "Output rinex file [default: excluded_sats.o]"));
	

	std::string sep =  "-----------------------------------------------------------------------";
	
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "                      GNSS SATELLITES EXCLUSION                       " << std::endl;
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	
	int szsq = 15;   // Taille des symboles

	std::string aName, aDir, aPat, mDir;
	
	StdCorrecNameOrient(mDirOri,mDir);
	
	std::vector<std::string> IMAGES;
	std::vector<std::string> SKY;
	std::vector<GPSTime> TIMESTAMPS;	
	
	double TIME_TOLERANCE_IMCAPTURE_GPS = 0.5;
	if (EAMIsInit(&mTolerance)){
		TIME_TOLERANCE_IMCAPTURE_GPS = std::stod(mTolerance);
	}
	
	// Fichier auxiliaire
	std::ofstream aux;
	bool aux_register = false;
	if (EAMIsInit(&mAux)){
		if (mAux == "1") {
			aux.open ("exclude_sats.aux");
			aux_register = true;
		}
	}
	
	// ---------------------------------------------------------------
	// Liste des satellites utilisables
	// ---------------------------------------------------------------
	std::vector<std::string> SATS = Utils::getSupportedSatellites();
	
		
	// ---------------------------------------------------------------
	// Offset éventuel
	// ---------------------------------------------------------------
	double offset_x = 0.0;
	double offset_y = 0.0;
	
	if (EAMIsInit(&mOffset)){
		mOffset = mOffset.substr(1,mOffset.length()-1);
		std::vector<string> offsets = Utils::tokenize(mOffset, ",");
		offset_x = std::stod(offsets.at(0)); 
		offset_y = std::stod(offsets.at(1));
	}
	
	// ---------------------------------------------------------------
	// Lecture des images
	// ---------------------------------------------------------------
	if (EAMIsInit(&ImPattern)){
		mEASF.Init(ImPattern);
        mICNM = mEASF.mICNM;
        mDir = mEASF.mDir;
	}
	
	size_t N = mEASF.SetIm()->size();
	
	// ---------------------------------------------------------------
	// Dossier de sortie (optionnel)
	// ---------------------------------------------------------------
	if (EAMIsInit(&mPlotFolder)){
		ELISE_fp::MkDir(mPlotFolder);
	}
	
	// ---------------------------------------------------------------
	// Récupération des masques à partir de la clé
	// ---------------------------------------------------------------
	SplitDirAndFile(aDir,aPat,ImPattern);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    for (unsigned i=0; i<N; i++){
		aName = mEASF.SetIm()[0][i];
		IMAGES.push_back(aName);
		std::string mask_name = "";
		if (EAMIsInit(&mMasqKey)){
			std::vector<std::string> aInput = {aName};
			mask_name = aICNM->Direct(mMasqKey,aInput)[0];
		}
		if (ELISE_fp::exist_file(mask_name)){
			SKY.push_back(mask_name);
		}else{
			SKY.push_back("");
		}
	}
	
	// ---------------------------------------------------------------
	// Impression console pour confirmation des inputs
	// ---------------------------------------------------------------
	printf ("Number of images:    %3d \n", (int)(IMAGES.size()));
	printf ("Number of sky masks: %3d \n", (int)(SKY.size()));
	
	std::cout << sep << std::endl;

	std::cout << "Supported satellites: ";
	
	for (unsigned i=0; i<SATS.size(); i++){
		if (i % 16 == 11)  std::cout << std::endl << "  ";
		std::cout << SATS.at(i) << " ";
	}
	
	std::cout << std::endl;
	std::cout << sep << std::endl;
	
	// ---------------------------------------------------------------
	// Ajout et retrait de satellites (optionnels)
	// ---------------------------------------------------------------
	std::vector<std::string> add_sats;
	std::vector<std::string> rem_sats;
	if (EAMIsInit(&mAddSat)) add_sats = Utils::tokenize(mAddSat.substr(1,mAddSat.size()-2), ",");
	if (EAMIsInit(&mRemSat)) rem_sats = Utils::tokenize(mRemSat.substr(1,mRemSat.size()-2), ",");
	
	if (add_sats.size() > 0) std::cout << "Add satellites: ";
	for (unsigned i=0; i<add_sats.size(); i++) std::cout << add_sats.at(i) << " ";
	if (rem_sats.size() > 0) std::cout << "Rem satellites: ";
	for (unsigned i=0; i<rem_sats.size(); i++) std::cout << rem_sats.at(i) << " ";
	
	std::cout << std::endl;
	std::cout << sep << std::endl;
	std::cout << std::endl;

	// ---------------------------------------------------------------
	// Chargement des fichiers d'obs. / nav. rinex et sp3
	// ---------------------------------------------------------------

	// Observations GNSS
	ObservationData obs = RinexReader::readObsFile(mRinexObs); 
	std::cout << std::endl;
	
	// Removing Beidou
	obs.removeConstellation("C");
	
	// Secondes intercalaires
	int leap_seconds = obs.getLeapSeconds();
	
	// Sous-échantillonnage
	if (EAMIsInit(&mSlice)) {
		obs.slice(std::stoi(mSlice));
	}
	
	// Ephémérides
	NavigationDataSet ephemeris;
	
	// GPS
	if (mNavFile.substr(mNavFile.size()-4,4) == ".sp3"){
		ephemeris.loadGpsPreciseEphemeris(mNavFile);
	} else{
		ephemeris.loadGpsEphemeris(mNavFile);
	}
	
	
	// GLONASS
	if (EAMIsInit(&mGloNavFile)) {
		std::cout << std::endl;
		if (mGloNavFile.substr(mGloNavFile.size()-4,4) == ".sp3"){
			ephemeris.loadGlonassPreciseEphemeris(mGloNavFile);
		} else{
			ephemeris.loadGlonassEphemeris(mGloNavFile);
		}
	}
	// GALILEO
	if (EAMIsInit(&mGalNavFile)) {
		std::cout << std::endl;
		if (mGalNavFile.substr(mGalNavFile.size()-4,4) == ".sp3"){
			ephemeris.loadGalileoPreciseEphemeris(mGalNavFile);
		} else{
			ephemeris.loadGalileoEphemeris(mGalNavFile);
		}
	}
	
	std::cout << sep << std::endl;
	std::cout << std::endl;
	
	// ---------------------------------------------------------------
	// Préparation aux changement de système de coordonnées
	// ---------------------------------------------------------------
	std::string code_direct = mSysCode + "@GeoC";
	std::string code_inverse = "GeoC@" + mSysCode;
	cChSysCo * tfdir = cChSysCo::Alloc(code_direct, "");
	cChSysCo * tfinv = cChSysCo::Alloc(code_inverse, "");
	
	// ---------------------------------------------------------------
	// Parcours des images et récupération des dates
	// ---------------------------------------------------------------	
	std::string exiv_cmd = "exiv2 -g Date";
	for (unsigned i=0; i<N; i++){
		
		std::string cmd = exiv_cmd +" "+ IMAGES.at(i);
		GPSTime timestamp = str2GPSTime(execCmdOutput(cmd.c_str()).substr(60,23));
		timestamp = timestamp.addSeconds(leap_seconds);
		TIMESTAMPS.push_back(timestamp);
		
	}	

	// ---------------------------------------------------------------
	// Préparation et récupération des centres
	// ---------------------------------------------------------------
	std::vector<Pt3dr> centres2proj;
	for (unsigned j=0; j<TIMESTAMPS.size(); j++){
		
		// Position du centre de prise de vue
		std::string aNameCam="Ori-"+mDirOri+"/Orientation-"+IMAGES.at(j)+".xml";
		cOrientationConique aOriConique = StdGetFromPCP(aNameCam,OrientationConique);
		Pt3dr pt = aOriConique.Externe().Centre();
		pt.x += offset_x; pt.y += offset_y;
		centres2proj.push_back(pt);
		
	}
	
	std::vector<Pt3dr> CENTRES = tfdir->Src2Cibl(centres2proj);
	
	double avg_dt_error = 0.0;
	int avg_dt_error_cnt = 0;
	
	// ---------------------------------------------------------------
	// Parcours des slots d'observations
	// ---------------------------------------------------------------
	for (unsigned i=0; i<obs.getNumberOfObservationSlots(); i++){
		
		ObservationSlot& slot = obs.getObservationSlots().at(i);
		std::vector<std::string> signal_received; 
		
		// -----------------------------------------------------------
		// Recherche des images de dates "proches"
		// -----------------------------------------------------------
		GPSTime topt = TIMESTAMPS.at(0);
		double current_diff = std::abs(slot.getTimestamp() - topt);
		for (unsigned j=1; j<TIMESTAMPS.size(); j++){
			current_diff =  std::abs(slot.getTimestamp() - TIMESTAMPS.at(j));
			if (current_diff < std::abs(slot.getTimestamp() - topt)){
				topt = TIMESTAMPS.at(j);
			}
		}
		
		// Date de la prise de vue trop lointaine // obs GPS
		double delta_t = std::abs(topt - slot.getTimestamp());
		if (delta_t > TIME_TOLERANCE_IMCAPTURE_GPS) continue;
	
		avg_dt_error += delta_t;
		avg_dt_error_cnt ++;
	
		// -----------------------------------------------------------
		// Au moins une image trouvée
		// -----------------------------------------------------------
		
		// GPS satellites
		std::vector<std::string> temp = slot.getSatellitesConstellation("G");
		for (unsigned nb=0; nb<temp.size(); nb++) signal_received.push_back(temp.at(nb));
		
	
		// Glonass satellites
		if (EAMIsInit(&mGloNavFile)){
			temp = slot.getSatellitesConstellation("R");
			for (unsigned nb=0; nb<temp.size(); nb++) signal_received.push_back(temp.at(nb));
		}
		
		// Galileo satellites
		if (EAMIsInit(&mGalNavFile)){
			temp = slot.getSatellitesConstellation("E");
			for (unsigned nb=0; nb<temp.size(); nb++) signal_received.push_back(temp.at(nb));
		}
		
		// Ajout de satellites
		if (EAMIsInit(&mAddSat)){
			for (unsigned ii=0; ii<add_sats.size(); ii++){
				signal_received.push_back(add_sats.at(ii));
			}
		}	 
		
		// Retrait de satellites
		std::vector<std::string> new_list;
		bool to_be_removed = false;
		for (unsigned ii=0; ii<signal_received.size(); ii++){
			
			// Retrait par l'utilisateur
			for (unsigned jj=0; jj<rem_sats.size(); jj++){
				to_be_removed = (signal_received.at(ii) == rem_sats.at(jj));
				if (to_be_removed) break;
			}
				
			// Retrait par absence d'éphéméride
			to_be_removed = !ephemeris.hasEphemeris(signal_received.at(ii), slot.getTimestamp());
				
			// Retrait
			if (!to_be_removed) new_list.push_back(signal_received.at(ii));
		}
			
		signal_received = new_list;
			
		
		// Pré-calcul éphémérides
		std::vector<ECEFCoords> SAT_POS;
		for (unsigned k=0; k<signal_received.size(); k++){
			std::string prn = signal_received.at(k);
			SAT_POS.push_back(ephemeris.computeSatellitePos(prn, slot.getTimestamp()));  
		} 
		
		int counter = 0;
		int  sat_vis_field = 0;
		int deleted_sats = 0;
		
		// -----------------------------------------------------------
		// Parcours des images séléctionnées
		// -----------------------------------------------------------
		for (unsigned j=0; j<TIMESTAMPS.size(); j++){
			
			// Test de synchronisation GPS <-> caméra
			if (TIMESTAMPS.at(j) - topt !=  0) continue; 
			counter ++;
			
			// Centre de prise de vue
			Pt3dr centre_geoc = CENTRES.at(j);
			ECEFCoords camera_pos(centre_geoc.x, centre_geoc.y, centre_geoc.z);
	
			// Modèle de caméra
			cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
			CamStenope * aCam = CamOrientGenFromFile("Ori-"+mDirOri+"/Orientation-"+IMAGES.at(j)+".xml", anICNM);

			// -----------------------------------------------------
			// Parcours de tous les satellites
			// -----------------------------------------------------
			std::vector<Pt3dr> coords2proj;
			for (unsigned k=0; k<signal_received.size(); k++){
			
				ECEFCoords sat_pos = SAT_POS.at(k);
				
				double factor_height = APPROX_HEIGHT_SAT_TARGETS/sat_pos.distanceTo(camera_pos);
				
				// Réduction distance caméra <-> satellite
				sat_pos.X = centre_geoc.x + factor_height*(sat_pos.X-centre_geoc.x);
				sat_pos.Y = centre_geoc.y + factor_height*(sat_pos.Y-centre_geoc.y);
				sat_pos.Z = centre_geoc.z + factor_height*(sat_pos.Z-centre_geoc.z);
				
				Pt3dr sat(sat_pos.X, sat_pos.Y, sat_pos.Z);
				coords2proj.push_back(sat);
		
			}
			
			// Conversion système véhicule 
			std::vector<int> sat_in_mask;  // [1=ciel, 0=bati et -1=inconnu]
			std::vector<Pt3dr> coords_projected = tfinv->Src2Cibl(coords2proj);
			std::vector<Pt2dr> coords_on_image;
		
			
			// -----------------------------------------------------
			// Cas 1 : le masque de ciel existe
			// -----------------------------------------------------			
			if (SKY.at(j) != ""){
				
				// Récupération du masque
				Tiff_Im aTF = Tiff_Im::StdConvGen(SKY.at(j),3,false);
				Pt2di im_size = aTF.sz();
				int Nx = im_size.x; 
				int Ny = im_size.y;
				Im2D_U_INT1  aImR(Nx,Ny);
   	 			Im2D_U_INT1  aImG(Nx,Ny);
    			Im2D_U_INT1  aImB(Nx,Ny);

				ELISE_COPY(
       				aTF.all_pts(),
       				aTF.in(),
       				Virgule(aImR.out(),aImG.out(),aImB.out())
    			);
				
				
			    U_INT1 ** MASK_R = aImR.data();
				
				// Projection système image 	
				for (unsigned k=0; k<signal_received.size(); k++){
				
					Pt3dr temp = coords2proj.at(k);
					Pt3dr sat_ori = coords_projected.at(k);	
					ECEFCoords sat_ecef(temp.x, temp.y, temp.z);
					sat_ori.x -= offset_x; sat_ori.y -= offset_y;
				
					// Test d'élévation et de visibilité image
					if (camera_pos.elevationTo(sat_ecef) < 0) continue;
					if (!aCam->PIsVisibleInImage(sat_ori)) continue;
		
					
					// Projection dans l'image
					Pt2dr im_point = aCam->Ter2Capteur(sat_ori);
					sat_vis_field ++;
					
					// Test satellite dans le masque de ciel
					coords_on_image.push_back(im_point);
					sat_in_mask.push_back((MASK_R[(int)im_point.y][(int)im_point.x] == 0)?0:1);
					
					// Suppression éventuelle du satellite
					if (sat_in_mask.back() == 0){
						slot.removeSatellite(signal_received.at(k));
						deleted_sats ++;
					}
					
					// Fichier auxiliaire
					if (aux_register){
						aux << slot.getTimestamp() << " " << signal_received.at(k) << " " << IMAGES.at(j) << " ";
						aux << Utils::formatNumber(coords_on_image.back().x,"%8.3f") << " ";
						aux << Utils::formatNumber(coords_on_image.back().y,"%8.3f") << " ";
						aux << sat_in_mask.back() << std::endl;
					}
					
				}
		
			} else {
			
			// -----------------------------------------------------
			// Cas 2 : le masque de ciel n'existe pas
			// -----------------------------------------------------	
		
				// Projection système image 	
				for (unsigned k=0; k<signal_received.size(); k++){
				
					Pt3dr temp = coords2proj.at(k);
					Pt3dr sat_ori = coords_projected.at(k);	
					ECEFCoords sat_ecef(temp.x, temp.y, temp.z);
					sat_ori.x -= offset_x; sat_ori.y -= offset_y;
					
				
					// Test d'élévation et visibilité sur l'image
					if (camera_pos.elevationTo(sat_ecef) < 0) continue;
					if (!aCam->PIsVisibleInImage(sat_ori)) continue;
					
					// Projection sur l'image
					Pt2dr im_point = aCam->Ter2Capteur(sat_ori);
					coords_on_image.push_back(im_point);
					sat_in_mask.push_back(-1);
					sat_vis_field ++;
					
					// Fichier auxiliaire
					if (aux_register){
						aux << slot.getTimestamp() << " " << signal_received.at(k) << " " << IMAGES.at(j) << " ";
						aux << Utils::formatNumber(coords_on_image.back().x,"%8.3f") << " ";
						aux << Utils::formatNumber(coords_on_image.back().y,"%8.3f") << " ";
						aux << sat_in_mask.back() << std::endl;
					}
					
				}
			}
			
			
	
			
			// -----------------------------------------------------
			// Tracé du satellite (optionnel)
			// -----------------------------------------------------
			
			if (EAMIsInit(&mPlotFolder)){
				std::string aNameOut = IMAGES.at(j);
				aNameOut = aNameOut.substr(0, aNameOut.size()-4);
				aNameOut = mPlotFolder + "/" + aNameOut + "_" + slot.getTimestamp().to_formal_string() + "_sats.tif";
				Tiff_Im aTF = Tiff_Im::StdConvGen(IMAGES.at(j),3,false);
				Pt2di im_size = aTF.sz();
				int Nx = im_size.x; 
				int Ny = im_size.y;
				Im2D_U_INT1  aImR(Nx,Ny);
   	 			Im2D_U_INT1  aImG(Nx,Ny);
    			Im2D_U_INT1  aImB(Nx,Ny);
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
				
				for (int iy=0; iy<Ny; iy++) {
        			for (int ix=0; ix<Nx; ix++) {
						aDataROut[iy][ix] = aDataR[iy][ix];
						aDataGOut[iy][ix] = aDataG[iy][ix];
						aDataBOut[iy][ix] = aDataB[iy][ix];
					}
				}
				
				for (unsigned k=0; k<coords_on_image.size(); k++){
					for (int iy=-szsq; iy<=szsq; iy++) {
        				for (int ix=-szsq; ix<=szsq; ix++) {
							int px = coords_on_image.at(k).x;
							int py = coords_on_image.at(k).y;
							int valx = std::min(std::max(px+ix, 0), Nx-1);
							int valy = std::min(std::max(py+iy, 0), Ny-1);
							
							if ((szsq-std::abs(ix) <= 2 ) || (szsq-std::abs(iy) <= 2)){
								aDataROut[valy][valx] = 0;
								aDataGOut[valy][valx] = 0;
								aDataBOut[valy][valx] = 0;
							}else{
								aDataROut[valy][valx] = (sat_in_mask.at(k) ==  0)? 255:0;
								aDataGOut[valy][valx] = (sat_in_mask.at(k) == +1)? 255:0;
								aDataBOut[valy][valx] = (sat_in_mask.at(k) == -1)? 255:0;
							}
						}
					}
				}
				
				
				Pt2di aSzOut = Pt2di(Nx, Ny);
				Tiff_Im aTOut(aNameOut.c_str(), aSzOut, GenIm::u_int1, Tiff_Im::No_Compr, Tiff_Im::RGB);
				ELISE_COPY(
					aTOut.all_pts(), 
					Virgule(aImROut.in(), aImGOut.in(), aImBOut.in()), 
					aTOut.out()
				);
		
			}
			
		}
		
		// Bilan de l'époque GPS
		int pc = sat_vis_field==0?0:deleted_sats*100.0/sat_vis_field;
		std::cout << "GNSS epoch [" << slot.getTimestamp()  << "]: " << counter << " image(s)  dt = " << Utils::formatNumber(delta_t, "%6.3f");
		std::cout << " sec  " << deleted_sats << "/" << sat_vis_field << " [" << Utils::formatNumber(pc, "%2d") << "%] sats removed" << std::endl;		
	
	}
	
	aux.close();
	
	std::cout << sep << std::endl;
	avg_dt_error /= avg_dt_error_cnt;
	int percent = avg_dt_error_cnt*100.0/obs.getNumberOfObservationSlots();
	std::cout << "Number of processed epochs: " << avg_dt_error_cnt << " [" << Utils::formatNumber(percent, "%5.2f") << " %]" << std::endl; 
	std::cout << "Mean GNSS-camera synchronization error: " << Utils::formatNumber(avg_dt_error, "%6.3f") << " sec" << std::endl; 
	
	// Ecritude du fichier rinex de sortie
	std::vector<std::string> split_name_rinex = Utils::tokenize(mRinexObs, ".");
	std::string out_file_name = split_name_rinex.at(0) + "_sky_masked." + split_name_rinex.at(1);
	if (EAMIsInit(&mOutRinex)){
		out_file_name = mOutRinex;
	}
	
	std::cout << sep << std::endl;
	obs.printRinexFile(out_file_name);
	std::cout << sep << std::endl;
	
}




// --------------------------------------------------------------------------------------
// Script de détection de masque de ciel
// Appel du code python de neural net (cf Imran Lokhat)
// L'installeur ne fonctionne pour l'instant que sous Linux (problème avec appel cmd)
// --------------------------------------------------------------------------------------
cAppli_YannSkyMask::cAppli_YannSkyMask(int argc, char ** argv){
	
	
	 ElInitArgMain(argc,argv,
        LArgMain()  <<  EAMC(ImPattern,"Image pattern")
				    <<  EAMC(outDir,"Output folder"),
        LArgMain()  <<  EAM(mProba,"Prob", "0", "Probabilistic prediction")
				    <<  EAM(mThresh,"Thresh", "-1", "Decision threshold")
				    <<  EAM(mInv,"Inv", "0", "Inverse prediction")
				   	<<  EAM(mFilter,"Filter", "0", "Filtering (artifacts relative size)")
				 	<<  EAM(mInstall,"Install", "0", "Install neural net program")
				    <<  EAM(mUninstall,"Uninstall", "0", "Uninstall neural net program"));
	
	// Main base directories
	std::string root_folder = "~/.local/share/mm3d";
	std::string skymask_folder = root_folder+"/skymask";
	std::string nn_location = skymask_folder;
	
	// Python requirements
	std::vector<std::string> REQ = {"tensorflow", "Keras", "albumentations", "segmentation-models"};
	std::vector<std::string> VER = {"2.2", "2.3.0", "0.4.6", "1.0.1"};
	//std::vector<std::string> VER = {"1.13.2", "2.3.0", "0.4.6", "1.0.1"};
	
	
	
	if (EAMIsInit(&mUninstall)){
		std::cout << "Uninstalling SkyMask...";
		System("rm -r "+skymask_folder);
		std::cout << "done" << std::endl;
		return;
	}
	
	if (EAMIsInit(&mInstall)){
		
		std::cout << "Installing SkyMask..." << std::endl;
		
		// Downloading scripts
		System("mkdir -p "+skymask_folder);
		System("wget http://recherche.ign.fr/labos/cogit/demo/skymask/predict.py");
		System("wget http://recherche.ign.fr/labos/cogit/demo/skymask/best_model.h5");
		System("wget http://recherche.ign.fr/labos/cogit/demo/skymask/requirements.txt");
		System("wget http://recherche.ign.fr/labos/cogit/demo/skymask/readme");
		
		System("mv predict.py "+skymask_folder);
		System("mv best_model.h5 "+skymask_folder);
		System("mv requirements.txt "+skymask_folder);
		System("mv readme "+skymask_folder);
		
		// Installing python library
		std::string answer;
		for (unsigned i=0; i<REQ.size(); i++){
			std::cout << "Install "+REQ.at(i)+" "+VER.at(i)+" ? [O/N] ";
			std::cin >> answer;
			if ((answer == "O") || (answer == "o")){
				std::cout << "Installing "+REQ.at(i) << std::endl;
				std::string cmd = "pip3 install "+ REQ.at(i)+"=="+VER.at(i);
				System(cmd);
			}
			std::cout << std::endl;
		}
		
		// Down-grading h5py to earlier version
		System("pip install 'h5py==2.10.0' --force-reinstall");
		
		return;
		
	}

	std::vector<std::string> IMAGES;

	if (EAMIsInit(&ImPattern)){
		mEASF.Init(ImPattern);
		mICNM = mEASF.mICNM;
		mDir = mEASF.mDir;
	}

	if (!EAMIsInit(&mInv)) mInv = "0";
	if (!EAMIsInit(&mProba)) mProba = "0";
	if (!EAMIsInit(&mThresh)) mThresh = "-1";
	if (!EAMIsInit(&mFilter)) mFilter = "0";

	size_t N = mEASF.SetIm()->size();

	for (unsigned i=0; i<N; i++){
		IMAGES.push_back(mEASF.SetIm()[0][i]);
	}

	// Copie temporaire des images
	std::cout << "------------------------------------------" << std::endl;
	std::cout << "Downsizing images " << std::endl;
	std::cout << "------------------------------------------" << std::endl;


	std::string raw_dir = nn_location+"/raw_images";
	std::string temp_output_folder = nn_location+"/output";

	System("rm -r -f "+raw_dir);
	System("mkdir "+raw_dir);

	System("rm -r -f "+temp_output_folder);
	System("mkdir "+temp_output_folder);

	ELISE_fp::MkDir(outDir);

	std::string cmd;
	std::vector<std::string> SIZES_X;
	std::vector<std::string> SIZES_Y;

	for (unsigned i=0; i<N; i++){
		std::string im_size = execCmdOutput(("identify "+IMAGES.at(i)).c_str());
		std::string dim = Utils::tokenize(im_size," ").at(2);
		std::vector<std::string> dims = Utils::tokenize(dim, "x");
		SIZES_X.push_back(dims.at(0)); 
		SIZES_Y.push_back(dims.at(1));
		std::cout << IMAGES.at(i) << " [" << dims.at(0) << " x " << dims.at(1) << "]" << std::endl;
		cmd = "convert -resize 480x360! "+IMAGES.at(i)+" "+raw_dir+"/"+IMAGES.at(i)+".png";
		System(cmd);
	}


	// Appel du réseau de neurones
	std::cout << "------------------------------------------" << std::endl;
	std::cout << "Sky estimation with neural network" << std::endl;
	std::cout << "------------------------------------------" << std::endl;

	System("cd "+nn_location+" && python predict.py -i raw_images/ -o output/ -p "+mProba+" -r "+mInv+" -t"+mThresh + " -f "+mFilter);

	// Rappatriemment des images
	std::cout << "------------------------------------------" << std::endl;
	std::cout << "Upsizing images " << std::endl;
	std::cout << "------------------------------------------" << std::endl;
	for (unsigned i=0; i<N; i++){
		std::cout << IMAGES.at(i)  << std::endl;
		std::string x_px = ""+SIZES_X.at(i);
		std::string y_px = ""+SIZES_Y.at(i);
		cmd = "convert -compress None -resize "+x_px+"x"+y_px+"! "+temp_output_folder+"/"+IMAGES.at(i)+".png "+outDir+"/"+IMAGES.at(i);
		System(cmd);
	}

	System("rm "+raw_dir+"/*");
	System("rm "+temp_output_folder+"/*");
		
}




// --------------------------------------------------------------------------------------
// Script de test
// --------------------------------------------------------------------------------------
cAppli_YannScript::cAppli_YannScript(int argc, char ** argv){
	
	
	
	ElInitArgMain(argc,argv,
        LArgMain()  <<  EAMC(ImPattern,"Rinex rover station observation file")
				    <<  EAMC(aPostIn,"Rinex navigation file"),
        LArgMain()  <<  EAM(mOut,"Ref", "file.o", "Rinex base station observation file"));
	
	
	ObservationData rover = RinexReader::readObsFile(ImPattern);
    ObservationData base = RinexReader::readObsFile(mOut);
    NavigationData nav  = RinexReader::readNavFile(aPostIn);

    rover.removeSatellite("G24");
    base.removeSatellite("G24");
    base.removeSatellite("G10");
    base.removeSatellite("G19");

    NavigationDataSet eph = NavigationDataSet();
    eph.addGpsEphemeris(nav);

    Solution solution = Algorithms::triple_difference_kalman(rover, base, eph, base.getApproxAntennaPosition());

    std::cout << "------------------------------------------------------------------------------" << std::endl;
    std::cout << rover.getApproxAntennaPosition() << std::endl;
    std::cout << solution.getPosition() - rover.getApproxAntennaPosition() << std::endl;
    std::cout << "------------------------------------------------------------------------------" << std::endl;


	/*
	std::string prn;
	std::string dec;
	std::string mDateIni;
	std::string mDateFin;
	
	 ElInitArgMain(argc,argv,
        LArgMain()  <<  EAMC(ImPattern,"Navigation file (nav/sp3)")
				    <<  EAMC(prn,"Satellite PRN")
				    <<  EAMC(mDateIni,"Initial date [dd/mm/yyyy hh:mm:ss.msc]")
				    <<  EAMC(mDateFin,"Final date [dd/mm/yyyy hh:mm:ss.msc]"),
        LArgMain()  <<  EAM(dec, "Dec", "1", "Time interval"));
	
	
	NavigationData nav;
	SP3NavigationData nav_sp3;
	
	std::string ext = ImPattern.substr(ImPattern.size()-3, ImPattern.size()-1);
	if (ext == "sp3"){
		nav_sp3 = SP3Reader::readNavFile(ImPattern);  
	} else{
		nav = RinexReader::readNavFile(ImPattern);  
	}
	
	GPSTime time = str2GPSTime(mDateIni);
	GPSTime end  = str2GPSTime(mDateFin);
	
	double interval = 1;
	if (EAMIsInit(&dec)) interval = std::stof(dec);
	
	ECEFCoords pos;
									   
	while (time <= end){
		
		if (ext == "sp3"){
			pos	= nav_sp3.computeSatellitePos(prn, time);
		} else{
			pos	= nav.computeSatellitePos(prn, time);
		}
		std::cout << prn << " " << time << " " << pos << std::endl;
		time = time.addSeconds(interval);
	}
	
	*/
	
	
	
	/*
	
	 ElInitArgMain(argc,argv,
        LArgMain()  <<  EAMC(ImPattern,"Rinex rover station observation file")
				    <<  EAMC(aPostIn,"Rinex navigation file"),
        LArgMain()  <<  EAM(mOut,"Ref", "file.o", "Rinex base station observation file"));
	
	

	
	
	
	ObservationData rover = RinexReader::readObsFile(ImPattern);
	NavigationData nav    = RinexReader::readNavFile(aPostIn);  
	
	Solution solution;
	ObservationData base;
	std::vector<ECEFCoords> position;
	
	ECEFCoords ground_truth = rover.getApproxAntennaPosition();
	
	
	if (EAMIsInit(&mOut)){
		base = RinexReader::readObsFile(mOut); 
	}	
	
	std::vector<GeoCoords> trajectory;

	for (unsigned i=0; i<rover.getNumberOfObservationSlots(); i++){

		ObservationSlot roverSlot = rover.getObservationSlots().at(i);
		
		if (EAMIsInit(&mOut)){
			ObservationSlot baseSlot  =  base.getObservationSlots().at(i);
			solution = Algorithms::estimateDifferentialPosition(roverSlot, baseSlot, base.getApproxAntennaPosition(), nav);
			
			int nb_sat = static_cast<int> (solution.getNumberOfUsedSatellites());
			
			std::cout << roverSlot.getTimestamp() << " " << solution.getPosition().toGeoCoords() << " ";
			std::cout << Utils::formatNumber(solution.getPosition().distanceTo(ground_truth), "%6.3f")  << " ";
			std::cout << Utils::formatNumber(nb_sat, "%02d sats") << " ";
			std::cout << Utils::formatNumber(solution.getPDOP(), "%3.2f") << " ";
			std::cout << Utils::formatNumber(solution.getDeltaTime()*1e9, "%3.1f ns") << std::endl;
			
		} else{
			solution = Algorithms::estimateState(roverSlot, nav);
			std::cout << solution << " " << solution.getDeltaTime()*1e9 << " ns ";
			std::cout << solution.getClockDrift()*1e6 << " us/s" << std::endl;
		}

		
		trajectory.push_back(solution.getPosition().toGeoCoords());
		position.push_back(solution.getPosition());

	}

	std::cout << "Mean position: " << Statistics::mean(position).toENUCoords(ground_truth) << std::endl;
	std::cout << "Std pos: " << Statistics::sd(position) << std::endl;
	std::cout << "Mean position error: " << Statistics::mean(position).distanceTo(ground_truth) << std::endl;
	std::cout << "Mean pos: " << Statistics::mean(position).toGeoCoords() << std::endl; 
	
		
	//std::cout << GeoCoords::makeWKT(trajectory) << std::endl;
	*/
	
	

	
	/*
	ElInitArgMain(argc,argv,
        LArgMain()  <<  EAMC(ImPattern,"Rinex rover station observation file")
				    <<  EAMC(aPostIn,"Rinex navigation file"),
        LArgMain()  <<  EAM(mOut,"Ref", "file.o", "Rinex base station observation file"));
	
	
	ObservationData obs1 = RinexReader::readObsFile(ImPattern);
    ObservationData obs2 = RinexReader::readObsFile(aPostIn);

    for (int i=0; i<obs1.getNumberOfObservationSlots(); i++){

        ObservationSlot slot1 = obs1.getObservationSlots().at(i);
        ObservationSlot slot2 = obs2.getObservationSlots().at(i);

        if (!(slot1.getTimestamp() == slot2.getTimestamp())){
            std::cout << "Error: rinex files are not synchronized [" << slot1.getTimestamp() << "]" << std::endl;
            break;
        }

        std::vector<std::string> SAT1 = slot1.getSatellites();
        std::vector<std::string> SAT2 = slot2.getSatellites();

        int N1 = SAT1.size();
        int N2 = SAT2.size();

        int G1 = 0; int R1 = 0; int E1 = 0;
        int G2 = 0; int R2 = 0; int E2 = 0;

        for (unsigned j=0; j<SAT1.size(); j++){
            if (SAT1.at(j).substr(0,1) == "G") G1++;
            if (SAT1.at(j).substr(0,1) == "R") R1++;
            if (SAT1.at(j).substr(0,1) == "E") E1++;
        }

        for (unsigned j=0; j<SAT2.size(); j++){
            if (SAT2.at(j).substr(0,1) == "G") G2++;
            if (SAT2.at(j).substr(0,1) == "R") R2++;
            if (SAT2.at(j).substr(0,1) == "E") E2++;
        }

        std::cout << slot1.getTimestamp() << " "<< N1 << " " << N2 << " ";
        std::cout << G1 << " " << G2 << " ";
        std::cout << R1 << " " << R2 << " ";
        std::cout << E1 << " " << E2 << " ";
        std::cout << std::endl;
		
	}
	*/

	
	
	// --------------------------------------------------------------------------------
	// Test décompte des points homologues
	// --------------------------------------------------------------------------------
	/*
	 ElInitArgMain(argc,argv,
        LArgMain()  <<  EAMC(ImPattern,"Image Pattern")
				    <<  EAMC(aPostIn,"Homol postfix"),
        LArgMain()  <<  EAM(mOut,"Out", "file.o", "Output file"));
	
	
	cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
	
	std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(aPostIn)
                       +  std::string("@")
                       +  std::string("dat");
	
	std::string aDir;
	std::string aPat;
	SplitDirAndFile(aDir,aPat,ImPattern);
	
	const std::vector<std::string> *  aVN = anICNM->Get(ImPattern);
	for (int aKN1=0; aKN1<int(aVN->size()); aKN1++){
		int count = 0;
		std::string aNameIm1 = (*aVN)[aKN1];
		for (int aKN2 = 0 ; aKN2<int(aVN->size()); aKN2++){
            
            std::string aNameIm2 = (*aVN)[aKN2];
			std::string aNameIn = aDir + anICNM->Assoc1To2(aKHIn,aNameIm1,aNameIm2,true);
			bool ExistFileIn =  ELISE_fp::exist_file(aNameIn);
			
			if (ExistFileIn){
				ElPackHomologue aPackIn =  ElPackHomologue::FromFile(aNameIn);
				for (ElPackHomologue::const_iterator itP=aPackIn.begin(); itP!=aPackIn.end() ; itP++){
					count ++;
				}
			}
		}
		std::cout << aNameIm1 << " " << count << std::endl;
	}
	
	*/
	
}



// ----------------------------------------------------------------------------
//  A chaque commande MicMac correspond une fonction ayant la signature du main
//  Le lien se fait dans src/CBinaires/mm3d.cpp
// ----------------------------------------------------------------------------

// (1) Complétion des timestamps
int CPP_YannSetTimestamps(int argc,char ** argv){
   cAppli_YannSetTimestamps(argc,argv);
   return EXIT_SUCCESS;
}


// (2) Exclusion des satellites
int CPP_YannExcludeSats(int argc,char ** argv){
   cAppli_YannExcludeSats(argc,argv);
   return EXIT_SUCCESS;
}

// (3) Fonctions de détection de ciel
int CPP_YannSkyMask(int argc,char ** argv){
   cAppli_YannSkyMask(argc,argv);
   return EXIT_SUCCESS;
}

// (4) Fonctions de test 
int CPP_YannScript(int argc,char ** argv){
   cAppli_YannScript(argc,argv);
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
