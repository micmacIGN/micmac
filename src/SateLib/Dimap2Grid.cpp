#include "StdAfx.h"
#include "RPC.h"

int Dimap2Grid_main(int argc, char **argv)
{
    std::string aNameFile, aNameIm; // RPC Dimap .xml file and image associated
	std::string inputSyst = "+proj=longlat +datum=WGS84"; //input syst proj4
	std::string targetSyst;//output syst proj4
	std::string refineCoef = "";
	bool binaire = true;
	double altiMin, altiMax;
    int nbLayers;

    double stepPixel = 100.f;
    double stepCarto = 50.f;

	ElInitArgMain
		(
		argc, argv,
		LArgMain() << EAMC(aNameFile, "RPC Dimap file")
		<< EAMC(aNameIm, "Name of image (to generate appropriatelly named GRID file)")
		<< EAMC(altiMin, "min altitude (ellipsoidal)")
		<< EAMC(altiMax, "max altitude (ellipsoidal)")
		<< EAMC(nbLayers, "number of layers (min 4)")
		<< EAMC(targetSyst, "targetSyst - target system in Proj4 format (ex : \"+proj=utm +zone=32 +north +datum=WGS84 +units=m +no_defs\""),
		LArgMain()
		//caracteristique du systeme geodesique saisies sans espace (+proj=utm +zone=10 +north +datum=WGS84...)
		<< EAM(stepPixel, "stepPixel", true, "Step in pixel (Def=100pix)")
		<< EAM(stepCarto, "stepCarto", true, "Step in m (carto) (Def=50m)")
		<< EAM(refineCoef, "refineCoef", true, "File of Coef to refine Grid")
		<< EAM(binaire, "Bin", true, "Export Grid in binaries (Def=True)")
		);

	//Reading Dimap and setting up RPC object
	RPC aRPC;
	aRPC.ReadDimap(aNameFile);
	cout << "Dimap File read" << endl;
	aRPC.info();

	//Computing Grid
	aRPC.RPC2Grid(nbLayers, altiMin, altiMax, refineCoef, aNameIm, stepPixel, stepCarto, targetSyst, inputSyst, binaire);

    return EXIT_SUCCESS;
}

