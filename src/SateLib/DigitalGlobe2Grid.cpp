#include "StdAfx.h"
#include "RPC.h"

int DigitalGlobe2Grid_main(int argc, char **argv)
{
    std::string aNameFile; // .RPB file from Digital Globe
    std::string inputSyst = "+proj=longlat +datum=WGS84 "; //input syst proj4
    std::string targetSyst;//output syst proj4
	std::string refineCoef="";
    bool binaire = true;
    double altiMin, altiMax;
    int nbLayers;

    double stepPixel = 100.f;
    double stepCarto = 50.f;

    ElInitArgMain
        (
        argc, argv,
        LArgMain() << EAMC(aNameFile, "RPB from DigitalGlobe file")
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

    //Reading Inverse RPC, computing Direct RPC and setting up RPC object
    RPC aRPC;
    aRPC.ReadRPB(aNameFile);
    cout << "RPB File read" << endl;

	//Generating a 50*50*50 grid on the normalized space with random normalized heights
	Pt3di aGridSz(50, 50, 50);
	vector<Pt3dr> aGridGeoNorm = aRPC.GenerateNormGrid(aGridSz);//50 is the size of grid for generated GCPs (50*50)

	//Converting the points to image space
	vector<Pt3dr> aGridImNorm;
	for (u_int i = 0; i < aGridGeoNorm.size(); i++)
	{
		aGridImNorm.push_back(aRPC.InverseRPCNorm(aGridGeoNorm[i]));
	}
	
	aRPC.GCP2Direct(aGridGeoNorm, aGridImNorm);
    cout << "Direct RPC estimated" << endl;
    aRPC.ReconstructValidity();
    aRPC.info();



    //Computing Grid
    std::string aNameIm = StdPrefix(aNameFile) + ".TIF";
    aRPC.RPC2Grid(nbLayers, altiMin, altiMax, refineCoef, aNameIm, stepPixel, stepCarto, targetSyst, inputSyst, binaire);


    return EXIT_SUCCESS;
}

