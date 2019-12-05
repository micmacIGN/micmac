#include "api_mm3d.h"
#include <iostream>

CamStenope * CamOrientFromFile(std::string filename)
{
	cElemAppliSetFile anEASF(filename);
	return CamOrientGenFromFile(filename,anEASF.mICNM);
}

void createIdealCamXML(double focale, Pt2dr aPP, Pt2di aSz, std::string oriName, std::string imgName, std::string idCam, ElRotation3D &orient, double prof=0, double rayonUtile=0)
{
    std::vector<double> paramFocal;
    cCamStenopeDistPolyn anIdealCam(true,focale,aPP,ElDistortionPolynomiale::DistId(3,1.0),paramFocal);
    if (prof!=0)
        anIdealCam.SetProfondeur(prof);
    anIdealCam.SetSz(aSz);
    anIdealCam.SetIdentCam(idCam);
    if (rayonUtile>0)
        anIdealCam.SetRayonUtile(rayonUtile,30);
    anIdealCam.SetOrientation(orient);
    anIdealCam.SetIncCentre(Pt3dr(1,1,1));

    MakeFileXML(anIdealCam.StdExportCalibGlob(),oriName+"/Orientation-" + NameWithoutDir(oriName) + imgName + ".tif.xml","MicMacForAPERO");
}


