#include "api_mm3d.h"
#include <iostream>

extern bool TheExitOnBrkp;
void mm3d_init()
{
	TheExitOnBrkp =true;
	std::cout<<"mm3d initialized."<<std::endl;
}


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
    ELISE_fp::MkDirSvp(oriName);
    MakeFileXML(anIdealCam.StdExportCalibGlob(),oriName+"/Orientation-" + NameWithoutDir(oriName) + imgName + ".tif.xml","MicMacForAPERO");
}

ElRotation3D list2rot(std::vector<double> l)
{
    ElMatrix<REAL> mat(3,3);
    int k=0;
    for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
        {
            mat(i,j)=l.at(k);
            k++;
        }
    
    ElRotation3D r(Pt3dr(),mat,false);
    return r;
}

std::vector<double> rot2list(ElRotation3D &r)
{
    std::vector<double> l;
    l.resize(9);
    int k=0;
    ElMatrix<REAL> mat=r.Mat();
    for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
        {
            l[k]=mat(i,j);
            k++;
        }
    
    return l;
}

ElRotation3D quaternion2rot(double a, double b, double c, double d)
{
    double n=sqrt(a*a+b*b+c*c+d*d);
    std::cout<<"Quat norm: "<<n<<"\n";
    a/=n;b/=n;c/=n;d/=n;
	ElMatrix<REAL> mat(3,3);
	mat(0,0)=a*a+b*b-c*c-d*d;
	mat(0,1)=2*b*c-2*a*d;
	mat(0,2)=2*a*c+2*b*d;
	mat(1,0)=2*a*d+2*b*c;
	mat(1,1)=a*a-b*b+c*c-d*d;
	mat(1,2)=2*c*d-2*a*b;
	mat(2,0)=2*b*d-2*a*c;
	mat(2,1)=2*a*b+2*c*d;
	mat(2,2)=a*a-b*b-c*c+d*d;
	ElRotation3D r(Pt3dr(),mat,false);
	return r;
}

std::vector<std::string> getFileSet(std::string dir, std::string pattern)
{
    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(dir);
    return *(aICNM->Get(pattern));
}
