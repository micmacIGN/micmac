#ifndef SIMUROLSHUT_H
#define SIMUROLSHUT_H

#include "StdAfx.h"
#include "../spline.h"
#include "../../uti_files/CPP_XifGps2Xml.h"
#include "../../uti_phgrm/TiepTri/TiepTri.h"
#include "../../uti_phgrm/TiepTri/MultTieP.h"


class cAppli_CamXifDate;

class cIm_CamXifDate : public cIm_XifDate
{
    public:
        cIm_CamXifDate(cInterfChantierNameManipulateur* aICNM,const std::string & aName, const std::string & aOri, cElHour & aBeginTime);

        CamStenope * mCam;
};

class cAppli_CamXifDate : public cAppli_XifDate
{
    public:
        cAppli_CamXifDate(const std::string & aFullName, std::string & aOri);
        std::vector<Pt3dr> GetCamCenter();
        std::vector<double> GetCamCenterComponent(int a);
        CamStenope * Cam(const std::string & aName);
//        CamStenope * GetNewCam(const std::string &aName,double aTime);

        std::map<std::string,cIm_CamXifDate> mVIm;
        tk::spline mS_x;
        tk::spline mS_y;
        tk::spline mS_z;
};

class cSetPMul_Cam
{
    public:
        cSetPMul_Cam(cSetTiePMul * pSH,cSetPMul1ConfigTPM & aCnf,cAppli_CamXifDate & anAppli);
        void Reproj(); // pseudo_intersect + reproj on cam with rolshut correction

    private:
        std::vector<CamStenope *> mVCam;
        cSetPMul1ConfigTPM        mCnf;
        cSetTiePMul *             m_pSH;
};

#endif // SIMUROLSHUT_H

