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

class cSetTiePMul_Cam
{
    public:
        cSetTiePMul_Cam(const std::string &aSH,const cAppli_CamXifDate & anAppli);
        void Reech_RS(const double & aRSSpeed, const std::string &aSHOut);
    private:
        cSetTiePMul               m_SetTiePMul;
        cSetTiePMul *             m_pSH;
        const cAppli_CamXifDate   m_Appli;
};

#endif // SIMUROLSHUT_H

