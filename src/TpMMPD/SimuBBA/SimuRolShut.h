#ifndef SIMUROLSHUT_H
#define SIMUROLSHUT_H

#include "StdAfx.h"
#include "../spline.h"
#include "../../uti_files/CPP_XifGps2Xml.h"
#include "../../uti_phgrm/TiepTri/TiepTri.h"
#include "../../uti_phgrm/TiepTri/MultTieP.h"

typedef pair<std::string,std::string> Key;

class cAppli_CamXifDate;

class cIm_CamXifDate : public cIm_XifDate
{
    public:
        cIm_CamXifDate(cInterfChantierNameManipulateur* aICNM, const std::string & aName, const std::string & aOri, cElHour & aBeginTime, const Pt3d<double> &aVitesse);

        CamStenope * mCam;
        Pt3dr        mVitesse;
};

class cAppli_CamXifDate : public cAppli_XifDate
{
    public:
        cAppli_CamXifDate(const std::string & aFullName, std::string & aOri, const std::string &aCalcV);
        std::vector<Pt3dr> GetCamCenter();
        std::vector<double> GetCamCenterComponent(int a);
        CamStenope * Cam(const std::string & aName);
        const Pt2di CamSz() const;
//        CamStenope * GetNewCam(const std::string &aName,double aTime);

        std::map<std::string,cIm_CamXifDate> mVIm;
        std::string                          mOri;
        tk::spline                           mS_x;
        tk::spline                           mS_y;
        tk::spline                           mS_z;
};

class cSetTiePMul_Cam
{
    public:
        cSetTiePMul_Cam(const std::string &aSH,const cAppli_CamXifDate & anAppli);
        void ReechRS_SH(const double & aRSSpeed, const std::string &aSHOut);
        void TestOri(double aTimeEcart);
    private:
        cSetTiePMul *             m_pSH;
        std::string               mSH;
        const cAppli_CamXifDate   m_Appli;
};

class cPtIm_CamXifDate
{
    public:
        cPtIm_CamXifDate(Pt2dr &aPtIm, cIm_CamXifDate &aIm_CamXifDate);

        Pt2dr mPtIm;
        const cIm_CamXifDate mIm_CamXifDate;
};

class cSetOfMesureAppuisFlottants_Cam
{
    public:
        cSetOfMesureAppuisFlottants_Cam(const std::string &aMAFIn,const cAppli_CamXifDate & anAppli);
        std::map<Key, Pt2dr> ReechRS_MAF(const double aRSSpeed); //<pair<ImName,PtName>,Pt2dr>
        void Export_MAF(const std::string & aMAFOut, const std::map<Key,Pt2dr> &aMap);
    private:
        const cAppli_CamXifDate   m_Appli;
        cSetOfMesureAppuisFlottants mDico;
        std::map<std::string,std::vector<cPtIm_CamXifDate>> mVPtIm; //<PtName,std::vector<cPtIm_CamXifDate>>
};


#endif // SIMUROLSHUT_H

